"""Document model: image + masks + undo/redo command history.

All state lives in full-resolution numpy buffers; the UI keeps its own
display-only preview. Commands store bounding-box patches only, so history
memory stays bounded across many strokes and removals.
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from .config import HISTORY_LIMIT
from .image_io import WorkingImage

MASK_REMOVAL = "removal"
MASK_PROTECT = "protect"
MASK_IDS = (MASK_REMOVAL, MASK_PROTECT)


class Command(Protocol):
    description: str

    def undo(self) -> None: ...

    def redo(self) -> None: ...


class History:
    """Undo/redo stacks with a soft memory cap."""

    def __init__(self, limit: int = HISTORY_LIMIT) -> None:
        self._undo: list[Command] = []
        self._redo: list[Command] = []
        self._limit = limit

    def push(self, command: Command) -> None:
        self._undo.append(command)
        if len(self._undo) > self._limit:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self) -> bool:
        if not self._undo:
            return False
        cmd = self._undo.pop()
        cmd.undo()
        self._redo.append(cmd)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        cmd = self._redo.pop()
        cmd.redo()
        self._undo.append(cmd)
        return True

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()


class MaskLayer:
    """A full-resolution 8-bit coverage mask (0..255)."""

    def __init__(self, height: int, width: int) -> None:
        self.data = np.zeros((height, width), dtype=np.uint8)

    def clear(self) -> None:
        self.data.fill(0)

    def bbox(self) -> tuple[int, int, int, int] | None:
        """(x0, y0, x1, y1) half-open, or None when empty."""
        return mask_bbox(self.data)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# --- brush stroke math ------------------------------------------------------

def stamp_segment(
    mask: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    radius: float,
    hardness: float,
    erase: bool,
    out_bbox: list[int] | None = None,
) -> tuple[int, int, int, int]:
    """Stamp a soft round brush along segment p0->p1 into mask.

    hardness 0..1: fraction of the radius that is fully opaque, smoothstep
    falloff to zero at the rim. Returns the dirty bbox (x0, y0, x1, y1).
    """
    h, w = mask.shape
    r = float(max(1.0, radius))
    x0 = int(max(0, np.floor(min(p0[0], p1[0]) - r - 1)))
    y0 = int(max(0, np.floor(min(p0[1], p1[1]) - r - 1)))
    x1 = int(min(w, np.ceil(max(p0[0], p1[0]) + r + 2)))
    y1 = int(min(h, np.ceil(max(p0[1], p1[1]) + r + 2)))
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0)

    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    ax, ay = float(p0[0]), float(p0[1])
    bx, by = float(p1[0]), float(p1[1])
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 <= 1e-9:
        dist = np.hypot(xs - ax, ys - ay)
    else:
        t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / len2, 0.0, 1.0)
        dist = np.hypot(xs - (ax + t * dx), ys - (ay + t * dy))

    hardness = float(min(1.0, max(0.0, hardness)))
    inner = r * hardness
    feather = max(r - inner, 1e-6)
    alpha = np.clip((r - dist) / feather, 0.0, 1.0)
    alpha = np.where(dist <= inner, 1.0, alpha)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep

    region = mask[y0:y1, x0:x1]
    if erase:
        region[:] = np.rint(region.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    else:
        np.maximum(region, np.rint(alpha * 255.0).astype(np.uint8), out=region)

    bbox = (x0, y0, x1, y1)
    if out_bbox is not None:
        _merge_bbox(out_bbox, bbox)
    return bbox


def _merge_bbox(base: list[int], other: tuple[int, int, int, int]) -> None:
    if base[2] <= base[0] or base[3] <= base[1]:
        base[:] = list(other)
    else:
        base[0] = min(base[0], other[0])
        base[1] = min(base[1], other[1])
        base[2] = max(base[2], other[2])
        base[3] = max(base[3], other[3])


# --- commands ---------------------------------------------------------------

class StrokeCommand:
    """One brush stroke on one mask (paint or erase)."""

    description = "brush stroke"

    def __init__(
        self,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        before: np.ndarray,
        after: np.ndarray,
    ) -> None:
        self._mask = mask
        self._bbox = bbox
        self._before = before
        self._after = after

    def undo(self) -> None:
        x0, y0, x1, y1 = self._bbox
        self._mask[y0:y1, x0:x1] = self._before

    def redo(self) -> None:
        x0, y0, x1, y1 = self._bbox
        self._mask[y0:y1, x0:x1] = self._after


class StrokeBuilder:
    """Accumulates brush stamps while capturing the before-state lazily.

    The before-patch grows with the stroke's bounding box; newly covered rows/
    columns are copied from the mask *before* the current segment is stamped,
    so undo restores the exact pre-stroke state at bounded memory cost.
    """

    def __init__(
        self,
        mask: np.ndarray,
        radius: float,
        hardness: float,
        erase: bool,
    ) -> None:
        self._mask = mask
        self._radius = radius
        self._hardness = hardness
        self._erase = erase
        self._bbox: list[int] = [0, 0, 0, 0]
        self._before: np.ndarray | None = None
        self._last: tuple[float, float] | None = None
        self._dirty = False

    def add_point(self, x: float, y: float) -> tuple[int, int, int, int]:
        p1 = (float(x), float(y))
        p0 = self._last if self._last is not None else p1
        self._capture_before(p0, p1)
        bbox = stamp_segment(
            self._mask, p0, p1, self._radius, self._hardness, self._erase
        )
        self._last = p1
        self._dirty = True
        return bbox

    def _capture_before(self, p0: tuple[float, float], p1: tuple[float, float]) -> None:
        r = float(max(1.0, self._radius)) + 2.0
        x0 = int(max(0, np.floor(min(p0[0], p1[0]) - r)))
        y0 = int(max(0, np.floor(min(p0[1], p1[1]) - r)))
        x1 = int(min(self._mask.shape[1], np.ceil(max(p0[0], p1[0]) + r + 1)))
        y1 = int(min(self._mask.shape[0], np.ceil(max(p0[1], p1[1]) + r + 1)))
        if x1 <= x0 or y1 <= y0:
            return
        if self._before is None:
            self._before = self._mask[y0:y1, x0:x1].copy()
            self._bbox = [x0, y0, x1, y1]
            return
        bx0, by0, bx1, by1 = self._bbox
        nx0, ny0 = min(bx0, x0), min(by0, y0)
        nx1, ny1 = max(bx1, x1), max(by1, y1)
        if (nx0, ny0, nx1, ny1) == (bx0, by0, bx1, by1):
            return  # fully covered already
        # Snapshot the ring before this segment touches it: copy the union
        # slice from the mask, then restore the already-captured core.
        ring = self._mask[ny0:ny1, nx0:nx1].copy()
        ring[by0 - ny0 : by1 - ny0, bx0 - nx0 : bx1 - nx0] = self._before
        self._before = ring
        self._bbox = [nx0, ny0, nx1, ny1]

    def finish(self) -> StrokeCommand | None:
        if not self._dirty or self._before is None:
            return None
        x0, y0, x1, y1 = self._bbox
        after = self._mask[y0:y1, x0:x1].copy()
        return StrokeCommand(self._mask, (x0, y0, x1, y1), self._before, after)


class ClearMasksCommand:
    """Clears both masks (remembering their previous content as bbox patches)."""

    description = "clear masks"

    def __init__(self, removal: MaskLayer, protect: MaskLayer) -> None:
        self._removal = removal
        self._protect = protect
        self._items = _snapshot_masks(removal, protect)

    def undo(self) -> None:
        _restore_masks(self._removal, self._protect, self._items)

    def redo(self) -> None:
        self._removal.clear()
        self._protect.clear()


def _snapshot_masks(
    removal: MaskLayer, protect: MaskLayer
) -> list[tuple[MaskLayer, tuple[int, int, int, int] | None, np.ndarray | None]]:
    items = []
    for layer in (removal, protect):
        bb = layer.bbox()
        patch = layer.data[bb[1] : bb[3], bb[0] : bb[2]].copy() if bb else None
        items.append((layer, bb, patch))
    return items


def _restore_masks(
    removal: MaskLayer,
    protect: MaskLayer,
    items: list[tuple[MaskLayer, tuple[int, int, int, int] | None, np.ndarray | None]],
) -> None:
    removal.clear()
    protect.clear()
    for layer, bb, patch in items:
        if bb is not None and patch is not None:
            layer.data[bb[1] : bb[3], bb[0] : bb[2]] = patch


class RemovalCommand:
    """One committed Remove pass: new pixels + masks cleared afterwards."""

    description = "remove object"

    def __init__(
        self,
        pixels: np.ndarray,
        removal: MaskLayer,
        protect: MaskLayer,
        bbox: tuple[int, int, int, int],
        before_pixels: np.ndarray,
        after_pixels: np.ndarray,
    ) -> None:
        self._pixels = pixels
        self._removal = removal
        self._protect = protect
        self._bbox = bbox
        self._before_pixels = before_pixels
        self._after_pixels = after_pixels
        self._mask_items = _snapshot_masks(removal, protect)

    def undo(self) -> None:
        x0, y0, x1, y1 = self._bbox
        self._pixels[y0:y1, x0:x1] = self._before_pixels
        _restore_masks(self._removal, self._protect, self._mask_items)

    def redo(self) -> None:
        x0, y0, x1, y1 = self._bbox
        self._pixels[y0:y1, x0:x1] = self._after_pixels
        self._removal.clear()
        self._protect.clear()


# --- document ---------------------------------------------------------------

class ImageDocument:
    """Holds the working image, the two masks, and the session history."""

    def __init__(self, image: WorkingImage) -> None:
        self.image = image
        self.removal = MaskLayer(image.height, image.width)
        self.protect = MaskLayer(image.height, image.width)
        self.history = History()

    # -- masks ---------------------------------------------------------------
    def mask(self, mask_id: str) -> MaskLayer:
        if mask_id == MASK_REMOVAL:
            return self.removal
        if mask_id == MASK_PROTECT:
            return self.protect
        raise KeyError(mask_id)

    def begin_stroke(
        self, mask_id: str, radius: float, hardness: float, erase: bool
    ) -> StrokeBuilder:
        return StrokeBuilder(self.mask(mask_id).data, radius, hardness, erase)

    def commit_stroke(self, builder: StrokeBuilder) -> bool:
        cmd = builder.finish()
        if cmd is None:
            return False
        self.history.push(cmd)
        return True

    def clear_masks(self) -> None:
        if not (self.removal.data.any() or self.protect.data.any()):
            return
        self.history.push(ClearMasksCommand(self.removal, self.protect))
        self.removal.clear()
        self.protect.clear()

    # -- overlap resolution ---------------------------------------------------
    def overlap_count(self) -> int:
        return int(np.count_nonzero(self.removal.data & self.protect.data))

    def resolve_overlap(self, prefer: str) -> None:
        """Resolve mask overlap: 'protect' or 'removal' wins (as one command)."""
        overlap = self.removal.data & self.protect.data
        if not overlap.any():
            return
        before_r = self.removal.data.copy()
        before_p = self.protect.data.copy()
        if prefer == "protect":
            self.removal.data[overlap > 0] = 0
        elif prefer == "removal":
            self.protect.data[overlap > 0] = 0
        else:
            raise ValueError(prefer)
        cmd = _MaskDeltaCommand(
            self.removal, self.protect, before_r, before_p,
            self.removal.data.copy(), self.protect.data.copy(),
        )
        self.history.push(cmd)

    # -- removal --------------------------------------------------------------
    def apply_removal(self, new_pixels: np.ndarray) -> None:
        """Commit an inpaint result: swap in pixels, clear masks, record undo."""
        x0, y0, x1, y1 = mask_bbox(self.removal.data) or (0, 0, 0, 0)
        # Expand bbox to include protected pixels so undo restores everything.
        pb = mask_bbox(self.protect.data)
        if pb is not None:
            x0, y0 = min(x0, pb[0]), min(y0, pb[1])
            x1, y1 = max(x1, pb[2]), max(y1, pb[3])
        if x1 <= x0 or y1 <= y0:
            return
        before = self.image.pixels[y0:y1, x0:x1].copy()
        after = new_pixels[y0:y1, x0:x1].copy()
        cmd = RemovalCommand(
            self.image.pixels, self.removal, self.protect,
            (x0, y0, x1, y1), before, after,
        )
        self.image.pixels[y0:y1, x0:x1] = after
        cmd.redo()  # clears masks (pixel part is already applied)
        # redo() re-writes pixels with the same content — idempotent.
        self.history.push(cmd)

    # -- history --------------------------------------------------------------
    def undo(self) -> bool:
        return self.history.undo()

    def redo(self) -> bool:
        return self.history.redo()


class _MaskDeltaCommand:
    description = "resolve mask overlap"

    def __init__(
        self,
        removal: MaskLayer,
        protect: MaskLayer,
        before_r: np.ndarray,
        before_p: np.ndarray,
        after_r: np.ndarray,
        after_p: np.ndarray,
    ) -> None:
        self._removal = removal
        self._protect = protect
        self._before_r, self._before_p = before_r, before_p
        self._after_r, self._after_p = after_r, after_p

    def undo(self) -> None:
        self._removal.data[:] = self._before_r
        self._protect.data[:] = self._before_p

    def redo(self) -> None:
        self._removal.data[:] = self._after_r
        self._protect.data[:] = self._after_p
