"""Inpainting: windowed fill generation with mask constraints.

The service is engine-agnostic. Windows are native-resolution MODEL_INPUT_SIZE
(512) tiles with TILE_OVERLAP overlap and feathered blending — the ONNX LaMa
export has a fixed 512x512 input contract (REQUIREMENTS.md §7).

Guarantees:
- protected pixels are never overwritten (composite writes removal pixels only);
- protected pixels are never used as fill source (they are masked as 'unknown'
  together with the hole before inference);
- unselected pixels are bit-exact.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import cv2
import numpy as np

from .config import (
    CONTEXT_MARGIN,
    FEATHER_SIGMA,
    MODEL_IMAGE_INPUT,
    MODEL_INPUT_SIZE,
    MODEL_MASK_INPUT,
    MODEL_OUTPUT,
    TILE_OVERLAP,
)
from .errors import CancelledError, InpaintError

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]  # (windows done, windows total)


class Engine(Protocol):
    name: str
    input_size: int

    def fill(self, tile: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """tile: float32 (S, S, 3) in 0..1; mask: float32 (S, S), 1 = to fill.

        Returns float32 (S, S, 3) in 0..1.
        """
        ...


class OnnxLamaEngine:
    """LaMa ONNX (Carve/LaMa-ONNX): fixed inputs image/mask -> output."""

    name = "LaMa (AI)"

    def __init__(self, session, input_size: int = MODEL_INPUT_SIZE) -> None:
        self._session = session
        self.input_size = int(input_size)
        self._out_name = MODEL_OUTPUT

    def fill(self, tile: np.ndarray, mask: np.ndarray) -> np.ndarray:
        s = self.input_size
        if tile.shape[0] != s or tile.shape[1] != s:
            raise InpaintError(f"OnnxLamaEngine expects {s}x{s} tiles, got {tile.shape}.")
        image_t = tile.transpose(2, 0, 1)[None].astype(np.float32)
        mask_t = mask.astype(np.float32)[None, None]
        outputs = self._session.run(
            None, {MODEL_IMAGE_INPUT: image_t, MODEL_MASK_INPUT: mask_t}
        )
        names = [o.name for o in self._session.get_outputs()]
        if self._out_name in names:
            out = outputs[names.index(self._out_name)]
        else:  # pragma: no cover - defensive
            out = outputs[0]
        out = out[0].transpose(1, 2, 0)
        return np.clip(out, 0.0, 1.0).astype(np.float32)


class ClassicalEngine:
    """Telea inpainting via OpenCV — offline fallback when the AI model is not
    downloaded yet. Lower quality, but keeps the app usable and tests fast."""

    name = "Classical (fallback)"

    def __init__(self, input_size: int = MODEL_INPUT_SIZE) -> None:
        self.input_size = int(input_size)

    def fill(self, tile: np.ndarray, mask: np.ndarray) -> np.ndarray:
        u8 = np.clip(tile * 255.0 + 0.5, 0, 255).astype(np.uint8)
        m = np.where(mask > 0.5, 255, 0).astype(np.uint8)
        if not m.any():
            return tile.astype(np.float32)
        filled = cv2.inpaint(u8, m, 4, cv2.INPAINT_TELEA)
        return filled.astype(np.float32) / 255.0


@dataclass
class InpaintStats:
    engine: str
    windows: int
    elapsed_s: float


class InpaintService:
    """Fills the removal mask of an image using an Engine."""

    def __init__(self, engine: Engine | None = None) -> None:
        self.engine: Engine = engine or ClassicalEngine()

    def inpaint(
        self,
        pixels: np.ndarray,
        removal: np.ndarray,
        protect: np.ndarray | None = None,
        progress: ProgressFn | None = None,
        cancel: threading.Event | None = None,
    ) -> tuple[np.ndarray, InpaintStats]:
        """Return (new_pixels, stats). Input masks are HxW uint8/bool."""
        t0 = time.monotonic()
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise InpaintError(f"Expected (H, W, 3) pixels, got {pixels.shape}.")
        rem = np.asarray(removal) > 0
        if protect is None:
            prot = np.zeros_like(rem)
        else:
            prot = np.asarray(protect) > 0
        # Defensive: protect always wins inside the service.
        rem = rem & ~prot
        if not rem.any():
            raise InpaintError("Removal mask is empty.")

        effective = rem | prot
        ys, xs = np.nonzero(rem)
        m0, m1 = int(ys.min()), int(ys.max()) + 1
        n0, n1 = int(xs.min()), int(xs.max()) + 1
        y0 = max(0, m0 - CONTEXT_MARGIN)
        x0 = max(0, n0 - CONTEXT_MARGIN)
        y1 = min(pixels.shape[0], m1 + CONTEXT_MARGIN)
        x1 = min(pixels.shape[1], n1 + CONTEXT_MARGIN)

        region = pixels[y0:y1, x0:x1].astype(np.float32) / 65535.0
        reg_eff = effective[y0:y1, x0:x1]
        reg_rem = rem[y0:y1, x0:x1]
        h, w = region.shape[:2]
        size = int(self.engine.input_size)
        overlap = min(TILE_OVERLAP, size // 2)
        step = size - overlap

        # Pad small dimensions up to the window size; padding is marked as
        # 'unknown' for the model so it never trusts synthetic edge pixels.
        hp, wp = max(h, size), max(w, size)
        pad_y, pad_x = hp - h, wp - w
        if pad_y or pad_x:
            mode = "reflect" if min(h, w) > 1 else "edge"
            region = np.pad(region, ((0, pad_y), (0, pad_x), (0, 0)), mode=mode)
            reg_eff = np.pad(reg_eff, ((0, pad_y), (0, pad_x)), mode="constant",
                             constant_values=True)

        acc = np.zeros((hp, wp, 3), dtype=np.float32)
        wsum = np.zeros((hp, wp), dtype=np.float32)

        y_starts = _window_starts(hp, size, step)
        x_starts = _window_starts(wp, size, step)
        total = len(y_starts) * len(x_starts)
        done = 0

        for wy in y_starts:
            for wx in x_starts:
                if cancel is not None and cancel.is_set():
                    raise CancelledError("Removal cancelled.")
                tile = region[wy : wy + size, wx : wx + size]
                mask = reg_eff[wy : wy + size, wx : wx + size].astype(np.float32)
                if not mask.any():
                    done += 1
                    if progress is not None:
                        progress(done, total)
                    continue
                out = self.engine.fill(tile, mask)
                wgt = _window_weights(size, wy, wx, hp, wp)
                acc[wy : wy + size, wx : wx + size] += out * wgt[:, :, None]
                wsum[wy : wy + size, wx : wx + size] += wgt
                done += 1
                if progress is not None:
                    progress(done, total)

        filled = acc / np.maximum(wsum, 1e-6)[:, :, None]
        filled = filled[:h, :w]
        region = region[:h, :w]

        # Composite: only removal pixels; soft seam *inside* the mask edge so
        # protected/unselected pixels stay bit-exact.
        out_region = region.copy()
        blend = filled
        if FEATHER_SIGMA > 0:
            soft = cv2.GaussianBlur(
                reg_rem.astype(np.float32), (0, 0), FEATHER_SIGMA
            )
            alpha = np.minimum(soft, reg_rem.astype(np.float32))
        else:
            alpha = reg_rem.astype(np.float32)
        out_region = out_region * (1.0 - alpha[:, :, None]) + blend * alpha[:, :, None]

        new_pixels = pixels.copy()
        filled_u16 = np.clip(np.rint(out_region * 65535.0), 0, 65535).astype(np.uint16)
        region_view = new_pixels[y0:y1, x0:x1]
        region_view[reg_rem] = filled_u16[reg_rem]

        stats = InpaintStats(
            engine=self.engine.name,
            windows=total,
            elapsed_s=time.monotonic() - t0,
        )
        return new_pixels, stats


def _window_starts(total: int, size: int, step: int) -> list[int]:
    if total <= size:
        return [0]
    starts = list(range(0, total - size, step))
    if starts[-1] != total - size:
        starts.append(total - size)
    return starts


def _window_weights(size: int, wy: int, wx: int, hp: int, wp: int) -> np.ndarray:
    """Linear edge ramps so overlapping windows blend smoothly."""
    ramp = np.ones(size, dtype=np.float32)
    ov = min(TILE_OVERLAP, size // 2)
    if ov > 0:
        if wx > 0:
            ramp[:ov] = np.linspace(0.0, 1.0, ov, endpoint=False, dtype=np.float32)
        if wx + size < wp:
            ramp[-ov:] = np.minimum(
                ramp[-ov:], np.linspace(1.0, 0.0, ov, endpoint=False, dtype=np.float32)
            )
    ramp_y = np.ones(size, dtype=np.float32)
    if ov > 0:
        if wy > 0:
            ramp_y[:ov] = np.linspace(0.0, 1.0, ov, endpoint=False, dtype=np.float32)
        if wy + size < hp:
            ramp_y[-ov:] = np.minimum(
                ramp_y[-ov:],
                np.linspace(1.0, 0.0, ov, endpoint=False, dtype=np.float32),
            )
    return np.outer(ramp_y, ramp)


def build_default_engine(model_path=None):
    """Best available engine: LaMa ONNX when the model file exists."""
    if model_path is not None:
        try:
            from .runtime import create_onnx_session

            session = create_onnx_session(model_path)
            engine: Engine = OnnxLamaEngine(session)
            log.info("Inpaint engine: %s (%s)", engine.name, model_path)
            return engine
        except Exception as exc:
            log.warning("Could not load ONNX model (%s); using classical fallback.", exc)
    return ClassicalEngine()
