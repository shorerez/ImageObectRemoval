"""Inpainting: progressive windowed fill generation with mask constraints.

The service is engine-agnostic and works at native resolution with the fixed
MODEL_INPUT_SIZE (512) window of the LaMa ONNX export. A crop around the
removal bbox carries a CONTEXT_MARGIN ring of real image context, and the hole
is filled in three stages:

1. **Whole-hole seed** — when the hole is larger than the multiscale trigger
   the crop is downscaled until the hole fits inside one window with real
   surroundings, filled in one shot, and upscaled. This seed has no structure
   detail, but it is plausible content everywhere, so the native stage never
   sees the removed object as context.
2. **Coarse-to-fine ladder** — the seed is only ~15-30% of native scale and
   its structure is too soft to be worth extending, so between the seed and
   native scale the fill is rebuilt at intermediate scales (×1.8 per level,
   at most `ladder_levels`). Each level sees the previous level's fill as
   context and re-fills the hole band by band from the real boundary inward,
   which sharpens the coarse structure (step lines, water swell) against
   context that finally has some detail in it.
3. **Progressive native refinement** — the removal region is re-filled at
   native resolution band by band, from the real boundary inward. Each pass
   masks only the pixels within `peel_band` of content the model can trust
   (real pixels, or already-refined ones), so no window is ever asked to fill
   an almost-empty 512x512 tile — the failure mode that turned large holes
   into a mosaic of flat tile-sized patches. Every pixel is written exactly
   once, by the window whose territory it belongs to, and later windows see
   earlier fills as context, so structure (e.g. step edges) is extended from
   the surroundings step by step. The band is deliberately thin (20% of a
   window): the first window of a pass is then anchored on real pixels, not
   on a coarse guess, which is what lets real structure propagate inward.

Runtime tuning lives in quality.ini (see config.quality).

Guarantees:
- protected pixels are never overwritten (composite writes removal pixels only);
- protected pixels are never used as fill source (they are masked as 'unknown'
  in every engine call, at every stage);
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
    MODEL_IMAGE_INPUT,
    MODEL_INPUT_SIZE,
    MODEL_MASK_INPUT,
    MODEL_OUTPUT,
    TILE_OVERLAP,
    quality,
)
from .diagnostics import DiagnosticCapture
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
    """LaMa ONNX (Carve/LaMa-ONNX): fixed inputs image/mask -> output.

    The Carve export outputs values in 0..255 (its demo casts output directly
    to uint8), while most exports output 0..1. The engine probes the output
    scale at construction with a synthetic self-test, which also detects
    broken execution providers (e.g. a CUDA EP producing garbage).
    """

    name = "LaMa (AI)"

    def __init__(self, session, input_size: int = MODEL_INPUT_SIZE) -> None:
        self._session = session
        self.input_size = int(input_size)
        self._out_name = MODEL_OUTPUT
        self._scale = self._probe()

    def _forward(self, tile: np.ndarray, mask: np.ndarray) -> np.ndarray:
        s = self.input_size
        if tile.shape[0] != s or tile.shape[1] != s:
            raise InpaintError(f"OnnxLamaEngine expects {s}x{s} tiles, got {tile.shape}.")
        image_t = np.ascontiguousarray(tile.transpose(2, 0, 1)[None], dtype=np.float32)
        mask_t = np.ascontiguousarray(mask.astype(np.float32)[None, None])
        outputs = self._session.run(
            None, {MODEL_IMAGE_INPUT: image_t, MODEL_MASK_INPUT: mask_t}
        )
        names = [o.name for o in self._session.get_outputs()]
        if self._out_name in names:
            out = outputs[names.index(self._out_name)]
        else:  # pragma: no cover - defensive
            out = outputs[0]
        return out[0].transpose(1, 2, 0).astype(np.float32)

    def _probe(self) -> float:
        """Self-test: detect output scale (0..1 vs 0..255) and provider health.

        A constant 0.5-gray tile is inpainted; the model must reconstruct the
        context (non-hole area) to roughly its input level. The ratio between
        output and input context level reveals the scale. Raises InpaintError
        when the provider produces garbage (NaN / absurd magnitudes).
        """
        s = self.input_size
        tile = np.full((s, s, 3), 0.5, dtype=np.float32)
        mask = np.zeros((s, s), dtype=np.float32)
        q = s // 4
        mask[q : s - q, q : s - q] = 1.0
        out = self._forward(tile, mask)
        if not np.isfinite(out).all():
            raise InpaintError("model self-test failed: non-finite output")
        ratio = float(np.mean(out[mask < 0.5]) / 0.5)
        if not (0.2 < ratio < 400.0):
            raise InpaintError(
                f"model self-test failed: implausible output level ({ratio:.1f}x)"
            )
        scale = 255.0 if ratio > 10.0 else 1.0
        hole = float(np.median(out[mask > 0.5])) / scale
        if not (0.02 < hole < 0.98):
            raise InpaintError(
                f"model self-test failed: fill saturated ({hole:.2f})"
            )
        log.info("LaMa self-test OK: output scale = %g", scale)
        return scale

    def fill(self, tile: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = self._forward(tile, mask)
        return np.clip(out / self._scale, 0.0, 1.0).astype(np.float32)


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


def _plan_region(shape, ys, xs, size, context_margin):
    """Crop bounds around the removal bbox.

    The region is expanded so it covers at least one full inference window and
    carries `context_margin` px of real image context per side (clamped to the
    image). This ring is what the model sees around the hole — too little
    context makes fills look pasted and unnatural.
    """
    h, w = shape[:2]
    m0, m1 = int(ys.min()), int(ys.max()) + 1
    n0, n1 = int(xs.min()), int(xs.max()) + 1
    pad_y = max(size - (m1 - m0), 2 * context_margin)
    pad_x = max(size - (n1 - n0), 2 * context_margin)
    y0 = max(0, m0 - pad_y // 2)
    y1 = min(h, m1 + pad_y - pad_y // 2)
    x0 = max(0, n0 - pad_x // 2)
    x1 = min(w, n1 + pad_x - pad_x // 2)
    return y0, y1, x0, x1


def _context_scale(hole_h, hole_w, size, trigger, floor):
    """Downscale factor for the whole-hole seed pass (1.0 = disabled).

    The model sees `size` px per window. When the hole is much larger than
    that, a native window is mostly hole with barely any surroundings, so the
    seed can only give the scene roughly the right structure and color. This
    scale makes the hole fit into a window with real context around it, which
    is what the progressive native stage then refines.
    """
    long_side = max(hole_h, hole_w)
    target = size * trigger
    if long_side <= target:
        return 1.0
    return max(floor, target / long_side)


def _ladder_scales(seed_scale: float, ratio: float, max_levels: int) -> list[float]:
    """Intermediate scales between the whole-hole seed and native (1.0).

    Each level multiplies the scale by `ratio` (at most `max_levels` of them)
    and stops before native scale, which the progressive stage always runs at
    full resolution. Returns ascending scales, possibly empty.
    """
    scales: list[float] = []
    scale = float(seed_scale)
    for _ in range(max(0, int(max_levels))):
        scale *= max(1.05, float(ratio))
        if scale >= 0.95:
            break
        scales.append(scale)
    return scales


def _resize_mask(mask: np.ndarray, w: int, h: int, scale: float) -> np.ndarray:
    """Bool mask resized for a ladder level (True = part of the hole)."""
    small = cv2.resize(
        mask.astype(np.float32),
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return small > 0.5


def _window_starts(total: int, size: int, step: int) -> list[int]:
    if total <= size:
        return [0]
    starts = list(range(0, total - size, step))
    if starts[-1] != total - size:
        starts.append(total - size)
    return starts


def _count_windows(h: int, w: int, size: int, step: int) -> int:
    """Number of inference windows covering a h x w canvas."""
    return len(_window_starts(max(h, size), size, step)) * len(
        _window_starts(max(w, size), size, step)
    )


def _count_windows_over_mask(mask: np.ndarray, size: int, step: int) -> int:
    """Number of grid windows that touch at least one masked pixel."""
    h, w = mask.shape[:2]
    n = 0
    for wy in _window_starts(max(h, size), size, step):
        for wx in _window_starts(max(w, size), size, step):
            window = mask[wy : wy + size, wx : wx + size]
            if window.size and window.any():
                n += 1
    return n


def _report(progress, done, total) -> None:
    if progress is not None:
        progress(min(done, total), total)


def _axis_weights(starts, idx, size, band):
    """1-D blend weights: full weight inside a window's territory, a `band`-
    wide linear crossfade at the border between neighbouring windows."""
    start = starts[idx]
    pos = np.arange(start, start + size, dtype=np.float32)
    w = np.ones(size, dtype=np.float32)
    width = max(float(band), 1.0)
    if idx > 0:
        border = (start + starts[idx - 1] + size) / 2.0
        w = np.minimum(w, np.clip((pos - border) / width + 0.5, 0.0, 1.0))
    if idx + 1 < len(starts):
        border = (start + starts[idx + 1] + size) / 2.0
        w = np.minimum(w, np.clip(0.5 - (pos - border) / width, 0.0, 1.0))
    return w


def _window_weights(y_starts, iy, x_starts, ix, size, band):
    return np.outer(
        _axis_weights(y_starts, iy, size, band),
        _axis_weights(x_starts, ix, size, band),
    )


def _tiled_fill(engine, region, reg_eff, q, cancel, progress, total, done):
    """Windowed inference over one canvas. Returns (filled, done, calls).

    `filled` is float32 (h, w, 3) in 0..1: the weighted blend of the engine's
    window predictions. Padding is marked 'unknown' for the model so it never
    trusts synthetic edge pixels. Windows that touch no masked pixel are
    skipped, and those areas keep the input image instead of ending up as a
    hard-edged black (0) zero-zone that would show up as a fake seam.
    """
    h, w = region.shape[:2]
    size = int(engine.input_size)
    step = max(1, size - min(TILE_OVERLAP, size // 2))
    hp, wp = max(h, size), max(w, size)
    pad_y, pad_x = hp - h, wp - w
    if pad_y or pad_x:
        mode = "reflect" if min(h, w) > 1 else "edge"
        region = np.pad(region, ((0, pad_y), (0, pad_x), (0, 0)), mode=mode)
        reg_eff = np.pad(
            reg_eff, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=True
        )

    acc = np.zeros((hp, wp, 3), dtype=np.float32)
    wsum = np.zeros((hp, wp), dtype=np.float32)
    y_starts = _window_starts(hp, size, step)
    x_starts = _window_starts(wp, size, step)
    band = float(q["blend_band"])
    calls = 0

    for iy, wy in enumerate(y_starts):
        for ix, wx in enumerate(x_starts):
            if cancel is not None and cancel.is_set():
                raise CancelledError("Removal cancelled.")
            tile = region[wy : wy + size, wx : wx + size]
            mask = reg_eff[wy : wy + size, wx : wx + size].astype(np.float32)
            if not mask.any():
                done += 1
                _report(progress, done, total)
                continue
            out = engine.fill(tile, mask)
            wgt = _window_weights(y_starts, iy, x_starts, ix, size, band)
            acc[wy : wy + size, wx : wx + size] += out * wgt[:, :, None]
            wsum[wy : wy + size, wx : wx + size] += wgt
            calls += 1
            done += 1
            _report(progress, done, total)

    filled = np.where(
        (wsum > 0)[:, :, None],
        acc / np.maximum(wsum, 1e-6)[:, :, None],
        region,
    )
    return filled[:h, :w], done, calls


def _progressive_fill(engine, fill, reg_rem, prot, q, cancel, progress, total, done):
    """Refine the removal region at native resolution, band by band.

    `fill` already holds plausible content in the removal region (the
    whole-hole seed), so pixels are only re-synthesized once the front of
    trusted content has reached them: a call masks the removal pixels within
    `peel_band` of a real or already-refined pixel (clipped further when
    needed) and at most `peel_max_unknown` of any window, and nothing else. The masked band is
    therefore always a rim of the remaining hole, the model always sees real
    context or its own earlier fill around it, and the fill front advances
    from the removal boundary inward instead of asking a window to invent a
    full, mostly-empty tile.

    Returns (fill, done, calls, forced) where `calls` records every window
    that was filled (for diagnostics/metadata) and `forced` counts calls that
    had to exceed the masked-share cap (whole-image removals, or a hole that
    is nothing but protected area).
    """
    h, w = fill.shape[:2]
    size = int(engine.input_size)
    step = max(1, size - min(TILE_OVERLAP, size // 2))
    hp, wp = max(h, size), max(w, size)
    if (hp, wp) != (h, w):
        # A crop smaller than one window (small images) is padded with the
        # crop's average color: flat color is safe context (a reflection could
        # otherwise mirror the removed object into the model's view) and the
        # model still sees real pixels all around the hole itself.
        pad_color = fill.reshape(-1, 3).mean(axis=0)
        padded = np.empty((hp, wp, 3), dtype=np.float32)
        padded[...] = pad_color
        padded[:h, :w] = fill
        fill = padded
        reg_rem = np.pad(reg_rem, ((0, hp - h), (0, wp - w)))
        prot = np.pad(prot, ((0, hp - h), (0, wp - w)))

    y_starts = _window_starts(hp, size, step)
    x_starts = _window_starts(wp, size, step)
    band_px = max(1.0, float(q["peel_band"]) * size)
    room_max = int(float(q["peel_max_unknown"]) * size * size)
    band_w = float(q["blend_band"])
    # Territories tile the canvas: every pixel is owned by the window with the
    # largest blend weight (max in each axis), so it is written exactly once
    # and overlapping window predictions never fight over a pixel.
    y_terr = [
        _axis_weights(y_starts, iy, size, band_w) >= 0.5 for iy in range(len(y_starts))
    ]
    x_terr = [
        _axis_weights(x_starts, ix, size, band_w) >= 0.5 for ix in range(len(x_starts))
    ]
    territories = []
    for iy, wy in enumerate(y_starts):
        for ix, wx in enumerate(x_starts):
            terr = np.outer(y_terr[iy], x_terr[ix])
            prot_count = int(prot[wy : wy + size, wx : wx + size].sum())
            territories.append((wy, wx, terr, prot_count))

    unknown = reg_rem | prot  # engine-invisible: unrefined removal + protect
    todo = reg_rem.copy()     # removal pixels still showing the seed only
    calls: list[dict] = []
    forced = 0
    while todo.any():
        if cancel is not None and cancel.is_set():
            raise CancelledError("Removal cancelled.")
        # Distance to the nearest trusted pixel, over the neighbourhood of the
        # pixels that are left (cheap, and exact for the band test).
        rows, cols = todo.any(axis=1), todo.any(axis=0)
        y_lo, y_hi = np.flatnonzero(rows)[[0, -1]]
        x_lo, x_hi = np.flatnonzero(cols)[[0, -1]]
        pad_px = int(band_px) + 2
        sy0, sy1 = max(0, int(y_lo) - pad_px), min(hp, int(y_hi) + 1 + pad_px)
        sx0, sx1 = max(0, int(x_lo) - pad_px), min(wp, int(x_hi) + 1 + pad_px)
        sub = np.ascontiguousarray(unknown[sy0:sy1, sx0:sx1], dtype=np.uint8)
        dist = np.zeros((hp, wp), dtype=np.float32)
        dist[sy0:sy1, sx0:sx1] = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
        writable = np.zeros_like(todo)
        writable[sy0:sy1, sx0:sx1] = todo[sy0:sy1, sx0:sx1] & (dist[sy0:sy1, sx0:sx1] <= band_px)
        if not writable.any():  # defensive: a band can hide behind protect
            writable = todo.copy()

        best = None      # best call that honours the masked-share cap
        best_raw = None  # fallback when the cap cannot be honoured at all
        for wy, wx, terr, prot_count in territories:
            base = writable[wy : wy + size, wx : wx + size] & terr
            n = int(base.sum())
            if not n:
                continue
            raw_key = (n, -(n + prot_count))
            if best_raw is None or raw_key > best_raw[0]:
                best_raw = (raw_key, wy, wx, base, (n + prot_count) / float(size * size))
            room = room_max - prot_count
            if room <= 0:
                continue
            target = base
            if n > room:
                # Clip the band to the pixels closest to trusted content so the
                # model never faces a mostly-empty tile.
                d_win = dist[wy : wy + size, wx : wx + size]
                keep = np.argpartition(d_win[base], room - 1)[:room]
                yy, xx = np.nonzero(base)
                target = np.zeros_like(base)
                target[yy[keep], xx[keep]] = True
                n = room
            share = (n + prot_count) / float(size * size)
            key = (n, -share)
            if best is None or key > best[0]:
                best = (key, wy, wx, target, share)

        if best is None:
            forced += 1
            chosen = best_raw
        else:
            chosen = best
        if chosen is None:  # pragma: no cover - defensive
            log.warning(
                "Progressive fill: no window accepted the remaining %d pixels; "
                "leaving them at their seed values.",
                int(todo.sum()),
            )
            break
        _key, wy, wx, target, share = chosen
        tile = fill[wy : wy + size, wx : wx + size]
        mask = (target | prot[wy : wy + size, wx : wx + size]).astype(np.float32)
        out = engine.fill(tile.copy(), mask)
        calls.append(
            {
                "window": [int(wx), int(wy)],
                "pixels": int(target.sum()),
                "window_hole_share": round(float(share), 4),
                "window_unrefined_share": round(
                    float(unknown[wy : wy + size, wx : wx + size].mean()), 4
                ),
                "forced": best is None,
            }
        )
        out_window = fill[wy : wy + size, wx : wx + size]
        out_window[target] = out[target]
        todo[wy : wy + size, wx : wx + size][target] = False
        unknown[wy : wy + size, wx : wx + size][target] = False
        done += 1
        _report(progress, done, total)

    return fill[:h, :w], done, calls, forced


def _disc(radius: int):
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * int(radius) + 1, 2 * int(radius) + 1)
    )


def _boundary_bands(reg_rem, width):
    """(inner, outer) boolean bands: `width` px inside and outside the mask edge."""
    u8 = reg_rem.astype(np.uint8)
    k = _disc(max(int(width), 1))
    inner = reg_rem & ~(cv2.erode(u8, k) > 0)
    outer = (cv2.dilate(u8, k) > 0) & ~reg_rem
    return inner, outer


def _harmonize_fill(fill, region, reg_rem, strength, band):
    """Match the fill's color statistics at the hole boundary to the real
    surroundings (Lab mean/std) and apply the correction to the whole fill.

    Compares fill pixels just INSIDE the mask against real pixels just OUTSIDE
    it; a global scene-wide comparison would dilute the correction to nothing.
    """
    if strength <= 0:
        return fill
    inner, outer = _boundary_bands(reg_rem, band)
    if int(inner.sum()) < 50 or int(outer.sum()) < 50:
        return fill
    lab_f = cv2.cvtColor(fill, cv2.COLOR_RGB2Lab)
    lab_r = cv2.cvtColor(region, cv2.COLOR_RGB2Lab)
    src = lab_r[outer].astype(np.float32)  # real surroundings
    dst = lab_f[inner].astype(np.float32)  # fill at its boundary
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    ratio = np.clip(src.std(axis=0) / np.maximum(dst.std(axis=0), 1e-3), 0.5, 2.0)
    centered = lab_f[reg_rem] - dst_mean
    lab_f[reg_rem] = (
        dst_mean
        + centered * (1.0 + (ratio - 1.0) * strength)
        + (src_mean - dst_mean) * strength
    )
    return np.clip(cv2.cvtColor(lab_f, cv2.COLOR_Lab2RGB), 0.0, 1.0)


def _synth_texture(fill, region, reg_rem, radius, sigma, strength):
    """Add scene-matched grain to the fill: high-frequency noise whose band
    energies match the real texture around the hole, so the fill keeps the
    scene's grain instead of looking blurred and pasted."""
    if strength <= 0 or sigma <= 0:
        return fill
    u8 = reg_rem.astype(np.uint8)
    ring = (cv2.dilate(u8, _disc(int(radius))) > 0) & ~reg_rem
    if int(ring.sum()) < 100:
        return fill
    sig = float(sigma)
    r_fine = region - cv2.GaussianBlur(region, (0, 0), sig)
    r_blur = cv2.GaussianBlur(region, (0, 0), sig)
    r_coarse = r_blur - cv2.GaussianBlur(r_blur, (0, 0), 2.0 * sig)
    rng = np.random.default_rng(12345)  # deterministic output
    noise = rng.standard_normal(region.shape).astype(np.float32)
    n_fine = noise - cv2.GaussianBlur(noise, (0, 0), sig)
    n_blur = cv2.GaussianBlur(noise, (0, 0), sig)
    n_coarse = n_blur - cv2.GaussianBlur(n_blur, (0, 0), 2.0 * sig)
    grain = np.zeros_like(fill)
    for c in range(3):
        for src, ref in ((n_fine, r_fine), (n_coarse, r_coarse)):
            target = float(ref[..., c][ring].std())
            have = float(src[..., c].std())
            if target > 1e-5 and have > 1e-5:
                grain[..., c] += src[..., c] * (target / have)
    out = fill.copy()
    out[reg_rem] = fill[reg_rem] + strength * grain[reg_rem]
    return out


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

        q = quality()
        effective = rem | prot
        ys, xs = np.nonzero(rem)
        size = int(self.engine.input_size)
        step = max(1, size - min(TILE_OVERLAP, size // 2))

        y0, y1, x0, x1 = _plan_region(
            pixels.shape, ys, xs, size, int(q["context_margin"])
        )
        region = pixels[y0:y1, x0:x1].astype(np.float32) / 65535.0
        reg_eff = effective[y0:y1, x0:x1]
        reg_rem = rem[y0:y1, x0:x1]
        h, w = region.shape[:2]

        # Opt-in local diagnostics (marker file in app_data_dir); best-effort,
        # never raises, never alters the removal (see diagnostics module).
        diag = DiagnosticCapture.maybe(self.engine.name, size, q)
        if diag is not None:
            diag.add_image("00-original.png", pixels[y0:y1, x0:x1])
            diag.add_image("mask.png", reg_rem)

        # --- stage 1: whole-hole seed ---------------------------------------
        # The seed only has to be plausible content (no detail): it keeps the
        # removed object from ever being visible as context to the progressive
        # stage and gives that stage a soft hint of the global structure.
        scale = 1.0
        if float(q["multiscale"]) > 0:
            scale = _context_scale(
                int(ys.max() - ys.min() + 1),
                int(xs.max() - xs.min() + 1),
                size,
                float(q["multiscale_trigger"]),
                float(q["multiscale_min"]),
            )
        # The seed reads a wider crop than the native stage when the image
        # allows it: at the seed scale the hole must land in a window with real
        # context around it, otherwise the model fills a mostly-empty tile and
        # the seed degrades to the smooth mush the old context pass produced.
        hole_h = int(ys.max() - ys.min() + 1)
        hole_w = int(xs.max() - xs.min() + 1)
        seed_margin = max(
            int(q["context_margin"]),
            int((0.98 * size / scale - max(hole_h, hole_w)) // 2),
        )
        sy0, sy1, sx0, sx1 = _plan_region(
            pixels.shape, ys, xs, size, seed_margin
        )
        ssw = max(1, int(round((sx1 - sx0) * scale)))
        ssh = max(1, int(round((sy1 - sy0) * scale)))
        seed_planned = _count_windows(ssh, ssw, size, step) if scale < 1.0 else 0
        if scale < 1.0:
            if float(q["multiscale"]) > 0:
                seed_engine = self.engine
                seed_kind = "ai-whole-hole"
                reason = "hole exceeds native-window trigger"
            else:
                # multiscale disabled but the hole is still too large for a
                # single conditioned window: a cheap classical seed keeps the
                # removed pixels from leaking into the progressive stage.
                seed_engine = ClassicalEngine(size)
                seed_kind = "classical-whole-hole"
                reason = "multiscale disabled; classical seed required"
            log.info(
                "Large removal: %s seed at %d%% (%dx%d crop -> %dx%d canvas)",
                seed_kind, round(scale * 100), sx1 - sx0, sy1 - sy0, ssw, ssh,
            )
            context_pass_info = {
                "ran": True,
                "seeder": seed_kind,
                "downscale": float(scale),
                "scaled_size": {"width": ssw, "height": ssh},
                "seed_crop": {"y0": sy0, "y1": sy1, "x0": sx0, "x1": sx1},
                "reason": reason,
            }
        elif float(q["multiscale"]) > 0:
            context_pass_info = {
                "ran": False,
                "reason": "hole fits within the native window",
            }
        else:
            context_pass_info = {
                "ran": False,
                "reason": "disabled in quality settings",
            }

        # Coarse-to-fine ladder between the seed and native resolution. The
        # seed is a blurry 15-30% scale guess, so at native scale the model
        # would see no structure around the hole to extend; rebuilding the
        # hole at intermediate scales first puts real-ish structure into that
        # context, which the native stage then sharpens band by band.
        ladder_levels: list[float] = []
        if scale < 1.0 and float(q["multiscale"]) > 0 and float(q["ladder"]) > 0:
            ladder_levels = _ladder_scales(
                scale, float(q["ladder_ratio"]), int(q["ladder_levels"])
            )

        mask_windows = _count_windows_over_mask(reg_eff, size, step)
        band_px = max(1.0, float(q["peel_band"]) * size)
        # Progress estimate: every window that touches the hole is filled at
        # least once, and a call writes roughly half a band's worth of pixels.
        # `done` is clamped, so a wrong estimate only affects the bar's pace.
        def _peel_estimate(hole_px: int) -> int:
            return max(1, int(np.ceil(hole_px / max(1.0, 0.55 * band_px * size))))

        ladder_estimate = 0
        for level_scale in ladder_levels:
            level_hole = int((reg_rem.sum()) * level_scale * level_scale)
            ladder_estimate += max(
                _peel_estimate(level_hole),
                _count_windows_over_mask(
                    _resize_mask(reg_rem, w, h, level_scale), size, step
                ),
            )
        total = seed_planned + ladder_estimate + _peel_estimate(int(reg_rem.sum()))
        done = 0
        seed_calls = 0
        fill = region.copy()
        if scale < 1.0:
            seed_region = pixels[sy0:sy1, sx0:sx1].astype(np.float32) / 65535.0
            seed_eff = effective[sy0:sy1, sx0:sx1]
            small = cv2.resize(seed_region, (ssw, ssh), interpolation=cv2.INTER_AREA)
            small_eff = (
                cv2.resize(
                    seed_eff.astype(np.float32),
                    (ssw, ssh),
                    interpolation=cv2.INTER_LINEAR,
                )
                > 0.5
            )
            seed_small, done, seed_calls = _tiled_fill(
                seed_engine, small, small_eff, q, cancel, progress, total, done
            )
            seed_full = cv2.resize(
                seed_small, (sx1 - sx0, sy1 - sy0), interpolation=cv2.INTER_CUBIC
            )
            seed = seed_full[y0 - sy0 : y0 - sy0 + h, x0 - sx0 : x0 - sx0 + w]
            # only the unknown pixels are seeded; real pixels stay bit-exact
            fill[reg_eff] = seed[reg_eff]
            if diag is not None:
                diag.add_image("02-context.png", seed)

        # --- stage 1.5: coarse-to-fine ladder -------------------------------
        # Each level re-fills the whole hole at a higher scale, band by band
        # from the boundary inward, with the previous level's fill as context.
        # Pixels the model trusts are therefore never more than one band away
        # from real content, and the structure of the coarse fill gets
        # re-decided (sharper) against real surroundings before native scale
        # ever runs.
        ladder_info: list[dict] = []
        reg_prot = prot[y0:y1, x0:x1]
        unknown_region = reg_rem | reg_prot
        for level_scale in ladder_levels:
            level_w = max(1, int(round(w * level_scale)))
            level_h = max(1, int(round(h * level_scale)))
            small = cv2.resize(
                fill, (level_w, level_h), interpolation=cv2.INTER_AREA
            )
            # Protect conservatively (any coarse pixel that overlaps protected
            # area stays unknown) so protected content is never model context.
            small_prot = (
                cv2.resize(
                    reg_prot.astype(np.float32), (level_w, level_h),
                    interpolation=cv2.INTER_AREA,
                )
                > 0.0
            )
            small_rem = _resize_mask(reg_rem, w, h, level_scale)
            small_fill, done, level_calls, level_forced = _progressive_fill(
                self.engine, small, small_rem, small_prot, q, cancel,
                progress, total, done,
            )
            upscaled = cv2.resize(
                small_fill, (w, h), interpolation=cv2.INTER_CUBIC
            )
            # Only unknown pixels are replaced: outside the hole the real
            # pixels of the crop stay exactly as the model saw them.
            fill[unknown_region] = upscaled[unknown_region]
            holes = [c["window_hole_share"] for c in level_calls]
            ladder_info.append(
                {
                    "scale": round(float(level_scale), 4),
                    "canvas": [int(level_w), int(level_h)],
                    "calls": len(level_calls),
                    "forced_calls": int(level_forced),
                    "max_hole_share_used": max(holes) if holes else 0.0,
                }
            )
            if diag is not None:
                diag.add_image(
                    f"02b-ladder-{int(round(level_scale * 100)):02d}.png", upscaled
                )
            if level_forced:
                log.warning(
                    "Ladder fill at %d%%: %d/%d windows had to run with more "
                    "than %.0f%% masked pixels.",
                    round(level_scale * 100), level_forced, len(level_calls),
                    float(q["peel_max_unknown"]) * 100,
                )

        # --- stage 2: progressive native refinement -------------------------
        fill, done, calls, forced = _progressive_fill(
            self.engine, fill, reg_rem, reg_prot, q, cancel,
            progress, total, done,
        )
        _report(progress, total, total)
        ladder_calls = sum(level["calls"] for level in ladder_info)
        if forced:
            log.warning(
                "Progressive fill: %d/%d windows had to run with more than "
                "%.0f%% masked pixels (large compact hole or protected area).",
                forced, len(calls), float(q["peel_max_unknown"]) * 100,
            )
        if diag is not None:
            diag.add_image("01-native.png", fill)
            holes = [c["window_hole_share"] for c in calls]
            progressive_info = {
                "band_px": int(round(float(q["peel_band"]) * size)),
                "max_window_hole_share": float(q["peel_max_unknown"]),
                "calls": len(calls),
                "forced_calls": int(forced),
                "windows_overlapping_mask": int(mask_windows),
                "max_hole_share_used": max(holes) if holes else 0.0,
                "call_log": calls,
            }
        else:
            progressive_info = None

        # Color + texture: the fill must sit in the scene's color statistics
        # and carry its grain, or the patch reads as a pasted blur.
        fill = _harmonize_fill(
            fill, region, reg_rem,
            float(q["harmonize"]), float(q["harmonize_band"]),
        )
        fill = _synth_texture(
            fill, region, reg_rem,
            float(q["texture_radius"]), float(q["texture_sigma"]),
            float(q["texture"]),
        )
        fill = np.clip(fill, 0.0, 1.0).astype(np.float32)

        # Composite: only removal pixels; soft seam *inside* the mask edge so
        # protected/unselected pixels stay bit-exact.
        out_region = region.copy()
        feather = float(q["feather_sigma"])
        if feather > 0:
            soft = cv2.GaussianBlur(reg_rem.astype(np.float32), (0, 0), feather)
            alpha = np.minimum(soft, reg_rem.astype(np.float32))
        else:
            alpha = reg_rem.astype(np.float32)
        out_region = out_region * (1.0 - alpha[:, :, None]) + fill * alpha[:, :, None]

        new_pixels = pixels.copy()
        filled_u16 = np.clip(np.rint(out_region * 65535.0), 0, 65535).astype(np.uint16)
        region_view = new_pixels[y0:y1, x0:x1]
        region_view[reg_rem] = filled_u16[reg_rem]

        if diag is not None:
            diag.add_image("04-final.png", new_pixels[y0:y1, x0:x1])
            diag.finish(
                crop=(y0, y1, x0, x1),
                image_size=pixels.shape[:2],
                mask={
                    "removal_pixels_in_crop": int(reg_rem.sum()),
                    "protect_pixels_in_crop": int(prot[y0:y1, x0:x1].sum()),
                },
                context_pass=context_pass_info,
                ladder=ladder_info or None,
                progressive=progressive_info,
            )

        stats = InpaintStats(
            engine=self.engine.name,
            windows=seed_calls + ladder_calls + len(calls),
            elapsed_s=time.monotonic() - t0,
        )
        return new_pixels, stats


def build_default_engine(model_path=None):
    """Best available engine: LaMa ONNX when the model file exists.

    Tries CUDA first, then CPU, and runs the engine self-test for each: a
    provider that produces garbage is rejected so the app never silently
    writes junk pixels. Falls back to the classical engine when nothing works.
    """
    if model_path is not None:
        from .runtime import create_onnx_session

        for providers in (
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ):
            try:
                session = create_onnx_session(model_path, providers=providers)
                engine: Engine = OnnxLamaEngine(session)
                log.info("Inpaint engine: %s (%s, %s)", engine.name, model_path,
                         session.get_providers()[0])
                return engine
            except Exception as exc:
                log.warning(
                    "Engine init with %s failed (%s); trying next option.",
                    providers[0], exc,
                )
    return ClassicalEngine()
