"""Inpainting: windowed fill generation with mask constraints.

The service is engine-agnostic. Native-resolution MODEL_INPUT_SIZE (512)
windows with TILE_OVERLAP overlap cover the removal area; the crop carries a
CONTEXT_MARGIN ring of real image context so the model sees the scene around
the hole. For holes larger than the window, a second pass runs at a downscale
factor where the whole hole fits into a window with real surroundings; the
result provides structure and the native pass provides fine detail
(frequency blend). Runtime tuning lives in quality.ini (see config.quality).

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
    MODEL_IMAGE_INPUT,
    MODEL_INPUT_SIZE,
    MODEL_MASK_INPUT,
    MODEL_OUTPUT,
    TILE_OVERLAP,
    quality,
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
    """Downscale factor for the whole-context pass (1.0 = disabled).

    The model sees `size` px per window. When the hole is much larger than
    that, every native window is mostly hole with barely any surroundings and
    the fill degrades to mush. The context pass feeds the model the whole
    scene at a scale where the hole fits into a window with real context.
    """
    long_side = max(hole_h, hole_w)
    target = size * trigger
    if long_side <= target:
        return 1.0
    return max(floor, target / long_side)


def _window_starts(total: int, size: int, step: int) -> list[int]:
    if total <= size:
        return [0]
    starts = list(range(0, total - size, step))
    if starts[-1] != total - size:
        starts.append(total - size)
    return starts


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
    """Windowed inference over one canvas. Returns (filled, done).

    `filled` is float32 (h, w, 3) in 0..1: the weighted blend of the engine's
    window predictions. Padding is marked 'unknown' for the model so it never
    trusts synthetic edge pixels.
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

    for iy, wy in enumerate(y_starts):
        for ix, wx in enumerate(x_starts):
            if cancel is not None and cancel.is_set():
                raise CancelledError("Removal cancelled.")
            tile = region[wy : wy + size, wx : wx + size]
            mask = reg_eff[wy : wy + size, wx : wx + size].astype(np.float32)
            if not mask.any():
                done += 1
                if progress is not None:
                    progress(done, total)
                continue
            out = engine.fill(tile, mask)
            wgt = _window_weights(y_starts, iy, x_starts, ix, size, band)
            acc[wy : wy + size, wx : wx + size] += out * wgt[:, :, None]
            wsum[wy : wy + size, wx : wx + size] += wgt
            done += 1
            if progress is not None:
                progress(done, total)

    filled = acc / np.maximum(wsum, 1e-6)[:, :, None]
    return filled[:h, :w], done


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

        def _count(ph, pw):
            hp_, wp_ = max(ph, size), max(pw, size)
            return len(_window_starts(hp_, size, step)) * len(
                _window_starts(wp_, size, step)
            )

        scale = 1.0
        if float(q["multiscale"]) > 0:
            scale = _context_scale(
                int(ys.max() - ys.min() + 1),
                int(xs.max() - xs.min() + 1),
                size,
                float(q["multiscale_trigger"]),
                float(q["multiscale_min"]),
            )
        sw = max(1, int(round(w * scale)))
        sh = max(1, int(round(h * scale)))
        total = _count(h, w)
        if scale < 1.0:
            total += _count(sh, sw)
            log.info(
                "Large removal: context pass at %d%% (%dx%d -> %dx%d)",
                round(scale * 100), w, h, sw, sh,
            )

        done = 0
        native, done = _tiled_fill(
            self.engine, region, reg_eff, q, cancel, progress, total, done
        )

        fill = native
        if scale < 1.0:
            small = cv2.resize(region, (sw, sh), interpolation=cv2.INTER_AREA)
            small_eff = (
                cv2.resize(
                    reg_eff.astype(np.float32),
                    (sw, sh),
                    interpolation=cv2.INTER_LINEAR,
                )
                > 0.5
            )
            ctx_small, done = _tiled_fill(
                self.engine, small, small_eff, q, cancel, progress, total, done
            )
            ctx = cv2.resize(ctx_small, (w, h), interpolation=cv2.INTER_CUBIC)
            # Structure from the whole-context pass, fine detail from the
            # native pass — either alone looks wrong (mush vs. seams).
            sigma = float(q["freq_sigma"])
            if sigma > 0:
                detail = native - cv2.GaussianBlur(native, (0, 0), sigma)
            else:
                detail = np.zeros_like(native)
            fill = ctx + detail

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

        stats = InpaintStats(
            engine=self.engine.name,
            windows=total,
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
