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
    """LaMa ONNX (Carve/LaMa-ONNX): fixed inputs image/mask -> output.

    The published ``lama_fp32.onnx`` export takes 0..1 inputs but returns RGB
    on a **0..255** scale (the upstream demo casts its output straight to
    ``uint8``). Re-exports such as ``sapienkit/LaMa-ONNX`` document the same
    contract, while other exports answer in 0..1. Clipping the raw output to
    [0, 1] therefore turns every fill solid white, so the scale is *measured*
    once in the constructor with a synthetic self-test instead of assumed.
    """

    name = "LaMa (AI)"

    #: level/probe ratio above which the output must be 0..255 (a 1x engine
    #: answers ~1x, a 255x engine ~255x, so the margin is huge either way)
    _SCALE_RATIO = 8.0
    #: a sane fill stays inside [-2, 4] once divided by the detected scale
    _MAX_SANE = 4.0
    _MIN_SANE = -2.0

    def __init__(self, session, input_size: int = MODEL_INPUT_SIZE) -> None:
        self._session = session
        self.input_size = int(input_size)
        self._out_name = MODEL_OUTPUT
        self.output_scale = self._detect_output_scale()

    # ------------------------------------------------------------- inference
    def _run(self, tile: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """One inference call; returns float32 (S, S, 3) at the model's scale."""
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
        out = np.asarray(out[0]).transpose(1, 2, 0)
        return out.astype(np.float32, copy=False)

    def _detect_output_scale(self) -> float:
        """Measure whether the model answers in 0..1 or 0..255.

        Feeds a uniform 0.5 gray tile with a center hole: a working engine
        echoes the known (unmasked) pixels, so the output level divided by the
        0.5 input is 1x or 255x. Non-finite or absurd answers mean the provider
        is broken and the caller should reject it.
        """
        s = self.input_size
        probe_level = 0.5
        tile = np.full((s, s, 3), probe_level, dtype=np.float32)
        mask = np.zeros((s, s), dtype=np.float32)
        h0, h1 = s // 4, s - s // 4
        mask[h0:h1, h0:h1] = 1.0

        out = self._run(tile, mask)
        if out.ndim != 3 or out.shape[0] != s or out.shape[1] != s or out.shape[2] < 3:
            raise InpaintError(
                f"ONNX output shape {out.shape} is not ({s}, {s}, >=3)."
            )
        if not np.isfinite(out).all():
            raise InpaintError("ONNX output contains non-finite values (NaN/Inf).")

        known = mask < 0.5
        # A working engine echoes the known pixels, so the context level divided
        # by the 0.5 probe input is 1x or 255x. The full-output mean is a second
        # opinion for engines that only return the filled region.
        measured = max(
            float(np.mean(out[known])) if known.any() else 0.0,
            float(np.mean(out)),
        )
        if not np.isfinite(measured):  # pragma: no cover - defensive
            raise InpaintError("ONNX output level is not finite.")
        scale = 255.0 if measured / probe_level > self._SCALE_RATIO else 1.0

        scaled = out / scale
        lo, hi = float(scaled.min()), float(scaled.max())
        if hi > self._MAX_SANE or lo < self._MIN_SANE:
            raise InpaintError(
                "ONNX output is out of range after scale detection "
                f"(level {measured:.3g}, range {lo:.3g}..{hi:.3g})."
            )
        log.info(
            "LaMa ONNX output scale detected: %.0fx (probe level %.4g -> %.4g)",
            scale, probe_level, measured,
        )
        return scale

    def fill(self, tile: np.ndarray, mask: np.ndarray) -> np.ndarray:
        s = self.input_size
        if tile.shape[0] != s or tile.shape[1] != s:
            raise InpaintError(f"OnnxLamaEngine expects {s}x{s} tiles, got {tile.shape}.")
        out = self._run(tile, mask)
        out = out / self.output_scale
        if not np.isfinite(out).all():
            raise InpaintError("LaMa ONNX returned non-finite values.")
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
    """Best available engine: LaMa ONNX (CUDA first, then CPU), else classical.

    Each provider gets its own attempt: a session that loads but produces
    unusable output (bad provider, corrupt weights) must not stop the app from
    starting, so the next provider is tried and the classical engine is the
    last resort.
    """
    if model_path is not None:
        from .runtime import available_providers, create_onnx_session

        avail = available_providers()
        providers = [
            p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail
        ] or ["CPUExecutionProvider"]
        for provider in providers:
            try:
                session = create_onnx_session(model_path, providers=[provider])
                engine: Engine = OnnxLamaEngine(session)
                log.info(
                    "Inpaint engine: %s (%s, provider=%s, output scale=%.0fx)",
                    engine.name, model_path, provider, engine.output_scale,
                )
                return engine
            except Exception as exc:
                log.warning(
                    "ONNX engine unavailable with %s (%s); trying the next provider.",
                    provider, exc,
                )
    return ClassicalEngine()
