"""Opt-in, fully local diagnostic capture of one removal run.

Disabled by default. When the marker file ``diagnostics.enabled`` exists in
``config.app_data_dir()`` (on Windows: ``%LOCALAPPDATA%\\ObjectRemover``),
every removal run writes a uniquely named folder under
``<app data>/diagnostics/`` holding the per-stage fills of the inpainting
pipeline as 8-bit PNGs plus ``metadata.json``, so the failing stage can be
compared offline.

Guarantees:
- fully local: files are written only under the local diagnostics folder,
  no upload or network activity;
- best-effort: any file/directory/write error is logged and swallowed, so
  enabling capture can never abort or alter the actual removal;
- read-only with respect to the pipeline: capture never mutates the arrays
  it is given, so removal output is bit-identical with capture on or off.

Limitation: diagnostic images are 8-bit PNGs and may not preserve the
ICC/color-management appearance of the original working buffer. Compare
structure and relative color between stages, not against the exported file.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from .config import APP_VERSION, app_data_dir

log = logging.getLogger(__name__)

MARKER_FILENAME = "diagnostics.enabled"
DIAGNOSTICS_DIRNAME = "diagnostics"
PNG_NOTE = (
    "Diagnostic images are 8-bit PNGs and may not preserve the "
    "ICC/color-management appearance of the original. All files are written "
    "locally only; nothing is uploaded."
)


def diagnostics_enabled() -> bool:
    """True only when the opt-in marker file exists. Never raises."""
    try:
        return (app_data_dir() / MARKER_FILENAME).is_file()
    except Exception:
        log.exception("diagnostics: marker check failed; capture stays off")
        return False


def _to_u8(arr: np.ndarray) -> np.ndarray:
    """8-bit view of a stage image (round-to-nearest, clipped)."""
    a = np.asarray(arr)
    if a.dtype == np.bool_:
        return a.astype(np.uint8) * 255
    if a.dtype == np.uint8:
        return a
    if a.dtype == np.uint16:
        return np.rint(a.astype(np.float32) / 257.0).astype(np.uint8)
    return np.clip(np.rint(a.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)


def _write_png(path: Path, image: np.ndarray) -> None:
    Image.fromarray(_to_u8(image)).save(path, format="PNG")


def _write_json(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _create_run_dir(root: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{stamp}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


class DiagnosticCapture:
    """Per-run collector: writes stage PNGs as they are recorded and
    ``metadata.json`` at the end. Every method is best-effort and never
    raises, so diagnostics cannot interfere with the removal itself."""

    def __init__(self, engine: dict, quality: dict, run_dir: Path) -> None:
        self._engine = engine
        self._quality = quality
        self._run_dir = run_dir
        self._images: list[str] = []

    @classmethod
    def maybe(
        cls, engine_name: str, engine_input_size: int, quality: dict
    ) -> "DiagnosticCapture | None":
        """Create a capture when the marker file exists; never raises.

        Returns None when capture is disabled or setup fails (logged).
        """
        try:
            if not diagnostics_enabled():
                return None
            run_dir = _create_run_dir(app_data_dir() / DIAGNOSTICS_DIRNAME)
            log.info("diagnostics: capturing removal run to %s", run_dir)
            return cls(
                {"name": str(engine_name), "input_size": int(engine_input_size)},
                dict(quality),
                run_dir,
            )
        except Exception:
            log.exception(
                "diagnostics: capture setup failed; continuing without capture"
            )
            return None

    def add_image(self, name: str, image: np.ndarray) -> None:
        """Write one stage image immediately; errors are logged, never raised."""
        try:
            _write_png(self._run_dir / name, image)
            self._images.append(name)
        except Exception:
            log.exception("diagnostics: failed to write %s", name)

    def finish(
        self,
        *,
        crop: tuple[int, int, int, int],
        image_size: tuple[int, int],
        mask: dict,
        context_pass: dict,
        ladder: list[dict] | None = None,
        progressive: dict | None = None,
    ) -> None:
        """Write ``metadata.json``; errors are logged, never raised."""
        try:
            y0, y1, x0, x1 = (int(v) for v in crop)
            meta = {
                "app_version": APP_VERSION,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "engine": self._engine,
                "quality": self._quality,
                "image": {"height": int(image_size[0]), "width": int(image_size[1])},
                "crop": {"y0": y0, "y1": y1, "x0": x0, "x1": x1},
                "mask": mask,
                "context_pass": context_pass,
                "ladder": ladder,
                "progressive_fill": progressive,
                "images_written": list(self._images),
                "png_note": PNG_NOTE,
            }
            _write_json(self._run_dir / "metadata.json", meta)
        except Exception:
            log.exception("diagnostics: failed to write metadata.json")
