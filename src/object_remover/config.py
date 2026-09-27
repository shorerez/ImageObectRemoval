"""Application configuration and platform paths."""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "ObjectRemover"
APP_VERSION = "0.1.0"

# --- AI model (LaMa ONNX, Apache-2.0 weights, Carve/LaMa-ONNX export) ---
MODEL_FILENAME = "lama_fp32.onnx"
MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
# Pinned SHA-256 of MODEL_URL (the LFS object hash published for
# lama_fp32.onnx, 208,044,816 bytes). Downloads whose hash does not match are
# rejected; set to None to fall back to trust-on-first-use.
MODEL_SHA256: str | None = (
    "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"
)
MODEL_INPUT_SIZE = 512  # fixed input shape of the ONNX export
MODEL_IMAGE_INPUT = "image"
MODEL_MASK_INPUT = "mask"
MODEL_OUTPUT = "output"

# --- Inpainting ---
TILE_OVERLAP = 128          # overlap between native-resolution inference windows
CONTEXT_MARGIN = 32         # extra context around the removal bounding box
FEATHER_SIGMA = 1.0         # seam feather (px) inside the removal-mask edge

# --- UI ---
PREVIEW_MAX_SIDE = 4096     # preview pyramid cap; masks stay full resolution
DEFAULT_BRUSH_SIZE = 80     # full-resolution pixels
MIN_BRUSH_SIZE = 2
MAX_BRUSH_SIZE = 800
DEFAULT_HARDNESS = 0.5      # 0 = very soft .. 1 = hard edge

REMOVAL_COLOR = (255, 64, 64)   # overlay RGB
PROTECT_COLOR = (64, 220, 96)
OVERLAY_ALPHA = 110             # 0..255

JPG_QUALITY_DEFAULT = 95
JPG_QUALITY_MIN = 50
JPG_QUALITY_MAX = 100

HISTORY_LIMIT = 200

SUPPORTED_INPUT_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg"}
SUPPORTED_JPG_SUFFIXES = {".jpg", ".jpeg"}
SUPPORTED_TIFF_SUFFIXES = {".tif", ".tiff"}


def app_data_dir() -> Path:
    """Per-user application data directory (created on demand)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(
            Path.home() / "AppData" / "Local"
        )
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    path = Path(base) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = app_data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path
