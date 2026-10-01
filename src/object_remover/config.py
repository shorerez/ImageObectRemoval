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
# Pinned SHA-256 of MODEL_URL. None = trust-on-first-use: the hash recorded on
# the first successful download is verified on every later load.
MODEL_SHA256: str | None = (
    "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"
)
MODEL_INPUT_SIZE = 512  # fixed input shape of the ONNX export
MODEL_IMAGE_INPUT = "image"
MODEL_MASK_INPUT = "mask"
MODEL_OUTPUT = "output"

# --- Inpainting quality ---
TILE_OVERLAP = 128          # overlap between native-resolution inference windows
CONTEXT_MARGIN = 192        # real-context ring around the removal bbox (px/side)
FEATHER_SIGMA = 1.0         # seam feather (px) inside the removal-mask edge
TILE_BLEND_BAND = 32        # crossfade width at window-territory borders (px)
MULTISCALE_TRIGGER = 0.6    # hole/window ratio that triggers the whole-hole seed
MULTISCALE_MIN = 0.125      # smallest downscale factor of the whole-hole seed
PEEL_BAND = 0.6             # band width (fraction of the window) refined per pass
PEEL_MAX_UNKNOWN = 0.75     # hard cap on the masked share of a single window
FREQ_SIGMA = 8.0            # deprecated: kept so existing quality.ini files parse
HARMONIZE = 1.0             # 0..1; Lab mean/std match of fill to surroundings
HARMONIZE_BAND = 24         # px; boundary band used for the color match
TEXTURE = 1.0               # 0..1; high-frequency grain transplanted into fill
TEXTURE_RADIUS = 96         # px; ring around the hole sampled for grain
TEXTURE_SIGMA = 4.0         # px; high-pass cutoff of the transplanted grain

# Runtime quality tuning: defaults overridden by quality.ini in app_data_dir().
# The file is re-read on every removal pass, so tuning needs no restart:
#   harmonize=1.0         # 0 = no color match .. 1 = full boundary-statistics match
#   harmonize_band=24     # boundary band width for the color match
#   texture=1.0           # 0 = no grain transplant .. 1 = full scene grain
#   texture_radius=96     # how far around the hole grain is sampled
#   texture_sigma=4.0     # grain size; lower = finer grain
#   blend_band=32         # window crossfade width; lower = sharper seams
#   context_margin=192    # real context ring; higher = better but slower
#   multiscale=1          # 0 = classical whole-hole seed instead of the AI seed
#   peel_band=0.6         # band width per native pass (fraction of the window);
#                         # lower = more, better-conditioned passes
#   peel_max_unknown=0.75 # never ask a window to fill more than this share of it
#   feather_sigma=1.0     # seam feather inside the mask edge
# freq_sigma is accepted (old quality.ini files) but no longer used.
_QUALITY_DEFAULTS = {
    "context_margin": float(CONTEXT_MARGIN),
    "feather_sigma": float(FEATHER_SIGMA),
    "blend_band": float(TILE_BLEND_BAND),
    "multiscale": 1.0,
    "multiscale_trigger": float(MULTISCALE_TRIGGER),
    "multiscale_min": float(MULTISCALE_MIN),
    "peel_band": float(PEEL_BAND),
    "peel_max_unknown": float(PEEL_MAX_UNKNOWN),
    "freq_sigma": float(FREQ_SIGMA),
    "harmonize": float(HARMONIZE),
    "harmonize_band": float(HARMONIZE_BAND),
    "texture": float(TEXTURE),
    "texture_radius": float(TEXTURE_RADIUS),
    "texture_sigma": float(TEXTURE_SIGMA),
}


def quality() -> dict:
    """Effective quality settings: defaults + quality.ini overrides.

    Unknown keys and invalid values are ignored; a missing file means defaults.
    """
    q = dict(_QUALITY_DEFAULTS)
    try:
        text = (app_data_dir() / "quality.ini").read_text(encoding="utf-8")
    except OSError:
        return q
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";", "[")) or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().lower()
        if key not in q:
            continue
        try:
            q[key] = float(value.strip())
        except ValueError:
            pass
    return q


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
