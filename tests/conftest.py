"""Shared fixtures: sys.path setup, synthetic images, Qt app fixture."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def rgb16(rng) -> np.ndarray:
    """Random 64x48x3 uint16 image."""
    return rng.integers(0, 65536, size=(64, 48, 3), dtype=np.uint16)


@pytest.fixture()
def gradient8() -> np.ndarray:
    """Smooth 64x64x3 uint8 gradient (good inpainting test content)."""
    y, x = np.mgrid[0:64, 0:64].astype(np.float32)
    img = np.stack([x * 4, y * 4, (x + y) * 2], axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture()
def tiff_path(tmp_path, rgb16) -> Path:
    import tifffile

    path = tmp_path / "sample.tif"
    tifffile.imwrite(path, rgb16, photometric="rgb")
    return path


@pytest.fixture()
def qt_app():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
