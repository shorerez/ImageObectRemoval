"""Tests for the quality pipeline: context expansion, multi-scale pass,
runtime quality.ini overrides."""
from __future__ import annotations

import numpy as np

from object_remover import config, inpaint as mod
from object_remover.inpaint import InpaintService, _context_scale, _plan_region


class DummyEngine:
    """Records calls; fills the hole with zeros."""

    name = "dummy"

    def __init__(self, input_size=512):
        self.input_size = input_size
        self.calls = []

    def fill(self, tile, mask):
        self.calls.append((tile.shape, mask.copy()))
        return np.zeros_like(tile)


def _img(h, w, value=30000):
    return np.full((h, w, 3), value, np.uint16)


def test_plan_region_expands_to_full_window():
    """A small hole in a large image must get one full window of real context."""
    ys, xs = np.nonzero(np.pad(np.ones((100, 100), bool), 950))
    y0, y1, x0, x1 = _plan_region((2000, 2000), ys, xs, 512, 192)
    assert (y1 - y0, x1 - x0) == (512, 512)
    # the window is centred on the hole
    assert y0 <= 950 and y1 >= 1050 and x0 <= 950 and x1 >= 1050


def test_plan_region_keeps_context_ring_for_large_hole():
    ys, xs = np.nonzero(np.pad(np.ones((800, 800), bool), 600))
    y0, y1, x0, x1 = _plan_region((2000, 2000), ys, xs, 512, 192)
    assert y0 <= 600 - 192 and y1 >= 1400 + 192
    assert x0 <= 600 - 192 and x1 >= 1400 + 192


def test_plan_region_clamps_to_image():
    ys, xs = np.nonzero(np.ones((40, 40), bool))
    y0, y1, x0, x1 = _plan_region((40, 40), ys, xs, 512, 192)
    assert (y0, y1, x0, x1) == (0, 40, 0, 40)


def test_context_scale_bounds():
    assert _context_scale(100, 80, 512, 0.6, 0.125) == 1.0
    s = _context_scale(700, 600, 512, 0.6, 0.125)
    assert 0.4 < s < 0.45
    assert _context_scale(20000, 20000, 512, 0.6, 0.125) == 0.125


def test_large_hole_runs_scaled_context_pass(monkeypatch):
    """A hole larger than the window must trigger a downscaled context pass."""
    seen = []
    orig = mod._tiled_fill

    def spy(engine, region, reg_eff, q, cancel, progress, total, done):
        seen.append(region.shape[:2])
        return orig(engine, region, reg_eff, q, cancel, progress, total, done)

    monkeypatch.setattr(mod, "_tiled_fill", spy)
    engine = DummyEngine()
    svc = InpaintService(engine)
    pixels = _img(1600, 1600)
    removal = np.zeros((1600, 1600), np.uint8)
    removal[400:1300, 400:1300] = 255  # 900x900 hole

    out, stats = svc.inpaint(pixels, removal, None)
    assert len(seen) == 2  # native pass + scaled context pass
    assert max(seen[1]) < max(seen[0])
    # invariants: only removal pixels change
    changed = np.any(out != pixels, axis=2)
    assert changed[400:1300, 400:1300].all()
    assert not changed[:400].any()
    assert not changed[1300:].any()
    assert stats.windows > 1


def test_small_hole_single_window_single_pass():
    engine = DummyEngine()
    svc = InpaintService(engine)
    pixels = _img(2000, 2000)
    removal = np.zeros((2000, 2000), np.uint8)
    removal[1000:1100, 1000:1100] = 255
    _out, stats = svc.inpaint(pixels, removal, None)
    assert stats.windows == 1
    assert len(engine.calls) == 1
    assert engine.calls[0][0][:2] == (512, 512)


def test_quality_ini_overrides(monkeypatch, tmp_path):
    (tmp_path / "quality.ini").write_text(
        "# tune\nfreq_sigma = 3.5\nharmonize=0\nmultiscale = 0\nbogus = 9\n[skip]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "app_data_dir", lambda: tmp_path)
    q = config.quality()
    assert q["freq_sigma"] == 3.5
    assert q["harmonize"] == 0.0
    assert q["multiscale"] == 0.0
    assert q["blend_band"] == config._QUALITY_DEFAULTS["blend_band"]


def test_quality_defaults_without_ini(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "app_data_dir", lambda: tmp_path)
    assert config.quality() == config._QUALITY_DEFAULTS
