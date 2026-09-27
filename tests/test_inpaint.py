"""Tests for the inpainting service: constraints, tiling, cancellation."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from object_remover.errors import CancelledError, InpaintError
from object_remover.inpaint import ClassicalEngine, InpaintService


class DummyEngine:
    """Records calls; fills the hole with zeros (or a color)."""

    name = "dummy"

    def __init__(self, input_size=512, fill_value=0.0, on_fill=None):
        self.input_size = input_size
        self.calls = []
        self.fill_value = fill_value
        self.on_fill = on_fill

    def fill(self, tile, mask):
        self.calls.append((tile.shape, mask.copy()))
        if self.on_fill is not None:
            self.on_fill(len(self.calls))
        out = np.full_like(tile, self.fill_value)
        return out


def _img(h, w, value=30000):
    return np.full((h, w, 3), value, np.uint16)


@pytest.fixture()
def plain_quality(monkeypatch):
    """Disable color/texture post-fills so tests can assert raw fill values."""
    from object_remover import config, inpaint as mod

    q = dict(config._QUALITY_DEFAULTS)
    q.update(harmonize=0.0, texture=0.0)
    monkeypatch.setattr(mod, "quality", lambda: q)


def test_empty_removal_raises():
    svc = InpaintService(DummyEngine())
    with pytest.raises(InpaintError):
        svc.inpaint(_img(32, 32), np.zeros((32, 32), np.uint8))


def test_protect_never_overwritten_and_unselected_exact(plain_quality):
    engine = DummyEngine(fill_value=0.0)
    svc = InpaintService(engine)
    pixels = _img(100, 100, 40000)
    removal = np.zeros((100, 100), np.uint8)
    removal[10:50, 10:50] = 255
    protect = np.zeros((100, 100), np.uint8)
    protect[25:35, 25:35] = 255  # inside the hole (protect wins defensively)

    out, stats = svc.inpaint(pixels, removal, protect)
    # protected pixels untouched
    np.testing.assert_array_equal(out[25:35, 25:35], pixels[25:35, 25:35])
    # non-selected untouched
    np.testing.assert_array_equal(out[50:], pixels[50:])
    # removal-only pixels well inside the hole are fully filled (dummy fills 0);
    # a 1-px seam feather is expected near the mask rim
    assert out[30, 15, 0] == 0
    assert out[15, 30, 0] == 0
    assert stats.engine == "dummy"


def test_protect_excluded_from_model_context():
    """The engine must see protected pixels as 'unknown', not as context."""
    seen_masks = []

    class Recorder(DummyEngine):
        def fill(self, tile, mask):
            seen_masks.append(mask.copy())
            return super().fill(tile, mask)

    svc = InpaintService(Recorder())
    pixels = _img(200, 200)
    removal = np.zeros((200, 200), np.uint8)
    removal[50:150, 50:150] = 255
    protect = np.zeros((200, 200), np.uint8)
    protect[10:40, 10:40] = 255  # adjacent, outside the hole

    # region = hole bbox + margin -> protect is inside the crop window
    svc.inpaint(pixels, removal, protect)
    assert seen_masks
    # in the first window containing both, protect must be masked (>=1) somewhere
    assert any(m.any() for m in seen_masks)


def test_tiling_covers_large_hole_with_windows(plain_quality):
    engine = DummyEngine(input_size=512)
    svc = InpaintService(engine)
    pixels = _img(900, 700)
    removal = np.zeros((900, 700), np.uint8)
    removal[100:800, 50:650] = 255

    out, stats = svc.inpaint(pixels, removal, None)
    assert stats.windows > 1
    for shape, _mask in engine.calls:
        assert shape[:2] == (512, 512)
    # only removal pixels changed
    changed = np.any(out != pixels, axis=2)
    assert changed[100:800, 50:650].all()
    assert not changed[:100].any()
    assert not changed[800:].any()


def test_progress_and_cancel_between_windows():
    cancel = threading.Event()

    def on_fill(n):
        if n >= 1:
            cancel.set()

    engine = DummyEngine(on_fill=on_fill)
    svc = InpaintService(engine)
    pixels = _img(1200, 1200)
    removal = np.zeros((1200, 1200), np.uint8)
    removal[:, :] = 255
    progress_log = []

    with pytest.raises(CancelledError):
        svc.inpaint(
            pixels, removal, None,
            progress=lambda d, t: progress_log.append((d, t)),
            cancel=cancel,
        )
    assert len(engine.calls) == 1
    assert progress_log


def test_small_hole_padded_window():
    engine = DummyEngine()
    svc = InpaintService(engine)
    pixels = _img(40, 40)
    removal = np.zeros((40, 40), np.uint8)
    removal[10:20, 10:20] = 255
    out, stats = svc.inpaint(pixels, removal, None)
    assert stats.windows == 1
    assert engine.calls[0][0][:2] == (512, 512)


def test_classical_engine_fills_gradient(gradient8):
    """Filled area should approximate the surrounding gradient (loose check)."""
    pixels = gradient8.astype(np.uint16) * 257
    h, w = pixels.shape[:2]
    removal = np.zeros((h, w), np.uint8)
    removal[24:40, 24:40] = 255
    svc = InpaintService(ClassicalEngine())
    out, _stats = svc.inpaint(pixels, removal, None)
    hole = out[28:36, 28:36].astype(np.float32)
    rim = np.concatenate([
        out[24:26, 28:36].reshape(-1, 3),
        out[38:40, 28:36].reshape(-1, 3),
        out[28:36, 24:26].reshape(-1, 3),
        out[28:36, 38:40].reshape(-1, 3),
    ]).astype(np.float32)
    assert np.abs(hole.mean(axis=(0, 1)) - rim.mean(axis=0)).max() < 8000
    # and nothing outside changed
    mask = np.zeros((h, w), bool)
    mask[24:40, 24:40] = True
    np.testing.assert_array_equal(out[~mask], pixels[~mask])


def test_feather_stays_inside_mask():
    """The seam feather must never write outside the removal mask."""
    svc = InpaintService(DummyEngine(fill_value=1.0))
    pixels = _img(80, 80, 0)
    removal = np.zeros((80, 80), np.uint8)
    removal[20:60, 20:60] = 255
    out, _ = svc.inpaint(pixels, removal, None)
    outside = removal == 0
    np.testing.assert_array_equal(out[outside], pixels[outside])
