"""Tests for the opt-in diagnostic capture: marker gating, saved artifacts,
metadata, failure isolation, and bit-identical removal output."""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest
from PIL import Image

from object_remover import config, diagnostics
from object_remover.diagnostics import DiagnosticCapture, diagnostics_enabled
from object_remover.inpaint import InpaintService


class DummyEngine:
    """Deterministic engine: fills every hole with zeros."""

    name = "dummy"
    input_size = 512

    def fill(self, tile, mask):
        return np.zeros_like(tile)


def _img(h, w, value=30000):
    return np.full((h, w, 3), value, np.uint16)


def _run_removal(h=200, w=200, hole=(80, 120, 80, 120), protect=None):
    """One deterministic removal run; returns the output pixels."""
    svc = InpaintService(DummyEngine())
    pixels = _img(h, w)
    removal = np.zeros((h, w), np.uint8)
    y0, y1, x0, x1 = hole
    removal[y0:y1, x0:x1] = 255
    out, stats = svc.inpaint(pixels, removal, protect)
    return out, stats


@pytest.fixture()
def app_dir(tmp_path, monkeypatch):
    """Isolated app-data dir for both quality.ini and diagnostics."""
    monkeypatch.setattr(config, "app_data_dir", lambda: tmp_path)
    monkeypatch.setattr(diagnostics, "app_data_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture()
def plain_quality(monkeypatch):
    """Disable color/texture corrections so raw fill values are observable."""
    q = dict(config._QUALITY_DEFAULTS)
    q.update(harmonize=0.0, texture=0.0)
    monkeypatch.setattr("object_remover.inpaint.quality", lambda: q)
    return q


def _enable(app_dir):
    (app_dir / "diagnostics.enabled").write_text("", encoding="utf-8")


def _runs(app_dir):
    root = app_dir / "diagnostics"
    return sorted(root.iterdir()) if root.exists() else []


# --- gating -----------------------------------------------------------------

def test_disabled_by_default_writes_nothing(app_dir):
    assert not diagnostics_enabled()
    _run_removal()
    assert _runs(app_dir) == []


def test_marker_enables_capture(app_dir):
    _enable(app_dir)
    assert diagnostics_enabled()


# --- artifacts + metadata ---------------------------------------------------

def test_enabled_writes_expected_files_and_metadata(app_dir, plain_quality):
    _enable(app_dir)
    out, stats = _run_removal()

    runs = _runs(app_dir)
    assert len(runs) == 1
    names = sorted(p.name for p in runs[0].iterdir())
    assert names == sorted([
        "00-original.png",
        "mask.png",
        "01-native.png",
        "04-final.png",
        "metadata.json",
    ])  # small hole: no 02-context.png (no whole-hole seed was needed)

    meta = json.loads((runs[0] / "metadata.json").read_text(encoding="utf-8"))
    assert meta["engine"] == {"name": "dummy", "input_size": 512}
    # metadata records the effective quality settings of the run
    assert meta["quality"] == plain_quality
    assert meta["image"] == {"height": 200, "width": 200}
    # 40x40 hole padded to a full 512 window, clamped to the image
    assert meta["crop"] == {"y0": 0, "y1": 200, "x0": 0, "x1": 200}
    assert meta["context_pass"]["ran"] is False
    assert meta["mask"] == {
        "removal_pixels_in_crop": 40 * 40,
        "protect_pixels_in_crop": 0,
    }
    # the progressive stage records what it actually did, window by window
    prog = meta["progressive_fill"]
    assert prog["calls"] == 1
    assert prog["forced_calls"] == 0
    assert prog["max_hole_share_used"] <= prog["max_window_hole_share"]
    assert len(prog["call_log"]) == 1
    assert prog["call_log"][0]["pixels"] == 40 * 40
    assert sorted(meta["images_written"]) == sorted(names[:-1])
    assert "8-bit" in meta["png_note"]

    # mask.png is the exact effective removal mask over the crop
    mask_img = np.asarray(Image.open(runs[0] / "mask.png"))
    assert mask_img.shape == (200, 200)
    assert set(np.unique(mask_img)) <= {0, 255}
    assert mask_img[80:120, 80:120].all()
    assert not mask_img[:80].any()
    assert not mask_img[120:].any()

    # 00-original is the cropped original region (uint16 30000 -> 8-bit)
    orig_img = np.asarray(Image.open(runs[0] / "00-original.png"))
    assert orig_img.shape == (200, 200, 3)
    assert (orig_img == np.rint(30000 / 257.0)).all()

    # 04-final is the final result crop: filled inside, original outside
    final_img = np.asarray(Image.open(runs[0] / "04-final.png"))
    assert (final_img[95:105, 95:105] == 0).all()  # dummy fill value
    assert (final_img[:75, :] == np.rint(30000 / 257.0)).all()

    # one folder per run, uniquely named
    _run_removal()
    second = _runs(app_dir)
    assert len(second) == 2
    assert second[0].name != second[1].name


def test_whole_hole_seed_captured_when_it_runs(app_dir):
    _enable(app_dir)
    # 900x900 hole in a 1600x1600 image: larger than the window trigger
    _run_removal(1600, 1600, (400, 1300, 400, 1300))

    runs = _runs(app_dir)
    assert len(runs) == 1
    assert (runs[0] / "02-context.png").is_file()
    meta = json.loads((runs[0] / "metadata.json").read_text(encoding="utf-8"))
    assert meta["quality"] == config._QUALITY_DEFAULTS
    cp = meta["context_pass"]
    assert cp["ran"] is True
    assert cp["seeder"] == "ai-whole-hole"
    assert 0.0 < cp["downscale"] < 1.0
    crop = meta["crop"]
    assert cp["scaled_size"]["width"] < crop["x1"] - crop["x0"]
    assert cp["scaled_size"]["height"] < crop["y1"] - crop["y0"]
    # the seed writes plausible content everywhere, then native windows refine
    prog = meta["progressive_fill"]
    assert prog["calls"] > 1
    assert prog["band_px"] == 307  # peel_band * 512
    assert all(
        c["window_hole_share"] <= prog["max_window_hole_share"] for c in prog["call_log"]
    )
    assert sorted(meta["images_written"]) == sorted([
        "00-original.png",
        "mask.png",
        "01-native.png",
        "02-context.png",
        "04-final.png",
    ])


# --- isolation: diagnostics never affect the removal ------------------------

def test_enabled_capture_does_not_change_output_pixels(app_dir):
    def run():
        svc = InpaintService(DummyEngine())
        pixels = _img(300, 300)
        removal = np.zeros((300, 300), np.uint8)
        removal[80:220, 80:220] = 255
        protect = np.zeros((300, 300), np.uint8)
        protect[140:180, 140:180] = 255  # overlap: protect wins
        out, _stats = svc.inpaint(pixels, removal, protect)
        return out

    out_off = run()  # capture disabled (no marker yet)
    _enable(app_dir)
    out_on = run()
    np.testing.assert_array_equal(out_on, out_off)
    assert _runs(app_dir)  # capture really happened


def test_write_failure_logged_and_removal_unchanged(app_dir, monkeypatch, caplog):
    reference, _ = _run_removal()  # capture disabled
    _enable(app_dir)

    def boom(*args, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(diagnostics, "_write_png", boom)
    monkeypatch.setattr(diagnostics, "_write_json", boom)
    with caplog.at_level(logging.ERROR, logger="object_remover.diagnostics"):
        out, stats = _run_removal()
    np.testing.assert_array_equal(out, reference)
    assert stats.engine == "dummy"
    assert any("diagnostic" in r.getMessage().lower() for r in caplog.records)


def test_run_dir_failure_logged_and_removal_unchanged(app_dir, monkeypatch, caplog):
    reference, _ = _run_removal()  # capture disabled
    _enable(app_dir)

    def boom(*args, **kwargs):
        raise OSError("simulated mkdir failure")

    monkeypatch.setattr(diagnostics, "_create_run_dir", boom)
    with caplog.at_level(logging.ERROR, logger="object_remover.diagnostics"):
        out, _stats = _run_removal()
    np.testing.assert_array_equal(out, reference)
    assert _runs(app_dir) == []
    assert any("diagnostic" in r.getMessage().lower() for r in caplog.records)


def test_marker_check_failure_keeps_capture_off(app_dir, monkeypatch, caplog):
    """A broken marker check must fail soft: capture off, error logged."""

    def boom():
        raise OSError("simulated marker check failure")

    monkeypatch.setattr(diagnostics, "app_data_dir", boom)
    with caplog.at_level(logging.ERROR, logger="object_remover.diagnostics"):
        assert diagnostics_enabled() is False
        assert DiagnosticCapture.maybe("dummy", 512, {}) is None
    assert any("diagnostic" in r.getMessage().lower() for r in caplog.records)
