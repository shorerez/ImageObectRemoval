"""Tests for the quality pipeline: crop planning, whole-hole seed, the
coarse-to-fine ladder, progressive native refinement, color harmonization,
texture, quality.ini overrides."""
from __future__ import annotations

import cv2
import numpy as np

from object_remover import config, inpaint as mod
from object_remover.inpaint import (
    InpaintService,
    _context_scale,
    _ladder_scales,
    _plan_region,
)

_ORIG_PROGRESSIVE_FILL = mod._progressive_fill


class DummyEngine:
    """Records calls; fills the hole with zeros."""

    name = "dummy"

    def __init__(self, input_size=512):
        self.input_size = input_size
        self.calls = []

    def fill(self, tile, mask):
        self.calls.append((tile.shape, mask.copy()))
        return np.zeros_like(tile)


class FlatEngine:
    """Fills every tile with a constant (smooth, texture-free output)."""

    name = "flat"
    input_size = 512

    def __init__(self, value=0.5):
        self.value = value

    def fill(self, tile, mask):
        return np.full_like(tile, self.value)


def _img(h, w, value=30000):
    return np.full((h, w, 3), value, np.uint16)


def _quality(monkeypatch, **over):
    q = dict(config._QUALITY_DEFAULTS)
    q.update(over)
    monkeypatch.setattr(mod, "quality", lambda: q)
    return q


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


def test_ladder_scales_climb_towards_native():
    assert _ladder_scales(1.0, 1.8, 2) == []
    assert _ladder_scales(0.3413, 1.8, 2) == [0.61434]
    scales = _ladder_scales(0.13, 1.8, 2)
    assert len(scales) == 2
    assert 0.13 < scales[0] < scales[1] < 0.95
    assert abs(scales[1] - scales[0] * 1.8) < 1e-9
    assert _ladder_scales(0.6, 1.8, 1) == []  # 1.08 >= 0.95: no extra level


def _big_hole_case():
    pixels = _img(1600, 1600)
    removal = np.zeros((1600, 1600), np.uint8)
    removal[400:1300, 400:1300] = 255  # 900x900 hole, far above the trigger
    return pixels, removal


class Recorder(DummyEngine):
    """Records every window the engine is asked to fill."""

    def __init__(self, input_size=512):
        super().__init__(input_size)
        self.masks = []

    def fill(self, tile, mask):
        self.masks.append(mask.copy())
        return super().fill(tile, mask)


def test_large_hole_seeds_then_refines_natively(monkeypatch):
    """A hole larger than the trigger is seeded from a downscaled canvas and
    then refined with native-resolution windows."""
    _quality(monkeypatch, harmonize=0.0, texture=0.0)
    seen = []
    orig = mod._tiled_fill

    def spy(engine, region, reg_eff, q, cancel, progress, total, done):
        seen.append(region.shape[:2])
        return orig(engine, region, reg_eff, q, cancel, progress, total, done)

    monkeypatch.setattr(mod, "_tiled_fill", spy)
    engine = Recorder()
    svc = InpaintService(engine)
    pixels, removal = _big_hole_case()

    out, stats = svc.inpaint(pixels, removal, None)
    # exactly one seed pass, on the downscaled canvas
    assert len(seen) == 1
    assert max(seen[0]) < 512
    # ... followed by native-resolution refinement windows only
    assert engine.calls
    assert all(shape[:2] == (512, 512) for shape, _mask in engine.calls)
    # invariants: only removal pixels change
    changed = np.any(out != pixels, axis=2)
    assert changed[400:1300, 400:1300].all()
    assert not changed[:400].any()
    assert not changed[1300:].any()
    # every engine call is counted, the seed window included
    assert stats.windows == len(engine.calls) > 1


def test_no_window_is_asked_to_fill_a_mostly_empty_tile(monkeypatch):
    """The failure mode of the old single native pass: windows that are almost
    entirely hole. The progressive stage must never exceed the cap, and the
    seed must run with real context around the hole, not a padded tile."""
    q = _quality(monkeypatch, harmonize=0.0, texture=0.0)
    engine = Recorder()
    svc = InpaintService(engine)
    pixels, removal = _big_hole_case()
    svc.inpaint(pixels, removal, None)

    shares = [float(m.mean()) for m in engine.masks]
    assert len(shares) > 1  # seed window first, then the progressive stage
    seed_share, peel_shares = shares[0], shares[1:]
    assert max(peel_shares) <= q["peel_max_unknown"] + 1e-6
    assert seed_share < 0.6


class MarkerEngine:
    """Fills every window with its own call number (1/255 per call) and keeps
    a cheap per-call record: masked share and how much of the visible tile is
    still the blurry whole-hole seed (marker of call 1)."""

    name = "marker"
    input_size = 512

    def __init__(self):
        self.n = 0
        self.stats = []  # index, mask_share, seed_share

    def fill(self, tile, mask):
        self.n += 1
        seed_share = float(
            ((np.abs(tile[..., 0] - 1.0 / 255.0) < 1e-6) & (mask < 0.5)).mean()
        )
        self.stats.append((self.n, float(mask.mean()), seed_share))
        return np.full_like(tile, self.n / 255.0)


def _marked_run(monkeypatch, **over):
    """Run the big-hole case; returns a dict with the quality dict, the call
    index map, the pixels/mask, the engine (with per-call conditioning) and
    the call range of every progressive stage (spied)."""
    q = _quality(monkeypatch, harmonize=0.0, texture=0.0, **over)
    engine = MarkerEngine()
    stages: list[dict] = []

    def spy(eng, fill, reg_rem, prot, qq, cancel, progress, total, done):
        first = len(eng.stats)
        result = _ORIG_PROGRESSIVE_FILL(
            eng, fill, reg_rem, prot, qq, cancel, progress, total, done
        )
        stages.append(
            {"canvas": fill.shape[:2], "first": first, "calls": len(eng.stats) - first}
        )
        return result

    monkeypatch.setattr(mod, "_progressive_fill", spy)
    svc = InpaintService(engine)
    pixels, removal = _big_hole_case()
    out, _stats = svc.inpaint(pixels, removal, None)
    idx = np.rint(out[..., 0].astype(np.float32) / 65535.0 * 255.0).astype(int)
    return {
        "q": q, "idx": idx, "pixels": pixels, "removal": removal,
        "engine": engine, "stages": stages,
    }


def test_first_native_windows_stay_near_the_real_boundary(monkeypatch):
    """The first windows of the native stage may only mask pixels within one
    peel band of real image content — that is what keeps structure
    continuation possible."""
    run = _marked_run(monkeypatch)
    native = run["stages"][-1]  # native scale is always the last stage
    hole = run["removal"] > 0
    dist = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 5)
    band = run["q"]["peel_band"] * 512
    first_round = hole & (run["idx"] == native["first"] + 1)  # marker is 1-based
    assert first_round.any()
    assert int(dist[first_round].max()) <= band + 1.5


def test_ladder_rebuilds_the_hole_before_native_scale(monkeypatch):
    """Between the whole-hole seed and native scale the fill is rebuilt at
    intermediate scales (coarse to fine), so the native stage stands on
    structure rather than on the blurry seed."""
    run = _marked_run(monkeypatch)
    stages = run["stages"]
    assert len(stages) > 1  # seed, ladder level(s), native
    native_h, native_w = stages[-1]["canvas"]
    ladder = stages[:-1]
    for stage in ladder:
        h, w = stage["canvas"]
        assert 0 < h < native_h and 0 < w < native_w  # finer than native
        assert stage["calls"] > 0
    # scales ascend by ladder_ratio and stay below native
    scales = [h / native_h for h, _w in (s["canvas"] for s in ladder)]
    assert scales == sorted(scales)
    assert len(scales) <= int(run["q"]["ladder_levels"])
    ratio = run["q"]["ladder_ratio"]
    for a, b in zip(scales, scales[1:]):
        assert abs(b - a * ratio) < 1e-3


def test_ladder_disabled_jumps_straight_to_native(monkeypatch):
    run = _marked_run(monkeypatch, ladder=0.0)
    assert len(run["stages"]) == 1


def test_native_stage_never_sees_only_the_blurry_seed(monkeypatch):
    """With the ladder enabled the first native call must already stand on
    refined (mid-scale) content, not on the 20-30% whole-hole seed: that is
    the difference between extending structure and extending a blur."""
    run = _marked_run(monkeypatch)
    first_native = run["engine"].stats[run["stages"][-1]["first"]]
    mask_share, seed_share = first_native[1], first_native[2]
    assert mask_share <= run["q"]["peel_max_unknown"] + 1e-6
    assert seed_share < 0.05  # the visible tile is refined content, not seed


def test_without_the_ladder_the_native_stage_stands_on_the_seed(monkeypatch):
    """Control for the previous test: with ladder=0 the very first native
    window is mostly the blurry whole-hole seed. The ladder is what removes
    that conditioning; this is measured, not assumed."""
    run = _marked_run(monkeypatch, ladder=0.0)
    native_first = run["stages"][0]["first"]
    seed_share = run["engine"].stats[native_first][2]
    assert seed_share > 0.2


def test_large_hole_is_refined_over_several_rounds(monkeypatch):
    """A hole far deeper than the peel band cannot be closed by one round: the
    pixels in its middle are only reachable once the front of trusted content
    has advanced, so they carry a later call number."""
    run = _marked_run(monkeypatch)
    idx, removal = run["idx"], run["removal"]
    native_first = run["stages"][-1]["first"] + 1  # 1-based marker of call 1
    hole = removal > 0
    band = int(run["q"]["peel_band"] * 512)
    dist = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 5)
    core = hole & (dist > band + 64)  # deeper than one peel band
    assert core.any()
    assert (idx[core] >= native_first).all()  # never in the first native round
    assert idx[core].min() > idx[hole & (idx >= native_first)].min()
    # every hole pixel was rewritten by the native stage (the ladder left the
    # coarse fill there): no pixel may be left behind because no window
    # claimed it
    interior = hole & (dist >= 4)  # the 1 px seam feather blends at the rim
    assert (idx[interior] >= native_first).all()
    assert set(np.unique(idx[hole])) <= set(range(1, 300))


def test_removed_content_never_becomes_model_context(monkeypatch):
    """The object being removed must never be visible to the model: inside the
    removal region every native window either masks the pixel or finds seed
    content there, never the original pixels."""
    _quality(monkeypatch, harmonize=0.0, texture=0.0)

    class FlatEngine:
        name = "flat"
        input_size = 512

        def __init__(self):
            self.tiles = []

        def fill(self, tile, mask):
            self.tiles.append((tile.copy(), mask.copy()))
            return np.full_like(tile, 0.25)

    engine = FlatEngine()
    svc = InpaintService(engine)
    pixels = np.full((1600, 1600, 3), int(0.2 * 65535), np.uint16)
    removal = np.zeros((1600, 1600), np.uint8)
    removal[400:1300, 400:1300] = 255
    pixels[removal > 0] = 65535  # the "object": pure white
    svc.inpaint(pixels, removal, None)

    native_tiles = [(t, m) for t, m in engine.tiles if t.shape == (512, 512, 3)]
    assert native_tiles  # the progressive stage really ran here
    for tile, mask in native_tiles:
        leaked = (tile[..., 0] > 0.9) & (mask < 0.5)
        assert not leaked.any()


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


def test_harmonize_matches_boundary_color():
    """A green engine fill on a blue scene must be color-corrected toward blue."""
    pixels = np.zeros((900, 900, 3), np.uint16)
    pixels[..., 0], pixels[..., 1], pixels[..., 2] = 12000, 24000, 52000
    removal = np.zeros((900, 900), np.uint8)
    removal[350:550, 350:550] = 255

    class GreenEngine(FlatEngine):
        name = "green"

        def fill(self, tile, mask):
            out = np.zeros_like(tile)
            out[..., 1] = 1.0
            return out

    svc = InpaintService(GreenEngine())
    out, _ = svc.inpaint(pixels, removal, None)
    hole = out[380:520, 380:520].astype(np.float32)
    ring = np.concatenate(
        [out[330:345, 380:520].reshape(-1, 3), out[555:570, 380:520].reshape(-1, 3)]
    ).astype(np.float32)
    assert np.abs(hole.mean(axis=(0, 1)) - ring.mean(axis=0)).max() < 3000
    # outside the mask stays bit-exact
    assert (out[:350] == pixels[:350]).all()


def test_texture_adds_scene_matched_grain(monkeypatch):
    """A smooth fill on a noisy scene must receive the scene's grain."""
    _quality(monkeypatch, harmonize=0.0)
    rng = np.random.default_rng(7)
    noise = np.clip(0.45 + 0.08 * rng.standard_normal((900, 900, 3)), 0, 1)
    pixels = (noise * 65535).astype(np.uint16)
    removal = np.zeros((900, 900), np.uint8)
    removal[350:550, 350:550] = 255

    svc = InpaintService(FlatEngine(0.5))
    out, _ = svc.inpaint(pixels, removal, None)
    out32 = out.astype(np.float32) / 65535.0
    hp = out32 - cv2.GaussianBlur(out32, (0, 0), 4.0)
    hole_hp = float(hp[380:520, 380:520].std())
    ring_hp = float(hp[250:330, 380:520].std())
    assert ring_hp > 0.01
    assert hole_hp > 0.5 * ring_hp


def test_quality_ini_overrides(monkeypatch, tmp_path):
    (tmp_path / "quality.ini").write_text(
        "# tune\nfreq_sigma = 3.5\nharmonize=0\ntexture = 0.4\nbogus = 9\n[skip]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "app_data_dir", lambda: tmp_path)
    q = config.quality()
    assert q["freq_sigma"] == 3.5
    assert q["harmonize"] == 0.0
    assert q["texture"] == 0.4
    assert q["blend_band"] == config._QUALITY_DEFAULTS["blend_band"]


def test_quality_defaults_without_ini(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "app_data_dir", lambda: tmp_path)
    assert config.quality() == config._QUALITY_DEFAULTS
