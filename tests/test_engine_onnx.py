"""ONNX engine tests using tiny stand-in models with LaMa's exact I/O.

Output scale matters: the published Carve/LaMa-ONNX export answers in 0..255
while other exports answer in 0..1, so it has to be measured, not assumed.
Covered behaviours:
- 1x scale    -> output = image*(1-mask) + 0.5*mask
- 255x scale  -> the same, times 255 (what lama_fp32.onnx actually returns)
- hole-only   -> 255x output that leaves the context at 0 (still usable)
- broken      -> NaN, absurd magnitudes, wrong channel count: must be rejected
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

from object_remover.errors import InpaintError
from object_remover.inpaint import InpaintService, OnnxLamaEngine, build_default_engine

SIZE = 512


def _build_tiny_lama(
    path: Path,
    out_scale: float = 1.0,
    garbage: str | None = None,
) -> Path:
    """Write a ~0.5 kB ONNX model with LaMa's contract (image/mask -> output).

    out_scale multiplies the result (255.0 reproduces the Carve export).
    garbage replaces the output: "nan", "absurd", "hole_only" or "mono".
    """
    from onnx import TensorProto, helper

    img = helper.make_tensor_value_info(
        "image", TensorProto.FLOAT, [1, 3, SIZE, SIZE]
    )
    mask = helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, SIZE, SIZE])
    out = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, SIZE, SIZE]
    )
    consts = [
        helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0]),
        helper.make_tensor("half", TensorProto.FLOAT, [1], [0.5]),
        helper.make_tensor("half3", TensorProto.FLOAT, [1, 3, 1, 1], [0.5] * 3),
        helper.make_tensor("scale", TensorProto.FLOAT, [1], [float(out_scale)]),
        helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0]),
        helper.make_tensor("big", TensorProto.FLOAT, [1], [1e9]),
    ]
    if garbage == "nan":  # 0/0 -> NaN, broadcast over the image
        nodes = [
            helper.make_node("Div", ["zero", "zero"], ["nan"]),
            helper.make_node("Mul", ["image", "nan"], ["output"]),
        ]
    elif garbage == "absurd":  # finite but wildly out of any sane range
        nodes = [helper.make_node("Mul", ["image", "big"], ["output"])]
    elif garbage == "hole_only":  # returns only the fill; context stays 0
        nodes = [
            helper.make_node("Cast", ["mask"], ["maskc"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["maskc", "half3"], ["mixed"]),
            helper.make_node("Mul", ["mixed", "scale"], ["output"]),
        ]
    elif garbage == "mono":  # wrong channel count: not an RGB image
        nodes = [helper.make_node("Cast", ["mask"], ["output"], to=TensorProto.FLOAT)]
    else:
        nodes = [
            helper.make_node("Cast", ["mask"], ["maskc"], to=TensorProto.FLOAT),
            helper.make_node("Sub", ["one", "maskc"], ["invm"]),
            helper.make_node("Mul", ["image", "invm"], ["kept"]),
            helper.make_node("Mul", ["maskc", "half"], ["fill"]),
            helper.make_node("Add", ["kept", "fill"], ["mixed"]),
            helper.make_node("Mul", ["mixed", "scale"], ["output"]),
        ]
    graph = helper.make_graph(nodes, "tiny_lama", [img, mask], [out], consts)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 11
    onnx.save(model, str(path))
    return path


@pytest.fixture(scope="module")
def tiny_lama(tmp_path_factory) -> Path:
    """0..1 output scale: output = image*(1-mask) + 0.5*mask."""
    return _build_tiny_lama(tmp_path_factory.mktemp("models") / "tiny_lama.onnx")


@pytest.fixture(scope="module")
def tiny_lama_255(tmp_path_factory) -> Path:
    """0..255 output scale — the behaviour of the real Carve/LaMa-ONNX export."""
    return _build_tiny_lama(
        tmp_path_factory.mktemp("models") / "tiny_lama_255.onnx", out_scale=255.0
    )


@pytest.fixture(scope="module")
def hole_only_lama(tmp_path_factory) -> Path:
    """0..255 output that only covers the hole (context stays 0)."""
    return _build_tiny_lama(
        tmp_path_factory.mktemp("models") / "hole_only.onnx",
        out_scale=255.0,
        garbage="hole_only",
    )


@pytest.fixture(scope="module")
def mono_lama(tmp_path_factory) -> Path:
    """Single-channel output: the wrong shape for an RGB fill."""
    return _build_tiny_lama(tmp_path_factory.mktemp("models") / "mono.onnx", garbage="mono")


@pytest.fixture(scope="module")
def garbage_lama(tmp_path_factory) -> Path:
    """Emits NaN — a broken provider must never be used for fills."""
    return _build_tiny_lama(
        tmp_path_factory.mktemp("models") / "garbage.onnx", garbage="nan"
    )


@pytest.fixture(scope="module")
def absurd_lama(tmp_path_factory) -> Path:
    """Emits finite but absurd values (1e9) — also unusable."""
    return _build_tiny_lama(
        tmp_path_factory.mktemp("models") / "absurd.onnx", garbage="absurd"
    )


def _session(path: Path):
    from object_remover.runtime import create_onnx_session

    return create_onnx_session(path)


def _tile_and_mask():
    tile = np.zeros((SIZE, SIZE, 3), np.float32)
    mask = np.zeros((SIZE, SIZE), np.float32)
    mask[100:200, 100:200] = 1.0
    return tile, mask


# --- output-scale detection -------------------------------------------------


def test_engine_fill_contract(tiny_lama):
    engine = OnnxLamaEngine(_session(tiny_lama))
    assert engine.output_scale == pytest.approx(1.0)
    tile, mask = _tile_and_mask()
    out = engine.fill(tile, mask)
    assert out.shape == (SIZE, SIZE, 3)
    assert out[150, 150, 0] == pytest.approx(0.5)  # filled
    assert out[10, 10, 0] == pytest.approx(0.0)    # context kept


def test_engine_detects_255_output_scale(tiny_lama_255):
    """The Carve export's 0..255 output must be detected, not clipped to white."""
    engine = OnnxLamaEngine(_session(tiny_lama_255))
    assert engine.output_scale == pytest.approx(255.0)

    tile, mask = _tile_and_mask()
    out = engine.fill(tile, mask)
    assert out[150, 150, 0] == pytest.approx(0.5, abs=1e-6)  # 127.5/255
    assert out[10, 10, 0] == pytest.approx(0.0, abs=1e-6)
    assert not np.all(out > 0.99), "fill must not saturate to white"


def test_service_with_255_model_fills_gray_not_white(tiny_lama_255):
    """End-to-end regression: a 0..255 model used to fill every hole white."""
    svc = InpaintService(OnnxLamaEngine(_session(tiny_lama_255)))
    pixels = np.zeros((300, 300, 3), np.uint16)  # black image
    removal = np.zeros((300, 300), np.uint8)
    removal[100:200, 100:200] = 255

    out, stats = svc.inpaint(pixels, removal)
    assert stats.engine == "LaMa (AI)"
    # 0.5 gray in 16-bit is ~32768; the old clipping produced 65535 (white)
    assert out[150, 150, 0] == pytest.approx(32768, abs=3000)
    assert out[150, 150, 0] < 60000
    # untouched pixels stay bit-exact
    np.testing.assert_array_equal(out[:50], pixels[:50])


def test_scale_detected_when_only_the_hole_is_returned(hole_only_lama):
    """An engine that answers 0 outside the hole must still be measured as 255x.

    The service only uses the fill inside the removal mask, so such a model is
    usable — it must not be mistaken for a 1x engine (which would clip to white).
    """
    engine = OnnxLamaEngine(_session(hole_only_lama))
    assert engine.output_scale == pytest.approx(255.0)
    tile, mask = _tile_and_mask()
    out = engine.fill(tile, mask)
    assert out[150, 150, 0] == pytest.approx(0.5, abs=1e-6)


def test_engine_rejects_wrong_channel_count(mono_lama):
    """A single-channel output is not an RGB fill and must be rejected."""
    with pytest.raises(InpaintError, match="output shape"):
        OnnxLamaEngine(_session(mono_lama))


# --- broken providers -------------------------------------------------------


def test_engine_rejects_nan_output(garbage_lama):
    with pytest.raises(InpaintError, match="non-finite"):
        OnnxLamaEngine(_session(garbage_lama))


def test_engine_rejects_absurd_output(absurd_lama):
    with pytest.raises(InpaintError, match="out of range"):
        OnnxLamaEngine(_session(absurd_lama))


def test_build_default_engine_falls_back_on_nan_output(garbage_lama):
    """A model that loads but emits garbage must not become the engine."""
    engine = build_default_engine(garbage_lama)
    assert engine.name == "Classical (fallback)"


def test_build_default_engine_falls_back_on_absurd_output(absurd_lama):
    engine = build_default_engine(absurd_lama)
    assert engine.name == "Classical (fallback)"


def test_fallback_engine_still_fills(garbage_lama):
    """The classical fallback keeps the app usable when the ONNX model is bad."""
    svc = InpaintService(build_default_engine(garbage_lama))
    pixels = np.full((200, 200, 3), 20000, np.uint16)
    removal = np.zeros((200, 200), np.uint8)
    removal[80:120, 80:120] = 255
    out, stats = svc.inpaint(pixels, removal)
    assert stats.engine == "Classical (fallback)"
    assert abs(int(out[100, 100, 0]) - 20000) < 5000


# --- engine selection -------------------------------------------------------


def test_build_default_engine_prefers_onnx(tiny_lama):
    engine = build_default_engine(tiny_lama)
    assert isinstance(engine, OnnxLamaEngine)


def test_build_default_engine_tries_available_providers(tiny_lama, monkeypatch):
    """CUDA first, then CPU; only providers the runtime reports are attempted."""
    import sys

    from object_remover.runtime import create_onnx_session as real_create

    attempts: list[list[str]] = []

    class _FakeRuntime:
        @staticmethod
        def available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        @staticmethod
        def create_onnx_session(path, providers=None):
            attempts.append(list(providers or []))
            if providers == ["CUDAExecutionProvider"]:
                raise RuntimeError("CUDA not usable here")
            return real_create(tiny_lama)

    monkeypatch.setitem(sys.modules, "object_remover.runtime", _FakeRuntime)
    engine = build_default_engine(tiny_lama)
    assert isinstance(engine, OnnxLamaEngine)
    assert attempts == [["CUDAExecutionProvider"], ["CPUExecutionProvider"]]


def test_build_default_engine_falls_back_without_model(tmp_path):
    engine = build_default_engine(tmp_path / "missing.onnx")
    assert engine.name == "Classical (fallback)"


def test_service_with_onnx_engine(tiny_lama):
    engine = OnnxLamaEngine(_session(tiny_lama))
    svc = InpaintService(engine)
    pixels = np.full((300, 300, 3), 20000, np.uint16)
    removal = np.zeros((300, 300), np.uint8)
    removal[100:200, 100:200] = 255
    protect = np.zeros((300, 300), np.uint8)
    protect[280:290, 280:290] = 255

    out, stats = svc.inpaint(pixels, removal, protect)
    assert stats.engine == "LaMa (AI)"
    # hole filled toward model output (0.5 in 0..1 => ~32768)
    assert out[150, 150, 0] > 25000
    # protected + unselected untouched
    np.testing.assert_array_equal(out[280:290, 280:290], pixels[280:290, 280:290])
    np.testing.assert_array_equal(out[:50], pixels[:50])
