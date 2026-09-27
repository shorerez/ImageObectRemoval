"""ONNX engine tests using a tiny stand-in model with LaMa's exact I/O."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

from object_remover.errors import InpaintError
from object_remover.inpaint import InpaintService, OnnxLamaEngine, build_default_engine


def _make_tiny_lama(path: Path, scale: float = 1.0) -> Path:
    """A 0.5-kB ONNX model with LaMa's contract, output scaled by ``scale``.

    output = (image * (1 - mask) + 0.5 * mask) * scale

    ``scale=1`` mimics the PyTorch 0..1 convention, ``scale=255`` the
    Carve/LaMa-ONNX export (its demo casts the raw output straight to uint8).
    """
    from onnx import TensorProto, helper

    img = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 512, 512])
    mask = helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, 512, 512])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 512, 512])
    nodes = [
        helper.make_node("Cast", ["mask"], ["maskc"], to=TensorProto.FLOAT),
        helper.make_node("Sub", ["one", "maskc"], ["invm"]),
        helper.make_node("Mul", ["image", "invm"], ["kept"]),
        helper.make_node("Mul", ["maskc", "half"], ["fill"]),
        helper.make_node("Add", ["kept", "fill"], ["mixed"]),
        helper.make_node("Mul", ["mixed", "scale"], ["output"]),
    ]
    consts = [
        helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0]),
        helper.make_tensor("half", TensorProto.FLOAT, [1], [0.5]),
        helper.make_tensor("scale", TensorProto.FLOAT, [1], [float(scale)]),
    ]
    graph = helper.make_graph(nodes, "tiny_lama", [img, mask], [out], consts)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 11
    onnx.save(model, str(path))
    return path


@pytest.fixture(scope="module")
def tiny_lama(tmp_path_factory) -> Path:
    """0..1 output (PyTorch convention)."""
    return _make_tiny_lama(tmp_path_factory.mktemp("models") / "tiny_lama.onnx", 1.0)


@pytest.fixture(scope="module")
def tiny_lama_255(tmp_path_factory) -> Path:
    """0..255 output (Carve/LaMa-ONNX export)."""
    return _make_tiny_lama(tmp_path_factory.mktemp("models") / "tiny_lama_255.onnx", 255.0)


@pytest.fixture(scope="module")
def tiny_lama_garbage(tmp_path_factory) -> Path:
    """Absurd output level — no plausible scale."""
    return _make_tiny_lama(
        tmp_path_factory.mktemp("models") / "tiny_lama_garbage.onnx", 1e30
    )


@pytest.fixture(scope="module")
def tiny_lama_nan(tmp_path_factory) -> Path:
    """Non-finite output."""
    return _make_tiny_lama(
        tmp_path_factory.mktemp("models") / "tiny_lama_nan.onnx", float("nan")
    )


def test_engine_fill_contract(tiny_lama):
    from object_remover.runtime import create_onnx_session

    engine = OnnxLamaEngine(create_onnx_session(tiny_lama))
    assert engine.output_scale == pytest.approx(1.0)
    tile = np.zeros((512, 512, 3), np.float32)
    mask = np.zeros((512, 512), np.float32)
    mask[100:200, 100:200] = 1.0
    out = engine.fill(tile, mask)
    assert out.shape == (512, 512, 3)
    assert out[150, 150, 0] == pytest.approx(0.5)  # filled
    assert out[10, 10, 0] == pytest.approx(0.0)    # context kept


def test_engine_detects_255_output_scale(tiny_lama_255):
    """A 0..255 model must be normalised, not clipped to white."""
    from object_remover.runtime import create_onnx_session

    engine = OnnxLamaEngine(create_onnx_session(tiny_lama_255))
    assert engine.output_scale == pytest.approx(255.0)

    tile = np.zeros((512, 512, 3), np.float32)
    mask = np.zeros((512, 512), np.float32)
    mask[100:200, 100:200] = 1.0
    out = engine.fill(tile, mask)
    assert out[150, 150, 0] == pytest.approx(0.5)  # filled, not 1.0 (white)
    assert out[10, 10, 0] == pytest.approx(0.0)    # context kept

    # end to end through the service: the hole lands on ~0.5, not saturated
    svc = InpaintService(engine)
    pixels = np.full((300, 300, 3), 20000, np.uint16)
    removal = np.zeros((300, 300), np.uint8)
    removal[100:200, 100:200] = 255
    out16, stats = svc.inpaint(pixels, removal, np.zeros((300, 300), np.uint8))
    assert stats.engine == "LaMa (AI)"
    assert out16[150, 150, 0] == pytest.approx(32768, abs=64)


def test_engine_rejects_absurd_and_nonfinite_output(tiny_lama_garbage, tiny_lama_nan):
    from object_remover.runtime import create_onnx_session

    with pytest.raises(InpaintError):
        OnnxLamaEngine(create_onnx_session(tiny_lama_garbage))
    with pytest.raises(InpaintError):
        OnnxLamaEngine(create_onnx_session(tiny_lama_nan))


def test_build_default_engine_falls_back_on_garbage_model(tiny_lama_garbage):
    engine = build_default_engine(tiny_lama_garbage)
    assert engine.name == "Classical (fallback)"


def test_build_default_engine_prefers_onnx(tiny_lama):
    engine = build_default_engine(tiny_lama)
    assert isinstance(engine, OnnxLamaEngine)


def test_build_default_engine_accepts_255_model(tiny_lama_255):
    engine = build_default_engine(tiny_lama_255)
    assert isinstance(engine, OnnxLamaEngine)
    assert engine.output_scale == pytest.approx(255.0)


def test_build_default_engine_falls_back_without_model(tmp_path):
    engine = build_default_engine(tmp_path / "missing.onnx")
    assert engine.name == "Classical (fallback)"


def test_service_with_onnx_engine(tiny_lama):
    from object_remover.runtime import create_onnx_session

    engine = OnnxLamaEngine(create_onnx_session(tiny_lama))
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
