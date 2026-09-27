"""ONNX engine tests using a tiny stand-in model with LaMa's exact I/O."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

from object_remover.inpaint import InpaintService, OnnxLamaEngine, build_default_engine


@pytest.fixture(scope="module")
def tiny_lama(tmp_path_factory) -> Path:
    """A 0.5-kB ONNX model with LaMa's contract: output = image*(1-mask)+0.5*mask."""
    from onnx import TensorProto, helper

    img = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 512, 512])
    mask = helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, 512, 512])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 512, 512])
    nodes = [
        helper.make_node("Cast", ["mask"], ["maskc"], to=TensorProto.FLOAT),
        helper.make_node("Sub", ["one", "maskc"], ["invm"]),
        helper.make_node("Mul", ["image", "invm"], ["kept"]),
        helper.make_node("Mul", ["maskc", "half"], ["fill"]),
        helper.make_node("Add", ["kept", "fill"], ["output"]),
    ]
    consts = [
        helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0]),
        helper.make_tensor("half", TensorProto.FLOAT, [1], [0.5]),
    ]
    graph = helper.make_graph(nodes, "tiny_lama", [img, mask], [out], consts)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 11
    path = tmp_path_factory.mktemp("models") / "tiny_lama.onnx"
    onnx.save(model, str(path))
    return path


def test_engine_fill_contract(tiny_lama):
    from object_remover.runtime import create_onnx_session

    engine = OnnxLamaEngine(create_onnx_session(tiny_lama))
    tile = np.zeros((512, 512, 3), np.float32)
    mask = np.zeros((512, 512), np.float32)
    mask[100:200, 100:200] = 1.0
    out = engine.fill(tile, mask)
    assert out.shape == (512, 512, 3)
    assert out[150, 150, 0] == pytest.approx(0.5)  # filled
    assert out[10, 10, 0] == pytest.approx(0.0)    # context kept


def test_build_default_engine_prefers_onnx(tiny_lama):
    engine = build_default_engine(tiny_lama)
    assert isinstance(engine, OnnxLamaEngine)


def test_build_default_engine_falls_back_without_model(tmp_path):
    engine = build_default_engine(tmp_path / "missing.onnx")
    assert engine.name == "Classical (fallback)"


def test_service_with_onnx_engine(tiny_lama, monkeypatch):
    from object_remover import config
    from object_remover import inpaint as mod
    from object_remover.runtime import create_onnx_session

    q = dict(config._QUALITY_DEFAULTS)
    q.update(harmonize=0.0, texture=0.0)  # assert raw engine output
    monkeypatch.setattr(mod, "quality", lambda: q)

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


def _make_model(tmp_path, nodes, consts, name):
    from onnx import TensorProto, helper

    img = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 512, 512])
    mask = helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, 512, 512])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 512, 512])
    graph = helper.make_graph(nodes, name, [img, mask], [out], consts)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 11
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, str(path))
    return path


def test_output_scale_autodetect_0_255(tmp_path):
    """Carve-style contract: 0..1 input, 0..255 output must be auto-detected."""
    from onnx import TensorProto, helper

    from object_remover.runtime import create_onnx_session

    nodes = [
        helper.make_node("Cast", ["mask"], ["maskc"], to=TensorProto.FLOAT),
        helper.make_node("Sub", ["one", "maskc"], ["invm"]),
        helper.make_node("Mul", ["image", "invm"], ["kept"]),
        helper.make_node("Mul", ["maskc", "half"], ["fill"]),
        helper.make_node("Add", ["kept", "fill"], ["out01"]),
        helper.make_node("Mul", ["out01", "s255"], ["output"]),
    ]
    consts = [
        helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0]),
        helper.make_tensor("half", TensorProto.FLOAT, [1], [0.5]),
        helper.make_tensor("s255", TensorProto.FLOAT, [1], [255.0]),
    ]
    path = _make_model(tmp_path, nodes, consts, "lama255")
    engine = OnnxLamaEngine(create_onnx_session(path))
    assert engine._scale == 255.0
    tile = np.zeros((512, 512, 3), np.float32)
    mask = np.zeros((512, 512), np.float32)
    mask[100:200, 100:200] = 1.0
    out = engine.fill(tile, mask)
    # model fills 128 (0..255 units) -> engine must return ~0.5 in 0..1 units
    assert out[150, 150, 0] == pytest.approx(0.5, abs=0.01)


def test_broken_provider_falls_back_to_classical(tmp_path):
    """A model/provider producing garbage must never reach the pixels."""
    from onnx import TensorProto, helper

    nodes = [
        helper.make_node("Mul", ["image", "boom"], ["output"]),
    ]
    consts = [helper.make_tensor("boom", TensorProto.FLOAT, [1], [1.0e8])]
    path = _make_model(tmp_path, nodes, consts, "broken")
    engine = build_default_engine(path)
    assert engine.name == "Classical (fallback)"
