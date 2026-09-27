"""Tests for ModelManager: download, checksum (trust-on-first-use), status."""
from __future__ import annotations

import threading

import pytest

from object_remover.errors import ModelDownloadError
from object_remover.models import ModelManager, ModelStatus


@pytest.fixture()
def fake_model(tmp_path):
    src = tmp_path / "fake_model.bin"
    src.write_bytes(b"MODEL" * (1 << 18))  # 1.25 MB (> status threshold)
    return src


def _manager(tmp_path, source: str, pinned=None) -> ModelManager:
    return ModelManager(
        directory=tmp_path / "models",
        url=source,
        filename="test_model.bin",
        pinned_sha256=pinned,
    )


def test_status_missing_then_ready(tmp_path, fake_model):
    mgr = _manager(tmp_path, fake_model.as_uri())
    assert mgr.status() == ModelStatus.MISSING
    mgr.download()
    assert mgr.status() == ModelStatus.READY
    assert mgr.path.is_file()
    assert mgr.hash_sidecar.is_file()
    assert len(mgr.hash_sidecar.read_text().strip()) == 64


def test_download_progress(tmp_path, fake_model):
    mgr = _manager(tmp_path, fake_model.as_uri())
    calls = []
    mgr.download(progress=lambda d, t: calls.append((d, t)))
    assert calls
    assert calls[-1][0] == fake_model.stat().st_size
    assert calls[-1][1] == fake_model.stat().st_size


def test_tofu_mismatch_rejected(tmp_path, fake_model):
    mgr = _manager(tmp_path, fake_model.as_uri())
    mgr.download()
    good = mgr.path.read_bytes()
    # attacker/corruption: same URL now serves different content
    fake_model.write_bytes(b"EVIL!" * (1 << 18))
    with pytest.raises(ModelDownloadError, match="checksum"):
        mgr.download()
    # the previously verified model stays in place and valid
    assert mgr.path.read_bytes() == good
    assert mgr.status() == ModelStatus.READY


def test_pinned_hash_enforced(tmp_path, fake_model):
    mgr = _manager(tmp_path, fake_model.as_uri(), pinned="0" * 64)
    with pytest.raises(ModelDownloadError, match="checksum"):
        mgr.download()


def test_config_pin_is_the_published_lama_hash(tmp_path, fake_model):
    """The default pin is the published lama_fp32.onnx hash and is enforced."""
    from object_remover.config import MODEL_SHA256

    assert MODEL_SHA256 == (
        "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"
    )
    # ModelManager picks the pin up from config by default (no pinned= argument)
    mgr = ModelManager(
        directory=tmp_path / "models",
        url=fake_model.as_uri(),
        filename="lama_fp32.onnx",
    )
    with pytest.raises(ModelDownloadError, match="checksum"):
        mgr.download()
    assert not mgr.path.exists()


def test_cancel_before_download(tmp_path, fake_model):
    mgr = _manager(tmp_path, fake_model.as_uri())
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ModelDownloadError, match="cancelled"):
        mgr.download(cancel=cancel)


def test_bad_url_reports_error(tmp_path):
    mgr = _manager(tmp_path, "file:///nonexistent/nope.bin")
    with pytest.raises(ModelDownloadError):
        mgr.download()
    assert mgr.last_error
