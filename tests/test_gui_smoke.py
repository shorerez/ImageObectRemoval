"""GUI smoke test (offscreen): window, canvas, full remove flow via jobs."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from object_remover.document import MASK_REMOVAL, ImageDocument
from object_remover.image_io import WorkingImage, export_jpg
from object_remover.inpaint import InpaintService
from object_remover.ui.main_window import MainWindow


@pytest.fixture()
def window(qt_app):
    win = MainWindow(auto_startup=False)
    yield win
    win.close()
    win._jobs.cancel_all()


def _make_tiff(tmp_path, size=300):
    import tifffile

    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.stack([x * 200, y * 200, (x + y) * 100], axis=-1).astype(np.uint16)
    path = tmp_path / "gui.tif"
    tifffile.imwrite(path, img, photometric="rgb")
    return path


def test_window_open_and_paint(window, tmp_path):
    path = _make_tiff(tmp_path)
    window.open_path(path)
    doc = window._doc
    assert doc is not None
    assert (doc.image.width, doc.image.height) == (300, 300)

    # programmatic stroke through the document API (as the canvas would)
    builder = doc.begin_stroke(MASK_REMOVAL, 20, 0.8, False)
    builder.add_point(150, 150)
    builder.add_point(170, 160)
    assert doc.commit_stroke(builder)
    window._canvas.refresh_overlays()
    assert doc.removal.data.any()
    window._update_action_state()
    assert window._act_undo.isEnabled()


def test_full_remove_flow_with_job(window, qt_app, tmp_path):
    path = _make_tiff(tmp_path, size=400)
    window.open_path(path)
    doc = window._doc

    builder = doc.begin_stroke(MASK_REMOVAL, 25, 1.0, False)
    builder.add_point(200, 200)
    doc.commit_stroke(builder)

    original = doc.image.pixels.copy()
    window._on_remove()
    assert window._busy

    # spin the event loop until the job completes (classical engine, fast)
    import time

    deadline = time.time() + 30
    while window._busy and time.time() < deadline:
        qt_app.processEvents()
        time.sleep(0.01)
    assert not window._busy

    assert not doc.removal.data.any()  # masks cleared after removal
    changed = np.any(doc.image.pixels != original, axis=2)
    assert changed.any()
    # unselected pixels untouched
    assert not changed[:100].any()

    # undo restores pixels and masks
    window._on_undo()
    np.testing.assert_array_equal(doc.image.pixels, original)
    assert doc.removal.data.any()

    # export works
    out = tmp_path / "export.jpg"
    export_jpg(doc.image, out, quality=90)
    assert out.stat().st_size > 0


def _preview_rgb(canvas) -> np.ndarray:
    """The canvas image preview as an (H, W, 3) uint8 array."""
    from PySide6.QtGui import QImage

    pix = canvas._image_item.pixmap()
    assert not pix.isNull(), "canvas has no image preview"
    img = pix.toImage().convertToFormat(QImage.Format_RGB888)
    buf = np.frombuffer(
        img.constBits(), np.uint8, count=img.sizeInBytes()
    ).reshape(img.height(), img.bytesPerLine())
    return buf[:, : img.width() * 3].reshape(img.height(), img.width(), 3).copy()


def _make_tiff_with_block(tmp_path, size=300):
    """Gradient with a bright block in the middle so a fill is clearly visible."""
    import tifffile

    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.stack([x * 200, y * 200, (x + y) * 100], axis=-1)
    c = size // 2
    img[c - 40 : c + 40, c - 40 : c + 40] = 65535.0
    path = tmp_path / "block.tif"
    tifffile.imwrite(path, img.astype(np.uint16), photometric="rgb")
    return path


def _wait_for_job(window, qt_app, timeout=30.0) -> None:
    import time

    deadline = time.time() + timeout
    while window._busy and time.time() < deadline:
        qt_app.processEvents()
        time.sleep(0.01)
    assert not window._busy, "removal job did not finish"


def test_canvas_preview_follows_remove_and_undo(window, qt_app, tmp_path, no_modal):
    """The image preview must redraw after Remove and restore after Undo."""
    from object_remover.ui.canvas import _u16_to_u8

    path = _make_tiff_with_block(tmp_path)
    window.open_path(path)
    canvas, doc = window._canvas, window._doc

    before = _preview_rgb(canvas)
    # the preview mirrors the document pixels (300 px < PREVIEW_MAX_SIDE)
    np.testing.assert_array_equal(before, _u16_to_u8(doc.image.pixels))

    builder = doc.begin_stroke(MASK_REMOVAL, 35, 1.0, False)
    builder.add_point(150, 150)
    assert doc.commit_stroke(builder)

    window._on_remove()
    assert window._busy
    _wait_for_job(window, qt_app)

    after = _preview_rgb(canvas)
    assert not np.array_equal(before, after), "preview kept stale pixels after Remove"
    np.testing.assert_array_equal(after, _u16_to_u8(doc.image.pixels))

    # Undo restores both the pixels and the preview
    window._on_undo()
    np.testing.assert_array_equal(_preview_rgb(canvas), before)

    # ... and Redo re-applies them
    window._on_redo()
    np.testing.assert_array_equal(_preview_rgb(canvas), after)


def _fake_dialog_class(result: int, **props):
    """A QDialog subclass whose exec() returns `result` (DialogCode value)."""
    from PySide6.QtWidgets import QDialog

    class _FakeDialog(QDialog):
        def __init__(self, *args, **kwargs):
            super().__init__(None)

        def exec(self):  # noqa: N802 - Qt naming
            return result

    for name, value in props.items():
        setattr(_FakeDialog, name, property(lambda self, v=value: v))
    return _FakeDialog


@pytest.fixture()
def no_modal(monkeypatch):
    """Turn blocking message boxes into failures (they would hang the run)."""
    from PySide6.QtWidgets import QMessageBox

    def _fail(*args, **kwargs):
        raise AssertionError(f"unexpected modal dialog: {args[1:3]}")

    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_fail))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_fail))


def test_export_dialog_accepted_uses_dialogcode(
    window, qt_app, tmp_path, monkeypatch, no_modal
):
    """_on_export must compare exec() with QDialog.DialogCode.Accepted.

    PySide6 removed the unscoped `dlg.Accepted` alias from dialog *instances*
    (AttributeError on 6.11), so the old comparison crashed instead of
    exporting.
    """
    from PySide6.QtWidgets import QDialog

    import object_remover.ui.main_window as mw

    window.open_path(_make_tiff(tmp_path, size=64))
    out = tmp_path / "exported.jpg"

    monkeypatch.setattr(
        mw,
        "ExportDialog",
        _fake_dialog_class(
            int(QDialog.DialogCode.Accepted),
            export_path=out, as_jpg=True, jpg_quality=90,
        ),
    )
    window._on_export()
    assert window._busy, "accepted export dialog must start the export job"
    _wait_for_job(window, qt_app)
    assert out.is_file() and out.stat().st_size > 0


def test_export_dialog_rejected_does_not_export(window, qt_app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog

    import object_remover.ui.main_window as mw

    window.open_path(_make_tiff(tmp_path, size=64))
    out = tmp_path / "not_exported.jpg"
    monkeypatch.setattr(
        mw,
        "ExportDialog",
        _fake_dialog_class(
            int(QDialog.DialogCode.Rejected),
            export_path=out, as_jpg=True, jpg_quality=90,
        ),
    )
    window._on_export()
    assert not window._busy
    assert not out.exists()


def test_remove_overlap_dialog_accepted_uses_dialogcode(
    window, qt_app, tmp_path, monkeypatch, no_modal
):
    """Same fix on the overlap branch of _on_remove."""
    from PySide6.QtWidgets import QDialog

    import object_remover.ui.main_window as mw
    from object_remover.document import MASK_PROTECT

    window.open_path(_make_tiff(tmp_path, size=200))
    doc = window._doc
    # Partially overlapping strokes: removing the overlap must still leave a
    # removal mask, otherwise there is nothing to inpaint.
    for mask_id, point, radius in (
        (MASK_REMOVAL, (90, 100), 30),
        (MASK_PROTECT, (125, 100), 14),
    ):
        builder = doc.begin_stroke(mask_id, radius, 1.0, False)
        builder.add_point(*point)
        assert doc.commit_stroke(builder)
    assert doc.overlap_count() > 0

    monkeypatch.setattr(
        mw,
        "OverlapDialog",
        _fake_dialog_class(int(QDialog.DialogCode.Accepted), choice="protect"),
    )
    window._on_remove()
    assert window._busy, "accepted overlap dialog must start the removal job"
    _wait_for_job(window, qt_app)
    assert doc.overlap_count() == 0
    assert not doc.removal.data.any()  # removal ran to completion


def test_overlap_flow(window, tmp_path):
    from object_remover.document import MASK_PROTECT

    path = _make_tiff(tmp_path)
    window.open_path(path)
    doc = window._doc
    for mask_id in (MASK_REMOVAL, MASK_PROTECT):
        builder = doc.begin_stroke(mask_id, 20, 1.0, False)
        builder.add_point(150, 150)
        doc.commit_stroke(builder)
    assert doc.overlap_count() > 0
    doc.resolve_overlap("protect")
    assert doc.overlap_count() == 0


def test_canvas_mouse_painting(window, qt_app, tmp_path):
    """Real mouse events on the canvas must paint a stroke into the mask."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    path = _make_tiff(tmp_path, size=300)
    window.open_path(path)
    canvas = window._canvas
    doc = window._doc

    # map full-res coords -> widget coords through the view transform
    def widget_point(fx: float, fy: float) -> QPoint:
        scene = canvas.mapFromScene(fx * canvas._scale, fy * canvas._scale)
        return scene

    p1 = widget_point(120, 120)
    p2 = widget_point(180, 150)

    window.show()
    QTest.mousePress(canvas.viewport(), Qt.LeftButton, Qt.NoModifier, p1)
    for t in np.linspace(0, 1, 8)[1:]:
        pt = widget_point(120 + 60 * t, 120 + 30 * t)
        QTest.mouseMove(canvas.viewport(), pt)
    QTest.mouseRelease(canvas.viewport(), Qt.LeftButton, Qt.NoModifier, p2)

    assert doc.removal.data.any(), "mouse stroke must paint the removal mask"
    assert doc.history.can_undo
    window._on_undo()
    assert not doc.removal.data.any()
