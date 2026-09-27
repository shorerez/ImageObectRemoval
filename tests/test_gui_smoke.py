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


def _wait_idle(window, qt_app, timeout: float = 30.0) -> None:
    """Spin the event loop until the current job finishes."""
    import time

    deadline = time.time() + timeout
    while window._busy and time.time() < deadline:
        qt_app.processEvents()
        time.sleep(0.01)
    assert not window._busy, "remove job did not finish in time"


def _preview_bytes(window) -> bytes:
    """Byte content of the canvas image preview (RGBA, no padding)."""
    from PySide6.QtGui import QImage

    img = window._canvas._image_item.pixmap().toImage()
    img = img.convertToFormat(QImage.Format_RGBA8888)
    return bytes(img.constBits())[: img.sizeInBytes()]


def test_canvas_image_preview_refreshes_after_remove_and_undo(window, qt_app, tmp_path):
    """Remove/Undo must rebuild the image preview, not only the mask overlays."""
    path = _make_tiff(tmp_path, size=400)
    window.open_path(path)
    doc = window._doc

    builder = doc.begin_stroke(MASK_REMOVAL, 25, 1.0, False)
    builder.add_point(200, 200)
    doc.commit_stroke(builder)
    window._canvas.refresh_overlays()

    before = _preview_bytes(window)

    window._on_remove()
    _wait_idle(window, qt_app)

    after_remove = _preview_bytes(window)
    assert after_remove != before, "canvas image preview is stale after Remove"

    window._on_undo()
    assert _preview_bytes(window) == before, "canvas image preview not restored by Undo"

    window._on_redo()
    assert _preview_bytes(window) == after_remove, "canvas image preview not re-applied by Redo"


def test_overlap_dialog_accept_path_uses_class_dialogcode(window, qt_app, tmp_path, monkeypatch):
    """`dlg.Accepted` is gone in PySide6 6.11 — the comparison must use the enum."""
    from PySide6.QtWidgets import QDialog

    from object_remover.document import MASK_PROTECT
    from object_remover.ui.dialogs import OverlapDialog

    path = _make_tiff(tmp_path)
    window.open_path(path)
    doc = window._doc
    for mask_id in (MASK_REMOVAL, MASK_PROTECT):
        builder = doc.begin_stroke(mask_id, 20, 1.0, False)
        builder.add_point(150, 150)
        doc.commit_stroke(builder)
    assert doc.overlap_count() > 0

    def fake_exec(self):
        # "removal wins" keeps the removal mask usable (both blobs are identical
        # here, so "protect wins" would legitimately empty it and fail the job).
        self._choice = OverlapDialog.REMOVAL
        return int(QDialog.DialogCode.Accepted)

    monkeypatch.setattr(OverlapDialog, "exec", fake_exec)
    window._on_remove()  # raised AttributeError on PySide6 6.11 before the fix
    _wait_idle(window, qt_app)
    assert doc.overlap_count() == 0


def test_export_dialog_accept_path_uses_class_dialogcode(window, qt_app, tmp_path, monkeypatch):
    """Same PySide6 6.11 fix on the export path (`_on_export`)."""
    from PySide6.QtWidgets import QDialog

    from object_remover.ui.dialogs import ExportDialog

    path = _make_tiff(tmp_path)
    window.open_path(path)
    target = tmp_path / "out.jpg"

    def fake_exec(self):
        self._path_edit.setText(str(target))
        return int(QDialog.DialogCode.Accepted)

    monkeypatch.setattr(ExportDialog, "exec", fake_exec)
    window._on_export()  # raised AttributeError on PySide6 6.11 before the fix
    _wait_idle(window, qt_app)
    assert target.stat().st_size > 0
