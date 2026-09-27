"""Main window: toolbar, canvas, status bar, and job orchestration."""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSlider,
    QToolBar,
    QWidget,
)

from ..config import (
    APP_NAME,
    DEFAULT_HARDNESS,
    MAX_BRUSH_SIZE,
    MIN_BRUSH_SIZE,
    SUPPORTED_INPUT_SUFFIXES,
)
from ..document import MASK_PROTECT, MASK_REMOVAL, ImageDocument
from ..errors import CancelledError
from ..image_io import export_jpg, export_tiff, load_image
from ..inpaint import ClassicalEngine, InpaintService, build_default_engine
from ..jobs import JobHandle, JobSupervisor
from ..models import ModelManager
from ..runtime import device_summary
from .canvas import MaskCanvas
from .dialogs import ExportDialog, OverlapDialog

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, auto_startup: bool = True) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1440, 900)

        self._doc: ImageDocument | None = None
        self._manager = ModelManager()
        self._service = InpaintService(ClassicalEngine())
        self._busy = False

        self._canvas = MaskCanvas(self)
        self.setCentralWidget(self._canvas)

        self._jobs = JobSupervisor(self)
        self._job_cbs: dict[JobHandle, tuple] = {}
        self._jobs.job_progress.connect(self._on_job_progress)
        self._jobs.job_succeeded.connect(self._on_job_succeeded)
        self._jobs.job_failed.connect(self._on_job_failed)
        self._jobs.job_cancelled.connect(self._on_job_cancelled)

        self._build_toolbar()
        self._build_statusbar()
        self._canvas.stroke_committed.connect(self._update_action_state)
        self._canvas.view_info_changed.connect(self._status_message)
        self._update_action_state()

        if auto_startup:
            QTimer.singleShot(0, self._startup)

    # ------------------------------------------------------------------ UI
    def _build_toolbar(self) -> None:
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        self._act_open = QAction("Open…", self)
        self._act_open.setShortcut(QKeySequence.Open)
        self._act_open.triggered.connect(self._on_open)
        tb.addAction(self._act_open)

        tb.addSeparator()
        self._mask_combo = QComboBox()
        self._mask_combo.addItem("Removal mask (red)", MASK_REMOVAL)
        self._mask_combo.addItem("Protect mask (green)", MASK_PROTECT)
        self._mask_combo.currentIndexChanged.connect(self._on_mask_mode)
        tb.addWidget(self._mask_combo)

        self._act_eraser = QAction("Eraser", self)
        self._act_eraser.setCheckable(True)
        self._act_eraser.setShortcut("E")
        self._act_eraser.toggled.connect(self._canvas.set_eraser)
        tb.addAction(self._act_eraser)

        tb.addWidget(QLabel("  Size "))
        self._size_slider = QSlider(Qt.Horizontal)
        self._size_slider.setRange(MIN_BRUSH_SIZE, MAX_BRUSH_SIZE)
        self._size_slider.setValue(int(self._canvas.brush_size()))
        self._size_slider.setFixedWidth(120)
        self._size_slider.valueChanged.connect(self._on_brush_size)
        tb.addWidget(self._size_slider)
        self._size_label = QLabel(f"{int(self._canvas.brush_size())} px")
        tb.addWidget(self._size_label)

        tb.addWidget(QLabel("  Hardness "))
        self._hardness_slider = QSlider(Qt.Horizontal)
        self._hardness_slider.setRange(0, 100)
        self._hardness_slider.setValue(int(DEFAULT_HARDNESS * 100))
        self._hardness_slider.setFixedWidth(100)
        self._hardness_slider.valueChanged.connect(self._on_hardness)
        tb.addWidget(self._hardness_slider)

        tb.addSeparator()
        self._act_remove = QAction("Remove", self)
        self._act_remove.setShortcut(QKeySequence("Ctrl+Return"))
        self._act_remove.triggered.connect(self._on_remove)
        tb.addAction(self._act_remove)

        self._act_clear = QAction("Clear masks", self)
        self._act_clear.triggered.connect(self._on_clear_masks)
        tb.addAction(self._act_clear)

        tb.addSeparator()
        self._act_undo = QAction("Undo", self)
        self._act_undo.setShortcut(QKeySequence.Undo)
        self._act_undo.triggered.connect(self._on_undo)
        tb.addAction(self._act_undo)

        self._act_redo = QAction("Redo", self)
        self._act_redo.setShortcut(QKeySequence.Redo)
        self._act_redo.triggered.connect(self._on_redo)
        tb.addAction(self._act_redo)

        tb.addSeparator()
        self._act_export = QAction("Export…", self)
        self._act_export.setShortcut(QKeySequence("Ctrl+E"))
        self._act_export.triggered.connect(self._on_export)
        tb.addAction(self._act_export)

    def _build_statusbar(self) -> None:
        self._status = QLabel("Open a TIFF or JPG to start.")
        self.statusBar().addWidget(self._status, 1)
        self._device_label = QLabel(device_summary())
        self.statusBar().addPermanentWidget(self._device_label)

    def _status_message(self, text: str) -> None:
        self._status.setText(text)

    # ----------------------------------------------------------- startup/model
    def _startup(self) -> None:
        if self._manager.is_ready():
            self._load_engine()
            return
        answer = QMessageBox.question(
            self,
            "AI model",
            "Download the AI quality model now (~208 MB, one time)?\n\n"
            "Without it the app runs with a classical fallback engine that is\n"
            "noticeably lower quality. You can also download it later — the app\n"
            "will ask again on the next start.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            self._download_model()
        else:
            self._status_message(
                "Classical fallback engine active (AI model not downloaded)."
            )

    def _download_model(self) -> None:
        self._set_busy(True, "Downloading AI model…")

        def job(progress, cancel):
            return self._manager.download(progress, cancel)

        self._run_job(
            job,
            on_ok=lambda _res: self._load_engine(),
            on_fail=self._on_model_failed,
        )

    def _on_model_failed(self, message: str) -> None:
        self._set_busy(False)
        QMessageBox.warning(
            self,
            "Model download failed",
            f"{message}\n\nThe app will use the classical fallback engine for now.",
        )
        self._status_message("Classical fallback engine active (model download failed).")

    def _load_engine(self) -> None:
        self._set_busy(True, "Loading AI model…")
        model_path = self._manager.path

        def job(progress, cancel):
            return build_default_engine(model_path)

        def ok(engine):
            self._service = InpaintService(engine)
            self._set_busy(False)
            self._status_message(f"Ready — engine: {engine.name}")

        self._run_job(job, on_ok=ok, on_fail=self._on_model_failed)

    # ------------------------------------------------------------------ jobs
    def _run_job(self, fn, on_ok=None, on_fail=None, on_cancel=None, on_progress=None):
        handle = self._jobs.run(fn)
        self._job_cbs[handle] = (on_ok, on_fail, on_cancel, on_progress)
        return handle

    def _on_job_progress(self, handle: JobHandle, done: int, total: int) -> None:
        cbs = self._job_cbs.get(handle)
        if cbs and cbs[3]:
            cbs[3](done, total)
        else:
            pct = int(100 * done / total) if total else 0
            self._status_message(f"Working… {pct}%")

    def _on_job_succeeded(self, handle: JobHandle, result) -> None:
        cbs = self._job_cbs.pop(handle, None)
        self._set_busy(False)
        if cbs and cbs[0]:
            cbs[0](result)

    def _on_job_failed(self, handle: JobHandle, message: str) -> None:
        cbs = self._job_cbs.pop(handle, None)
        self._set_busy(False)
        if cbs and cbs[1]:
            cbs[1](message)
        else:
            QMessageBox.critical(self, "Error", message)

    def _on_job_cancelled(self, handle: JobHandle) -> None:
        cbs = self._job_cbs.pop(handle, None)
        self._set_busy(False)
        if cbs and cbs[2]:
            cbs[2]()
        self._status_message("Cancelled.")

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        self._busy = busy
        self._canvas.setEnabled(not busy)
        if message:
            self._status_message(message)
        self._update_action_state()

    # ------------------------------------------------------------------ file
    def _on_open(self) -> None:
        filters = "Images (*.tif *.tiff *.jpg *.jpeg)"
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", filters)
        if path:
            self.open_path(path)

    def open_path(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            QMessageBox.critical(
                self,
                "Unsupported file",
                f"'{path.name}' is not a supported input.\n"
                "ObjectRemover opens TIFF and JPG files.",
            )
            return
        try:
            image = load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not open image", str(exc))
            return
        self._doc = ImageDocument(image)
        self._canvas.set_document(self._doc)
        self.setWindowTitle(f"{APP_NAME} — {path.name} — {image.width}×{image.height}")
        self._status_message(
            f"Loaded {path.name} ({image.width}×{image.height}, 16-bit working buffer). "
            "Paint the area to remove, then click Remove."
        )
        self._update_action_state()

    # ------------------------------------------------------------------ edit
    def _on_mask_mode(self, index: int) -> None:
        self._canvas.set_active_mask(self._mask_combo.itemData(index))

    def _on_brush_size(self, value: int) -> None:
        self._canvas.set_brush_size(float(value))
        self._size_label.setText(f"{value} px")

    def _on_hardness(self, value: int) -> None:
        self._canvas.set_hardness(value / 100.0)

    def _on_undo(self) -> None:
        if self._doc and self._doc.undo():
            self._canvas.refresh_overlays()
            self._update_action_state()

    def _on_redo(self) -> None:
        if self._doc and self._doc.redo():
            self._canvas.refresh_overlays()
            self._update_action_state()

    def _on_clear_masks(self) -> None:
        if self._doc:
            self._doc.clear_masks()
            self._canvas.refresh_overlays()
            self._update_action_state()

    # ---------------------------------------------------------------- remove
    def _on_remove(self) -> None:
        doc = self._doc
        if doc is None or self._busy:
            return
        if not doc.removal.data.any():
            self._status_message("Paint the area to remove first (red mask).")
            return
        overlap = doc.overlap_count()
        if overlap:
            dlg = OverlapDialog(overlap, self)
            if dlg.exec() != dlg.Accepted or dlg.choice is None:
                self._status_message("Remove cancelled — fix the masks and try again.")
                return
            doc.resolve_overlap(dlg.choice)
            self._canvas.refresh_overlays()

        self._set_busy(True, "Removing selection…")
        service = self._service
        pixels = doc.image.pixels
        removal = doc.removal.data
        protect = doc.protect.data

        def job(progress, cancel):
            return service.inpaint(pixels, removal, protect, progress, cancel)

        self._run_job(
            job,
            on_ok=self._on_remove_done,
            on_fail=lambda m: QMessageBox.critical(self, "Remove failed", m),
        )

    def _on_remove_done(self, result) -> None:
        new_pixels, stats = result
        if self._doc is None:
            return
        self._doc.apply_removal(new_pixels)
        self._canvas.refresh_overlays()
        self._update_action_state()
        self._status_message(
            f"Removed in {stats.elapsed_s:.1f}s ({stats.engine}, {stats.windows} windows). "
            "Repeat on other areas or Export when done."
        )

    # ---------------------------------------------------------------- export
    def _on_export(self) -> None:
        doc = self._doc
        if doc is None or self._busy:
            return
        default_dir = str(Path(doc.image.source_path).parent) if doc.image.source_path else ""
        dlg = ExportDialog(default_dir, self)
        if dlg.exec() != dlg.Accepted:
            return
        target = dlg.export_path
        as_jpg = dlg.as_jpg
        quality = dlg.jpg_quality

        if doc.image.source_path and target == Path(doc.image.source_path):
            answer = QMessageBox.question(
                self,
                "Overwrite source?",
                "The export path is the source image itself. Overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        self._set_busy(True, f"Exporting {target.name}…")
        image = doc.image

        def job(progress, cancel):
            if as_jpg:
                export_jpg(image, target, quality)
            else:
                export_tiff(image, target)
            return target

        def ok(path):
            self._status_message(f"Exported: {path}")

        self._run_job(
            job,
            on_ok=ok,
            on_fail=lambda m: QMessageBox.critical(self, "Export failed", m),
        )

    # ----------------------------------------------------------------- state
    def _update_action_state(self) -> None:
        has_doc = self._doc is not None and not self._busy
        self._act_open.setEnabled(not self._busy)
        self._act_remove.setEnabled(has_doc)
        self._act_clear.setEnabled(has_doc)
        self._act_export.setEnabled(has_doc)
        self._act_undo.setEnabled(has_doc and self._doc.history.can_undo)
        self._act_redo.setEnabled(has_doc and self._doc.history.can_redo)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._jobs.cancel_all()
        super().closeEvent(event)
