"""Modal dialogs: mask-overlap resolution and export options."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..config import JPG_QUALITY_DEFAULT, JPG_QUALITY_MAX, JPG_QUALITY_MIN


class OverlapDialog(QDialog):
    """Ask the user which mask wins where removal and protect overlap."""

    PROTECT = "protect"
    REMOVAL = "removal"

    def __init__(self, overlap_pixels: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mask overlap")
        self.setModal(True)
        self._choice: str | None = None

        layout = QVBoxLayout(self)
        label = QLabel(
            f"The removal and protect masks overlap in {overlap_pixels:,} pixels.\n"
            "Which one should win in the overlapping area?"
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        buttons = QHBoxLayout()
        protect_btn = QPushButton("Protect wins")
        protect_btn.clicked.connect(self._on_protect)
        removal_btn = QPushButton("Removal wins")
        removal_btn.clicked.connect(self._on_removal)
        cancel_btn = QPushButton("Cancel and fix by hand")
        cancel_btn.clicked.connect(self.reject)
        for b in (protect_btn, removal_btn, cancel_btn):
            buttons.addWidget(b)
        layout.addLayout(buttons)

    def _on_protect(self) -> None:
        self._choice = self.PROTECT
        self.accept()

    def _on_removal(self) -> None:
        self._choice = self.REMOVAL
        self.accept()

    @property
    def choice(self) -> str | None:
        return self._choice


class ExportDialog(QDialog):
    """Pick export path, format and JPG quality."""

    def __init__(self, default_dir: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export image")
        self.setModal(True)
        self._default_dir = default_dir or str(Path.home())

        layout = QVBoxLayout(self)
        form = QFormLayout()

        path_row = QWidget()
        row = QHBoxLayout(path_row)
        row.setContentsMargins(0, 0, 0, 0)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Output file path…")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self._path_edit, 1)
        row.addWidget(browse)
        form.addRow("File:", path_row)

        self._format = QComboBox()
        self._format.addItems(["JPG (sRGB, 8-bit)", "TIFF (16-bit, lossless)"])
        self._format.currentIndexChanged.connect(self._on_format_changed)
        form.addRow("Format:", self._format)

        quality_row = QWidget()
        qrow = QHBoxLayout(quality_row)
        qrow.setContentsMargins(0, 0, 0, 0)
        self._quality = QSlider(Qt.Horizontal)
        self._quality.setRange(JPG_QUALITY_MIN, JPG_QUALITY_MAX)
        self._quality.setValue(JPG_QUALITY_DEFAULT)
        self._quality_label = QLabel(f"{JPG_QUALITY_DEFAULT}")
        self._quality.valueChanged.connect(
            lambda v: self._quality_label.setText(str(v))
        )
        qrow.addWidget(self._quality, 1)
        qrow.addWidget(self._quality_label)
        self._quality_row = quality_row
        form.addRow("JPG quality:", quality_row)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_format_changed(self, index: int) -> None:
        self._quality_row.setVisible(index == 0)
        path = self._path_edit.text().strip()
        if path:
            p = Path(path)
            if index == 0 and p.suffix.lower() in {".tif", ".tiff"}:
                self._path_edit.setText(str(p.with_suffix(".jpg")))
            elif index == 1 and p.suffix.lower() in {".jpg", ".jpeg"}:
                self._path_edit.setText(str(p.with_suffix(".tif")))

    def _browse(self) -> None:
        jpg_sel = "JPEG Images (*.jpg *.jpeg)"
        tif_sel = "TIFF Images (*.tif *.tiff)"
        path, selected = QFileDialog.getSaveFileName(
            self,
            "Export image",
            self._default_dir,
            f"{jpg_sel};;{tif_sel}",
            jpg_sel if self._format.currentIndex() == 0 else tif_sel,
        )
        if path:
            self._path_edit.setText(path)
            if "TIFF" in selected:
                self._format.setCurrentIndex(1)
            elif "JPEG" in selected:
                self._format.setCurrentIndex(0)

    def _validate_accept(self) -> None:
        path = self._path_edit.text().strip()
        if not path:
            self._path_edit.setFocus()
            return
        p = Path(path)
        if self._format.currentIndex() == 0 and p.suffix.lower() not in {".jpg", ".jpeg"}:
            self._path_edit.setText(str(p.with_suffix(".jpg")))
        elif self._format.currentIndex() == 1 and p.suffix.lower() not in {".tif", ".tiff"}:
            self._path_edit.setText(str(p.with_suffix(".tif")))
        self.accept()

    @property
    def export_path(self) -> Path:
        return Path(self._path_edit.text().strip())

    @property
    def as_jpg(self) -> bool:
        return self._format.currentIndex() == 0

    @property
    def jpg_quality(self) -> int:
        return int(self._quality.value())
