"""MaskCanvas: large-image view with brush painting on full-resolution masks.

Display uses a downsampled preview pyramid (PREVIEW_MAX_SIDE); masks live at
full resolution in the document. During a brush drag overlay updates are
coalesced (~20 Hz) so huge images stay smooth.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsView

from ..config import (
    DEFAULT_BRUSH_SIZE,
    DEFAULT_HARDNESS,
    MAX_BRUSH_SIZE,
    MIN_BRUSH_SIZE,
    OVERLAY_ALPHA,
    PREVIEW_MAX_SIDE,
    PROTECT_COLOR,
    REMOVAL_COLOR,
)
from ..document import MASK_PROTECT, MASK_REMOVAL, ImageDocument


def _u16_to_u8(pixels16: np.ndarray) -> np.ndarray:
    return np.rint(pixels16.astype(np.float32) / 257.0).astype(np.uint8)


class MaskCanvas(QGraphicsView):
    stroke_committed = Signal()     # after a stroke joins the history
    view_info_changed = Signal(str) # status-bar hint (zoom etc.)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene_items_init()
        self._doc: ImageDocument | None = None
        self._scale = 1.0            # full-res -> preview
        self._preview_size = (1, 1)
        self._active_mask = MASK_REMOVAL
        self._brush_size = float(DEFAULT_BRUSH_SIZE)
        self._hardness = float(DEFAULT_HARDNESS)
        self._eraser = False

        self._builder = None
        self._panning = False
        self._pan_start: QPoint | None = None
        self._hardness_drag: tuple[int, float] | None = None
        self._space_down = False

        self._overlay_arrays: dict[str, np.ndarray] = {}
        self._overlay_images: dict[str, QImage] = {}
        self._dirty: QRect | None = None
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(50)
        self._flush_timer.timeout.connect(self._flush_overlays)

        self.setRenderHints(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor(32, 32, 36)))
        self.setFocusPolicy(Qt.StrongFocus)

    def _scene_items_init(self) -> None:
        from PySide6.QtWidgets import QGraphicsScene

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._image_item = QGraphicsPixmapItem()
        self._removal_item = QGraphicsPixmapItem()
        self._protect_item = QGraphicsPixmapItem()
        for item in (self._image_item, self._removal_item, self._protect_item):
            self._scene.addItem(item)
        self._cursor_item = QGraphicsEllipseItem()
        self._cursor_item.setPen(QPen(QColor(255, 255, 255, 200), 1.5))
        self._cursor_item.setBrush(QBrush(Qt.NoBrush))
        self._cursor_item.setZValue(10)
        self._cursor_item.setVisible(False)
        self._scene.addItem(self._cursor_item)

    # ------------------------------------------------------------------ setup
    @property
    def document(self) -> ImageDocument | None:
        return self._doc

    def set_document(self, doc: ImageDocument | None) -> None:
        self._doc = doc
        self._builder = None
        if doc is None:
            self._scene.setSceneRect(0, 0, 1, 1)
            self._image_item.setPixmap(QPixmap())
            self._removal_item.setPixmap(QPixmap())
            self._protect_item.setPixmap(QPixmap())
            return
        h, w = doc.image.height, doc.image.width
        s = min(1.0, PREVIEW_MAX_SIDE / max(h, w))
        pw, ph = max(1, int(round(w * s))), max(1, int(round(h * s)))
        self._scale = s
        self._preview_size = (pw, ph)

        for mask_id in (MASK_REMOVAL, MASK_PROTECT):
            arr = np.zeros((ph, pw, 4), dtype=np.uint8)
            color = REMOVAL_COLOR if mask_id == MASK_REMOVAL else PROTECT_COLOR
            arr[:, :, 0], arr[:, :, 1], arr[:, :, 2] = color
            self._overlay_arrays[mask_id] = arr
            self._overlay_images[mask_id] = QImage(
                arr.data, pw, ph, 4 * pw, QImage.Format_RGBA8888
            )
        self._removal_item.setPixmap(QPixmap.fromImage(self._overlay_images[MASK_REMOVAL]))
        self._protect_item.setPixmap(QPixmap.fromImage(self._overlay_images[MASK_PROTECT]))
        self.refresh_image()
        self._scene.setSceneRect(0, 0, pw, ph)
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def refresh_image(self) -> None:
        """Rebuild the display-only image preview from the document pixels.

        Needed after anything that changes pixels (Remove, undo/redo of a
        removal) — refresh_overlays() only redraws the two mask layers, so
        without this the canvas keeps showing the pre-removal pixels.
        """
        if self._doc is None:
            return
        w, h = self._doc.image.width, self._doc.image.height
        pw, ph = self._preview_size
        rgb8 = _u16_to_u8(self._doc.image.pixels)
        if (pw, ph) != (w, h):
            import cv2

            rgb8 = cv2.resize(rgb8, (pw, ph), interpolation=cv2.INTER_AREA)
        rgb8 = np.ascontiguousarray(rgb8)
        qimg = QImage(rgb8.data, pw, ph, 3 * pw, QImage.Format_RGB888).copy()
        self._image_item.setPixmap(QPixmap.fromImage(qimg))

    # ------------------------------------------------------------------ state
    def set_active_mask(self, mask_id: str) -> None:
        self._active_mask = mask_id

    def set_brush_size(self, size: float) -> None:
        self._brush_size = float(min(MAX_BRUSH_SIZE, max(MIN_BRUSH_SIZE, size)))
        self._update_cursor_shape()

    def set_hardness(self, hardness: float) -> None:
        self._hardness = float(min(1.0, max(0.0, hardness)))

    def set_eraser(self, on: bool) -> None:
        self._eraser = bool(on)

    def brush_size(self) -> float:
        return self._brush_size

    def hardness(self) -> float:
        return self._hardness

    # ----------------------------------------------------------------- coords
    def _full_from_view(self, pos: QPointF) -> tuple[float, float]:
        scene = self.mapToScene(pos.toPoint())
        s = self._scale if self._scale > 0 else 1.0
        return scene.x() / s, scene.y() / s

    def _view_zoom(self) -> float:
        return float(self.transform().m11())

    def _update_cursor_shape(self) -> None:
        if self._doc is None:
            self._cursor_item.setVisible(False)
            return
        s = self._scale if self._scale > 0 else 1.0
        r_full = self._brush_size / 2.0
        r_scene = r_full * s
        pos = self._cursor_item.pos()
        self._cursor_item.setRect(-r_scene, -r_scene, 2 * r_scene, 2 * r_scene)
        self._cursor_item.setPos(pos)
        self._cursor_item.setVisible(True)

    # ---------------------------------------------------------------- overlays
    def refresh_overlays(self) -> None:
        """Full overlay rebuild (after undo/redo/remove)."""
        if self._doc is None:
            return
        pw, ph = self._preview_size
        self._refresh_mask(MASK_REMOVAL, (0, 0, self._doc.image.width, self._doc.image.height))
        self._refresh_mask(MASK_PROTECT, (0, 0, self._doc.image.width, self._doc.image.height))
        self._dirty = None
        self._flush_overlays()

    def _mark_dirty(self, full_bbox: tuple[int, int, int, int]) -> None:
        if full_bbox[2] <= full_bbox[0] or full_bbox[3] <= full_bbox[1]:
            return
        pw, ph = self._preview_size
        x0, y0, x1, y1 = full_bbox
        sx = pw / self._doc.image.width
        sy = ph / self._doc.image.height
        rect = QRect(
            int(np.floor(x0 * sx)) - 1,
            int(np.floor(y0 * sy)) - 1,
            int(np.ceil(x1 * sx)) - int(np.floor(x0 * sx)) + 2,
            int(np.ceil(y1 * sy)) - int(np.floor(y0 * sy)) + 2,
        ).intersected(QRect(0, 0, pw, ph))
        self._dirty = rect if self._dirty is None else self._dirty.united(rect)
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def _refresh_mask(self, mask_id: str, full_bbox: tuple[int, int, int, int]) -> None:
        import cv2

        if self._doc is None:
            return
        arr = self._overlay_arrays[mask_id]
        mask = self._doc.mask(mask_id).data
        pw, ph = self._preview_size
        h, w = mask.shape
        x0, y0, x1, y1 = full_bbox
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return
        sx, sy = pw / w, ph / h
        px0 = int(np.floor(x0 * sx))
        py0 = int(np.floor(y0 * sy))
        px1 = int(np.ceil(x1 * sx))
        py1 = int(np.ceil(y1 * sy))
        px0, py0 = max(0, px0), max(0, py0)
        px1, py1 = min(pw, px1), min(ph, py1)
        if px1 <= px0 or py1 <= py0:
            return
        sub = mask[y0:y1, x0:x1]
        resized = cv2.resize(
            sub, (px1 - px0, py1 - py0), interpolation=cv2.INTER_NEAREST
        )
        arr[py0:py1, px0:px1, 3] = (resized.astype(np.uint16) * OVERLAY_ALPHA // 255).astype(
            np.uint8
        )

    def _flush_overlays(self) -> None:
        if self._doc is None:
            return
        if self._dirty is not None:
            self._refresh_rect_from_dirty()
            rect = self._dirty
            for mask_id, item in (
                (MASK_REMOVAL, self._removal_item),
                (MASK_PROTECT, self._protect_item),
            ):
                img = self._overlay_images[mask_id]
                pix = item.pixmap()
                if pix.isNull():
                    pix = QPixmap.fromImage(img.copy())
                painter = QPainter(pix)
                painter.drawImage(rect.topLeft(), img.copy(rect))
                painter.end()
                item.setPixmap(pix)
            self._dirty = None
        else:
            for mask_id, item in (
                (MASK_REMOVAL, self._removal_item),
                (MASK_PROTECT, self._protect_item),
            ):
                item.setPixmap(QPixmap.fromImage(self._overlay_images[mask_id].copy()))

    def _refresh_rect_from_dirty(self) -> None:
        """Recompute overlay alpha for the dirty rect from both masks."""
        import cv2

        rect = self._dirty
        pw, ph = self._preview_size
        h, w = self._doc.image.height, self._doc.image.width
        sx, sy = w / pw, h / ph
        fx0 = int(np.floor(rect.x() * sx))
        fy0 = int(np.floor(rect.y() * sy))
        fx1 = int(np.ceil((rect.x() + rect.width()) * sx))
        fy1 = int(np.ceil((rect.y() + rect.height()) * sy))
        fx0, fy0 = max(0, fx0), max(0, fy0)
        fx1, fy1 = min(w, fx1), min(h, fy1)
        if fx1 <= fx0 or fy1 <= fy0:
            return
        px0, py0 = rect.x(), rect.y()
        px1, py1 = rect.x() + rect.width(), rect.y() + rect.height()
        for mask_id in (MASK_REMOVAL, MASK_PROTECT):
            arr = self._overlay_arrays[mask_id]
            sub = self._doc.mask(mask_id).data[fy0:fy1, fx0:fx1]
            resized = cv2.resize(
                sub,
                (px1 - px0, py1 - py0),
                interpolation=cv2.INTER_NEAREST,
            )
            arr[py0:py1, px0:px1, 3] = (
                resized.astype(np.uint16) * OVERLAY_ALPHA // 255
            ).astype(np.uint8)

    # ------------------------------------------------------------------ mouse
    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._doc is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        pct = self._scale * self._view_zoom() * 100.0
        self.view_info_changed.emit(f"zoom {pct:.0f}%")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._doc is None:
            return
        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and self._space_down
        ):
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() == Qt.RightButton:
            self._hardness_drag = (event.position().toPoint().y(), self._hardness)
            return
        if event.button() == Qt.LeftButton:
            x, y = self._full_from_view(event.position())
            self._builder = self._doc.begin_stroke(
                self._active_mask, self._brush_size, self._hardness, self._eraser
            )
            bbox = self._builder.add_point(x, y)
            self._mark_dirty(bbox)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning and self._pan_start is not None:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        if self._hardness_drag is not None:
            start_y, start_h = self._hardness_drag
            dy = start_y - event.position().toPoint().y()
            self.set_hardness(start_h + dy / 200.0)
            self.view_info_changed.emit(f"hardness {self._hardness:.2f}")
            return
        if self._builder is not None and event.buttons() & Qt.LeftButton:
            x, y = self._full_from_view(event.position())
            bbox = self._builder.add_point(x, y)
            self._mark_dirty(bbox)
            return
        # keep brush cursor under pointer
        if self._doc is not None:
            scene = self.mapToScene(event.position().toPoint())
            self._cursor_item.setPos(scene)
            self._update_cursor_shape()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return
        if self._hardness_drag is not None and event.button() == Qt.RightButton:
            self._hardness_drag = None
            return
        if self._builder is not None and event.button() == Qt.LeftButton:
            builder = self._builder
            self._builder = None
            self._flush_overlays()
            if self._doc.commit_stroke(builder):
                self.stroke_committed.emit()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_down = True
            self.setCursor(Qt.OpenHandCursor)
            return
        if event.key() == Qt.Key_BracketLeft:
            self.set_brush_size(self._brush_size / 1.25)
            self.view_info_changed.emit(f"brush {self._brush_size:.0f} px")
            return
        if event.key() == Qt.Key_BracketRight:
            self.set_brush_size(self._brush_size * 1.25)
            self.view_info_changed.emit(f"brush {self._brush_size:.0f} px")
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_down = False
            self.setCursor(Qt.ArrowCursor)
            return
        super().keyReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._cursor_item.setVisible(False)
        super().leaveEvent(event)

    def enterEvent(self, event) -> None:
        if self._doc is not None:
            self._cursor_item.setVisible(True)
        super().enterEvent(event)
