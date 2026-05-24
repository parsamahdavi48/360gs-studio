"""Zoomable/pannable image label used by preview panes."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QLabel


class ZoomableImageLabel(QLabel):
    """QLabel-like image view with wheel zoom, drag pan, and double-click reset."""

    look_dragged = Signal(float, float)
    image_clicked = Signal(float, float)

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self._source_pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._drag_start: QPointF | None = None
        self._drag_last: QPointF | None = None
        self._drag_mode = "pan"
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)

    def set_drag_mode(self, mode: str) -> None:
        if mode not in {"pan", "look"}:
            return
        self._drag_mode = mode
        if self._drag_last is None:
            self.setCursor(Qt.OpenHandCursor)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API naming
        if hasattr(self, "_source_pixmap"):
            self._source_pixmap = None
            self._zoom = 1.0
            self._pan = QPointF(0.0, 0.0)
        super().setText(text)
        self.update()

    def set_source_pixmap(self, pixmap: QPixmap | None) -> None:
        self._source_pixmap = pixmap
        if pixmap is None:
            super().clear()
            self._zoom = 1.0
            self._pan = QPointF(0.0, 0.0)
        else:
            super().setText("")
            self._zoom = 1.0
            self._pan = QPointF(0.0, 0.0)
            self._clamp_pan()
        self.update()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def paintEvent(self, event) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, _smooth_pixmap_transform_for_scale(self._display_scale()))
        target = self._target_rect()
        painter.drawPixmap(target, self._source_pixmap, QRectF(self._source_pixmap.rect()))

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            super().wheelEvent(event)
            return

        old_scale = self._display_scale()
        if old_scale <= 0:
            return

        cursor = event.position()
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        image_point = (cursor - center - self._pan) / old_scale

        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self._zoom = max(1.0, min(12.0, self._zoom * factor))

        new_scale = self._display_scale()
        self._pan = cursor - center - image_point * new_scale
        self._clamp_pan()
        self.update()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._source_pixmap is not None and event.button() == Qt.LeftButton:
            self._drag_start = event.position()
            self._drag_last = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_last is not None:
            delta = event.position() - self._drag_last
            self._drag_last = event.position()
            if self._drag_mode == "look":
                self.look_dragged.emit(float(delta.x()), float(delta.y()))
                event.accept()
                return
            self._pan += delta
            self._clamp_pan()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._drag_last is not None:
            start = self._drag_start
            release = event.position()
            self._drag_last = None
            self._drag_start = None
            self.setCursor(Qt.OpenHandCursor)
            if start is not None and (release - start).manhattanLength() < 4:
                clicked = self._logical_point_from_widget(release)
                if clicked is not None:
                    self.image_clicked.emit(clicked.x(), clicked.y())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._source_pixmap is not None and event.button() == Qt.LeftButton:
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._clamp_pan()

    def _fit_scale(self) -> float:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return 1.0
        if self.width() <= 0 or self.height() <= 0:
            return 1.0
        return min(
            self.width() / float(self._source_pixmap.width()),
            self.height() / float(self._source_pixmap.height()),
        )

    def _display_scale(self) -> float:
        return self._fit_scale() * self._zoom

    def _target_rect(self) -> QRectF:
        if self._source_pixmap is None:
            return QRectF()
        scale = self._display_scale()
        draw_w = self._source_pixmap.width() * scale
        draw_h = self._source_pixmap.height() * scale
        left = (self.width() - draw_w) / 2.0 + self._pan.x()
        top = (self.height() - draw_h) / 2.0 + self._pan.y()
        return QRectF(left, top, draw_w, draw_h)

    def _logical_point_from_widget(self, point: QPointF) -> QPointF | None:
        if self._source_pixmap is None:
            return None
        target = self._target_rect()
        if not target.contains(point):
            return None
        x = (point.x() - target.left()) / max(target.width(), 1e-12) * self._source_pixmap.width()
        y = (point.y() - target.top()) / max(target.height(), 1e-12) * self._source_pixmap.height()
        return QPointF(float(x), float(y))

    def _clamp_pan(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            self._pan = QPointF(0.0, 0.0)
            return

        scale = self._display_scale()
        draw_w = self._source_pixmap.width() * scale
        draw_h = self._source_pixmap.height() * scale
        max_x = max(0.0, (draw_w - self.width()) / 2.0)
        max_y = max(0.0, (draw_h - self.height()) / 2.0)
        self._pan = QPointF(
            max(-max_x, min(max_x, self._pan.x())),
            max(-max_y, min(max_y, self._pan.y())),
        )


def _smooth_pixmap_transform_for_scale(scale: float) -> bool:
    return float(scale) < 1.0
