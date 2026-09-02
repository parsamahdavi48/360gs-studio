"""Lightweight native Qt globe selector for perspective view layouts."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from gs360studio.domain.models import ViewSpec


class ViewGlobeWidget(QWidget):
    view_selected = Signal(int)
    views_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(280, 280)
        self.setMouseTracking(True)
        self._views: list[ViewSpec] = []
        self._selected = -1
        self._orbit_yaw = -25.0
        self._orbit_pitch = 15.0
        self._drag_origin = None

    def set_views(self, views: list[ViewSpec], selected: int | None = None) -> None:
        self._views = list(views)
        if selected is not None:
            self._selected = max(-1, min(int(selected), len(self._views) - 1))
        elif self._selected >= len(self._views):
            self._selected = -1
        self.update()

    def selected_index(self) -> int:
        return self._selected

    def views(self) -> list[ViewSpec]:
        """Return an isolated snapshot of the current view layout."""
        return list(self._views)

    def _sphere_geometry(self) -> tuple[QPointF, float]:
        radius = max(20.0, min(self.width(), self.height()) * 0.40)
        return QPointF(self.width() / 2.0, self.height() / 2.0), radius

    def _project(self, yaw: float, pitch: float) -> tuple[QPointF, float]:
        yaw_rad = math.radians(yaw + self._orbit_yaw)
        pitch_rad = math.radians(pitch)
        x = math.cos(pitch_rad) * math.sin(yaw_rad)
        y = math.sin(pitch_rad)
        z = math.cos(pitch_rad) * math.cos(yaw_rad)
        orbit_pitch = math.radians(self._orbit_pitch)
        rotated_y = y * math.cos(orbit_pitch) - z * math.sin(orbit_pitch)
        rotated_z = y * math.sin(orbit_pitch) + z * math.cos(orbit_pitch)
        center, radius = self._sphere_geometry()
        return QPointF(center.x() + x * radius, center.y() - rotated_y * radius), rotated_z

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center, radius = self._sphere_geometry()
        painter.setPen(QPen(QColor("#41495A"), 2))
        painter.setBrush(QColor("#1C2028"))
        painter.drawEllipse(center, radius, radius)

        grid_pen = QPen(QColor("#353B49"), 1)
        painter.setPen(grid_pen)
        for latitude in (-60, -30, 0, 30, 60):
            previous = None
            for yaw in range(-180, 181, 5):
                point, depth = self._project(float(yaw), float(latitude))
                if previous is not None and depth >= -0.08 and previous[1] >= -0.08:
                    painter.drawLine(previous[0], point)
                previous = (point, depth)
        for longitude in range(-180, 180, 30):
            previous = None
            for pitch in range(-90, 91, 4):
                point, depth = self._project(float(longitude), float(pitch))
                if previous is not None and depth >= -0.08 and previous[1] >= -0.08:
                    painter.drawLine(previous[0], point)
                previous = (point, depth)

        ordered = sorted(enumerate(self._views), key=lambda pair: self._project(pair[1].yaw_deg, pair[1].pitch_deg)[1])
        for index, view in ordered:
            point, depth = self._project(view.yaw_deg, view.pitch_deg)
            if depth < -0.12:
                continue
            selected = index == self._selected
            color = QColor("#5AA7FF" if view.enabled else "#626B7C")
            if selected:
                painter.setPen(QPen(QColor("#F3F6FC"), 2))
                painter.setBrush(QColor("#A66FE0"))
                size = 8.0
            else:
                painter.setPen(QPen(color.lighter(125), 1))
                painter.setBrush(color)
                size = 5.5
            painter.drawEllipse(point, size, size)
        painter.end()

    def _nearest_view(self, position: QPointF, maximum_distance: float = 20.0) -> int:
        nearest = -1
        distance = maximum_distance
        for index, view in enumerate(self._views):
            point, depth = self._project(view.yaw_deg, view.pitch_deg)
            if depth < -0.12:
                continue
            candidate = math.hypot(position.x() - point.x(), position.y() - point.y())
            if candidate < distance:
                nearest, distance = index, candidate
        return nearest

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.LeftButton:
            nearest = self._nearest_view(event.position())
            if nearest >= 0:
                self._selected = nearest
                self.view_selected.emit(nearest)
                self.update()
                return
            self._drag_origin = event.position()
        elif event.button() == Qt.RightButton:
            nearest = self._nearest_view(event.position())
            if nearest >= 0:
                view = self._views[nearest]
                from dataclasses import replace

                self._views[nearest] = replace(view, enabled=not view.enabled)
                self._selected = nearest
                self.view_selected.emit(nearest)
                self.views_changed.emit()
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if self._drag_origin is None or not (event.buttons() & Qt.LeftButton):
            return
        delta = event.position() - self._drag_origin
        self._drag_origin = event.position()
        self._orbit_yaw += delta.x() * 0.45
        self._orbit_pitch = max(-80.0, min(80.0, self._orbit_pitch - delta.y() * 0.35))
        self.update()

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        self._drag_origin = None
