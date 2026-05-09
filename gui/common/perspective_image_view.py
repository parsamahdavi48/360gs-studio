"""Hybrid CPU/GPU image view for equirectangular perspective previews."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPolygonF, QVector3D, QWheelEvent
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QStackedLayout, QWidget

from gui.common.perspective_preview import PerspectiveParams, perspective_output_size
from gui.common.zoomable_image_label import ZoomableImageLabel

_GL_COLOR_BUFFER_BIT = 0x00004000
_GL_TRIANGLE_STRIP = 0x0005
_GL_FLOAT = 0x1406
_GL_TEXTURE0 = 0x84C0


@dataclass(frozen=True)
class PerspectiveLabelOverlay:
    label: str
    box: tuple[int, int, int, int]
    origin: tuple[int, int]
    color_bgr: tuple[int, int, int]
    highlighted: bool = False
    polygon: tuple[tuple[float, float], ...] = ()
    fill_alpha: float = 0.0
    polyline: tuple[tuple[float, float], ...] = ()
    dashed: bool = False
    point_radius: float = 0.0


def bgr_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        return QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.shape[1] * 4,
            QImage.Format_RGBA8888,
        ).copy()
    return QImage(
        rgb.data,
        rgb.shape[1],
        rgb.shape[0],
        rgb.shape[1] * 3,
        QImage.Format_RGB888,
    ).copy()


class PerspectiveGLImageView(QOpenGLWidget):
    """OpenGL-backed FOV perspective view over an equirectangular texture."""

    look_dragged = Signal(float, float)
    image_clicked = Signal(float, float)
    gpu_failed = Signal()

    _VERTEX_SHADER = """
        attribute vec2 a_position;

        void main() {
            gl_Position = vec4(a_position, 0.0, 1.0);
        }
    """

    _FRAGMENT_SHADER = """
        #ifdef GL_ES
        precision highp float;
        #endif

        uniform sampler2D u_texture;
        uniform float u_yaw_rad;
        uniform float u_pitch_rad;
        uniform float u_roll_rad;
        uniform float u_fov_rad;
        uniform vec3 u_texture_ray_sign;
        uniform vec2 u_viewport_origin;
        uniform vec2 u_viewport_size;

        const float PI = 3.1415926535897932384626433832795;

        void main() {
            vec2 view_coord = (gl_FragCoord.xy - u_viewport_origin) / u_viewport_size;
            float view_x = view_coord.x * 2.0 - 1.0;
            float view_y = view_coord.y * 2.0 - 1.0;
            float focal = 1.0 / tan(u_fov_rad * 0.5);
            vec3 ray = normalize(vec3(view_x, view_y, focal));

            float cr = cos(u_roll_rad);
            float sr = sin(u_roll_rad);
            vec3 rolled = vec3(
                cr * ray.x - sr * ray.y,
                sr * ray.x + cr * ray.y,
                ray.z
            );

            float cp = cos(u_pitch_rad);
            float sp = sin(u_pitch_rad);
            vec3 pitched = vec3(
                rolled.x,
                cp * rolled.y - sp * rolled.z,
                sp * rolled.y + cp * rolled.z
            );

            float cy = cos(u_yaw_rad);
            float sy = sin(u_yaw_rad);
            vec3 world = vec3(
                cy * pitched.x + sy * pitched.z,
                pitched.y,
                -sy * pitched.x + cy * pitched.z
            );

            vec3 texture_ray = world * u_texture_ray_sign;
            float lon = atan(texture_ray.x, texture_ray.z);
            float lat = asin(clamp(texture_ray.y, -1.0, 1.0));
            vec2 pano_uv = vec2((lon / PI + 1.0) * 0.5, 0.5 - lat / PI);
            gl_FragColor = texture2D(u_texture, pano_uv);
        }
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source_image = QImage()
        self._texture: QOpenGLTexture | None = None
        self._program: QOpenGLShaderProgram | None = None
        self._vbo: QOpenGLBuffer | None = None
        self._vao: QOpenGLVertexArrayObject | None = None
        self._functions = None
        self._initialized = False
        self._failed = False
        self._params = PerspectiveParams()
        self._texture_ray_sign = (1.0, 1.0, 1.0)
        self._logical_size = QSize(1, 1)
        self._overlays: list[PerspectiveLabelOverlay] = []
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._drag_start: QPointF | None = None
        self._drag_last: QPointF | None = None
        self._drag_mode = "look"
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip(text)

    def failed(self) -> bool:
        return self._failed

    def logical_image_size(self) -> QSize:
        return QSize(self._logical_size)

    def set_drag_mode(self, mode: str) -> None:
        if mode not in {"pan", "look"}:
            return
        self._drag_mode = mode
        if self._drag_last is None:
            self.setCursor(Qt.OpenHandCursor)

    def set_source_image(self, image: QImage, *, logical_size: QSize | None = None) -> None:
        self._source_image = image.convertToFormat(QImage.Format_RGBA8888)
        size = logical_size or QSize(
            max(1, min(self._source_image.width(), self._source_image.height())),
            max(1, min(self._source_image.width(), self._source_image.height())),
        )
        self._logical_size = QSize(max(1, size.width()), max(1, size.height()))
        if self._initialized and not self._failed:
            self.makeCurrent()
            try:
                self._upload_texture()
            finally:
                self.doneCurrent()
        self._clamp_pan()
        self.update()

    def set_perspective_params(self, params: PerspectiveParams) -> None:
        self._params = params
        self.update()

    def set_texture_ray_sign(self, sign: float | tuple[float, float, float]) -> None:
        if isinstance(sign, tuple):
            values = sign
        else:
            value = float(sign)
            values = (value, value, value)
        self._texture_ray_sign = tuple(-1.0 if float(value) < 0.0 else 1.0 for value in values[:3])
        if len(self._texture_ray_sign) != 3:
            self._texture_ray_sign = (1.0, 1.0, 1.0)
        self.update()

    def set_label_overlays(self, overlays: list[PerspectiveLabelOverlay]) -> None:
        self._overlays = list(overlays)
        self.update()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def initializeGL(self) -> None:  # noqa: N802 - Qt API
        try:
            self._functions = self.context().functions()
            self._functions.initializeOpenGLFunctions()
            self._program = QOpenGLShaderProgram(self)
            if not self._program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, self._VERTEX_SHADER):
                raise RuntimeError(self._program.log())
            if not self._program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, self._FRAGMENT_SHADER):
                raise RuntimeError(self._program.log())
            if not self._program.link():
                raise RuntimeError(self._program.log())

            vertices = np.array(
                [
                    -1.0, 1.0,
                    -1.0, -1.0,
                    1.0, 1.0,
                    1.0, -1.0,
                ],
                dtype=np.float32,
            )
            self._vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            if not self._vbo.create():
                raise RuntimeError("Failed to create OpenGL vertex buffer")
            self._vbo.bind()
            self._vbo.allocate(vertices.tobytes(), int(vertices.nbytes))

            self._vao = QOpenGLVertexArrayObject(self)
            self._vao.create()
            self._vao.bind()
            self._program.bind()
            self._vbo.bind()
            pos_loc = self._program.attributeLocation(b"a_position")
            self._program.enableAttributeArray(pos_loc)
            self._program.setAttributeBuffer(pos_loc, _GL_FLOAT, 0, 2, 2 * 4)
            self._program.release()
            self._vao.release()
            self._vbo.release()

            self._initialized = True
            self._upload_texture()
        except Exception:
            self._failed = True
            self.gpu_failed.emit()

    def paintGL(self) -> None:  # noqa: N802 - Qt API
        if self._failed or self._functions is None:
            return
        dpr = max(1.0, float(self.devicePixelRatioF()))
        full_w = max(1, int(round(self.width() * dpr)))
        full_h = max(1, int(round(self.height() * dpr)))
        self._functions.glViewport(0, 0, full_w, full_h)
        self._functions.glClearColor(0.0, 0.0, 0.0, 0.0)
        self._functions.glClear(_GL_COLOR_BUFFER_BIT)

        if self._texture is None or self._program is None or self._vao is None:
            return
        target = self._target_rect()
        if target.width() <= 0 or target.height() <= 0:
            return

        viewport_x = int(round(target.left() * dpr))
        viewport_y = int(round((self.height() - target.bottom()) * dpr))
        viewport_w = max(1, int(round(target.width() * dpr)))
        viewport_h = max(1, int(round(target.height() * dpr)))
        self._functions.glViewport(viewport_x, viewport_y, viewport_w, viewport_h)

        self._program.bind()
        self._functions.glActiveTexture(_GL_TEXTURE0)
        self._texture.bind(0)
        self._program.setUniformValue1i(self._program.uniformLocation(b"u_texture"), 0)
        self._program.setUniformValue1f(
            self._program.uniformLocation(b"u_yaw_rad"),
            float(np.deg2rad(float(self._params.yaw_deg))),
        )
        self._program.setUniformValue1f(
            self._program.uniformLocation(b"u_pitch_rad"),
            float(np.deg2rad(float(self._params.pitch_deg))),
        )
        self._program.setUniformValue1f(
            self._program.uniformLocation(b"u_roll_rad"),
            float(np.deg2rad(float(getattr(self._params, "roll_deg", 0.0)))),
        )
        self._program.setUniformValue1f(
            self._program.uniformLocation(b"u_fov_rad"),
            float(np.deg2rad(float(self._params.fov_deg))),
        )
        self._program.setUniformValue(
            self._program.uniformLocation(b"u_texture_ray_sign"),
            QVector3D(*self._texture_ray_sign),
        )
        self._program.setUniformValue(
            self._program.uniformLocation(b"u_viewport_origin"),
            float(viewport_x),
            float(viewport_y),
        )
        self._program.setUniformValue(
            self._program.uniformLocation(b"u_viewport_size"),
            float(viewport_w),
            float(viewport_h),
        )
        self._vao.bind()
        self._functions.glDrawArrays(_GL_TRIANGLE_STRIP, 0, 4)
        self._vao.release()
        self._texture.release()
        self._program.release()

        self._functions.glViewport(0, 0, full_w, full_h)
        if self._overlays:
            self._paint_overlays(target)

    def resizeGL(self, _width: int, _height: int) -> None:  # noqa: N802 - Qt API
        self._clamp_pan()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        if self._source_image.isNull():
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

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if not self._source_image.isNull() and event.button() == Qt.LeftButton:
            self._drag_start = event.position()
            self._drag_last = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
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

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
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

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if not self._source_image.isNull() and event.button() == Qt.LeftButton:
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _upload_texture(self) -> None:
        if self._source_image.isNull() or self._failed:
            return
        if self._texture is not None:
            self._texture.destroy()
            self._texture = None
        self._texture = QOpenGLTexture(self._source_image, QOpenGLTexture.MipMapGeneration.DontGenerateMipMaps)
        self._texture.setMinificationFilter(QOpenGLTexture.Filter.Linear)
        self._texture.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
        self._texture.setWrapMode(
            QOpenGLTexture.CoordinateDirection.DirectionS,
            QOpenGLTexture.WrapMode.Repeat,
        )
        self._texture.setWrapMode(
            QOpenGLTexture.CoordinateDirection.DirectionT,
            QOpenGLTexture.WrapMode.ClampToEdge,
        )

    def _fit_scale(self) -> float:
        if self._logical_size.width() <= 0 or self._logical_size.height() <= 0:
            return 1.0
        if self.width() <= 0 or self.height() <= 0:
            return 1.0
        return min(
            self.width() / float(self._logical_size.width()),
            self.height() / float(self._logical_size.height()),
        )

    def _display_scale(self) -> float:
        return self._fit_scale() * self._zoom

    def _target_rect(self) -> QRectF:
        scale = self._display_scale()
        draw_w = self._logical_size.width() * scale
        draw_h = self._logical_size.height() * scale
        left = (self.width() - draw_w) / 2.0 + self._pan.x()
        top = (self.height() - draw_h) / 2.0 + self._pan.y()
        return QRectF(left, top, draw_w, draw_h)

    def _logical_point_from_widget(self, point: QPointF) -> QPointF | None:
        target = self._target_rect()
        if not target.contains(point):
            return None
        logical_w = max(1.0, float(self._logical_size.width()))
        logical_h = max(1.0, float(self._logical_size.height()))
        x = (point.x() - target.left()) / max(target.width(), 1e-12) * logical_w
        y = (point.y() - target.top()) / max(target.height(), 1e-12) * logical_h
        return QPointF(float(x), float(y))

    def _clamp_pan(self) -> None:
        if self._logical_size.width() <= 0 or self._logical_size.height() <= 0:
            self._pan = QPointF(0.0, 0.0)
            return
        scale = self._display_scale()
        draw_w = self._logical_size.width() * scale
        draw_h = self._logical_size.height() * scale
        max_x = max(0.0, (draw_w - self.width()) / 2.0)
        max_y = max(0.0, (draw_h - self.height()) / 2.0)
        self._pan = QPointF(
            max(-max_x, min(max_x, self._pan.x())),
            max(-max_y, min(max_y, self._pan.y())),
        )

    def _paint_overlays(self, target: QRectF) -> None:
        logical_w = max(1.0, float(self._logical_size.width()))
        logical_h = max(1.0, float(self._logical_size.height()))
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setClipRect(target)
        painter.translate(target.left(), target.top())
        painter.scale(target.width() / logical_w, target.height() / logical_h)
        font = painter.font()
        font.setPixelSize(13)
        painter.setFont(font)
        for item in self._overlays:
            self._draw_overlay_label(painter, item)
        painter.end()

    def _draw_overlay_label(self, painter: QPainter, item: PerspectiveLabelOverlay) -> None:
        x1, y1, x2, y2 = item.box
        rect = QRectF(float(x1), float(y1), float(max(1, x2 - x1)), float(max(1, y2 - y1)))
        color = QColor(int(item.color_bgr[2]), int(item.color_bgr[1]), int(item.color_bgr[0]))
        if item.polyline:
            self._draw_polyline_overlay(painter, item, color)
            return

        if item.polygon:
            polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in item.polygon])
            fill = QColor(color)
            fill.setAlphaF(max(0.0, min(1.0, float(item.fill_alpha))))
            if item.fill_alpha > 0.0:
                painter.setPen(Qt.NoPen)
                painter.setBrush(fill)
                painter.drawPolygon(polygon)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0), 6 if item.highlighted else 4))
            painter.drawPolygon(polygon)
            painter.setPen(QPen(color, 3 if item.highlighted else 2))
            painter.drawPolygon(polygon)

            origin = QPointF(float(item.origin[0]), float(item.origin[1]))
            painter.setPen(QPen(QColor(0, 0, 0), 4 if item.highlighted else 3))
            painter.drawText(origin, item.label)
            painter.setPen(QPen(QColor(245, 245, 245) if item.highlighted else color, 1))
            painter.drawText(origin, item.label)
            return

        if item.highlighted:
            bg = QColor(color)
            bg.setAlphaF(0.42)
        else:
            bg = QColor(0, 0, 0, 178)
        painter.fillRect(rect, bg)

        painter.setPen(QPen(QColor(0, 0, 0), 4 if item.highlighted else 2))
        painter.drawRect(rect)
        painter.setPen(QPen(color if item.highlighted else QColor(245, 245, 245), 2 if item.highlighted else 1))
        painter.drawRect(rect)

        origin = QPointF(float(item.origin[0]), float(item.origin[1]))
        painter.setPen(QPen(QColor(0, 0, 0), 4 if item.highlighted else 3))
        painter.drawText(origin, item.label)
        painter.setPen(QPen(QColor(245, 245, 245) if item.highlighted else color, 1))
        painter.drawText(origin, item.label)

    def _draw_polyline_overlay(self, painter: QPainter, item: PerspectiveLabelOverlay, color: QColor) -> None:
        points = [QPointF(float(x), float(y)) for x, y in item.polyline]
        width = 3 if item.highlighted else 1
        pen = QPen(color, width)
        if item.dashed:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        for a, b in zip(points, points[1:], strict=False):
            painter.drawLine(a, b)
        if item.point_radius > 0.0 and points:
            center = points[-1]
            radius = float(item.point_radius)
            painter.setPen(QPen(QColor(0, 0, 0), max(2, width + 2)))
            painter.setBrush(color)
            painter.drawEllipse(center, radius, radius)
            painter.setPen(QPen(QColor(245, 245, 245), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(center, radius, radius)
        if item.label:
            origin = QPointF(float(item.origin[0]), float(item.origin[1]))
            painter.setPen(QPen(QColor(0, 0, 0), 4))
            painter.drawText(origin, item.label)
            painter.setPen(QPen(color, 1))
            painter.drawText(origin, item.label)


class PerspectiveImageView(QWidget):
    """Switches between the existing CPU pixmap view and GPU perspective view."""

    look_dragged = Signal(float, float)
    image_clicked = Signal(float, float)
    gpu_failed = Signal()

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cpu_view = ZoomableImageLabel(text)
        self._gpu_view: PerspectiveGLImageView | None = PerspectiveGLImageView(text)
        self._gpu_view.look_dragged.connect(self.look_dragged.emit)
        self._gpu_view.image_clicked.connect(self.image_clicked.emit)
        self._gpu_view.gpu_failed.connect(self._on_gpu_failed)
        self._gpu_failed = False
        self._drag_mode = "pan"

        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._cpu_view)
        self._stack.addWidget(self._gpu_view)
        self._stack.setCurrentWidget(self._cpu_view)

        self._cpu_view.look_dragged.connect(self.look_dragged.emit)
        self._cpu_view.image_clicked.connect(self.image_clicked.emit)

    @property
    def _source_pixmap(self):  # noqa: ANN001 - compatibility with existing preview tests
        return self._cpu_view._source_pixmap

    def setStyleSheet(self, style_sheet: str) -> None:  # noqa: N802 - Qt API
        super().setStyleSheet(style_sheet)
        self._cpu_view.setStyleSheet(style_sheet)
        if self._gpu_view is not None:
            self._gpu_view.setStyleSheet(style_sheet)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        self._cpu_view.setText(text)
        self._stack.setCurrentWidget(self._cpu_view)

    def set_source_pixmap(self, pixmap) -> None:  # noqa: ANN001 - QPixmap import kept local to callers
        self._cpu_view.set_source_pixmap(pixmap)
        self._stack.setCurrentWidget(self._cpu_view)

    def set_drag_mode(self, mode: str) -> None:
        if mode not in {"pan", "look"}:
            return
        self._drag_mode = mode
        self._cpu_view.set_drag_mode(mode)
        if self._gpu_view is not None:
            self._gpu_view.set_drag_mode(mode)

    def reset_view(self) -> None:
        self._cpu_view.reset_view()
        if self._gpu_view is not None:
            self._gpu_view.reset_view()

    def set_perspective_image_bgr(
        self,
        image: np.ndarray,
        params: PerspectiveParams,
        *,
        overlays: list[PerspectiveLabelOverlay] | None = None,
        logical_size: QSize | None = None,
        texture_ray_sign: float | tuple[float, float, float] = 1.0,
    ) -> bool:
        if self._gpu_view is None or self._gpu_failed or self._gpu_view.failed():
            return False
        size = logical_size or QSize(perspective_output_size(image), perspective_output_size(image))
        self._gpu_view.set_drag_mode(self._drag_mode)
        self._gpu_view.set_texture_ray_sign(texture_ray_sign)
        self._gpu_view.set_source_image(bgr_to_qimage(image), logical_size=size)
        self._gpu_view.set_perspective_params(params)
        self._gpu_view.set_label_overlays(overlays or [])
        self._stack.setCurrentWidget(self._gpu_view)
        return True

    def set_perspective_params(self, params: PerspectiveParams) -> bool:
        if not self.is_showing_gpu_perspective():
            return False
        assert self._gpu_view is not None
        self._gpu_view.set_perspective_params(params)
        return True

    def set_perspective_label_overlays(self, overlays: list[PerspectiveLabelOverlay]) -> bool:
        if not self.is_showing_gpu_perspective():
            return False
        assert self._gpu_view is not None
        self._gpu_view.set_label_overlays(overlays)
        return True

    def perspective_logical_size(self) -> QSize:
        if self._gpu_view is None:
            return QSize(0, 0)
        return self._gpu_view.logical_image_size()

    def is_showing_gpu_perspective(self) -> bool:
        return (
            self._gpu_view is not None
            and not self._gpu_failed
            and not self._gpu_view.failed()
            and self._stack.currentWidget() is self._gpu_view
        )

    def _on_gpu_failed(self) -> None:
        self._gpu_failed = True
        self._stack.setCurrentWidget(self._cpu_view)
        self.gpu_failed.emit()
