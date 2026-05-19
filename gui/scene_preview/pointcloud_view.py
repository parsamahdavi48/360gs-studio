"""OpenGL point-cloud view for read-only scene preview."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPolygonF, QVector3D, QWheelEvent
from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram, QOpenGLVertexArrayObject
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from core.pointcloud_io import DEFAULT_POINT_COLOR_RGBA, POINT_COLOR_ALPHA
from core.scene_preview import ScenePreviewCamera, ScenePreviewDataset, ScenePreviewPointCloud
from gui import i18n

GRID_X_AXIS_COLOR = QColor(255, 92, 92)
GRID_Z_AXIS_COLOR = QColor(90, 175, 245)
CAMERA_DOT_COLOR = QColor(95, 160, 255)
SELECTED_CAMERA_DOT_COLOR = QColor(255, 64, 160)
SELECTED_CAMERA_HALO_COLOR = QColor(255, 255, 255, 220)
CAMERA_OUTLINE_COLOR = QColor(7, 10, 14)
SELECTED_VIEW_RAY_COLOR = QColor(255, 230, 115, 220)
CAMERA_DOT_RADIUS = 4
SELECTED_CAMERA_DOT_RADIUS = 8

_GL_COLOR_BUFFER_BIT = 0x00004000
_GL_FLOAT = 0x1406
_GL_POINTS = 0x0000
_GL_BLEND = 0x0BE2
_GL_SRC_ALPHA = 0x0302
_GL_ONE_MINUS_SRC_ALPHA = 0x0303


class ScenePointCloudView(QOpenGLWidget):
    """GPU-backed orthographic point-cloud and camera pose preview."""

    gpu_pointcloud_failed = Signal()
    camera_selected = Signal(str)

    _POINT_VERTEX_SHADER = """
        #ifdef GL_ES
        precision highp float;
        #endif

        attribute vec3 a_position;
        attribute vec4 a_color;

        uniform vec3 u_center;
        uniform vec3 u_right;
        uniform vec3 u_up;
        uniform vec2 u_viewport_size;
        uniform float u_pixels_per_unit;
        uniform float u_point_size;

        varying vec4 v_color;

        void main() {
            vec3 rel = a_position - u_center;
            float screen_x = dot(rel, u_right) * u_pixels_per_unit;
            float screen_y = dot(rel, u_up) * u_pixels_per_unit;
            gl_Position = vec4(
                screen_x / max(u_viewport_size.x * 0.5, 1.0),
                screen_y / max(u_viewport_size.y * 0.5, 1.0),
                0.0,
                1.0
            );
            gl_PointSize = u_point_size;
            v_color = a_color;
        }
    """

    _POINT_FRAGMENT_SHADER = """
        #ifdef GL_ES
        precision mediump float;
        #endif

        varying vec4 v_color;

        void main() {
            gl_FragColor = v_color;
        }
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(520, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.OpenHandCursor)
        self._dataset: ScenePreviewDataset | None = None
        self._pointcloud: ScenePreviewPointCloud | None = None
        self._cameras: tuple[ScenePreviewCamera, ...] = ()
        self._selected_camera_id = ""
        self._selected_view_ray_direction: np.ndarray | None = None
        self._view_yaw_deg = 35.0
        self._view_pitch_deg = -28.0
        self._view_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._pixels_per_unit = 18.0
        self._grid_step = 1.0
        self._last_mouse: QPointF | None = None
        self._press_pos: QPointF | None = None
        self._press_button: Qt.MouseButton | None = None
        self._user_navigated = False
        self._gpu_failed = False
        self._gpu_initialized = False
        self._gpu_pointcloud_dirty = True
        self._gpu_point_count = 0
        self._gpu_functions = None
        self._gpu_program: QOpenGLShaderProgram | None = None
        self._gpu_vbo: QOpenGLBuffer | None = None
        self._gpu_vao: QOpenGLVertexArrayObject | None = None

    def sizeHint(self) -> QSize:
        return QSize(700, 460)

    def set_dataset(self, dataset: ScenePreviewDataset | None) -> None:
        self._dataset = dataset
        self._pointcloud = dataset.pointcloud if dataset is not None else None
        self._cameras = dataset.cameras if dataset is not None else ()
        if self._selected_camera_id and not any(camera.camera_id == self._selected_camera_id for camera in self._cameras):
            self._selected_camera_id = ""
            self._selected_view_ray_direction = None
        self._gpu_pointcloud_dirty = True
        self._fit_scene(force=True)
        self.update()

    def set_selected_camera_id(self, camera_id: str) -> None:
        self._selected_camera_id = str(camera_id or "")
        self.update()

    def set_selected_view_ray(self, direction: np.ndarray | None) -> None:
        if direction is None:
            self._selected_view_ray_direction = None
        else:
            self._selected_view_ray_direction = _normalized(
                np.asarray(direction, dtype=np.float64).reshape(3),
                fallback=(0.0, 0.0, 1.0),
            )
        self.update()

    def pointcloud_renderer_label(self) -> str:
        return "GPU unavailable" if self._gpu_failed else "GPU"

    def _pointcloud_gl_vertex_data(self) -> np.ndarray:
        return _pointcloud_vertex_data(self._pointcloud)

    def initializeGL(self) -> None:  # noqa: N802 - Qt API
        try:
            self._gpu_functions = self.context().functions()
            self._gpu_functions.initializeOpenGLFunctions()
            self._gpu_program = QOpenGLShaderProgram(self)
            if not self._gpu_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, self._POINT_VERTEX_SHADER):
                raise RuntimeError(self._gpu_program.log())
            if not self._gpu_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, self._POINT_FRAGMENT_SHADER):
                raise RuntimeError(self._gpu_program.log())
            if not self._gpu_program.link():
                raise RuntimeError(self._gpu_program.log())

            self._gpu_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            if not self._gpu_vbo.create():
                raise RuntimeError("Failed to create OpenGL point-cloud vertex buffer")
            self._gpu_vao = QOpenGLVertexArrayObject(self)
            if not self._gpu_vao.create():
                raise RuntimeError("Failed to create OpenGL point-cloud vertex array")

            self._gpu_vao.bind()
            self._gpu_program.bind()
            self._gpu_vbo.bind()
            stride = 7 * 4
            position_location = self._gpu_program.attributeLocation(b"a_position")
            color_location = self._gpu_program.attributeLocation(b"a_color")
            if position_location < 0 or color_location < 0:
                raise RuntimeError("OpenGL point-cloud shader attributes are missing")
            self._gpu_program.enableAttributeArray(position_location)
            self._gpu_program.setAttributeBuffer(position_location, _GL_FLOAT, 0, 3, stride)
            self._gpu_program.enableAttributeArray(color_location)
            self._gpu_program.setAttributeBuffer(color_location, _GL_FLOAT, 3 * 4, 4, stride)
            self._gpu_vbo.release()
            self._gpu_program.release()
            self._gpu_vao.release()
            self._gpu_initialized = True
            self._gpu_pointcloud_dirty = True
        except Exception:
            self._gpu_failed = True
            self._gpu_initialized = False
            self.gpu_pointcloud_failed.emit()

    def resizeGL(self, _width: int, _height: int) -> None:  # noqa: N802 - Qt API
        self._fit_scene(force=not self._user_navigated)

    def paintGL(self) -> None:  # noqa: N802 - Qt API
        self._clear_gl()
        self._draw_pointcloud_gpu()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self._draw_grid(painter)
        self._draw_selected_view_ray(painter)
        self._draw_cameras(painter)
        self._draw_world_axes(painter)
        self._draw_status(painter)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse = event.position()
        self._press_pos = event.position()
        self._press_button = event.button()
        if event.button() == Qt.LeftButton:
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_mouse is None:
            self._last_mouse = event.position()
            return
        delta = event.position() - self._last_mouse
        self._last_mouse = event.position()
        buttons = event.buttons()
        if buttons & Qt.LeftButton:
            self._user_navigated = True
            self._view_yaw_deg -= float(delta.x()) * 0.35
            self._view_pitch_deg = max(-85.0, min(85.0, self._view_pitch_deg + float(delta.y()) * 0.25))
            self.update()
        elif buttons & (Qt.RightButton | Qt.MiddleButton):
            self._user_navigated = True
            right, up, _forward = self._view_basis()
            self._view_center -= right * (float(delta.x()) / max(self._pixels_per_unit, 1e-6))
            self._view_center += up * (float(delta.y()) / max(self._pixels_per_unit, 1e-6))
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._press_button == Qt.LeftButton and self._press_pos is not None:
            delta = event.position() - self._press_pos
            if abs(float(delta.x())) <= 4.0 and abs(float(delta.y())) <= 4.0:
                camera_id = self._camera_id_at_screen_pos(event.position())
                if camera_id:
                    self._selected_camera_id = camera_id
                    self.camera_selected.emit(camera_id)
        self._last_mouse = None
        self._press_pos = None
        self._press_button = None
        self.setCursor(Qt.OpenHandCursor)

    def mouseDoubleClickEvent(self, _event: QMouseEvent) -> None:
        self._user_navigated = False
        self._fit_scene(force=True)
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self._user_navigated = True
        factor = 1.15 ** (float(delta) / 120.0)
        self._pixels_per_unit = max(0.001, min(200000.0, self._pixels_per_unit * factor))
        self.update()

    def _clear_gl(self) -> None:
        if self._gpu_functions is None:
            return
        dpr = max(1.0, float(self.devicePixelRatioF()))
        full_w = max(1, int(round(self.width() * dpr)))
        full_h = max(1, int(round(self.height() * dpr)))
        self._gpu_functions.glViewport(0, 0, full_w, full_h)
        self._gpu_functions.glClearColor(13.0 / 255.0, 17.0 / 255.0, 23.0 / 255.0, 1.0)
        self._gpu_functions.glClear(_GL_COLOR_BUFFER_BIT)

    def _upload_pointcloud_gl_data(self) -> None:
        if self._gpu_failed or not self._gpu_initialized or self._gpu_vbo is None:
            return
        data = self._pointcloud_gl_vertex_data()
        self._gpu_point_count = int(len(data))
        self._gpu_vbo.bind()
        if self._gpu_point_count:
            self._gpu_vbo.allocate(data.tobytes(), int(data.nbytes))
        else:
            self._gpu_vbo.allocate(0)
        self._gpu_vbo.release()
        self._gpu_pointcloud_dirty = False

    def _draw_pointcloud_gpu(self) -> bool:
        if (
            self._gpu_failed
            or not self._gpu_initialized
            or self._gpu_functions is None
            or self._gpu_program is None
            or self._gpu_vao is None
            or self._gpu_vbo is None
            or self._pointcloud is None
            or len(self._pointcloud.points) == 0
        ):
            return False
        if self._gpu_pointcloud_dirty:
            try:
                self._upload_pointcloud_gl_data()
            except Exception:
                self._gpu_failed = True
                self.gpu_pointcloud_failed.emit()
                return False
        if self._gpu_point_count <= 0:
            return False

        dpr = max(1.0, float(self.devicePixelRatioF()))
        full_w = max(1, int(round(self.width() * dpr)))
        full_h = max(1, int(round(self.height() * dpr)))
        right, up, _forward = self._view_basis()
        self._gpu_functions.glEnable(_GL_BLEND)
        self._gpu_functions.glBlendFunc(_GL_SRC_ALPHA, _GL_ONE_MINUS_SRC_ALPHA)
        self._gpu_program.bind()
        self._gpu_program.setUniformValue(
            self._gpu_program.uniformLocation(b"u_center"),
            QVector3D(float(self._view_center[0]), float(self._view_center[1]), float(self._view_center[2])),
        )
        self._gpu_program.setUniformValue(
            self._gpu_program.uniformLocation(b"u_right"),
            QVector3D(float(right[0]), float(right[1]), float(right[2])),
        )
        self._gpu_program.setUniformValue(
            self._gpu_program.uniformLocation(b"u_up"),
            QVector3D(float(up[0]), float(up[1]), float(up[2])),
        )
        self._gpu_program.setUniformValue(self._gpu_program.uniformLocation(b"u_viewport_size"), float(full_w), float(full_h))
        self._gpu_program.setUniformValue1f(
            self._gpu_program.uniformLocation(b"u_pixels_per_unit"),
            float(self._pixels_per_unit) * dpr,
        )
        self._gpu_program.setUniformValue1f(self._gpu_program.uniformLocation(b"u_point_size"), max(1.0, 1.35 * dpr))
        self._gpu_vao.bind()
        self._gpu_functions.glDrawArrays(_GL_POINTS, 0, self._gpu_point_count)
        self._gpu_vao.release()
        self._gpu_program.release()
        return True

    def _fit_scene(self, *, force: bool) -> None:
        if not force:
            return
        groups: list[np.ndarray] = []
        if self._pointcloud is not None and len(self._pointcloud.points):
            pc = np.asarray(self._pointcloud.points, dtype=np.float64)
            if len(pc) > 12000:
                pc = pc[np.linspace(0, len(pc) - 1, 12000, dtype=np.int64)]
            if len(pc) >= 20:
                lower = np.percentile(pc, 1.0, axis=0)
                upper = np.percentile(pc, 99.0, axis=0)
                inside = np.all((pc >= lower) & (pc <= upper), axis=1)
                if np.count_nonzero(inside) >= 10:
                    pc = pc[inside]
            groups.append(pc)
        if self._cameras:
            groups.append(np.asarray([camera.position for camera in self._cameras], dtype=np.float64))
        if not groups:
            self._view_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            self._pixels_per_unit = 18.0
            self._grid_step = 1.0
            return
        points = np.vstack(groups)
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        self._view_center = (minimum + maximum) * 0.5
        span = float(np.max(maximum - minimum))
        if span > 1e-9:
            viewport = max(240, min(max(self.width(), 1), max(self.height(), 1)))
            self._pixels_per_unit = max(0.001, min(200000.0, viewport / (span * 1.35)))
            self._grid_step = _nice_step(span / 8.0)

    def _view_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        yaw = np.deg2rad(self._view_yaw_deg)
        pitch = np.deg2rad(self._view_pitch_deg)
        forward = np.array(
            [np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)],
            dtype=np.float64,
        )
        right = np.array([-np.cos(yaw), 0.0, np.sin(yaw)], dtype=np.float64)
        return _right_handed_view_basis(
            right=right,
            up=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            forward=forward,
        )

    def _project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        right, up, forward = self._view_basis()
        rel = points - self._view_center
        depth = rel @ forward
        x = rel @ right * self._pixels_per_unit + self.width() * 0.5
        y = -(rel @ up) * self._pixels_per_unit + self.height() * 0.5
        return np.column_stack([x, y]), depth

    def _draw_grid(self, painter: QPainter) -> None:
        step = max(self._grid_step, 1e-6)
        visible_units = max(self.width(), self.height()) / max(self._pixels_per_unit, 1e-6)
        half = max(step * 6.0, visible_units * 0.75)
        x_min = math.floor((self._view_center[0] - half) / step) * step
        x_max = math.ceil((self._view_center[0] + half) / step) * step
        z_min = math.floor((self._view_center[2] - half) / step) * step
        z_max = math.ceil((self._view_center[2] + half) / step) * step
        max_lines = 160
        while ((x_max - x_min) / step + (z_max - z_min) / step) > max_lines:
            step *= 2.0
            x_min = math.floor((self._view_center[0] - half) / step) * step
            x_max = math.ceil((self._view_center[0] + half) / step) * step
            z_min = math.floor((self._view_center[2] - half) / step) * step
            z_max = math.ceil((self._view_center[2] + half) / step) * step

        for x in np.arange(x_min, x_max + step * 0.5, step):
            color = GRID_Z_AXIS_COLOR if abs(float(x)) <= step * 0.25 else QColor(55, 63, 72, 82)
            width = 2 if abs(float(x)) <= step * 0.25 else 1
            self._draw_world_line(painter, np.array([x, 0.0, z_min]), np.array([x, 0.0, z_max]), color, width)
        for z in np.arange(z_min, z_max + step * 0.5, step):
            color = GRID_X_AXIS_COLOR if abs(float(z)) <= step * 0.25 else QColor(55, 63, 72, 82)
            width = 2 if abs(float(z)) <= step * 0.25 else 1
            self._draw_world_line(painter, np.array([x_min, 0.0, z]), np.array([x_max, 0.0, z]), color, width)
        self._draw_marker(painter, np.array([0.0, 0.0, 0.0]), QColor(255, 80, 255), "O", radius=4)

    def _draw_cameras(self, painter: QPainter) -> None:
        if not self._cameras:
            return
        positions = np.asarray([camera.position for camera in self._cameras], dtype=np.float64)
        xy, _depth = self._project(positions)
        painter.setBrush(CAMERA_DOT_COLOR)
        for camera, point in zip(self._cameras, xy, strict=True):
            if not np.all(np.isfinite(point)):
                continue
            selected = camera.camera_id == self._selected_camera_id
            center = QPointF(float(point[0]), float(point[1]))
            color = SELECTED_CAMERA_DOT_COLOR if selected else CAMERA_DOT_COLOR
            radius = SELECTED_CAMERA_DOT_RADIUS if selected else CAMERA_DOT_RADIUS
            if selected:
                painter.setPen(QPen(SELECTED_CAMERA_HALO_COLOR, 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(center, radius + 4, radius + 4)
            painter.setPen(QPen(CAMERA_OUTLINE_COLOR, 3))
            painter.setBrush(color)
            painter.drawEllipse(center, radius, radius)
            painter.setPen(QPen(color, 1))
            painter.drawEllipse(center, radius, radius)
        painter.setBrush(Qt.NoBrush)

    def _draw_selected_view_ray(self, painter: QPainter) -> None:
        camera = self._selected_camera()
        if camera is None or self._selected_view_ray_direction is None:
            return
        length = max(self._scene_scale() * 0.035, 0.02)
        start = np.asarray(camera.position, dtype=np.float64)
        end = start + self._selected_view_ray_direction * length
        self._draw_world_arrow(
            painter,
            start,
            end,
            SELECTED_VIEW_RAY_COLOR,
            None,
            width=3,
        )

    def _selected_camera(self) -> ScenePreviewCamera | None:
        if not self._selected_camera_id:
            return None
        return next((camera for camera in self._cameras if camera.camera_id == self._selected_camera_id), None)

    def _camera_id_at_screen_pos(self, pos: QPointF, *, max_distance_px: float = 14.0) -> str | None:
        if not self._cameras:
            return None
        positions = np.asarray([camera.position for camera in self._cameras], dtype=np.float64)
        xy, _depth = self._project(positions)
        target = np.array([float(pos.x()), float(pos.y())], dtype=np.float64)
        finite = np.isfinite(xy).all(axis=1)
        if not np.any(finite):
            return None
        distances = np.linalg.norm(xy - target[None, :], axis=1)
        distances[~finite] = np.inf
        index = int(np.argmin(distances))
        if float(distances[index]) > float(max_distance_px):
            return None
        return self._cameras[index].camera_id

    def _draw_world_axes(self, painter: QPainter) -> None:
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        length = max(self._grid_step * 2.0, self._scene_scale() * 0.08)
        axes = (
            ("X", np.array([1.0, 0.0, 0.0], dtype=np.float64), QColor(255, 92, 92)),
            ("Y", np.array([0.0, 1.0, 0.0], dtype=np.float64), QColor(120, 245, 130)),
            ("Z", np.array([0.0, 0.0, 1.0], dtype=np.float64), QColor(96, 170, 255)),
        )
        for label, axis, color in axes:
            self._draw_world_arrow(painter, origin, origin + axis * length, color, f"+{label}", width=2)

    def _draw_status(self, painter: QPainter) -> None:
        point_count = 0 if self._pointcloud is None else int(self._pointcloud.points.shape[0])
        text = (
            i18n.t("SCENE_PREVIEW_STATUS").format(
                points=f"{point_count:,}",
                cameras=f"{len(self._cameras):,}",
                renderer=self.pointcloud_renderer_label(),
            )
            if point_count or self._cameras
            else i18n.t("SCENE_PREVIEW_STATUS_EMPTY")
        )
        painter.setPen(QPen(QColor(7, 10, 14), 3))
        painter.drawText(QPointF(14, 24), text)
        painter.setPen(QPen(QColor(224, 231, 241), 1))
        painter.drawText(QPointF(14, 24), text)

    def _scene_scale(self) -> float:
        visible_units = max(self.width(), self.height()) / max(self._pixels_per_unit, 1e-6)
        return max(visible_units, self._grid_step * 6.0)

    def _draw_world_line(
        self,
        painter: QPainter,
        a: np.ndarray,
        b: np.ndarray,
        color: QColor,
        width: int,
    ) -> None:
        xy, _depth = self._project(np.vstack([a, b]))
        if not np.all(np.isfinite(xy)):
            return
        painter.setPen(QPen(color, width))
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(QPointF(float(xy[0, 0]), float(xy[0, 1])), QPointF(float(xy[1, 0]), float(xy[1, 1])))

    def _draw_world_arrow(
        self,
        painter: QPainter,
        a: np.ndarray,
        b: np.ndarray,
        color: QColor,
        label: str | None,
        *,
        width: int,
    ) -> None:
        xy, _depth = self._project(np.vstack([a, b]))
        if not np.all(np.isfinite(xy)):
            return
        self._draw_screen_arrow(
            painter,
            QPointF(float(xy[0, 0]), float(xy[0, 1])),
            QPointF(float(xy[1, 0]), float(xy[1, 1])),
            color,
            label,
            width=width,
        )

    def _draw_screen_arrow(
        self,
        painter: QPainter,
        start: QPointF,
        end: QPointF,
        color: QColor,
        label: str | None,
        *,
        width: int,
    ) -> None:
        delta = end - start
        length = float((delta.x() ** 2 + delta.y() ** 2) ** 0.5)
        painter.setPen(QPen(QColor(7, 10, 14), width + 2))
        painter.drawLine(start, end)
        painter.setPen(QPen(color, width))
        painter.drawLine(start, end)
        if length >= 6.0:
            direction = np.array([float(delta.x()), float(delta.y())], dtype=np.float64) / length
            normal = np.array([-direction[1], direction[0]], dtype=np.float64)
            head_len = min(10.0, max(6.0, length * 0.22))
            head_w = head_len * 0.45
            end_np = np.array([float(end.x()), float(end.y())], dtype=np.float64)
            p1 = end_np - direction * head_len + normal * head_w
            p2 = end_np - direction * head_len - normal * head_w
            painter.setBrush(color)
            painter.setPen(QPen(QColor(7, 10, 14), width + 2))
            painter.drawPolygon(QPolygonF([end, QPointF(float(p1[0]), float(p1[1])), QPointF(float(p2[0]), float(p2[1]))]))
            painter.setPen(QPen(color, width))
            painter.drawPolygon(QPolygonF([end, QPointF(float(p1[0]), float(p1[1])), QPointF(float(p2[0]), float(p2[1]))]))
            label_pos = end + QPointF(float(direction[0]) * 6.0, float(direction[1]) * 6.0)
        else:
            label_pos = end + QPointF(6.0, -6.0)
        painter.setBrush(Qt.NoBrush)
        if label:
            painter.setPen(QPen(QColor(7, 10, 14), 3))
            painter.drawText(label_pos, label)
            painter.setPen(QPen(color, 1))
            painter.drawText(label_pos, label)

    def _draw_marker(self, painter: QPainter, point: np.ndarray, color: QColor, label: str, *, radius: int) -> None:
        xy, _depth = self._project(np.asarray(point, dtype=np.float64).reshape(1, 3))
        if not np.all(np.isfinite(xy)):
            return
        center = QPointF(float(xy[0, 0]), float(xy[0, 1]))
        painter.setPen(QPen(QColor(7, 10, 14), 3))
        painter.setBrush(color)
        painter.drawEllipse(center, radius, radius)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(color, 1))
        painter.drawText(center + QPointF(radius + 4, -radius - 4), label)


def _pointcloud_vertex_data(pointcloud: ScenePreviewPointCloud | None) -> np.ndarray:
    if pointcloud is None or len(pointcloud.points) == 0:
        return np.empty((0, 7), dtype=np.float32)
    points = np.asarray(pointcloud.points, dtype=np.float32).reshape(-1, 3)
    finite = np.all(np.isfinite(points), axis=1)
    if not np.all(finite):
        points = points[finite]
    data = np.empty((len(points), 7), dtype=np.float32)
    data[:, :3] = points
    if pointcloud.colors is None:
        data[:, 3:] = np.asarray(DEFAULT_POINT_COLOR_RGBA, dtype=np.float32)
    else:
        colors = np.asarray(pointcloud.colors, dtype=np.float32).reshape(-1, 3)
        if len(colors) != len(finite):
            raise ValueError("point cloud color count must match point count")
        colors = colors[finite] if not np.all(finite) else colors
        data[:, 3:6] = np.clip(colors, 0.0, 255.0) / 255.0
        data[:, 6] = POINT_COLOR_ALPHA
    return data


def _nice_step(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10.0**exponent
    scaled = value / base
    if scaled <= 1.0:
        factor = 1.0
    elif scaled <= 2.0:
        factor = 2.0
    elif scaled <= 5.0:
        factor = 5.0
    else:
        factor = 10.0
    return max(base * factor, 1e-6)


def _normalized(value: np.ndarray, *, fallback: tuple[float, float, float]) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12 or not np.isfinite(norm):
        return np.asarray(fallback, dtype=np.float64)
    return value / norm


def _right_handed_view_basis(
    *,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _normalized(np.asarray(forward, dtype=np.float64), fallback=(0.0, 0.0, 1.0))
    back = -forward
    right = np.asarray(right, dtype=np.float64) - back * float(np.asarray(right, dtype=np.float64) @ back)
    if float(np.linalg.norm(right)) <= 1e-12:
        right = np.cross(np.asarray(up, dtype=np.float64), back)
    right = _normalized(right, fallback=(1.0, 0.0, 0.0))
    up_from_basis = np.cross(back, right)
    up_from_basis = _normalized(up_from_basis, fallback=(0.0, 1.0, 0.0))
    if float(up_from_basis @ np.asarray(up, dtype=np.float64)) < 0.0:
        right = -right
        up_from_basis = np.cross(back, right)
        up_from_basis = _normalized(up_from_basis, fallback=(0.0, 1.0, 0.0))
    return right, up_from_basis, forward
