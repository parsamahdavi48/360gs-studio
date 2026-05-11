"""2D-painted 3D world debug view for AprilTag synthetic placement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPolygonF, QVector3D, QWheelEvent
from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram, QOpenGLVertexArrayObject
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from core.apriltag_geometry import tag_corners_sfm
from devtools.apriltag.cubemap_preview import CubemapFrameGroup, axis_preview_frustum_rays


GRID_X_AXIS_COLOR = QColor(255, 92, 92)
GRID_Z_AXIS_COLOR = QColor(90, 175, 245)
DEFAULT_POINT_COLOR_RGBA = (145.0 / 255.0, 155.0 / 255.0, 164.0 / 255.0, 130.0 / 255.0)
POINT_COLOR_ALPHA = 0.9

_GL_COLOR_BUFFER_BIT = 0x00004000
_GL_FLOAT = 0x1406
_GL_POINTS = 0x0000
_GL_BLEND = 0x0BE2
_GL_SRC_ALPHA = 0x0302
_GL_ONE_MINUS_SRC_ALPHA = 0x0303


@dataclass(frozen=True)
class PointCloudSample:
    points: np.ndarray
    colors: np.ndarray | None
    source_count: int


@dataclass(frozen=True)
class GridBounds:
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    core_x_min: float
    core_x_max: float
    core_z_min: float
    core_z_max: float
    step: float


_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def load_point_cloud_sample(path: Path, *, max_points: int | None = 80_000) -> PointCloudSample:
    """Load a deterministic sample, or all points when ``max_points`` is unset."""
    with path.open("rb") as f:
        fmt, vertex_count, properties = _read_ply_header(f)
        if vertex_count <= 0:
            return PointCloudSample(np.empty((0, 3), dtype=np.float32), None, 0)
        if fmt == "ascii":
            points, colors = _read_ascii_vertices(f, vertex_count, properties)
        elif fmt in {"binary_little_endian", "binary_big_endian"}:
            points, colors = _read_binary_vertices(f, vertex_count, properties, endian="<" if "little" in fmt else ">")
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")
    return _sample_points(points, colors, max_points=max_points, source_count=vertex_count)


def transform_point_cloud_sample(sample: PointCloudSample, matrix: np.ndarray | None) -> PointCloudSample:
    if matrix is None:
        return sample
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("point cloud transform must be a 4x4 matrix")
    points = np.asarray(sample.points, dtype=np.float64)
    transformed = points @ transform[:3, :3].T + transform[:3, 3]
    return PointCloudSample(transformed.astype(np.float32, copy=False), sample.colors, sample.source_count)


def _read_ply_header(f) -> tuple[str, int, list[tuple[str, str]]]:
    first = f.readline().decode("ascii", errors="replace").strip()
    if first != "ply":
        raise ValueError("Not a PLY file")

    fmt = ""
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    current_element = ""
    while True:
        raw = f.readline()
        if not raw:
            raise ValueError("PLY header ended unexpectedly")
        line = raw.decode("ascii", errors="replace").strip()
        if line == "end_header":
            break
        if not line or line.startswith("comment "):
            continue
        parts = line.split()
        if parts[:1] == ["format"] and len(parts) >= 2:
            fmt = parts[1]
        elif parts[:1] == ["element"] and len(parts) >= 3:
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[:1] == ["property"] and current_element == "vertex":
            if len(parts) >= 3 and parts[1] != "list":
                properties.append((parts[2], parts[1]))
    if not fmt:
        raise ValueError("PLY format is missing")
    if not properties:
        raise ValueError("PLY vertex properties are missing")
    return fmt, vertex_count, properties


def _property_indices(properties: list[tuple[str, str]]) -> tuple[int, int, int, tuple[int, int, int] | None]:
    names = [name.lower() for name, _type in properties]
    try:
        x_index, y_index, z_index = names.index("x"), names.index("y"), names.index("z")
    except ValueError as e:
        raise ValueError("PLY vertex must have x/y/z properties") from e

    color_sets = (
        ("red", "green", "blue"),
        ("r", "g", "b"),
        ("diffuse_red", "diffuse_green", "diffuse_blue"),
    )
    color_indices = None
    for keys in color_sets:
        if all(key in names for key in keys):
            color_indices = tuple(names.index(key) for key in keys)
            break
    return x_index, y_index, z_index, color_indices


def _read_ascii_vertices(
    f,
    vertex_count: int,
    properties: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray | None]:
    x_index, y_index, z_index, color_indices = _property_indices(properties)
    points = np.empty((vertex_count, 3), dtype=np.float32)
    colors = np.empty((vertex_count, 3), dtype=np.uint8) if color_indices is not None else None
    for index in range(vertex_count):
        line = f.readline().decode("ascii", errors="replace")
        values = line.split()
        points[index] = (float(values[x_index]), float(values[y_index]), float(values[z_index]))
        if colors is not None and color_indices is not None:
            colors[index] = tuple(max(0, min(255, int(float(values[i])))) for i in color_indices)
    return points, colors


def _read_binary_vertices(
    f,
    vertex_count: int,
    properties: list[tuple[str, str]],
    *,
    endian: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    dtype_fields = []
    for name, type_name in properties:
        dtype_code = _PLY_TYPES.get(type_name.lower())
        if dtype_code is None:
            raise ValueError(f"Unsupported PLY vertex property type: {type_name}")
        dtype_fields.append((name, np.dtype(dtype_code).newbyteorder(endian)))
    data = np.fromfile(f, dtype=np.dtype(dtype_fields), count=vertex_count)
    if len(data) != vertex_count:
        raise ValueError("PLY vertex data ended unexpectedly")
    _x_index, _y_index, _z_index, color_indices = _property_indices(properties)
    points = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32, copy=False)
    colors = None
    if color_indices is not None:
        names = [name for name, _type in properties]
        color_names = [names[i] for i in color_indices]
        colors = np.column_stack([data[name] for name in color_names])
        colors = np.clip(colors, 0, 255).astype(np.uint8, copy=False)
    return points, colors


def _sample_points(
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    max_points: int | None,
    source_count: int,
) -> PointCloudSample:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("point cloud must be an Nx3 array")
    finite = np.all(np.isfinite(points), axis=1)
    if not np.all(finite):
        points = points[finite]
        colors = colors[finite] if colors is not None else None
    limit = 0 if max_points is None else max(0, int(max_points))
    if limit and len(points) > limit:
        indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
        points = points[indices]
        colors = colors[indices] if colors is not None else None
    return PointCloudSample(points, colors, source_count)


def _pointcloud_vertex_data(pointcloud: PointCloudSample | None) -> np.ndarray:
    if pointcloud is None or len(pointcloud.points) == 0:
        return np.empty((0, 7), dtype=np.float32)
    points = np.asarray(pointcloud.points, dtype=np.float32).reshape(-1, 3)
    data = np.empty((len(points), 7), dtype=np.float32)
    data[:, :3] = points
    if pointcloud.colors is None:
        data[:, 3:] = np.asarray(DEFAULT_POINT_COLOR_RGBA, dtype=np.float32)
    else:
        colors = np.asarray(pointcloud.colors, dtype=np.float32).reshape(-1, 3)
        if len(colors) != len(points):
            raise ValueError("point cloud color count must match point count")
        data[:, 3:6] = np.clip(colors, 0.0, 255.0) / 255.0
        data[:, 6] = POINT_COLOR_ALPHA
    return data


class AprilTagWorldDebugView(QOpenGLWidget):
    """Orthographic world-space viewport synced with the AprilTag image preview."""

    camera_clicked = Signal(str)
    fixed_view_dragged = Signal(float, float)
    gpu_pointcloud_failed = Signal()

    _POINT_VERTEX_SHADER = """
        #ifdef GL_ES
        precision highp float;
        #endif

        attribute vec3 a_position;
        attribute vec4 a_color;

        uniform vec3 u_center;
        uniform vec3 u_right;
        uniform vec3 u_up;
        uniform vec3 u_forward;
        uniform vec2 u_viewport_size;
        uniform float u_pixels_per_unit;
        uniform float u_focal_px;
        uniform float u_projection_mode;
        uniform float u_point_size;
        uniform float u_screen_zoom;
        uniform vec2 u_screen_pan_px;

        varying vec4 v_color;

        void main() {
            vec3 rel = a_position - u_center;
            float view_x = dot(rel, u_right);
            float view_y = dot(rel, u_up);
            float depth = dot(rel, u_forward);
            float screen_x = 0.0;
            float screen_y = 0.0;
            if (u_projection_mode > 0.5) {
                if (depth <= 1.0e-8) {
                    gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
                    gl_PointSize = 0.0;
                    v_color = vec4(0.0, 0.0, 0.0, 0.0);
                    return;
                }
                screen_x = u_focal_px * (view_x / depth);
                screen_y = u_focal_px * (view_y / depth);
            } else {
                screen_x = view_x * u_pixels_per_unit;
                screen_y = view_y * u_pixels_per_unit;
            }
            screen_x = screen_x * u_screen_zoom + u_screen_pan_px.x;
            screen_y = screen_y * u_screen_zoom + u_screen_pan_px.y;
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(360, 320)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._pointcloud: PointCloudSample | None = None
        self._groups: tuple[CubemapFrameGroup, ...] = ()
        self._image_ray_groups: tuple[CubemapFrameGroup, ...] = ()
        self._face_ray_mode = "both"
        self._selected_group_name = ""
        self._tag_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._tag_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self._tag_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self._tag_size_m = 0.16
        self._true_scale = 0.25
        self._tag_validation_distance_sfm: float | None = None
        self._preview_yaw_deg = 0.0
        self._preview_pitch_deg = 0.0
        self._preview_roll_deg = 0.0
        self._preview_fov_deg = 90.0
        self._preview_to_world_matrix = np.eye(3, dtype=np.float64)
        self._grid_step = 2.0
        self._grid_extent = 20.0
        self._view_yaw_deg = 35.0
        self._view_pitch_deg = -28.0
        self._view_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._pixels_per_unit = 18.0
        self._fixed_view_basis: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._fixed_projection = "orthographic"
        self._fixed_perspective_fov_deg = 90.0
        self._fixed_screen_zoom_enabled = False
        self._fixed_screen_zoom = 1.0
        self._fixed_screen_pan = QPointF(0.0, 0.0)
        self._last_mouse: QPointF | None = None
        self._press_pos: QPointF | None = None
        self._press_button: Qt.MouseButton | None = None
        self._user_navigated = False
        self._fixed_navigation_enabled = True
        self._gpu_pointcloud_enabled = True
        self._gpu_failed = False
        self._gpu_initialized = False
        self._gpu_pointcloud_dirty = True
        self._gpu_point_count = 0
        self._gpu_functions = None
        self._gpu_program: QOpenGLShaderProgram | None = None
        self._gpu_vbo: QOpenGLBuffer | None = None
        self._gpu_vao: QOpenGLVertexArrayObject | None = None

    def sizeHint(self) -> QSize:
        return QSize(520, 420)

    def set_pointcloud(self, pointcloud: PointCloudSample | None) -> None:
        self._pointcloud = pointcloud
        self._gpu_pointcloud_dirty = True
        self._fit_scene_if_needed(force=not self._user_navigated)
        self.update()

    def set_gpu_pointcloud_enabled(self, enabled: bool) -> None:
        self._gpu_pointcloud_enabled = bool(enabled)
        self.update()

    def set_fixed_navigation_enabled(self, enabled: bool) -> None:
        self._fixed_navigation_enabled = bool(enabled)

    def set_fixed_screen_zoom_enabled(self, enabled: bool) -> None:
        self._fixed_screen_zoom_enabled = bool(enabled)
        if not self._fixed_screen_zoom_enabled:
            self._fixed_screen_zoom = 1.0
            self._fixed_screen_pan = QPointF(0.0, 0.0)
        self.update()

    def gpu_pointcloud_active(self) -> bool:
        return (
            self._gpu_pointcloud_enabled
            and not self._gpu_failed
            and self._gpu_initialized
            and self._gpu_point_count > 0
        )

    def pointcloud_renderer_label(self) -> str:
        if not self._gpu_pointcloud_enabled:
            return "CPU"
        if self._gpu_failed:
            return "CPU fallback"
        return "GPU"

    def _pointcloud_gl_vertex_data(self) -> np.ndarray:
        return _pointcloud_vertex_data(self._pointcloud)

    def set_groups(self, groups: tuple[CubemapFrameGroup, ...]) -> None:
        self._groups = groups
        if self._selected_group_name and not any(group.name == self._selected_group_name for group in groups):
            self._selected_group_name = ""
        self._fit_scene_if_needed(force=not self._user_navigated)
        self.update()

    def set_image_ray_groups(self, groups: tuple[CubemapFrameGroup, ...]) -> None:
        self._image_ray_groups = groups
        self.update()

    def set_face_ray_mode(self, mode: str) -> None:
        self._face_ray_mode = mode if mode in {"world", "image", "both"} else "both"
        self.update()

    def set_selected_group(self, group_name: str) -> None:
        self._selected_group_name = group_name
        self.update()

    def set_preview_params(self, *, yaw_deg: float, pitch_deg: float, fov_deg: float, roll_deg: float = 0.0) -> None:
        self._preview_yaw_deg = float(yaw_deg)
        self._preview_pitch_deg = float(pitch_deg)
        self._preview_roll_deg = float(roll_deg)
        self._preview_fov_deg = float(fov_deg)
        self.update()

    def set_preview_to_world_matrix(self, matrix: np.ndarray | None) -> None:
        if matrix is None:
            self._preview_to_world_matrix = np.eye(3, dtype=np.float64)
        else:
            value = np.asarray(matrix, dtype=np.float64)
            if value.shape == (4, 4):
                value = value[:3, :3]
            if value.shape != (3, 3):
                raise ValueError("preview-to-world matrix must be 3x3 or 4x4")
            self._preview_to_world_matrix = value.copy()
        self.update()

    def set_grid(self, *, step: float, extent: float) -> None:
        self._grid_step = max(0.1, float(step))
        self._grid_extent = max(self._grid_step, float(extent))
        self.update()

    def set_fixed_view(
        self,
        *,
        center: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        forward: np.ndarray,
        pixels_per_unit: float,
    ) -> None:
        self._view_center = np.asarray(center, dtype=np.float64).reshape(3)
        self._fixed_view_basis = _right_handed_view_basis(
            right=np.asarray(right, dtype=np.float64).reshape(3),
            up=np.asarray(up, dtype=np.float64).reshape(3),
            forward=np.asarray(forward, dtype=np.float64).reshape(3),
        )
        self._fixed_projection = "orthographic"
        self._pixels_per_unit = max(0.01, min(10000.0, float(pixels_per_unit)))
        self.update()

    def set_fixed_perspective_view(
        self,
        *,
        camera_position: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        forward: np.ndarray,
        fov_deg: float,
    ) -> None:
        self._view_center = np.asarray(camera_position, dtype=np.float64).reshape(3)
        self._fixed_view_basis = _right_handed_view_basis(
            right=np.asarray(right, dtype=np.float64).reshape(3),
            up=np.asarray(up, dtype=np.float64).reshape(3),
            forward=np.asarray(forward, dtype=np.float64).reshape(3),
        )
        self._fixed_projection = "perspective"
        self._fixed_perspective_fov_deg = max(1.0, min(179.0, float(fov_deg)))
        self.update()

    def clear_fixed_view(self) -> None:
        self._fixed_view_basis = None
        self._fixed_projection = "orthographic"
        self._fixed_screen_zoom = 1.0
        self._fixed_screen_pan = QPointF(0.0, 0.0)
        self.update()

    def set_tag(
        self,
        *,
        center: np.ndarray,
        normal: np.ndarray,
        up: np.ndarray,
        tag_size_m: float,
        true_scale: float,
    ) -> None:
        self._tag_center = np.asarray(center, dtype=np.float64).reshape(3)
        self._tag_normal = _normalized(np.asarray(normal, dtype=np.float64).reshape(3), fallback=(0.0, 0.0, -1.0))
        self._tag_up = _normalized(np.asarray(up, dtype=np.float64).reshape(3), fallback=(0.0, 1.0, 0.0))
        self._tag_size_m = max(1e-9, float(tag_size_m))
        self._true_scale = max(1e-12, float(true_scale))
        self.update()

    def set_tag_validation_distance(self, distance_sfm: float | None) -> None:
        if distance_sfm is None:
            self._tag_validation_distance_sfm = None
        else:
            value = float(distance_sfm)
            self._tag_validation_distance_sfm = value if np.isfinite(value) and value > 0.0 else None
        self.update()

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
                raise RuntimeError("Failed to create OpenGL point cloud vertex buffer")
            self._gpu_vao = QOpenGLVertexArrayObject(self)
            if not self._gpu_vao.create():
                raise RuntimeError("Failed to create OpenGL point cloud vertex array")

            self._gpu_vao.bind()
            self._gpu_program.bind()
            self._gpu_vbo.bind()
            stride = 7 * 4
            position_location = self._gpu_program.attributeLocation(b"a_position")
            color_location = self._gpu_program.attributeLocation(b"a_color")
            if position_location < 0 or color_location < 0:
                raise RuntimeError("OpenGL point cloud shader attributes are missing")
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
        self._clamp_fixed_screen_pan()

    def paintGL(self) -> None:  # noqa: N802 - Qt API
        gpu_drawn = self._draw_pointcloud_gpu()
        painter = QPainter(self)
        if not gpu_drawn:
            painter.fillRect(self.rect(), QColor(13, 17, 23))
        painter.setRenderHint(QPainter.Antialiasing, True)
        self._draw_grid(painter)
        if not gpu_drawn:
            self._draw_pointcloud(painter)
        self._draw_cameras(painter)
        self._draw_tag(painter)
        self._draw_world_axes(painter)
        self._draw_orientation_gizmo(painter)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse = event.position()
        self._press_pos = event.position()
        self._press_button = event.button()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_mouse is None:
            self._last_mouse = event.position()
            return
        delta = event.position() - self._last_mouse
        self._last_mouse = event.position()
        buttons = event.buttons()
        if buttons & Qt.LeftButton:
            self._user_navigated = True
            if self._fixed_view_basis is not None:
                self.fixed_view_dragged.emit(float(delta.x()), float(delta.y()))
            else:
                self._view_yaw_deg -= float(delta.x()) * 0.35
                self._view_pitch_deg = max(-85.0, min(85.0, self._view_pitch_deg + float(delta.y()) * 0.25))
                self.update()
        elif buttons & (Qt.RightButton | Qt.MiddleButton):
            if self._fixed_view_basis is not None and not self._fixed_navigation_enabled:
                event.accept()
                return
            self._user_navigated = True
            right, up, _forward = self._view_basis()
            self._view_center -= right * (float(delta.x()) / max(self._pixels_per_unit, 1e-6))
            self._view_center += up * (float(delta.y()) / max(self._pixels_per_unit, 1e-6))
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._press_button == Qt.LeftButton and self._press_pos is not None:
            delta = event.position() - self._press_pos
            if abs(float(delta.x())) <= 4.0 and abs(float(delta.y())) <= 4.0:
                group_name = self._camera_name_at_screen_pos(event.position())
                if group_name:
                    self.camera_clicked.emit(group_name)
        self._last_mouse = None
        self._press_pos = None
        self._press_button = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._fixed_view_basis is not None and not self._fixed_navigation_enabled:
            if self._fixed_screen_zoom_enabled:
                delta = event.angleDelta().y()
                if delta != 0:
                    self._apply_fixed_screen_wheel_zoom(event.position(), delta)
            event.accept()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.2 if delta > 0 else 1.0 / 1.2
        self._pixels_per_unit = max(0.01, min(10000.0, self._pixels_per_unit * factor))
        self._user_navigated = True
        self.update()

    def _fixed_screen_zoom_active(self) -> bool:
        return bool(self._fixed_view_basis is not None and self._fixed_screen_zoom_enabled)

    def _screen_center(self) -> QPointF:
        return QPointF(self.width() * 0.5, self.height() * 0.5)

    def _apply_fixed_screen_wheel_zoom(self, cursor: QPointF, delta: int) -> None:
        old_zoom = max(1e-12, float(self._fixed_screen_zoom))
        center = self._screen_center()
        anchored = (cursor - center - self._fixed_screen_pan) / old_zoom
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self._fixed_screen_zoom = max(1.0, min(12.0, old_zoom * factor))
        self._fixed_screen_pan = cursor - center - anchored * self._fixed_screen_zoom
        self._clamp_fixed_screen_pan()
        self.update()

    def _clamp_fixed_screen_pan(self) -> None:
        if self._fixed_screen_zoom <= 1.0:
            self._fixed_screen_zoom = 1.0
            self._fixed_screen_pan = QPointF(0.0, 0.0)
            return
        max_x = max(0.0, self.width() * (self._fixed_screen_zoom - 1.0) * 0.5)
        max_y = max(0.0, self.height() * (self._fixed_screen_zoom - 1.0) * 0.5)
        self._fixed_screen_pan = QPointF(
            max(-max_x, min(max_x, self._fixed_screen_pan.x())),
            max(-max_y, min(max_y, self._fixed_screen_pan.y())),
        )

    def _apply_fixed_screen_transform(self, xy: np.ndarray) -> np.ndarray:
        if not self._fixed_screen_zoom_active():
            return xy
        transformed = np.asarray(xy, dtype=np.float64).copy()
        center_x = self.width() * 0.5
        center_y = self.height() * 0.5
        transformed[:, 0] = (transformed[:, 0] - center_x) * self._fixed_screen_zoom + center_x + self._fixed_screen_pan.x()
        transformed[:, 1] = (transformed[:, 1] - center_y) * self._fixed_screen_zoom + center_y + self._fixed_screen_pan.y()
        return transformed

    def _base_screen_point(self, sx: float, sy: float) -> tuple[float, float]:
        if not self._fixed_screen_zoom_active():
            return float(sx), float(sy)
        center_x = self.width() * 0.5
        center_y = self.height() * 0.5
        zoom = max(1e-12, float(self._fixed_screen_zoom))
        base_x = ((float(sx) - center_x - self._fixed_screen_pan.x()) / zoom) + center_x
        base_y = ((float(sy) - center_y - self._fixed_screen_pan.y()) / zoom) + center_y
        return base_x, base_y

    def _fit_scene_if_needed(self, *, force: bool = False) -> None:
        if not force:
            return
        points: list[np.ndarray] = []
        if self._pointcloud is not None and len(self._pointcloud.points):
            pc = self._pointcloud.points
            if len(pc) > 5000:
                pc = pc[np.linspace(0, len(pc) - 1, 5000, dtype=np.int64)]
            pc = pc.astype(np.float64, copy=False)
            if len(pc) >= 20:
                lower = np.percentile(pc, 1.0, axis=0)
                upper = np.percentile(pc, 99.0, axis=0)
                inside = np.all((pc >= lower) & (pc <= upper), axis=1)
                if np.count_nonzero(inside) >= 10:
                    pc = pc[inside]
            points.append(pc)
        camera_positions = [group.camera_position_sfm for group in self._groups]
        if camera_positions:
            points.append(np.asarray(camera_positions, dtype=np.float64))
        if not points:
            self._view_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            self._pixels_per_unit = 18.0
            return
        all_points = np.vstack(points)
        minimum = np.min(all_points, axis=0)
        maximum = np.max(all_points, axis=0)
        self._view_center = (minimum + maximum) * 0.5
        span = float(np.max(maximum - minimum))
        if span > 1e-9:
            viewport = max(200, min(max(self.width(), 1), max(self.height(), 1)))
            self._pixels_per_unit = max(0.01, min(5000.0, viewport / (span * 1.4)))

    def _view_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._fixed_view_basis is not None:
            return self._fixed_view_basis
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
        if self._fixed_projection == "perspective":
            size = float(max(1, min(max(self.width(), 1), max(self.height(), 1))))
            focal = 0.5 * size / np.tan(np.deg2rad(self._fixed_perspective_fov_deg) * 0.5)
            view_x = rel @ right
            view_y = rel @ up
            xy = np.full((len(points), 2), np.nan, dtype=np.float64)
            valid = np.isfinite(depth) & (depth > 1e-8)
            with np.errstate(divide="ignore", invalid="ignore"):
                xy[valid, 0] = self.width() * 0.5 + focal * (view_x[valid] / depth[valid])
                xy[valid, 1] = self.height() * 0.5 - focal * (view_y[valid] / depth[valid])
            return self._apply_fixed_screen_transform(xy), depth
        x = rel @ right * self._pixels_per_unit + self.width() * 0.5
        y = -(rel @ up) * self._pixels_per_unit + self.height() * 0.5
        return self._apply_fixed_screen_transform(np.column_stack([x, y])), depth

    def _screen_ground_points(self, *, margin_px: float = 0.0) -> np.ndarray:
        width = float(max(1, self.width()))
        height = float(max(1, self.height()))
        margin = max(0.0, float(margin_px))
        screen_points = (
            (-margin, -margin),
            (width + margin, -margin),
            (width + margin, height + margin),
            (-margin, height + margin),
        )
        right, up, forward = self._view_basis()
        points: list[np.ndarray] = []
        if self._fixed_projection == "perspective":
            focal = 0.5 * float(max(1, min(self.width(), self.height()))) / np.tan(
                np.deg2rad(self._fixed_perspective_fov_deg) * 0.5
            )
            for sx, sy in screen_points:
                sx, sy = self._base_screen_point(float(sx), float(sy))
                ray = (
                    forward
                    + right * ((float(sx) - width * 0.5) / max(focal, 1e-12))
                    + up * (-(float(sy) - height * 0.5) / max(focal, 1e-12))
                )
                if abs(float(ray[1])) <= 1e-8:
                    continue
                t = -float(self._view_center[1]) / float(ray[1])
                if t < 0.0:
                    continue
                point = self._view_center + ray * t
                if np.all(np.isfinite(point)):
                    points.append(point)
        else:
            pixels_per_unit = max(self._pixels_per_unit, 1e-12)
            for sx, sy in screen_points:
                sx, sy = self._base_screen_point(float(sx), float(sy))
                offset_x = (float(sx) - width * 0.5) / pixels_per_unit
                offset_y = -(float(sy) - height * 0.5) / pixels_per_unit
                ray_origin = self._view_center + right * offset_x + up * offset_y
                if abs(float(forward[1])) <= 1e-8:
                    continue
                t = -float(ray_origin[1]) / float(forward[1])
                point = ray_origin + forward * t
                if np.all(np.isfinite(point)):
                    points.append(point)
        if not points:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(points, dtype=np.float64)

    def _fallback_grid_bounds(self) -> tuple[float, float, float, float]:
        visible_units = max(self.width(), self.height()) / max(self._pixels_per_unit, 1e-6)
        half = max(self._grid_extent, visible_units * 0.75)
        return (
            float(self._view_center[0] - half),
            float(self._view_center[0] + half),
            float(self._view_center[2] - half),
            float(self._view_center[2] + half),
        )

    def _grid_bounds(self) -> GridBounds:
        visible_units = max(self.width(), self.height()) / max(self._pixels_per_unit, 1e-6)
        fallback = self._fallback_grid_bounds()
        core_points = self._screen_ground_points(margin_px=0.0)
        falloff_margin_px = max(80.0, min(max(self.width(), self.height()) * 0.18, 180.0))
        expanded_points = self._screen_ground_points(margin_px=falloff_margin_px)
        if len(core_points) >= 3 and len(expanded_points) >= 3:
            core_x_min = float(np.min(core_points[:, 0]))
            core_x_max = float(np.max(core_points[:, 0]))
            core_z_min = float(np.min(core_points[:, 2]))
            core_z_max = float(np.max(core_points[:, 2]))
            x_min = float(np.min(expanded_points[:, 0]))
            x_max = float(np.max(expanded_points[:, 0]))
            z_min = float(np.min(expanded_points[:, 2]))
            z_max = float(np.max(expanded_points[:, 2]))
        else:
            x_min, x_max, z_min, z_max = fallback
            core_x_min, core_x_max, core_z_min, core_z_max = fallback

        center_x = float(self._view_center[0])
        center_z = float(self._view_center[2])
        minimum_half = max(self._grid_extent, visible_units * 0.75)
        x_min = min(x_min, center_x - minimum_half)
        x_max = max(x_max, center_x + minimum_half)
        z_min = min(z_min, center_z - minimum_half)
        z_max = max(z_max, center_z + minimum_half)
        core_x_min = min(core_x_min, center_x - minimum_half)
        core_x_max = max(core_x_max, center_x + minimum_half)
        core_z_min = min(core_z_min, center_z - minimum_half)
        core_z_max = max(core_z_max, center_z + minimum_half)

        max_span = max(self._grid_extent * 24.0, visible_units * 16.0, self._grid_step * 180.0)
        x_min, x_max = _clamp_interval(x_min, x_max, center_x, max_span)
        z_min, z_max = _clamp_interval(z_min, z_max, center_z, max_span)
        core_x_min = max(core_x_min, x_min)
        core_x_max = min(core_x_max, x_max)
        core_z_min = max(core_z_min, z_min)
        core_z_max = min(core_z_max, z_max)

        step = self._grid_step
        max_lines = 180
        while ((x_max - x_min) / step + (z_max - z_min) / step) > max_lines:
            step *= 2.0
        return GridBounds(
            x_min=x_min,
            x_max=x_max,
            z_min=z_min,
            z_max=z_max,
            core_x_min=core_x_min,
            core_x_max=core_x_max,
            core_z_min=core_z_min,
            core_z_max=core_z_max,
            step=step,
        )

    def _draw_grid(self, painter: QPainter) -> None:
        bounds = self._grid_bounds()
        step = bounds.step
        x_min = np.floor(bounds.x_min / step) * step
        x_max = np.ceil(bounds.x_max / step) * step
        z_min = np.floor(bounds.z_min / step) * step
        z_max = np.ceil(bounds.z_max / step) * step

        for x in np.arange(x_min, x_max + step * 0.5, step):
            if abs(x) <= step * 0.25:
                color = GRID_Z_AXIS_COLOR
                width = 2
            else:
                color = self._grid_falloff_color(
                    55,
                    63,
                    72,
                    value=float(x),
                    core_min=bounds.core_x_min,
                    core_max=bounds.core_x_max,
                    outer_min=bounds.x_min,
                    outer_max=bounds.x_max,
                )
                width = 1
            self._draw_world_line(painter, np.array([x, 0.0, z_min]), np.array([x, 0.0, z_max]), color, width)
        for z in np.arange(z_min, z_max + step * 0.5, step):
            if abs(z) <= step * 0.25:
                color = GRID_X_AXIS_COLOR
                width = 2
            else:
                color = self._grid_falloff_color(
                    55,
                    63,
                    72,
                    value=float(z),
                    core_min=bounds.core_z_min,
                    core_max=bounds.core_z_max,
                    outer_min=bounds.z_min,
                    outer_max=bounds.z_max,
                )
                width = 1
            self._draw_world_line(painter, np.array([x_min, 0.0, z]), np.array([x_max, 0.0, z]), color, width)
        self._draw_marker(painter, np.array([0.0, 0.0, 0.0]), QColor(255, 80, 255), "O", radius=5)

    @staticmethod
    def _grid_falloff_color(
        red: int,
        green: int,
        blue: int,
        *,
        value: float,
        core_min: float,
        core_max: float,
        outer_min: float,
        outer_max: float,
    ) -> QColor:
        if core_min <= value <= core_max:
            alpha = 115
        elif value < core_min:
            span = max(core_min - outer_min, 1e-12)
            alpha = 28 + 87 * max(0.0, min(1.0, (value - outer_min) / span))
        else:
            span = max(outer_max - core_max, 1e-12)
            alpha = 28 + 87 * max(0.0, min(1.0, (outer_max - value) / span))
        return QColor(int(red), int(green), int(blue), int(alpha))

    def _draw_world_axes(self, painter: QPainter) -> None:
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        length = max(self._grid_step * 2.0, self._grid_extent * 0.16)
        for label, axis, color in self._axis_definitions():
            end = origin + axis * length
            self._draw_world_arrow(painter, origin, end, color, f"+{label}", width=2)

    def _draw_orientation_gizmo(self, painter: QPainter) -> None:
        origin = QPointF(52.0, float(self.height()) - 48.0)
        length = 34.0
        endpoints = self._orientation_axis_points(origin, length)
        _right, _up, forward = self._view_basis()
        axes = sorted(
            self._axis_definitions(),
            key=lambda item: float(item[1] @ forward),
        )
        painter.setPen(QPen(QColor(255, 255, 255, 70), 1))
        painter.setBrush(QColor(7, 10, 14, 130))
        painter.drawEllipse(origin, 4.0, 4.0)
        for label, _axis, color in axes:
            self._draw_screen_arrow(painter, origin, endpoints[label], color, f"+{label}", width=2)

    def _orientation_axis_points(self, origin: QPointF, length: float) -> dict[str, QPointF]:
        right, up, _forward = self._view_basis()
        points: dict[str, QPointF] = {}
        for label, axis, _color in self._axis_definitions():
            delta = _screen_axis_delta(axis=axis, right=right, up=up, length=length)
            points[label] = QPointF(float(origin.x()) + float(delta[0]), float(origin.y()) + float(delta[1]))
        return points

    def _screen_basis_determinant(self) -> float:
        right, up, forward = self._view_basis()
        return float(np.linalg.det(np.column_stack([right, up, -forward])))

    @staticmethod
    def _axis_definitions() -> tuple[tuple[str, np.ndarray, QColor], ...]:
        return (
            ("X", np.array([1.0, 0.0, 0.0], dtype=np.float64), QColor(255, 92, 92)),
            ("Y", np.array([0.0, 1.0, 0.0], dtype=np.float64), QColor(120, 245, 130)),
            ("Z", np.array([0.0, 0.0, 1.0], dtype=np.float64), QColor(96, 170, 255)),
        )

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
            not self._gpu_pointcloud_enabled
            or self._gpu_failed
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
        self._gpu_functions.glViewport(0, 0, full_w, full_h)
        self._gpu_functions.glClearColor(13.0 / 255.0, 17.0 / 255.0, 23.0 / 255.0, 1.0)
        self._gpu_functions.glClear(_GL_COLOR_BUFFER_BIT)
        self._gpu_functions.glEnable(_GL_BLEND)
        self._gpu_functions.glBlendFunc(_GL_SRC_ALPHA, _GL_ONE_MINUS_SRC_ALPHA)

        right, up, forward = self._view_basis()
        if self._fixed_projection == "perspective":
            size = float(max(1, min(full_w, full_h)))
            focal = 0.5 * size / np.tan(np.deg2rad(self._fixed_perspective_fov_deg) * 0.5)
            projection_mode = 1.0
            pixels_per_unit = 1.0
        else:
            focal = 1.0
            projection_mode = 0.0
            pixels_per_unit = float(self._pixels_per_unit) * dpr

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
        self._gpu_program.setUniformValue(
            self._gpu_program.uniformLocation(b"u_forward"),
            QVector3D(float(forward[0]), float(forward[1]), float(forward[2])),
        )
        self._gpu_program.setUniformValue(self._gpu_program.uniformLocation(b"u_viewport_size"), float(full_w), float(full_h))
        self._gpu_program.setUniformValue1f(self._gpu_program.uniformLocation(b"u_pixels_per_unit"), float(pixels_per_unit))
        self._gpu_program.setUniformValue1f(self._gpu_program.uniformLocation(b"u_focal_px"), float(focal))
        self._gpu_program.setUniformValue1f(self._gpu_program.uniformLocation(b"u_projection_mode"), float(projection_mode))
        screen_zoom = float(self._fixed_screen_zoom) if self._fixed_screen_zoom_active() else 1.0
        self._gpu_program.setUniformValue1f(self._gpu_program.uniformLocation(b"u_screen_zoom"), screen_zoom)
        self._gpu_program.setUniformValue(
            self._gpu_program.uniformLocation(b"u_screen_pan_px"),
            float(self._fixed_screen_pan.x() * dpr) if self._fixed_screen_zoom_active() else 0.0,
            float(-self._fixed_screen_pan.y() * dpr) if self._fixed_screen_zoom_active() else 0.0,
        )
        self._gpu_program.setUniformValue1f(self._gpu_program.uniformLocation(b"u_point_size"), max(1.0, 1.35 * dpr))
        self._gpu_vao.bind()
        self._gpu_functions.glDrawArrays(_GL_POINTS, 0, self._gpu_point_count)
        self._gpu_vao.release()
        self._gpu_program.release()
        return True

    def _draw_pointcloud(self, painter: QPainter) -> None:
        if self._pointcloud is None or len(self._pointcloud.points) == 0:
            return
        points = self._pointcloud.points
        colors = self._pointcloud.colors
        if len(points) > 60_000:
            indices = np.linspace(0, len(points) - 1, 60_000, dtype=np.int64)
            points = points[indices]
            colors = colors[indices] if colors is not None else None
        xy, depth = self._project(points)
        margin = 8.0
        mask = (
            np.isfinite(xy).all(axis=1)
            & (xy[:, 0] >= -margin)
            & (xy[:, 1] >= -margin)
            & (xy[:, 0] <= self.width() + margin)
            & (xy[:, 1] <= self.height() + margin)
        )
        if not np.any(mask):
            return
        visible_xy = xy[mask]
        visible_depth = depth[mask]
        visible_colors = colors[mask] if colors is not None else None
        if len(visible_xy) > 60_000:
            order = np.argsort(visible_depth)
            selected = order[np.linspace(0, len(order) - 1, 60_000, dtype=np.int64)]
            visible_xy = visible_xy[selected]
            visible_colors = visible_colors[selected] if visible_colors is not None else None
        if visible_colors is None:
            painter.setPen(QPen(QColor(145, 155, 164, 130), 1))
            painter.drawPoints(QPolygonF([QPointF(float(x), float(y)) for x, y in visible_xy]))
            return
        quantized = (np.clip(visible_colors, 0, 255).astype(np.uint8) // 32) * 32 + 16
        keys = (
            quantized[:, 0].astype(np.uint32) << 16
            | quantized[:, 1].astype(np.uint32) << 8
            | quantized[:, 2].astype(np.uint32)
        )
        for key in np.unique(keys):
            group_mask = keys == key
            color = quantized[group_mask][0]
            painter.setPen(QPen(QColor(int(color[0]), int(color[1]), int(color[2]), 190), 1))
            painter.drawPoints(QPolygonF([QPointF(float(x), float(y)) for x, y in visible_xy[group_mask]]))

    def _draw_cameras(self, painter: QPainter) -> None:
        if not self._groups:
            return
        positions = np.asarray([group.camera_position_sfm for group in self._groups], dtype=np.float64)
        xy, _depth = self._project(positions)
        for group, point in zip(self._groups, xy, strict=True):
            if not np.all(np.isfinite(point)):
                continue
            selected = group.name == self._selected_group_name
            color = QColor(255, 218, 92) if selected else QColor(95, 160, 255)
            radius = 5 if selected else 3
            painter.setPen(QPen(QColor(7, 10, 14), 3))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(float(point[0]), float(point[1])), radius, radius)
            painter.setBrush(Qt.NoBrush)
        selected = self._selected_group()
        if selected is not None:
            image_ray_group = self._selected_image_ray_group()
            if self._face_ray_mode == "image" and image_ray_group is not None:
                self._draw_selected_image_rays(painter, image_ray_group, primary=True)
            else:
                self._draw_selected_face_rays(painter, selected)
            if self._face_ray_mode == "both" and image_ray_group is not None:
                self._draw_selected_image_rays(painter, image_ray_group)
            if self._fixed_projection != "perspective":
                self._draw_selected_frustum(painter, selected)
                self._draw_world_line(painter, selected.camera_position_sfm, self._tag_center, QColor(120, 230, 180), 1, dashed=True)

    def _camera_name_at_screen_pos(self, pos: QPointF, *, max_distance_px: float = 12.0) -> str | None:
        if not self._groups:
            return None
        positions = np.asarray([group.camera_position_sfm for group in self._groups], dtype=np.float64)
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
        return self._groups[index].name

    def _draw_selected_frustum(self, painter: QPainter, group: CubemapFrameGroup) -> None:
        position = group.camera_position_sfm
        forward, corner_rays = self._preview_frustum_rays_in_world()
        scene_scale = max(self._grid_step * 2.0, self._grid_extent * 0.12, self._tag_size_m / self._true_scale * 2.0)
        distance = max(0.5, scene_scale)
        plane_scales = distance / np.maximum(corner_rays @ forward, 1e-6)
        center = position + forward * distance
        corners = position + corner_rays * plane_scales[:, None]
        color = QColor(255, 218, 92)
        for corner in corners:
            self._draw_world_line(painter, position, corner, color, 2)
        for a, b in zip(corners, np.roll(corners, -1, axis=0), strict=True):
            self._draw_world_line(painter, a, b, color, 2)
        self._draw_world_line(painter, position, center + forward * distance, QColor(255, 245, 150), 2)

    def _draw_selected_face_rays(self, painter: QPainter, group: CubemapFrameGroup) -> None:
        for label, start, end, color in self._selected_face_ray_segments(group):
            self._draw_world_arrow(painter, start, end, color, label, width=2)

    def _draw_selected_image_rays(self, painter: QPainter, group: CubemapFrameGroup, *, primary: bool = False) -> None:
        for label, start, end, color in self._selected_image_ray_segments(group):
            if primary:
                solid = QColor(color)
                solid.setAlpha(255)
                self._draw_world_arrow(painter, start, end, solid, f"img {label}", width=2)
            else:
                self._draw_world_line(painter, start, end, color, 2, dashed=True)
                self._draw_marker(painter, end, color, f"img {label}", radius=3)

    def _selected_face_ray_segments(
        self,
        group: CubemapFrameGroup,
    ) -> tuple[tuple[str, np.ndarray, np.ndarray, QColor], ...]:
        position = np.asarray(group.camera_position_sfm, dtype=np.float64)
        scene_scale = max(self._grid_step * 2.0, self._grid_extent * 0.12, self._tag_size_m / self._true_scale * 2.0)
        length = max(0.5, scene_scale)
        start_distance = max(length * 0.06, 1e-4)
        colors = {
            "pz": QColor(255, 245, 150),
            "nz": QColor(210, 130, 255),
            "px": QColor(255, 100, 100),
            "nx": QColor(255, 150, 190),
            "top": QColor(120, 255, 130),
            "bottom": QColor(90, 220, 255),
            "py": QColor(120, 255, 130),
            "ny": QColor(90, 220, 255),
        }
        ordered_faces = ("pz", "px", "nx", "nz", "top", "bottom", "py", "ny")
        segments: list[tuple[str, np.ndarray, np.ndarray, QColor]] = []
        for face in ordered_faces:
            frame = group.frames_by_face.get(face)
            if frame is None:
                continue
            direction = np.array([0.0, 0.0, 1.0], dtype=np.float64) @ frame.camera_to_world_rotation.T
            direction = _normalized(direction, fallback=(0.0, 0.0, 1.0))
            segments.append(
                (
                    face,
                    position + direction * start_distance,
                    position + direction * length,
                    colors[face],
                )
            )
        return tuple(segments)

    def _selected_image_ray_segments(
        self,
        group: CubemapFrameGroup,
    ) -> tuple[tuple[str, np.ndarray, np.ndarray, QColor], ...]:
        position = np.asarray(group.camera_position_sfm, dtype=np.float64)
        scene_scale = max(self._grid_step * 2.0, self._grid_extent * 0.12, self._tag_size_m / self._true_scale * 2.0)
        length = max(0.5, scene_scale) * 0.78
        start_distance = max(length * 0.08, 1e-4)
        colors = {
            "pz": QColor(255, 245, 150, 150),
            "nz": QColor(210, 130, 255, 150),
            "px": QColor(255, 100, 100, 150),
            "nx": QColor(255, 150, 190, 150),
            "top": QColor(120, 255, 130, 150),
            "bottom": QColor(90, 220, 255, 150),
            "py": QColor(120, 255, 130, 150),
            "ny": QColor(90, 220, 255, 150),
        }
        ordered_faces = ("pz", "px", "nx", "nz", "top", "bottom", "py", "ny")
        segments: list[tuple[str, np.ndarray, np.ndarray, QColor]] = []
        for face in ordered_faces:
            frame = group.frames_by_face.get(face)
            if frame is None:
                continue
            direction = np.array([0.0, 0.0, 1.0], dtype=np.float64) @ frame.camera_to_world_rotation.T
            direction = _normalized(direction, fallback=(0.0, 0.0, 1.0))
            segments.append(
                (
                    face,
                    position + direction * start_distance,
                    position + direction * length,
                    colors[face],
                )
            )
        return tuple(segments)

    def _preview_frustum_rays_in_world(self) -> tuple[np.ndarray, np.ndarray]:
        forward, corner_rays = axis_preview_frustum_rays(
            output_size=129,
            yaw_deg=self._preview_yaw_deg,
            pitch_deg=self._preview_pitch_deg,
            roll_deg=self._preview_roll_deg,
            fov_deg=self._preview_fov_deg,
        )
        matrix = self._preview_to_world_matrix
        forward = _normalized(forward @ matrix, fallback=(0.0, 0.0, 1.0))
        corner_rays = corner_rays @ matrix
        norms = np.linalg.norm(corner_rays, axis=1, keepdims=True)
        corner_rays = corner_rays / np.maximum(norms, 1e-12)
        return forward, corner_rays

    def _draw_tag(self, painter: QPainter) -> None:
        try:
            corners = tag_corners_sfm(
                self._tag_center,
                self._tag_normal,
                self._tag_up,
                self._tag_size_m,
                self._true_scale,
            )
        except Exception:
            return
        self._draw_tag_validation_distance(painter)
        xy, _depth = self._project(corners)
        if not np.all(np.isfinite(xy)):
            return
        polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in xy])
        painter.setPen(QPen(QColor(0, 255, 180), 3))
        painter.setBrush(QColor(0, 255, 180, 35))
        painter.drawPolygon(polygon)
        painter.setBrush(Qt.NoBrush)
        self._draw_marker(painter, self._tag_center, QColor(0, 255, 180), "tag", radius=5)

        foot = np.array([self._tag_center[0], 0.0, self._tag_center[2]], dtype=np.float64)
        self._draw_world_line(painter, self._tag_center, foot, QColor(0, 255, 255), 2, dashed=True)
        self._draw_marker(painter, foot, QColor(0, 255, 255), "XZ", radius=5)

        for label, start, end, color, dashed in self._tag_orientation_segments():
            if dashed:
                self._draw_world_line(painter, start, end, color, 2, dashed=True)
                self._draw_marker(painter, end, color, label, radius=3)
            else:
                self._draw_world_arrow(painter, start, end, color, label, width=3 if label == "front" else 2)

    def _tag_orientation_segments(self) -> tuple[tuple[str, np.ndarray, np.ndarray, QColor, bool], ...]:
        side_sfm = self._tag_size_m / self._true_scale
        arrow_len = max(side_sfm, self._grid_step) * 0.85
        back_len = arrow_len * 0.45
        normal = _normalized(self._tag_normal, fallback=(0.0, 0.0, -1.0))
        up = self._tag_up - normal * float(self._tag_up @ normal)
        up = _normalized(up, fallback=(0.0, 1.0, 0.0))
        center = np.asarray(self._tag_center, dtype=np.float64)
        return (
            ("front", center, center + normal * arrow_len, QColor(255, 110, 110), False),
            ("back", center, center - normal * back_len, QColor(255, 110, 110, 130), True),
            ("up", center, center + up * arrow_len, QColor(125, 255, 125), False),
        )

    def _tag_validation_distance_circles(self, *, samples: int = 96) -> tuple[tuple[str, np.ndarray], ...]:
        radius = self._tag_validation_distance_sfm
        if radius is None or radius <= 0.0:
            return ()
        count = max(16, int(samples))
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=True, dtype=np.float64)
        cos = np.cos(angles)
        sin = np.sin(angles)
        center = np.asarray(self._tag_center, dtype=np.float64)
        xz = np.column_stack(
            [
                center[0] + radius * cos,
                np.full_like(cos, center[1]),
                center[2] + radius * sin,
            ]
        )
        xy = np.column_stack(
            [
                center[0] + radius * cos,
                center[1] + radius * sin,
                np.full_like(cos, center[2]),
            ]
        )
        yz = np.column_stack(
            [
                np.full_like(cos, center[0]),
                center[1] + radius * cos,
                center[2] + radius * sin,
            ]
        )
        return (("XZ", xz), ("XY", xy), ("YZ", yz))

    def _draw_tag_validation_distance(self, painter: QPainter) -> None:
        circles = self._tag_validation_distance_circles()
        if not circles:
            return
        color = QColor(255, 210, 90, 105)
        for _label, points in circles:
            self._draw_world_polyline(painter, points, color, 1, dashed=True)
        radius = float(self._tag_validation_distance_sfm or 0.0)
        self._draw_marker(
            painter,
            np.asarray(self._tag_center, dtype=np.float64) + np.array([radius, 0.0, 0.0], dtype=np.float64),
            QColor(255, 210, 90, 150),
            "max dist",
            radius=3,
        )

    def _selected_group(self) -> CubemapFrameGroup | None:
        if not self._selected_group_name:
            return None
        return next((group for group in self._groups if group.name == self._selected_group_name), None)

    def _selected_image_ray_group(self) -> CubemapFrameGroup | None:
        if not self._selected_group_name:
            return None
        return next((group for group in self._image_ray_groups if group.name == self._selected_group_name), None)

    def _draw_world_line(
        self,
        painter: QPainter,
        a: np.ndarray,
        b: np.ndarray,
        color: QColor,
        width: int,
        *,
        dashed: bool = False,
    ) -> None:
        xy, _depth = self._project(np.vstack([a, b]))
        if not np.all(np.isfinite(xy)):
            return
        pen = QPen(color, width)
        if dashed:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(QPointF(float(xy[0, 0]), float(xy[0, 1])), QPointF(float(xy[1, 0]), float(xy[1, 1])))

    def _draw_world_polyline(
        self,
        painter: QPainter,
        points: np.ndarray,
        color: QColor,
        width: int,
        *,
        dashed: bool = False,
    ) -> None:
        xy, _depth = self._project(points)
        finite = np.isfinite(xy).all(axis=1)
        if np.count_nonzero(finite) < 2:
            return
        pen = QPen(color, width)
        if dashed:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        for a, b, valid_a, valid_b in zip(xy[:-1], xy[1:], finite[:-1], finite[1:], strict=True):
            if not (valid_a and valid_b):
                continue
            painter.drawLine(QPointF(float(a[0]), float(a[1])), QPointF(float(b[0]), float(b[1])))

    def _draw_world_arrow(
        self,
        painter: QPainter,
        a: np.ndarray,
        b: np.ndarray,
        color: QColor,
        label: str,
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
        label: str,
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
            painter.setBrush(color)
            painter.drawEllipse(end, 4.0, 4.0)
            label_pos = end + QPointF(6.0, -6.0)
        painter.setBrush(Qt.NoBrush)
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


def _normalized(value: np.ndarray, *, fallback: tuple[float, float, float]) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12 or not np.isfinite(norm):
        return np.asarray(fallback, dtype=np.float64)
    return value / norm


def _clamp_interval(minimum: float, maximum: float, center: float, max_span: float) -> tuple[float, float]:
    lower = float(minimum)
    upper = float(maximum)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        half = max(float(max_span) * 0.5, 1e-6)
        return float(center) - half, float(center) + half
    span = upper - lower
    limit = max(float(max_span), 1e-6)
    if span <= limit:
        return lower, upper
    interval_center = float(np.clip(float(center), lower + limit * 0.5, upper - limit * 0.5))
    return interval_center - limit * 0.5, interval_center + limit * 0.5


def _screen_axis_delta(
    *,
    axis: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    length: float,
) -> np.ndarray:
    return np.array(
        [
            float(axis @ right) * float(length),
            -float(axis @ up) * float(length),
        ],
        dtype=np.float64,
    )


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
