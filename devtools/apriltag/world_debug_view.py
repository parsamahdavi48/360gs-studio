"""2D-painted 3D world debug view for AprilTag synthetic placement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPolygonF, QWheelEvent
from PySide6.QtWidgets import QWidget

from core.apriltag_geometry import tag_corners_sfm
from devtools.apriltag.cubemap_preview import CubemapFrameGroup, axis_preview_frustum_rays


@dataclass(frozen=True)
class PointCloudSample:
    points: np.ndarray
    colors: np.ndarray | None
    source_count: int


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


def load_point_cloud_sample(path: Path, *, max_points: int = 80_000) -> PointCloudSample:
    """Load a small deterministic sample of an ASCII or binary PLY point cloud."""
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
    max_points: int,
    source_count: int,
) -> PointCloudSample:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("point cloud must be an Nx3 array")
    finite = np.all(np.isfinite(points), axis=1)
    if not np.all(finite):
        points = points[finite]
        colors = colors[finite] if colors is not None else None
    limit = max(0, int(max_points))
    if limit and len(points) > limit:
        indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
        points = points[indices]
        colors = colors[indices] if colors is not None else None
    return PointCloudSample(points, colors, source_count)


class AprilTagWorldDebugView(QWidget):
    """Orthographic world-space viewport synced with the AprilTag image preview."""

    camera_clicked = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(360, 320)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._pointcloud: PointCloudSample | None = None
        self._groups: tuple[CubemapFrameGroup, ...] = ()
        self._selected_group_name = ""
        self._tag_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._tag_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self._tag_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self._tag_size_m = 0.16
        self._true_scale = 0.25
        self._preview_yaw_deg = 0.0
        self._preview_pitch_deg = 0.0
        self._preview_roll_deg = 0.0
        self._preview_fov_deg = 90.0
        self._grid_step = 2.0
        self._grid_extent = 20.0
        self._view_yaw_deg = 35.0
        self._view_pitch_deg = -28.0
        self._view_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._pixels_per_unit = 18.0
        self._last_mouse: QPointF | None = None
        self._press_pos: QPointF | None = None
        self._press_button: Qt.MouseButton | None = None
        self._user_navigated = False

    def sizeHint(self) -> QSize:
        return QSize(520, 420)

    def set_pointcloud(self, pointcloud: PointCloudSample | None) -> None:
        self._pointcloud = pointcloud
        self._fit_scene_if_needed(force=not self._user_navigated)
        self.update()

    def set_groups(self, groups: tuple[CubemapFrameGroup, ...]) -> None:
        self._groups = groups
        if self._selected_group_name and not any(group.name == self._selected_group_name for group in groups):
            self._selected_group_name = ""
        self._fit_scene_if_needed(force=not self._user_navigated)
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

    def set_grid(self, *, step: float, extent: float) -> None:
        self._grid_step = max(0.1, float(step))
        self._grid_extent = max(self._grid_step, float(extent))
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

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(13, 17, 23))
        painter.setRenderHint(QPainter.Antialiasing, True)
        self._draw_grid(painter)
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
                group_name = self._camera_name_at_screen_pos(event.position())
                if group_name:
                    self.camera_clicked.emit(group_name)
        self._last_mouse = None
        self._press_pos = None
        self._press_button = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.2 if delta > 0 else 1.0 / 1.2
        self._pixels_per_unit = max(0.01, min(10000.0, self._pixels_per_unit * factor))
        self._user_navigated = True
        self.update()

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
        yaw = np.deg2rad(self._view_yaw_deg)
        pitch = np.deg2rad(self._view_pitch_deg)
        forward = np.array(
            [np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)],
            dtype=np.float64,
        )
        forward = _normalized(forward, fallback=(0.0, 0.0, 1.0))
        right = np.array([np.cos(yaw), 0.0, -np.sin(yaw)], dtype=np.float64)
        right = _normalized(right, fallback=(1.0, 0.0, 0.0))
        up = np.cross(forward, right)
        up = _normalized(up, fallback=(0.0, 1.0, 0.0))
        # Match the Metashape/LichtFeld viewport convention: +X and +Y keep the
        # same screen direction, while +Z is displayed on the opposite side.
        screen_z_flip = np.array([1.0, 1.0, -1.0], dtype=np.float64)
        return right * screen_z_flip, up * screen_z_flip, forward * screen_z_flip

    def _project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        right, up, forward = self._view_basis()
        rel = points - self._view_center
        x = rel @ right * self._pixels_per_unit + self.width() * 0.5
        y = -(rel @ up) * self._pixels_per_unit + self.height() * 0.5
        depth = rel @ forward
        return np.column_stack([x, y]), depth

    def _draw_grid(self, painter: QPainter) -> None:
        visible_units = max(self.width(), self.height()) / max(self._pixels_per_unit, 1e-6)
        extent = max(self._grid_extent, visible_units * 0.75)
        step = self._grid_step
        while extent / step > 90:
            step *= 2.0
        x_min = np.floor((self._view_center[0] - extent) / step) * step
        x_max = np.ceil((self._view_center[0] + extent) / step) * step
        z_min = np.floor((self._view_center[2] - extent) / step) * step
        z_max = np.ceil((self._view_center[2] + extent) / step) * step

        for x in np.arange(x_min, x_max + step * 0.5, step):
            color = QColor(90, 175, 245) if abs(x) <= step * 0.25 else QColor(55, 63, 72)
            self._draw_world_line(painter, np.array([x, 0.0, z_min]), np.array([x, 0.0, z_max]), color, 2 if abs(x) <= step * 0.25 else 1)
        for z in np.arange(z_min, z_max + step * 0.5, step):
            color = QColor(245, 180, 90) if abs(z) <= step * 0.25 else QColor(55, 63, 72)
            self._draw_world_line(painter, np.array([x_min, 0.0, z]), np.array([x_max, 0.0, z]), color, 2 if abs(z) <= step * 0.25 else 1)
        self._draw_marker(painter, np.array([0.0, 0.0, 0.0]), QColor(255, 80, 255), "O", radius=5)

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
            screen = np.array(
                [
                    float(origin.x()) + float(axis @ right) * float(length),
                    float(origin.y()) - float(axis @ up) * float(length),
                ],
                dtype=np.float64,
            )
            points[label] = QPointF(float(screen[0]), float(screen[1]))
        return points

    @staticmethod
    def _axis_definitions() -> tuple[tuple[str, np.ndarray, QColor], ...]:
        return (
            ("X", np.array([1.0, 0.0, 0.0], dtype=np.float64), QColor(255, 92, 92)),
            ("Y", np.array([0.0, 1.0, 0.0], dtype=np.float64), QColor(120, 245, 130)),
            ("Z", np.array([0.0, 0.0, 1.0], dtype=np.float64), QColor(96, 170, 255)),
        )

    def _draw_pointcloud(self, painter: QPainter) -> None:
        if self._pointcloud is None or len(self._pointcloud.points) == 0:
            return
        xy, depth = self._project(self._pointcloud.points)
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
        if len(visible_xy) > 60_000:
            order = np.argsort(visible_depth)
            selected = order[np.linspace(0, len(order) - 1, 60_000, dtype=np.int64)]
            visible_xy = visible_xy[selected]
        painter.setPen(QPen(QColor(145, 155, 164, 130), 1))
        painter.drawPoints(QPolygonF([QPointF(float(x), float(y)) for x, y in visible_xy]))

    def _draw_cameras(self, painter: QPainter) -> None:
        if not self._groups:
            return
        positions = np.asarray([group.camera_position_sfm for group in self._groups], dtype=np.float64)
        xy, _depth = self._project(positions)
        for group, point in zip(self._groups, xy, strict=True):
            selected = group.name == self._selected_group_name
            color = QColor(255, 218, 92) if selected else QColor(95, 160, 255)
            radius = 5 if selected else 3
            painter.setPen(QPen(QColor(7, 10, 14), 3))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(float(point[0]), float(point[1])), radius, radius)
            painter.setBrush(Qt.NoBrush)
        selected = self._selected_group()
        if selected is not None:
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
        forward, corner_rays = axis_preview_frustum_rays(
            output_size=129,
            yaw_deg=self._preview_yaw_deg,
            pitch_deg=self._preview_pitch_deg,
            roll_deg=self._preview_roll_deg,
            fov_deg=self._preview_fov_deg,
        )
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
        xy, _depth = self._project(corners)
        polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in xy])
        painter.setPen(QPen(QColor(0, 255, 180), 3))
        painter.setBrush(QColor(0, 255, 180, 35))
        painter.drawPolygon(polygon)
        painter.setBrush(Qt.NoBrush)
        self._draw_marker(painter, self._tag_center, QColor(0, 255, 180), "tag", radius=5)

        foot = np.array([self._tag_center[0], 0.0, self._tag_center[2]], dtype=np.float64)
        self._draw_world_line(painter, self._tag_center, foot, QColor(0, 255, 255), 2, dashed=True)
        self._draw_marker(painter, foot, QColor(0, 255, 255), "XZ", radius=5)

        arrow_len = max(self._tag_size_m / self._true_scale, self._grid_step) * 0.8
        self._draw_world_line(painter, self._tag_center, self._tag_center + self._tag_normal * arrow_len, QColor(255, 110, 110), 2)
        self._draw_world_line(painter, self._tag_center, self._tag_center + self._tag_up * arrow_len, QColor(125, 255, 125), 2)

    def _selected_group(self) -> CubemapFrameGroup | None:
        if not self._selected_group_name:
            return None
        return next((group for group in self._groups if group.name == self._selected_group_name), None)

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
        pen = QPen(color, width)
        if dashed:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(QPointF(float(xy[0, 0]), float(xy[0, 1])), QPointF(float(xy[1, 0]), float(xy[1, 1])))

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
