"""Small PLY point-cloud reader for preview and validation tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_POINT_COLOR_RGBA = (145.0 / 255.0, 155.0 / 255.0, 164.0 / 255.0, 130.0 / 255.0)
POINT_COLOR_ALPHA = 0.9


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


def load_point_cloud_sample(path: Path, *, max_points: int | None = None) -> PointCloudSample:
    """Load a deterministic PLY sample, or all points when ``max_points`` is unset."""
    with Path(path).open("rb") as f:
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


def pointcloud_vertex_data(pointcloud: PointCloudSample | None) -> np.ndarray:
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
