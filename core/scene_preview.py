"""Read-only scene preview data model and loaders.

Coordinate contract:
- Preview world space is right-handed and Y-up.
- A preview camera stores explicit image axes instead of relying on a single
  external convention: ``right`` is image +X, ``up`` is image +Y, and
  ``forward`` is the viewing direction.
- Perspective preview projection maps camera-local +Y to physical screen up.
  Display-only leveling can be added on top, but loaders must preserve image up.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from core.pointcloud_io import load_point_cloud_sample
from core.scene_import_contracts import IMAGE_EXTS

SCENE_PREVIEW_WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class ScenePreviewCamera:
    camera_id: str
    label: str
    image_path: Path | None
    projection: str
    width: int
    height: int
    fl_x: float | None
    fl_y: float | None
    cx: float | None
    cy: float | None
    position: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    source: dict[str, Any]

    def world_to_camera(self, points_world: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=np.float64)
        delta = points.reshape(-1, 3) - self.position.reshape(1, 3)
        return np.column_stack(
            [
                delta @ self.right,
                delta @ self.up,
                delta @ self.forward,
            ]
        )

    def project_world_points(self, points_world: np.ndarray) -> np.ndarray | None:
        if self.fl_x is None or self.fl_y is None or self.cx is None or self.cy is None:
            return None
        camera_points = self.world_to_camera(points_world)
        z = camera_points[:, 2]
        if np.any(z <= 1e-8):
            return None
        u = float(self.fl_x) * (camera_points[:, 0] / z) + float(self.cx)
        v = float(self.cy) - float(self.fl_y) * (camera_points[:, 1] / z)
        return np.stack([u, v], axis=1).astype(np.float32)

    def world_up_roll_radians(self, world_up: np.ndarray = SCENE_PREVIEW_WORLD_UP) -> float | None:
        """Return display roll that would align projected world up with screen up."""
        up = np.asarray(world_up, dtype=np.float64)
        projected = up - self.forward * float(up @ self.forward)
        norm = float(np.linalg.norm(projected))
        if norm <= 1e-8:
            return None
        projected /= norm
        x = float(projected @ self.right)
        y = float(projected @ self.up)
        return math.atan2(x, y)


@dataclass(frozen=True)
class ScenePreviewPointCloud:
    points: np.ndarray
    colors: np.ndarray | None
    source_path: Path | None = None


@dataclass(frozen=True)
class ScenePreviewDataset:
    source_kind: str
    source_path: Path
    cameras: tuple[ScenePreviewCamera, ...]
    pointcloud: ScenePreviewPointCloud | None = None
    image_root: Path | None = None
    mask_root: Path | None = None
    coordinate_note: str = "preview_y_up"


def transform_preview_dataset(
    dataset: ScenePreviewDataset,
    *,
    camera_matrix: np.ndarray | None = None,
    pointcloud_matrix: np.ndarray | None = None,
    coordinate_note: str | None = None,
) -> ScenePreviewDataset:
    camera_transform = _display_matrix_or_none(camera_matrix)
    pointcloud_transform = _display_matrix_or_none(pointcloud_matrix)
    cameras = (
        tuple(_transform_preview_camera(camera, camera_transform) for camera in dataset.cameras)
        if camera_transform is not None
        else dataset.cameras
    )
    pointcloud = (
        _transform_preview_pointcloud(dataset.pointcloud, pointcloud_transform)
        if pointcloud_transform is not None
        else dataset.pointcloud
    )
    return replace(
        dataset,
        cameras=cameras,
        pointcloud=pointcloud,
        coordinate_note=coordinate_note or dataset.coordinate_note,
    )


def load_ply_preview_pointcloud(path: Path, *, max_points: int | None = None) -> ScenePreviewPointCloud:
    sample = load_point_cloud_sample(Path(path), max_points=max_points)
    return ScenePreviewPointCloud(points=sample.points, colors=sample.colors, source_path=Path(path))


def load_transforms_preview_dataset(
    transforms_json: Path,
    *,
    image_root: Path | None = None,
    mask_root: Path | None = None,
    pointcloud: ScenePreviewPointCloud | None = None,
) -> ScenePreviewDataset:
    transforms_json = Path(transforms_json)
    data = json.loads(transforms_json.read_text(encoding="utf-8"))
    root = Path(image_root) if image_root is not None else transforms_json.parent
    camera_model = str(data.get("camera_model") or "").strip().upper()
    projection = "equirectangular" if camera_model == "EQUIRECTANGULAR" else "pinhole"
    cameras: list[ScenePreviewCamera] = []
    for index, frame in enumerate(data.get("frames") or []):
        if not isinstance(frame, dict):
            continue
        file_path = str(frame.get("file_path") or "").strip()
        matrix = _matrix4(frame.get("transform_matrix"), f"frame {file_path or index}")
        intrinsics = _frame_intrinsics(frame, data, projection=projection)
        cameras.append(
            _camera_from_transform(
                camera_id=str(frame.get("name") or file_path or index),
                label=file_path or f"camera-{index + 1}",
                image_path=_resolve_path(root, file_path) if file_path else None,
                projection=projection,
                matrix=matrix,
                source={"format": "transforms", "frame_index": index, "camera_model": camera_model},
                **intrinsics,
            )
        )
    return ScenePreviewDataset(
        source_kind="transforms",
        source_path=transforms_json,
        cameras=tuple(cameras),
        pointcloud=pointcloud,
        image_root=root,
        mask_root=Path(mask_root) if mask_root is not None else None,
    )


def load_colmap_preview_dataset(
    model_dir: Path,
    *,
    images_dir: Path | None = None,
    masks_dir: Path | None = None,
    pointcloud: ScenePreviewPointCloud | None = None,
    opengl_camera: bool = False,
    sphere_as_equirectangular: bool = True,
) -> ScenePreviewDataset:
    from scripts.spheresfm_to_transforms import colmap_pose_to_c2w, read_model

    cameras_by_id, images_by_id, points_by_id, resolved_model_dir = read_model(Path(model_dir))
    root = Path(images_dir) if images_dir is not None else resolved_model_dir
    preview_cameras: list[ScenePreviewCamera] = []
    for image_id in sorted(images_by_id):
        image = images_by_id[image_id]
        camera = cameras_by_id.get(image.camera_id)
        if camera is None:
            continue
        c2w = colmap_pose_to_c2w(image, opengl_camera=opengl_camera)
        right = _normalized(c2w[:3, 0], f"COLMAP image {image_id} right")
        up_axis = c2w[:3, 1] if opengl_camera else -c2w[:3, 1]
        up = _normalized(up_axis, f"COLMAP image {image_id} up")
        forward = _normalized(c2w[:3, 2], f"COLMAP image {image_id} forward")
        intrinsics = _colmap_intrinsics(camera)
        projection = "equirectangular" if sphere_as_equirectangular and camera.model == "SPHERE" else "pinhole"
        if projection == "equirectangular":
            intrinsics = {**intrinsics, "fl_x": None, "fl_y": None, "cx": None, "cy": None}
        preview_cameras.append(
            ScenePreviewCamera(
                camera_id=str(image.image_id),
                label=image.name,
                image_path=_resolve_path(root, image.name),
                projection=projection,
                position=c2w[:3, 3].astype(np.float64),
                right=right,
                up=up,
                forward=forward,
                source={"format": "colmap", "image_id": image.image_id, "camera_id": image.camera_id},
                **intrinsics,
            )
        )
    pointcloud = pointcloud or _pointcloud_from_colmap_points(points_by_id)
    if pointcloud is None:
        ply = Path(resolved_model_dir) / "points3D.ply"
        if ply.is_file():
            pointcloud = load_ply_preview_pointcloud(ply)
    return ScenePreviewDataset(
        source_kind="colmap",
        source_path=resolved_model_dir,
        cameras=tuple(preview_cameras),
        pointcloud=pointcloud,
        image_root=root,
        mask_root=Path(masks_dir) if masks_dir is not None else None,
    )


def load_realityscan_preview_dataset(
    csv_path: Path,
    *,
    images_dir: Path | None = None,
    masks_dir: Path | None = None,
    pointcloud: ScenePreviewPointCloud | None = None,
) -> ScenePreviewDataset:
    from core.realityscan_to_transforms import read_realityscan_csv, resolve_image_path, row_to_transform

    csv_path = Path(csv_path)
    image_root = Path(images_dir) if images_dir is not None else csv_path.parent / "images"
    cameras: list[ScenePreviewCamera] = []
    for index, row in enumerate(read_realityscan_csv(csv_path)):
        image_path = resolve_image_path(image_root, row.name)
        intrinsics = _realityscan_intrinsics(row, image_path)
        cameras.append(
            _camera_from_transform(
                camera_id=str(index + 1),
                label=row.name,
                image_path=image_path,
                projection="pinhole",
                matrix=row_to_transform(row),
                source={"format": "realityscan_csv", "row_index": index},
                **intrinsics,
            )
        )
    return ScenePreviewDataset(
        source_kind="realityscan",
        source_path=csv_path,
        cameras=tuple(cameras),
        pointcloud=pointcloud,
        image_root=image_root,
        mask_root=Path(masks_dir) if masks_dir is not None else None,
        coordinate_note="realityscan_csv",
    )


def load_metashape_preview_dataset(
    xml_path: Path,
    *,
    images_dir: Path | None = None,
    masks_dir: Path | None = None,
    pointcloud: ScenePreviewPointCloud | None = None,
) -> ScenePreviewDataset:
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    chunk = _first_enabled_child(root, "chunk")
    if chunk is None:
        raise ValueError(f"Metashape XML has no chunk: {xml_path}")
    sensors = _parse_metashape_sensors(chunk)
    components = _parse_metashape_components(chunk)
    fallback = next(iter(sensors.values()), _default_intrinsics("pinhole"))
    image_lookup = _image_lookup(Path(images_dir)) if images_dir is not None else {}
    camera_parent = _xml_child(chunk, "cameras")
    preview_cameras: list[ScenePreviewCamera] = []
    if camera_parent is None:
        return ScenePreviewDataset(
            "metashape",
            xml_path,
            tuple(),
            pointcloud=pointcloud,
            image_root=images_dir,
            mask_root=Path(masks_dir) if masks_dir is not None else None,
        )
    for index, camera_el in enumerate(_xml_children(camera_parent, "camera")):
        transform_el = _xml_child(camera_el, "transform")
        if transform_el is None or not transform_el.text:
            continue
        matrix = np.asarray([float(part) for part in transform_el.text.split()], dtype=np.float64).reshape(4, 4)
        component_id = str(camera_el.get("component_id") or "")
        if component_id in components:
            matrix = components[component_id] @ matrix
        sensor = sensors.get(str(camera_el.get("sensor_id") or ""), fallback)
        label = str(camera_el.get("label") or f"camera-{index + 1}")
        image_path = _resolve_metashape_image(label, image_lookup)
        projection = str(sensor.get("projection") or "pinhole")
        intrinsics = {key: value for key, value in sensor.items() if key != "projection"}
        preview_cameras.append(
            _camera_from_transform(
                camera_id=str(camera_el.get("id") or index),
                label=label,
                image_path=image_path,
                projection=projection,
                matrix=matrix,
                image_y_axis="down",
                source={"format": "metashape", "camera_index": index, "enabled": camera_el.get("enabled") != "false"},
                **intrinsics,
            )
        )
    return ScenePreviewDataset(
        source_kind="metashape",
        source_path=xml_path,
        cameras=tuple(preview_cameras),
        pointcloud=pointcloud,
        image_root=images_dir,
        mask_root=Path(masks_dir) if masks_dir is not None else None,
    )


def _camera_from_transform(
    *,
    camera_id: str,
    label: str,
    image_path: Path | None,
    projection: str,
    matrix: np.ndarray,
    image_y_axis: str = "up",
    width: int,
    height: int,
    fl_x: float | None,
    fl_y: float | None,
    cx: float | None,
    cy: float | None,
    source: dict[str, Any],
) -> ScenePreviewCamera:
    y_axis = _normalized(matrix[:3, 1], f"{label} image y")
    up = -y_axis if str(image_y_axis).strip().lower() == "down" else y_axis
    return ScenePreviewCamera(
        camera_id=camera_id,
        label=label,
        image_path=image_path,
        projection=projection,
        width=width,
        height=height,
        fl_x=fl_x,
        fl_y=fl_y,
        cx=cx,
        cy=cy,
        position=matrix[:3, 3].astype(np.float64),
        right=_normalized(matrix[:3, 0], f"{label} right"),
        up=up,
        forward=_normalized(matrix[:3, 2], f"{label} forward"),
        source=source,
    )


def _transform_preview_camera(camera: ScenePreviewCamera, matrix: np.ndarray) -> ScenePreviewCamera:
    rotation = matrix[:3, :3]
    position = _transform_points(camera.position.reshape(1, 3), matrix, dtype=np.float64)[0]
    return replace(
        camera,
        position=position,
        right=_normalized(rotation @ camera.right, f"{camera.label} transformed right"),
        up=_normalized(rotation @ camera.up, f"{camera.label} transformed up"),
        forward=_normalized(rotation @ camera.forward, f"{camera.label} transformed forward"),
    )


def _transform_preview_pointcloud(
    pointcloud: ScenePreviewPointCloud | None,
    matrix: np.ndarray,
) -> ScenePreviewPointCloud | None:
    if pointcloud is None:
        return None
    points = _transform_points(pointcloud.points, matrix, dtype=np.float32)
    return ScenePreviewPointCloud(
        points=points,
        colors=pointcloud.colors,
        source_path=pointcloud.source_path,
    )


def _transform_points(points: np.ndarray, matrix: np.ndarray, *, dtype: Any) -> np.ndarray:
    source = np.asarray(points, dtype=dtype).reshape(-1, 3)
    rotation = matrix[:3, :3].astype(dtype)
    translation = matrix[:3, 3].astype(dtype)
    return source @ rotation.T + translation.reshape(1, 3)


def _display_matrix_or_none(matrix: np.ndarray | None) -> np.ndarray | None:
    if matrix is None:
        return None
    result = np.asarray(matrix, dtype=np.float64)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError("display transform must be a finite 4x4 matrix")
    if np.allclose(result, np.eye(4, dtype=np.float64)):
        return None
    return result


def _frame_intrinsics(frame: dict[str, Any], data: dict[str, Any], *, projection: str) -> dict[str, Any]:
    width = int(frame.get("w") or data.get("w") or 0)
    height = int(frame.get("h") or data.get("h") or 0)
    if projection == "equirectangular":
        return {"width": width, "height": height, "fl_x": None, "fl_y": None, "cx": None, "cy": None}
    fl_x = float(frame.get("fl_x") or data.get("fl_x") or 0.0)
    fl_y = float(frame.get("fl_y") or data.get("fl_y") or fl_x)
    cx = float(frame.get("cx") if frame.get("cx") is not None else data.get("cx", (width - 1) / 2.0))
    cy = float(frame.get("cy") if frame.get("cy") is not None else data.get("cy", (height - 1) / 2.0))
    return {"width": width, "height": height, "fl_x": fl_x, "fl_y": fl_y, "cx": cx, "cy": cy}


def _colmap_intrinsics(camera: Any) -> dict[str, Any]:
    width = int(camera.width)
    height = int(camera.height)
    params = tuple(float(value) for value in camera.params)
    if camera.model == "PINHOLE" and len(params) >= 4:
        fl_x, fl_y, cx, cy = params[:4]
    else:
        fl_x = params[0] if params else float(width) / 2.0
        fl_y = fl_x
        cx = params[1] if len(params) > 1 else (width - 1) / 2.0
        cy = params[2] if len(params) > 2 else (height - 1) / 2.0
    return {"width": width, "height": height, "fl_x": fl_x, "fl_y": fl_y, "cx": cx, "cy": cy}


def _realityscan_intrinsics(row: Any, image_path: Path) -> dict[str, Any]:
    from core.realityscan_to_transforms import camera_from_csv_row, image_size

    if image_path.is_file():
        try:
            width, height = image_size(image_path)
            camera = camera_from_csv_row(row, width, height)
            return {
                "width": int(camera["w"]),
                "height": int(camera["h"]),
                "fl_x": float(camera["fl_x"]),
                "fl_y": float(camera["fl_y"]),
                "cx": float(camera["cx"]),
                "cy": float(camera["cy"]),
            }
        except Exception:
            pass
    return {"width": 0, "height": 0, "fl_x": None, "fl_y": None, "cx": None, "cy": None}


def _pointcloud_from_colmap_points(points_by_id: dict[int, Any]) -> ScenePreviewPointCloud | None:
    if not points_by_id:
        return None
    ordered = [points_by_id[key] for key in sorted(points_by_id)]
    points = np.asarray([point.xyz for point in ordered], dtype=np.float32)
    colors = np.asarray([point.rgb for point in ordered], dtype=np.uint8)
    return ScenePreviewPointCloud(points=points, colors=colors)


def _parse_metashape_sensors(chunk: ET.Element) -> dict[str, dict[str, Any]]:
    sensors_parent = _xml_child(chunk, "sensors")
    if sensors_parent is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for sensor in _xml_children(sensors_parent, "sensor"):
        sensor_id = str(sensor.get("id") or "")
        calibration = _xml_child(sensor, "calibration")
        resolution = _parse_resolution(_xml_child(calibration, "resolution") if calibration is not None else None)
        resolution = resolution or _parse_resolution(_xml_child(sensor, "resolution")) or (1, 1)
        sensor_type = str(sensor.get("type") or (calibration.get("type") if calibration is not None else "") or "frame")
        width, height = resolution
        focal = _xml_float(_xml_child(calibration, "f") if calibration is not None else None)
        focal = focal or _xml_float(_xml_child(sensor, "f")) or float(max(width, height, 1))
        projection = "equirectangular" if sensor_type.lower() in {"spherical", "equirectangular"} else "pinhole"
        result[sensor_id] = _default_intrinsics(projection, width=width, height=height, focal=focal)
    return result


def _parse_metashape_components(chunk: ET.Element) -> dict[str, np.ndarray]:
    components_parent = _xml_child(chunk, "components")
    if components_parent is None:
        return {}
    result: dict[str, np.ndarray] = {}
    for component in _xml_children(components_parent, "component"):
        transform = _xml_child(component, "transform")
        matrix = _parse_component_transform(transform)
        component_id = str(component.get("id") or "")
        if component_id and matrix is not None:
            result[component_id] = matrix
    return result


def _parse_component_transform(transform: ET.Element | None) -> np.ndarray | None:
    if transform is None:
        return None
    rotation_values = _number_list(_xml_child(transform, "rotation"))
    translation_values = _number_list(_xml_child(transform, "translation"))
    if len(rotation_values) < 9 or len(translation_values) < 3:
        return None
    scale = _xml_float(_xml_child(transform, "scale")) or 1.0
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation_values[:9], dtype=np.float64).reshape(3, 3) * scale
    matrix[:3, 3] = np.asarray(translation_values[:3], dtype=np.float64)
    return matrix


def _default_intrinsics(projection: str, *, width: int = 1, height: int = 1, focal: float = 1.0) -> dict[str, Any]:
    if projection == "equirectangular":
        return {"projection": projection, "width": width, "height": height, "fl_x": None, "fl_y": None, "cx": None, "cy": None}
    return {
        "projection": projection,
        "width": width,
        "height": height,
        "fl_x": float(focal),
        "fl_y": float(focal),
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
    }


def _matrix4(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"Invalid 4x4 transform for {label}")
    return matrix


def _normalized(value: np.ndarray, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{label} must not be zero")
    return vector / norm


def _resolve_path(root: Path, file_path: str) -> Path:
    path = Path(file_path)
    return path if path.is_absolute() else root / path


def _image_lookup(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    lookup: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        lookup.setdefault(path.name.lower(), path)
        lookup.setdefault(path.stem.lower(), path)
    return lookup


def _resolve_metashape_image(label: str, lookup: dict[str, Path]) -> Path | None:
    if not lookup:
        return None
    return lookup.get(label.lower()) or lookup.get(Path(label).stem.lower())


def _tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _xml_child(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    target = name.lower()
    for child in element:
        if _tag_name(child.tag) == target:
            return child
    return None


def _xml_children(element: ET.Element | None, name: str) -> list[ET.Element]:
    if element is None:
        return []
    target = name.lower()
    return [child for child in element if _tag_name(child.tag) == target]


def _first_enabled_child(element: ET.Element, name: str) -> ET.Element | None:
    children = _xml_children(element, name)
    enabled = next((child for child in children if child.get("enabled") != "false"), None)
    if enabled is not None:
        return enabled
    return children[0] if children else None


def _xml_float(element: ET.Element | None) -> float | None:
    if element is None or element.text is None:
        return None
    try:
        value = float(element.text.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _number_list(element: ET.Element | None) -> list[float]:
    if element is None or element.text is None:
        return []
    values: list[float] = []
    for part in element.text.split():
        try:
            value = float(part)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _parse_resolution(element: ET.Element | None) -> tuple[int, int] | None:
    if element is None:
        return None
    try:
        width = int(element.get("width") or 0)
        height = int(element.get("height") or 0)
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height
