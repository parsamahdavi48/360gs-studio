from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.cubemap_image_io import (
    load_equirect,
    remap_with_channels,
    resolve_output_ext,
    save_image,
)
from core.cubemap_remap import (
    build_remap,
    rot4,
    rotation_matrix,
)
from core.dataset_export_plan import (
    EXPORT_ACTION_EXPAND_ERP_TO_VIEWS,
    EXPORT_ACTION_LINK_PINHOLE,
    EXPORT_ACTION_UNDISTORT_FRAME_TO_PINHOLE,
    build_metashape_dataset_export_plan,
)
from core.dataset_writer_colmap import (
    SPARSE_RELATIVE_DIR,
    ColmapCamera,
    ColmapImage,
    camera_signature,
    quaternion_from_matrix,
    replace_file_with_link_or_copy,
    write_colmap_text_dataset,
)
from core.metashape_coordinates import (
    metashape_camera_to_world,
    metashape_pointcloud_matrix,
)
from core.metashape_model import (
    CAMERA_MODEL_EQUIRECTANGULAR,
    MetashapeSensor,
    parse_metashape_model,
)
from core.metashape_nerf_dataset import AXIS_TRANSFORM_NONE, axis_transform_matrix, normalize_axis_transform
from core.orientation_correction import FINAL_ORIENTATION_NONE, final_orientation_matrix, normalize_final_orientation
from core.realityscan_to_transforms import transform_points, write_transformed_ply
from core.scene_inventory import build_scene_inventory
from core.transforms_to_colmap import read_ply_points, write_points3d_txt

DEFAULT_FOV_DEG = 90.0
DEFAULT_UNDISTORT_ALPHA = 1.0
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class MetashapeColmapExportResult:
    output_dir: Path
    sparse_dir: Path
    image_count: int
    camera_count: int
    action_counts: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def _notify_progress(callback: ProgressCallback | None, done: int, total: int) -> None:
    if callback is not None:
        callback(max(0, int(done)), max(0, int(total)))


def export_metashape_colmap_dataset(
    *,
    scene_dir: str | Path,
    images_dir: str | Path,
    masks_dir: str | Path | None,
    xml_path: str | Path,
    ply_path: str | Path | None,
    output_dir: str | Path,
    views: list[dict[str, Any]],
    output_scale: float,
    output_format: str = "jpg",
    output_bit_depth: str = "8",
    jpg_quality: int = 95,
    fov_deg: float = DEFAULT_FOV_DEG,
    undistort_alpha: float = DEFAULT_UNDISTORT_ALPHA,
    axis_transform: str = AXIS_TRANSFORM_NONE,
    final_orientation: str = FINAL_ORIENTATION_NONE,
    progress_callback: ProgressCallback | None = None,
) -> MetashapeColmapExportResult:
    scene = Path(scene_dir)
    images_root = Path(images_dir)
    masks_root = Path(masks_dir) if masks_dir else scene / "masks"
    output = Path(output_dir)
    output_images = output / "images"
    output_masks = output / "masks"
    sparse_dir = output / SPARSE_RELATIVE_DIR
    output_images.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    model = parse_metashape_model(xml_path)
    inventory = build_scene_inventory(scene, images_dir=images_root, masks_dir=masks_root)
    plan = build_metashape_dataset_export_plan(model, inventory)
    item_count = len(plan.items)
    progress_total = max(1, item_count + 1 + (1 if ply_path else 0))
    _notify_progress(progress_callback, 0, progress_total)

    cameras: list[ColmapCamera] = []
    images: list[ColmapImage] = []
    camera_ids: dict[tuple[Any, ...], int] = {}
    action_counts: dict[str, int] = {}
    world_transform = dataset_world_transform(axis_transform, final_orientation)

    camera_by_id = {camera.camera_id: camera for camera in model.cameras}
    for item_index, item in enumerate(plan.items, start=1):
        action_counts[item.action] = action_counts.get(item.action, 0) + 1
        camera = camera_by_id.get(item.camera_id)
        if camera is None or not item.image_rel_path:
            _notify_progress(progress_callback, item_index, progress_total)
            continue
        sensor = model.sensor_for_camera(camera)
        source_image = _resolve_inventory_path(scene, images_root, item.image_rel_path, standard_root_name="images")
        source_mask = (
            _resolve_inventory_path(scene, masks_root, item.mask_rel_path, standard_root_name="masks")
            if item.mask_rel_path
            else None
        )
        c2w = world_transform @ metashape_camera_to_world(model, camera)
        if item.action == EXPORT_ACTION_EXPAND_ERP_TO_VIEWS:
            _append_expanded_erp_records(
                source_image,
                source_mask,
                output_images,
                output_masks,
                views,
                output_scale,
                output_format,
                output_bit_depth,
                jpg_quality,
                fov_deg,
                c2w,
                cameras,
                images,
                camera_ids,
            )
        elif item.action == EXPORT_ACTION_LINK_PINHOLE:
            output_image = _linked_or_copied_output(source_image, images_root, output_images)
            output_mask = _copy_mask_if_available(source_mask, masks_root, output_masks, output_image)
            _append_colmap_image_record(
                output_image,
                output_images,
                "PINHOLE",
                *_pinhole_payload(sensor, source_image),
                c2w,
                cameras,
                images,
                camera_ids,
            )
            if output_mask is not None:
                output_masks.mkdir(parents=True, exist_ok=True)
        elif item.action == EXPORT_ACTION_UNDISTORT_FRAME_TO_PINHOLE:
            output_image, width, height, new_matrix = _undistort_frame_and_mask(
                sensor,
                source_image,
                source_mask,
                images_root,
                output_images,
                output_masks,
                output_format,
                output_bit_depth,
                jpg_quality,
                undistort_alpha,
            )
            _append_colmap_image_record(
                output_image,
                output_images,
                "PINHOLE",
                width,
                height,
                (
                    float(new_matrix[0, 0]),
                    float(new_matrix[1, 1]),
                    float(new_matrix[0, 2]),
                    float(new_matrix[1, 2]),
                ),
                c2w,
                cameras,
                images,
                camera_ids,
            )

        _notify_progress(progress_callback, item_index, progress_total)

    if not images:
        warning_text = "; ".join(plan.warnings)
        detail = f": {warning_text}" if warning_text else ""
        raise ValueError(f"No Metashape cameras were converted to COLMAP images{detail}")

    write_colmap_text_dataset(output, cameras, images)
    _notify_progress(progress_callback, item_count + 1, progress_total)
    if ply_path:
        ply = Path(ply_path)
        if ply.is_file():
            _write_colmap_pointcloud_files(ply, sparse_dir, world_transform)
        _notify_progress(progress_callback, progress_total, progress_total)
    _write_manifest(output, plan.source_kind, action_counts, plan.warnings)
    return MetashapeColmapExportResult(
        output_dir=output,
        sparse_dir=sparse_dir,
        image_count=len(images),
        camera_count=len(cameras),
        action_counts=action_counts,
        warnings=plan.warnings,
    )


def _resolve_inventory_path(scene: Path, root: Path, rel_path: str, *, standard_root_name: str) -> Path:
    raw = Path(rel_path)
    if raw.is_absolute():
        return raw
    candidates: list[Path] = []
    parts = raw.parts
    if parts and parts[0].casefold() == standard_root_name.casefold():
        candidates.append(root / Path(*parts[1:]) if len(parts) > 1 else root)
    candidates.append(scene / raw)
    candidates.append(root / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else root / raw


def _append_expanded_erp_records(
    source_image: Path,
    source_mask: Path | None,
    output_images: Path,
    output_masks: Path,
    views: list[dict[str, Any]],
    output_scale: float,
    output_format: str,
    output_bit_depth: str,
    jpg_quality: int,
    fov_deg: float,
    c2w: np.ndarray,
    cameras: list[ColmapCamera],
    images: list[ColmapImage],
    camera_ids: dict[tuple[Any, ...], int],
) -> None:
    source = load_equirect(str(source_image))
    input_size = (int(source.shape[1]), int(source.shape[0]))
    output_size = max(1, int(round(input_size[1] * float(output_scale))))
    out_ext = resolve_output_ext(source_image.suffix, output_format)
    mask = load_equirect(str(source_mask)) if source_mask is not None and source_mask.is_file() else None
    for view in views:
        name = str(view["name"])
        yaw = float(view["yaw"])
        pitch = float(view["pitch"])
        map_x, map_y = build_remap(input_size, fov_deg, yaw, pitch, output_size)
        output_image = output_images / source_image.parent.name / f"{source_image.stem}_{name}{out_ext}"
        if source_image.parent.name == "images":
            output_image = output_images / f"{source_image.stem}_{name}{out_ext}"
        output_image.parent.mkdir(parents=True, exist_ok=True)
        save_image(
            remap_with_channels(source, map_x, map_y),
            str(output_image),
            jpg_quality=jpg_quality,
            force_8bit=output_bit_depth == "8",
        )
        if mask is not None:
            output_mask = output_masks / output_image.relative_to(output_images).with_suffix(".png")
            output_mask.parent.mkdir(parents=True, exist_ok=True)
            converted_mask = remap_with_channels(mask, map_x, map_y)
            _threshold, binary = cv2.threshold(_ensure_gray(converted_mask), 127, 255, cv2.THRESH_BINARY)
            save_image(binary, str(output_mask), force_8bit=True)
        view_c2w = c2w @ rot4(rotation_matrix(yaw, pitch, True).T)
        focal = output_size / 2.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
        principal = (output_size - 1.0) / 2.0
        _append_colmap_image_record(
            output_image,
            output_images,
            "PINHOLE",
            output_size,
            output_size,
            (float(focal), float(focal), float(principal), float(principal)),
            view_c2w,
            cameras,
            images,
            camera_ids,
        )


def _append_colmap_image_record(
    output_image: Path,
    output_images: Path,
    model: str,
    width: int,
    height: int,
    params: tuple[float, ...],
    c2w: np.ndarray,
    cameras: list[ColmapCamera],
    images: list[ColmapImage],
    camera_ids: dict[tuple[Any, ...], int],
) -> None:
    signature = camera_signature(model, width, height, params)
    camera_id = camera_ids.get(signature)
    if camera_id is None:
        camera_id = len(cameras) + 1
        camera_ids[signature] = camera_id
        cameras.append(ColmapCamera(camera_id, model, int(width), int(height), params))
    r_cw, t_cw = _c2w_to_colmap_w2c(c2w)
    images.append(
        ColmapImage(
            image_id=len(images) + 1,
            qvec=quaternion_from_matrix(r_cw),
            tvec=t_cw,
            camera_id=camera_id,
            name=output_image.relative_to(output_images).as_posix(),
        )
    )


def _c2w_to_colmap_w2c(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_wc = c2w[:3, :3]
    t_wc = c2w[:3, 3]
    r_cw = r_wc.T
    t_cw = -r_cw @ t_wc
    return r_cw, t_cw


def _pinhole_payload(sensor: MetashapeSensor, image_path: Path) -> tuple[int, int, tuple[float, ...]]:
    width, height = _image_size(image_path)
    sx = width / sensor.width if sensor.width > 0 else 1.0
    sy = height / sensor.height if sensor.height > 0 else 1.0
    fl_x = float(sensor.params.get("fl_x") or sensor.params.get("f") or width)
    fl_y = float(sensor.params.get("fl_y") or fl_x)
    cx = float(sensor.params.get("cx") or width / 2.0)
    cy = float(sensor.params.get("cy") or height / 2.0)
    return width, height, (fl_x * sx, fl_y * sy, cx * sx, cy * sy)


def _undistort_frame_and_mask(
    sensor: MetashapeSensor,
    source_image: Path,
    source_mask: Path | None,
    images_root: Path,
    output_images: Path,
    output_masks: Path,
    output_format: str,
    output_bit_depth: str,
    jpg_quality: int,
    alpha: float,
) -> tuple[Path, int, int, np.ndarray]:
    image = cv2.imread(str(source_image), cv2.IMREAD_UNCHANGED)
    if image is None:
        image = load_equirect(str(source_image))
    width, height, params = _pinhole_payload(sensor, source_image)
    matrix = np.array([[params[0], 0.0, params[2]], [0.0, params[1], params[3]], [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = np.array(
        [
            float(sensor.params.get("k1", 0.0)),
            float(sensor.params.get("k2", 0.0)),
            float(sensor.params.get("p1", 0.0)),
            float(sensor.params.get("p2", 0.0)),
            float(sensor.params.get("k3", 0.0)),
            float(sensor.params.get("k4", 0.0)),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    new_matrix, _roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (width, height), alpha, (width, height))
    map_x, map_y = cv2.initUndistortRectifyMap(matrix, distortion, None, new_matrix, (width, height), cv2.CV_32FC1)
    undistorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rel = _relative_image_path(source_image, images_root)
    output_ext = resolve_output_ext(source_image.suffix, output_format)
    output_image = output_images / rel.with_name(f"{rel.stem}_undistorted{output_ext}")
    output_image.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        undistorted,
        str(output_image),
        jpg_quality=jpg_quality,
        force_8bit=output_bit_depth == "8",
    )

    mask = load_equirect(str(source_mask)) if source_mask is not None and source_mask.is_file() else None
    if mask is None and alpha > 0.0:
        mask = np.full((height, width), 255, dtype=np.uint8)
    if mask is not None:
        output_mask = output_masks / output_image.relative_to(output_images).with_suffix(".png")
        output_mask.parent.mkdir(parents=True, exist_ok=True)
        undistorted_mask = cv2.remap(_ensure_gray(mask), map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        _threshold, binary = cv2.threshold(undistorted_mask, 127, 255, cv2.THRESH_BINARY)
        save_image(binary, str(output_mask), force_8bit=True)
    return output_image, width, height, new_matrix


def _linked_or_copied_output(source: Path, images_root: Path, output_images: Path) -> Path:
    rel = _relative_image_path(source, images_root)
    destination = output_images / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    replace_file_with_link_or_copy(source, destination)
    return destination


def _copy_mask_if_available(source_mask: Path | None, masks_root: Path, output_masks: Path, output_image: Path) -> Path | None:
    if source_mask is None or not source_mask.is_file():
        return None
    rel = _relative_image_path(source_mask, masks_root) if masks_root else Path(source_mask.name)
    destination = output_masks / rel.with_suffix(".png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    replace_file_with_link_or_copy(source_mask, destination)
    return destination


def _relative_image_path(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def _image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        image = load_equirect(str(path))
    height, width = image.shape[:2]
    return int(width), int(height)


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return image[..., 3]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _write_manifest(output: Path, source_kind: str, action_counts: dict[str, int], warnings: tuple[str, ...]) -> None:
    payload = {
        "schema_version": 1,
        "kind": "metashape_colmap_dataset",
        "source_kind": source_kind,
        "action_counts": action_counts,
        "warnings": list(warnings),
        "images_dir": "images",
        "masks_dir": "masks",
        "sparse_dir": "sparse/0",
    }
    (output / "stechdrive_dataset_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_colmap_pointcloud_files(source_ply: Path, sparse_dir: Path, world_transform: np.ndarray | None = None) -> int:
    matrix = (np.eye(4, dtype=np.float64) if world_transform is None else world_transform) @ metashape_pointcloud_matrix()
    points, colors = read_ply_points(source_ply)
    transformed = transform_points(points, matrix)
    count = write_points3d_txt(sparse_dir / "points3D.txt", transformed, colors)
    write_transformed_ply(source_ply, sparse_dir / "points3D.ply", matrix)
    return count


def dataset_world_transform(axis_transform: object, final_orientation: object) -> np.ndarray:
    matrix = axis_transform_matrix(normalize_axis_transform(axis_transform))
    orientation = normalize_final_orientation(final_orientation)
    if orientation != FINAL_ORIENTATION_NONE:
        matrix = final_orientation_matrix(orientation) @ matrix
    return matrix


def metashape_model_requires_mixed_colmap_writer(xml_path: str | Path) -> bool:
    model = parse_metashape_model(xml_path)
    camera_models = {sensor.camera_model for sensor in model.sensors.values()}
    sensor_sizes = {(sensor.width, sensor.height) for sensor in model.sensors.values()}
    return (
        camera_models != {CAMERA_MODEL_EQUIRECTANGULAR}
        or len(sensor_sizes) > 1
        or any(sensor.has_distortion for sensor in model.sensors.values())
    )
