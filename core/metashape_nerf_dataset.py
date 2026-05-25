from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.dataset_export_plan import (
    EXPORT_ACTION_EXPAND_ERP_TO_VIEWS,
    EXPORT_ACTION_LINK_PINHOLE,
    EXPORT_ACTION_UNDISTORT_FRAME_TO_PINHOLE,
    build_metashape_dataset_export_plan,
)
from core.dataset_writer_nerf import write_nerf_json_ply_dataset
from core.metashape_coordinates import metashape_camera_to_world, metashape_pointcloud_matrix
from core.metashape_dataset_assets import (
    MetashapeDatasetAsset,
    distortion_coefficients,
    expand_erp_to_view_assets,
    image_size,
    link_pinhole_asset,
    pinhole_payload,
    resolve_inventory_path,
    undistort_frame_to_pinhole_asset,
)
from core.metashape_model import CAMERA_MODEL_EQUIRECTANGULAR, parse_metashape_model
from core.orientation_correction import (
    FINAL_ORIENTATION_LICHTFELD,
    FINAL_ORIENTATION_NONE,
    FINAL_ORIENTATION_STAGE_CUBEMAP_CLI,
    final_orientation_matrix,
    mark_final_orientation,
    normalize_final_orientation,
)
from core.realityscan_to_transforms import top_level_camera_payload, write_transformed_ply
from core.scene_inventory import build_scene_inventory

DEFAULT_FOV_DEG = 90.0
DEFAULT_UNDISTORT_ALPHA = 1.0
AXIS_TRANSFORM_POSTSHOT = "postshot"
AXIS_TRANSFORM_BRUSH = "brush"
AXIS_TRANSFORM_NONE = "none"
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class MetashapeNerfExportResult:
    output_dir: Path
    transforms_json: Path
    pointcloud: Path | None
    frame_count: int
    action_counts: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MetashapeNerfCompatibility:
    frame_count: int
    camera_group_count: int
    top_level_camera_group_count: int
    action_counts: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def lichtfeld_nerf_supported(self) -> bool:
        return self.camera_group_count <= 1


def _notify_progress(callback: ProgressCallback | None, done: int, total: int) -> None:
    if callback is not None:
        callback(max(0, int(done)), max(0, int(total)))


def export_metashape_nerf_dataset(
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
) -> MetashapeNerfExportResult:
    scene = Path(scene_dir)
    images_root = Path(images_dir)
    masks_root = Path(masks_dir) if masks_dir else scene / "masks"
    output = Path(output_dir)
    output_images = output / "images"
    output_masks = output / "masks"

    model = parse_metashape_model(xml_path)
    inventory = build_scene_inventory(scene, images_dir=images_root, masks_dir=masks_root)
    plan = build_metashape_dataset_export_plan(model, inventory)
    camera_by_id = {camera.camera_id: camera for camera in model.cameras}
    compatibility = analyze_metashape_nerf_compatibility(
        scene_dir=scene,
        images_dir=images_root,
        masks_dir=masks_root,
        xml_path=xml_path,
        views=views,
        output_scale=output_scale,
        fov_deg=fov_deg,
        undistort_alpha=undistort_alpha,
        _model=model,
        _plan=plan,
        _camera_by_id=camera_by_id,
    )
    if (
        is_lichtfeld_nerf_target(axis_transform=axis_transform, final_orientation=final_orientation)
        and not compatibility.lichtfeld_nerf_supported
    ):
        raise ValueError(lichtfeld_nerf_incompatible_message(compatibility))

    output_images.mkdir(parents=True, exist_ok=True)
    world_transform = dataset_world_transform(axis_transform, final_orientation)
    item_count = len(plan.items)
    progress_total = max(1, item_count + 1 + (1 if ply_path else 0))
    _notify_progress(progress_callback, 0, progress_total)

    frames: list[dict[str, Any]] = []
    action_counts: dict[str, int] = {}

    for item_index, item in enumerate(plan.items, start=1):
        action_counts[item.action] = action_counts.get(item.action, 0) + 1
        camera = camera_by_id.get(item.camera_id)
        if camera is None or not item.image_rel_path:
            _notify_progress(progress_callback, item_index, progress_total)
            continue
        sensor = model.sensor_for_camera(camera)
        source_image = resolve_inventory_path(scene, images_root, item.image_rel_path, standard_root_name="images")
        source_mask = (
            resolve_inventory_path(scene, masks_root, item.mask_rel_path, standard_root_name="masks")
            if item.mask_rel_path
            else None
        )
        c2w = world_transform @ metashape_camera_to_world(model, camera)
        assets: tuple[MetashapeDatasetAsset, ...]
        if item.action == EXPORT_ACTION_EXPAND_ERP_TO_VIEWS:
            assets = expand_erp_to_view_assets(
                source_image=source_image,
                source_mask=source_mask,
                images_root=images_root,
                output_images=output_images,
                output_masks=output_masks,
                views=views,
                output_scale=output_scale,
                output_format=output_format,
                output_bit_depth=output_bit_depth,
                jpg_quality=jpg_quality,
                fov_deg=fov_deg,
                c2w=c2w,
                action=item.action,
                source_camera_id=item.camera_id,
                source_camera_label=item.camera_label,
            )
        elif item.action == EXPORT_ACTION_LINK_PINHOLE:
            assets = (
                link_pinhole_asset(
                    sensor=sensor,
                    source_image=source_image,
                    source_mask=source_mask,
                    images_root=images_root,
                    masks_root=masks_root,
                    output_images=output_images,
                    output_masks=output_masks,
                    c2w=c2w,
                    action=item.action,
                    source_camera_id=item.camera_id,
                    source_camera_label=item.camera_label,
                ),
            )
        elif item.action == EXPORT_ACTION_UNDISTORT_FRAME_TO_PINHOLE:
            assets = (
                undistort_frame_to_pinhole_asset(
                    sensor=sensor,
                    source_image=source_image,
                    source_mask=source_mask,
                    images_root=images_root,
                    output_images=output_images,
                    output_masks=output_masks,
                    output_format=output_format,
                    output_bit_depth=output_bit_depth,
                    jpg_quality=jpg_quality,
                    alpha=undistort_alpha,
                    c2w=c2w,
                    action=item.action,
                    source_camera_id=item.camera_id,
                    source_camera_label=item.camera_label,
                ),
            )
        else:
            _notify_progress(progress_callback, item_index, progress_total)
            continue
        frames.extend(_asset_to_frame(asset, output) for asset in assets)
        _notify_progress(progress_callback, item_index, progress_total)

    if not frames:
        raise ValueError("No Metashape cameras were converted to NeRF frames")

    top_camera, top_camera_count, camera_group_count = top_level_camera_payload(frames)
    top_camera_model = str(top_camera.pop("camera_model", "PINHOLE") or "PINHOLE")
    payload: dict[str, Any] = {
        "camera_model": top_camera_model,
        **top_camera,
        "frames": frames,
        "source": {
            "type": "metashape_xml_ply",
            "xml_path": str(xml_path),
            "images_dir": str(images_root),
            "masks_dir": str(masks_root) if masks_root.is_dir() else "",
            "axis_transform": normalize_axis_transform(axis_transform),
            "axis_transform_matrix": axis_transform_matrix(axis_transform).tolist(),
            "final_orientation": normalize_final_orientation(final_orientation),
            "dataset_world_transform": world_transform.tolist(),
            "per_frame_intrinsics": True,
            "per_frame_camera_model": True,
            "top_level_camera_group_count": top_camera_count,
            "camera_group_count": camera_group_count,
            "source_kind": plan.source_kind,
            "action_counts": action_counts,
            "warnings": list(plan.warnings),
        },
    }
    if normalize_final_orientation(final_orientation) != FINAL_ORIENTATION_NONE:
        mark_final_orientation(payload, final_orientation, FINAL_ORIENTATION_STAGE_CUBEMAP_CLI)

    pointcloud_output: Path | None = None
    if ply_path:
        ply = Path(ply_path)
        if ply.is_file():
            pointcloud_output = output / "pointcloud.ply"
            write_transformed_ply(ply, pointcloud_output, world_transform @ metashape_pointcloud_matrix())
            payload["ply_file_path"] = pointcloud_output.name
        _notify_progress(progress_callback, item_count + 1, progress_total)

    write_result = write_nerf_json_ply_dataset(
        output,
        payload,
        transforms_name="transforms.json",
        pointcloud_name="pointcloud.ply",
        manifest={
            "source_kind": "metashape_xml_ply",
            "action_counts": action_counts,
            "warnings": list(plan.warnings),
            "images_dir": "images",
            "masks_dir": "masks",
        },
    )
    _notify_progress(progress_callback, progress_total, progress_total)
    return MetashapeNerfExportResult(
        output_dir=output,
        transforms_json=write_result.transforms_json,
        pointcloud=pointcloud_output or write_result.pointcloud,
        frame_count=write_result.frame_count,
        action_counts=action_counts,
        warnings=plan.warnings,
    )


def analyze_metashape_nerf_compatibility(
    *,
    scene_dir: str | Path,
    images_dir: str | Path,
    masks_dir: str | Path | None,
    xml_path: str | Path,
    views: list[dict[str, Any]],
    output_scale: float,
    fov_deg: float = DEFAULT_FOV_DEG,
    undistort_alpha: float = DEFAULT_UNDISTORT_ALPHA,
    _model: Any | None = None,
    _plan: Any | None = None,
    _camera_by_id: dict[str, Any] | None = None,
) -> MetashapeNerfCompatibility:
    scene = Path(scene_dir)
    images_root = Path(images_dir)
    masks_root = Path(masks_dir) if masks_dir else scene / "masks"
    model = _model or parse_metashape_model(xml_path)
    if _plan is None:
        inventory = build_scene_inventory(scene, images_dir=images_root, masks_dir=masks_root)
        plan = build_metashape_dataset_export_plan(model, inventory)
    else:
        plan = _plan
    camera_by_id = _camera_by_id or {camera.camera_id: camera for camera in model.cameras}

    frames: list[dict[str, Any]] = []
    action_counts: dict[str, int] = {}
    for item in plan.items:
        action_counts[item.action] = action_counts.get(item.action, 0) + 1
        camera = camera_by_id.get(item.camera_id)
        if camera is None or not item.image_rel_path:
            continue
        sensor = model.sensor_for_camera(camera)
        source_image = resolve_inventory_path(scene, images_root, item.image_rel_path, standard_root_name="images")
        if item.action == EXPORT_ACTION_EXPAND_ERP_TO_VIEWS:
            frames.extend(
                _planned_erp_frame_payloads(
                    source_image=source_image,
                    views=views,
                    output_scale=output_scale,
                    fov_deg=fov_deg,
                )
            )
        elif item.action == EXPORT_ACTION_LINK_PINHOLE:
            frames.append(_planned_pinhole_frame_payload(sensor=sensor, source_image=source_image))
        elif item.action == EXPORT_ACTION_UNDISTORT_FRAME_TO_PINHOLE:
            frames.append(
                _planned_undistorted_frame_payload(
                    sensor=sensor,
                    source_image=source_image,
                    alpha=undistort_alpha,
                )
            )

    if not frames:
        return MetashapeNerfCompatibility(
            frame_count=0,
            camera_group_count=0,
            top_level_camera_group_count=0,
            action_counts=action_counts,
            warnings=plan.warnings,
        )
    top_camera, top_camera_count, camera_group_count = top_level_camera_payload(frames)
    _ = top_camera
    return MetashapeNerfCompatibility(
        frame_count=len(frames),
        camera_group_count=camera_group_count,
        top_level_camera_group_count=top_camera_count,
        action_counts=action_counts,
        warnings=plan.warnings,
    )


def _planned_erp_frame_payloads(
    *,
    source_image: Path,
    views: list[dict[str, Any]],
    output_scale: float,
    fov_deg: float,
) -> list[dict[str, Any]]:
    _width, height = image_size(source_image)
    output_size = max(1, int(round(height * float(output_scale))))
    focal = output_size / 2.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    principal = (output_size - 1.0) / 2.0
    return [
        {
            "camera_model": "PINHOLE",
            "w": output_size,
            "h": output_size,
            "fl_x": float(focal),
            "fl_y": float(focal),
            "cx": float(principal),
            "cy": float(principal),
        }
        for view in views
        if bool(view.get("enabled", True))
    ]


def _planned_pinhole_frame_payload(*, sensor: Any, source_image: Path) -> dict[str, Any]:
    width, height, params = pinhole_payload(sensor, source_image)
    fl_x, fl_y, cx, cy = params[:4]
    return {
        "camera_model": "PINHOLE",
        "w": int(width),
        "h": int(height),
        "fl_x": float(fl_x),
        "fl_y": float(fl_y),
        "cx": float(cx),
        "cy": float(cy),
    }


def _planned_undistorted_frame_payload(*, sensor: Any, source_image: Path, alpha: float) -> dict[str, Any]:
    width, height, params = pinhole_payload(sensor, source_image)
    matrix = np.array([[params[0], 0.0, params[2]], [0.0, params[1], params[3]], [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = distortion_coefficients(sensor)
    new_matrix, _roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (width, height), alpha, (width, height))
    return {
        "camera_model": "PINHOLE",
        "w": int(width),
        "h": int(height),
        "fl_x": float(new_matrix[0, 0]),
        "fl_y": float(new_matrix[1, 1]),
        "cx": float(new_matrix[0, 2]),
        "cy": float(new_matrix[1, 2]),
    }


def is_lichtfeld_nerf_target(*, axis_transform: object, final_orientation: object) -> bool:
    return (
        normalize_axis_transform(axis_transform) == AXIS_TRANSFORM_NONE
        and normalize_final_orientation(final_orientation) == FINAL_ORIENTATION_LICHTFELD
    )


def lichtfeld_nerf_incompatible_message(compatibility: MetashapeNerfCompatibility) -> str:
    return (
        "LichtFeld の NeRF JSON/PLY 読み込みはフレームごとのカメラ内部パラメータを扱えないため、"
        f"複数カメラ設定のMetashape結果は安全に出力できません "
        f"(camera groups: {compatibility.camera_group_count}, frames: {compatibility.frame_count})。"
        "LichtFeldではMetashape → COLMAPデータセットを使うか、単一解像度・単一PINHOLE設定にそろえてください。"
    )


def _asset_to_frame(asset: MetashapeDatasetAsset, output_root: Path) -> dict[str, Any]:
    fl_x, fl_y, cx, cy = asset.params[:4]
    frame: dict[str, Any] = {
        "file_path": asset.image_path.relative_to(output_root).as_posix(),
        "transform_matrix": asset.c2w.tolist(),
        "camera_model": asset.camera_model,
        "w": int(asset.width),
        "h": int(asset.height),
        "fl_x": float(fl_x),
        "fl_y": float(fl_y),
        "cx": float(cx),
        "cy": float(cy),
        "source_file_path": asset.source_image.as_posix(),
        "source_camera_id": asset.source_camera_id,
        "source_camera_label": asset.source_camera_label,
        "export_action": asset.action,
    }
    if asset.view_name:
        frame["view_name"] = asset.view_name
    if asset.mask_path is not None:
        frame["mask_path"] = asset.mask_path.relative_to(output_root).as_posix()
    return frame


def normalize_axis_transform(value: object) -> str:
    raw = str(value or AXIS_TRANSFORM_NONE).strip().lower().replace("_", "-")
    if raw in {"", "none", "off", "false", "no"}:
        return AXIS_TRANSFORM_NONE
    if raw in {"brush"}:
        return AXIS_TRANSFORM_BRUSH
    if raw in {"postshot", "default"}:
        return AXIS_TRANSFORM_POSTSHOT
    raise ValueError(f"Unsupported axis transform: {value}")


def axis_transform_matrix(value: object) -> np.ndarray:
    mode = normalize_axis_transform(value)
    if mode == AXIS_TRANSFORM_NONE:
        return np.eye(4, dtype=np.float64)
    axis = np.array(
        [
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if mode == AXIS_TRANSFORM_BRUSH:
        brush_rot = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        axis = brush_rot @ axis
    return axis


def dataset_world_transform(axis_transform: object, final_orientation: object) -> np.ndarray:
    matrix = axis_transform_matrix(axis_transform)
    orientation = normalize_final_orientation(final_orientation)
    if orientation != FINAL_ORIENTATION_NONE:
        matrix = final_orientation_matrix(orientation) @ matrix
    return matrix


def metashape_model_requires_mixed_nerf_writer(xml_path: str | Path) -> bool:
    model = parse_metashape_model(xml_path)
    camera_models = {sensor.camera_model for sensor in model.sensors.values()}
    sensor_sizes = {(sensor.width, sensor.height) for sensor in model.sensors.values()}
    return (
        camera_models != {CAMERA_MODEL_EQUIRECTANGULAR}
        or len(sensor_sizes) > 1
        or any(sensor.has_distortion for sensor in model.sensors.values())
    )


def metashape_model_uses_multiple_camera_signatures(
    *,
    scene_dir: str | Path,
    images_dir: str | Path,
    masks_dir: str | Path | None,
    xml_path: str | Path,
    views: list[dict[str, Any]],
    output_scale: float,
    fov_deg: float = DEFAULT_FOV_DEG,
    undistort_alpha: float = DEFAULT_UNDISTORT_ALPHA,
) -> bool:
    compatibility = analyze_metashape_nerf_compatibility(
        scene_dir=scene_dir,
        images_dir=images_dir,
        masks_dir=masks_dir,
        xml_path=xml_path,
        views=views,
        output_scale=output_scale,
        fov_deg=fov_deg,
        undistort_alpha=undistort_alpha,
    )
    return compatibility.camera_group_count > 1


def result_summary(result: MetashapeNerfExportResult) -> str:
    return json.dumps(
        {
            "output_dir": str(result.output_dir),
            "transforms_json": str(result.transforms_json),
            "pointcloud": str(result.pointcloud) if result.pointcloud else "",
            "frame_count": result.frame_count,
            "action_counts": result.action_counts,
            "warnings": list(result.warnings),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
