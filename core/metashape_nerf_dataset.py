from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.dataset_export_plan import (
    EXPORT_ACTION_EXPAND_ERP_TO_VIEWS,
    EXPORT_ACTION_LINK_PINHOLE,
    EXPORT_ACTION_UNDISTORT_FRAME_TO_PINHOLE,
    build_metashape_dataset_export_plan,
)
from core.dataset_writer_nerf import write_nerf_json_ply_dataset
from core.metashape_dataset_assets import (
    MetashapeDatasetAsset,
    expand_erp_to_view_assets,
    link_pinhole_asset,
    metashape_camera_to_world,
    metashape_pointcloud_matrix,
    resolve_inventory_path,
    undistort_frame_to_pinhole_asset,
)
from core.metashape_model import CAMERA_MODEL_EQUIRECTANGULAR, parse_metashape_model
from core.orientation_correction import (
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


@dataclass(frozen=True, slots=True)
class MetashapeNerfExportResult:
    output_dir: Path
    transforms_json: Path
    pointcloud: Path | None
    frame_count: int
    action_counts: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


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
) -> MetashapeNerfExportResult:
    scene = Path(scene_dir)
    images_root = Path(images_dir)
    masks_root = Path(masks_dir) if masks_dir else scene / "masks"
    output = Path(output_dir)
    output_images = output / "images"
    output_masks = output / "masks"
    output_images.mkdir(parents=True, exist_ok=True)

    model = parse_metashape_model(xml_path)
    inventory = build_scene_inventory(scene, images_dir=images_root, masks_dir=masks_root)
    plan = build_metashape_dataset_export_plan(model, inventory)
    world_transform = dataset_world_transform(axis_transform, final_orientation)

    camera_by_id = {camera.camera_id: camera for camera in model.cameras}
    frames: list[dict[str, Any]] = []
    action_counts: dict[str, int] = {}

    for item in plan.items:
        action_counts[item.action] = action_counts.get(item.action, 0) + 1
        camera = camera_by_id.get(item.camera_id)
        if camera is None or not item.image_rel_path:
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
            continue
        frames.extend(_asset_to_frame(asset, output) for asset in assets)

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
    return MetashapeNerfExportResult(
        output_dir=output,
        transforms_json=write_result.transforms_json,
        pointcloud=pointcloud_output or write_result.pointcloud,
        frame_count=write_result.frame_count,
        action_counts=action_counts,
        warnings=plan.warnings,
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
