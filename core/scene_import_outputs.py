from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from core.nerf_dataset_paths import find_nerf_pointcloud_path, find_nerf_transforms_path
from core.scene_import_contracts import (
    EXTERNAL_IMPORT_KIND,
    IssueSummary,
    SceneImportCancelToken,
    SceneImportOptions,
    import_origin,
    is_external_import_record,
)
from core.scene_import_sources import first_existing_mask, image_size, iter_scene_images
from core.scene_layout import (
    STEP4_SETTINGS_VERSION,
    scene_images_dir,
    scene_masks_dir,
    scene_metashape_3dgut_dir,
    scene_metashape_cubemap_dir,
    scene_output_dir,
    scene_spheresfm_3dgut_dir,
    scene_spheresfm_cubemap_dir,
    step4_dataset_runs_path,
    step4_export_settings_path,
)
from core.scene_project import file_identity, load_json, scene_relative, update_project, utc_now_iso, write_json


def inspect_output_dataset(
    scene: Path,
    warnings: list[str],
    *,
    options: SceneImportOptions | None = None,
    cancel_token: SceneImportCancelToken | None = None,
) -> dict[str, Any]:
    options = options or SceneImportOptions()
    output = _active_output_dataset_root(scene)
    root_label = scene_relative(scene, output)
    images = iter_scene_images(output / "images", cancel_token)
    masks = iter_scene_images(output / "masks", cancel_token)
    transforms = find_nerf_transforms_path(output)
    transforms_path = transforms or output / "transforms.json"
    pointcloud = find_nerf_pointcloud_path(output, transforms_json=transforms) if transforms is not None else None
    pointcloud_path = pointcloud or output / "pointcloud.ply"
    data: dict[str, Any] = {}
    camera_model = ""
    frames: list[Any] = []
    if transforms is not None and transforms.is_file():
        try:
            data = json.loads(transforms.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"{root_label}/{transforms.name} could not be read: {exc}")
        if isinstance(data, dict):
            camera_model = str(data.get("camera_model") or "")
            raw_frames = data.get("frames")
            frames = raw_frames if isinstance(raw_frames, list) else []
    if cancel_token is not None:
        cancel_token.check_cancelled()
    if transforms is None and output.exists() and (images or masks):
        warnings.append(f"{root_label}/ exists but {root_label}/transforms.json was not found.")

    output_shape = infer_output_shape(camera_model, images)
    dataset_kind = "3dgut" if output_shape == "equirect_3dgut" else ("projection_views" if output_shape else "")
    image_sample = sample_paths(images, options.output_validation_sample_limit)

    if images and output_shape == "projected":
        validate_projected_output_images(
            images,
            warnings,
            images_label=f"{root_label}/images",
            sample_paths=image_sample,
            cancel_token=cancel_token,
        )
    if frames:
        validate_transform_frames(output, frames, warnings, image_paths=images, cancel_token=cancel_token)
    if images and masks:
        validate_output_masks(
            output,
            images,
            warnings,
            masks_label=f"{root_label}/masks",
            sample_paths=image_sample,
            cancel_token=cancel_token,
        )
    if output_shape == "equirect_3dgut" and (pointcloud is None or not pointcloud.is_file()):
        warnings.append(f"3DGUT-style output was detected, but {root_label}/pointcloud.ply was not found.")

    return {
        "root": output,
        "active": bool(images or (transforms is not None and transforms.is_file())),
        "images": images,
        "masks": masks,
        "transforms_json": transforms_path,
        "pointcloud": pointcloud_path,
        "camera_model": camera_model,
        "frames_count": len(frames),
        "output_shape": output_shape,
        "dataset_kind": dataset_kind,
        "validation_sample_count": len(image_sample),
        "validation_sample_limit": options.output_validation_sample_limit,
    }


def infer_output_shape(camera_model: str, images: list[Path]) -> str:
    model = camera_model.strip().upper()
    if model == "EQUIRECTANGULAR":
        return "equirect_3dgut"
    if model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        return "projected"
    if images and all_square(images[:24]):
        return "projected"
    return ""


def _active_output_dataset_root(scene: Path) -> Path:
    legacy = scene_output_dir(scene)
    configured = _configured_output_dataset_root(scene)
    candidates = [
        configured,
        scene_metashape_cubemap_dir(scene),
        scene_metashape_3dgut_dir(scene),
        scene_spheresfm_cubemap_dir(scene),
        scene_spheresfm_3dgut_dir(scene),
        legacy,
    ]
    for candidate in (path for path in candidates if path is not None):
        if find_nerf_transforms_path(candidate) is not None:
            return candidate
    for candidate in (path for path in candidates if path is not None):
        if (candidate / "images").is_dir():
            return candidate
    return legacy


def _configured_output_dataset_root(scene: Path) -> Path | None:
    settings_path = step4_export_settings_path(scene)
    if not settings_path.is_file():
        return None
    settings = load_json(settings_path, {})
    if not isinstance(settings, dict):
        return None
    output_dir = str(settings.get("output_dir") or "").strip()
    if output_dir:
        path = Path(output_dir)
        return path if path.is_absolute() else scene / path
    portable = settings.get("portable_output")
    if isinstance(portable, dict):
        portable_root = str(portable.get("root") or "").strip()
        if portable_root:
            return scene / portable_root
    return None


def all_square(paths: list[Path]) -> bool:
    if not paths:
        return False
    for path in paths:
        size = image_size(path)
        if size is None or size[0] != size[1]:
            return False
    return True


def sample_paths(paths: list[Path], limit: int) -> list[Path]:
    if not paths or limit <= 0 or len(paths) <= limit:
        return list(paths)
    if limit == 1:
        return [paths[0]]
    indexes = sorted({round(i * (len(paths) - 1) / (limit - 1)) for i in range(limit)})
    return [paths[index] for index in indexes]


def validate_projected_output_images(
    images: list[Path],
    warnings: list[str],
    *,
    images_label: str = "output/images",
    sample_paths: list[Path] | None = None,
    cancel_token: SceneImportCancelToken | None = None,
) -> None:
    targets = sample_paths if sample_paths is not None else images
    sizes: set[tuple[int, int]] = set()
    non_square = IssueSummary(f"{images_label} non-square projected images")
    unreadable = IssueSummary(f"{images_label} unreadable images")
    for index, path in enumerate(targets, start=1):
        if cancel_token is not None and index % 64 == 0:
            cancel_token.check_cancelled()
        size = image_size(path)
        if size is None:
            unreadable.add(path.name)
            continue
        sizes.add(size)
        if size[0] != size[1]:
            non_square.add(path.name)
    if len(sizes) > 1:
        examples = ", ".join(f"{w}x{h}" for w, h in sorted(sizes)[:6])
        warnings.append(f"{images_label} has mixed image sizes: {examples}")
    for issue in (non_square, unreadable):
        message = issue.message()
        if message:
            warnings.append(message)


def validate_transform_frames(
    output: Path,
    frames: list[Any],
    warnings: list[str],
    *,
    image_paths: list[Path] | None = None,
    cancel_token: SceneImportCancelToken | None = None,
) -> None:
    known_images = {normalize_frame_path(scene_relative(output, path)) for path in image_paths or []}
    missing = IssueSummary("transforms.json references missing images")
    invalid = IssueSummary("transforms.json has invalid transform matrices")
    for index, frame in enumerate(frames, start=1):
        if cancel_token is not None and index % 256 == 0:
            cancel_token.check_cancelled()
        if not isinstance(frame, dict):
            invalid.add("<non-object frame>")
            continue
        file_path = str(frame.get("file_path") or "").strip()
        normalized = normalize_frame_path(file_path)
        if normalized and known_images and normalized not in known_images:
            missing.add(file_path)
        elif normalized and not known_images and not (output / file_path).is_file():
            missing.add(file_path)
        matrix = frame.get("transform_matrix")
        if not valid_transform_matrix(matrix):
            invalid.add(file_path or "<empty file_path>")
    for issue in (missing, invalid):
        message = issue.message()
        if message:
            warnings.append(message)


def valid_transform_matrix(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            return False
        for item in row:
            try:
                number = float(item)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(number):
                return False
    return True


def normalize_frame_path(path: str) -> str:
    value = path.replace("\\", "/").strip()
    while value.startswith("./"):
        value = value[2:]
    return value


def validate_output_masks(
    output: Path,
    images: list[Path],
    warnings: list[str],
    *,
    masks_label: str = "output/masks",
    sample_paths: list[Path] | None = None,
    cancel_token: SceneImportCancelToken | None = None,
) -> None:
    masks_root = output / "masks"
    images_root = output / "images"
    targets = sample_paths if sample_paths is not None else images
    missing = IssueSummary(f"{masks_label} missing matching files")
    mismatch = IssueSummary(f"{masks_label} size mismatch")
    for index, image_path in enumerate(targets, start=1):
        if cancel_token is not None and index % 64 == 0:
            cancel_token.check_cancelled()
        mask_path = first_existing_mask(image_path, images_root, masks_root)
        if mask_path is None:
            missing.add(scene_relative(output, image_path))
            continue
        source_size = image_size(image_path)
        mask_size = image_size(mask_path)
        if source_size is not None and mask_size is not None and source_size != mask_size:
            mismatch.add(f"{scene_relative(output, image_path)} -> {scene_relative(output, mask_path)}")
    for issue in (missing, mismatch):
        message = issue.message()
        if message:
            warnings.append(message)


def write_external_step4_settings(scene: Path, import_id: str, output_info: dict[str, Any]) -> None:
    output = Path(output_info.get("root") or scene_output_dir(scene))
    root_rel = scene_relative(scene, output)
    transforms = find_nerf_transforms_path(output)
    pointcloud = find_nerf_pointcloud_path(output, transforms_json=transforms) if transforms is not None else None
    output_shape = str(output_info.get("output_shape") or "")
    dataset_kind = str(output_info.get("dataset_kind") or "")
    active = bool(output_info.get("active"))
    settings = {
        "app": "stechdrive-3dgs-utils",
        "settings_version": STEP4_SETTINGS_VERSION,
        "created_at": utc_now_iso(),
        "scene_dir": str(scene),
        "output_dir": str(output),
        "origin": import_origin(import_id),
        "portable_output": {
            "root": root_rel,
            "dataset_kind": dataset_kind,
            "active": active,
        },
        "export_method": "metashape",
        "output_shape": output_shape,
        "target_profile": "lichtfeld" if output_shape == "equirect_3dgut" else "custom",
        "effective_profile": "lichtfeld" if output_shape == "equirect_3dgut" else "custom",
        "axis_transform": "none",
        "fov": 90.0,
        "image_size": {"label": "Imported", "scale": 1.0},
        "view_config": {
            "mode": "external_import",
            "yaw_offset": 0.0,
            "yaw_slots": 0,
            "pitch_rows": [],
            "pitch_rows_text": "",
            "cube6_drop_top": False,
            "cube6_drop_bottom": False,
            "views": [],
        },
        "views_config_path": "",
        "views_config_snapshot": None,
        "conversion": {
            "yaw_offset_per_frame": 0.0,
            "output_format": "auto",
            "output_bit_depth": "source",
            "jpg_quality": 95,
            "invert_masks": False,
            "write_images": False,
            "write_masks": False,
            "no_image": True,
            "uses_source_images": output_shape == "equirect_3dgut",
            "uses_source_masks": output_shape == "equirect_3dgut" and (output / "masks").is_dir(),
            "export_colmap": False,
        },
        "postprocess": {
            "final_orientation": "none",
            "final_orientation_stage": "none",
            "final_orientation_matrix": None,
            "lichtfeld_final_orientation_correction": False,
            "lichtfeld_final_orientation_stage": "none",
            "lichtfeld_final_orientation_matrix": None,
        },
        "metashape_import": {
            "enabled": False,
            "use_ply": False,
            "images_dir": str(scene_images_dir(scene)),
            "xml": "",
            "ply": "",
            "ply_approved": False,
            "scale": 1.0,
            "no_fix_rotation": True,
        },
        "colmap_rig": {"enabled": False, "run_sfm": False},
        "spheresfm": {"enabled": False, "run_scope": "convert_only"},
        "training": {
            "enabled": False,
            "backend": "lichtfeld",
            "executable": "",
            "dataset_root": str(output),
            "images_dir": str(output / "images"),
            "masks_dir": str(output / "masks"),
            "colmap_sparse_dir": "",
            "output_dir": str(output),
            "lichtfeld_config": "",
        },
        "inputs": {
            "transforms_json": str(transforms or output / "transforms.json"),
            "masks_dir": str(scene_masks_dir(scene)),
            "ply_source": str(pointcloud) if pointcloud is not None and pointcloud.is_file() else "",
        },
        "registered_assets": {
            "images_dir": f"{root_rel}/images" if (output / "images").is_dir() else "",
            "masks_dir": f"{root_rel}/masks" if (output / "masks").is_dir() else "",
            "transforms_json": f"{root_rel}/{transforms.name}" if transforms is not None and transforms.is_file() else "",
            "pointcloud": f"{root_rel}/{pointcloud.name}" if pointcloud is not None and pointcloud.is_file() else "",
        },
        "output_files": {
            "settings": "_stechdrive/step4/export_settings.json",
            "views_config": "",
            "transforms_json": transforms.name if transforms is not None and transforms.is_file() else "",
            "images_dir": "images" if (output / "images").is_dir() else "",
            "masks_dir": "masks" if (output / "masks").is_dir() else "",
            "pointcloud": pointcloud.name if pointcloud is not None and pointcloud.is_file() else "",
            "colmap_rig_dir": "colmap_rig",
        },
    }
    write_json(step4_export_settings_path(scene), settings)


def replace_external_dataset_run(scene: Path, import_id: str, output_info: dict[str, Any]) -> None:
    path = step4_dataset_runs_path(scene)
    data = load_json(path, {"version": 1, "runs": []})
    runs = data.get("runs")
    if not isinstance(runs, list):
        runs = []
    kept = [run for run in runs if not is_external_import_record(run)]
    if output_info.get("active"):
        run_id = f"dataset_{import_id}"
        root = Path(output_info.get("root") or scene_output_dir(scene))
        root_rel = scene_relative(scene, root)
        transforms = find_nerf_transforms_path(root)
        pointcloud = find_nerf_pointcloud_path(root, transforms_json=transforms) if transforms is not None else None
        output_shape = str(output_info.get("output_shape") or "")
        kept.append(
            {
                "id": run_id,
                "created_at": utc_now_iso(),
                "route": EXTERNAL_IMPORT_KIND,
                "output_shape": output_shape,
                "target_profile": "lichtfeld" if output_shape == "equirect_3dgut" else "custom",
                "dataset_root": root_rel,
                "origin": import_origin(import_id),
                "artifacts": {
                    "root": root_rel,
                    "transforms_json": file_identity(transforms or root / "transforms.json"),
                    "pointcloud": file_identity(pointcloud or root / "pointcloud.ply"),
                    "images_dir": file_identity(root / "images"),
                    "masks_dir": file_identity(root / "masks"),
                    "colmap_sparse_dir": file_identity(root / "sparse"),
                },
                "settings": load_json(step4_export_settings_path(scene), {}),
            }
        )
        update_project(scene, "step4", {"last_dataset_run_id": run_id})
    write_json(path, {"version": 1, "runs": kept[-200:]})
