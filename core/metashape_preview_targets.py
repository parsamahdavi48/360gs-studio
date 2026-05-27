from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from core.dataset_export_plan import (
    EXPORT_ACTION_EXPAND_ERP_TO_VIEWS,
    EXPORT_ACTION_LINK_PINHOLE,
    EXPORT_ACTION_SKIP,
    EXPORT_ACTION_UNDISTORT_FRAME_TO_PINHOLE,
    build_metashape_dataset_export_plan,
)
from core.metashape_model import (
    CAMERA_MODEL_EQUIRECTANGULAR,
    CAMERA_MODEL_OPENCV,
    CAMERA_MODEL_PINHOLE,
    MetashapeCamera,
    parse_metashape_model,
)
from core.scene_import_contracts import IMAGE_EXTS
from core.scene_inventory import build_scene_inventory


@dataclass(frozen=True, slots=True)
class MetashapePreviewTargets:
    image_paths: tuple[Path, ...]
    equirect_paths: tuple[Path, ...]
    action_counts: dict[str, int]
    warnings: tuple[str, ...]


def build_metashape_preview_targets(
    *,
    scene_dir: str | Path,
    images_dir: str | Path,
    masks_dir: str | Path | None,
    xml_path: str | Path,
    strict: bool = False,
) -> MetashapePreviewTargets:
    scene = Path(scene_dir)
    images_root = Path(images_dir)
    masks_root = Path(masks_dir) if masks_dir else scene / "masks"
    model = parse_metashape_model(xml_path)
    if strict:
        inventory = build_scene_inventory(scene, images_dir=images_root, masks_dir=masks_root)
        plan = build_metashape_dataset_export_plan(model, inventory)
        images_by_rel = {image.rel_path: image.path for image in inventory.images}

        image_paths: list[Path] = []
        equirect_paths: list[Path] = []
        image_seen: set[str] = set()
        equirect_seen: set[str] = set()
        action_counts: Counter[str] = Counter()

        for item in plan.items:
            action_counts[item.action] += 1
            if item.action == EXPORT_ACTION_SKIP or not item.image_rel_path:
                continue
            image_path = images_by_rel.get(item.image_rel_path)
            if image_path is None or not image_path.is_file():
                continue
            image_key = _path_key(image_path)
            if image_key not in image_seen:
                image_seen.add(image_key)
                image_paths.append(image_path)
            if item.action == EXPORT_ACTION_EXPAND_ERP_TO_VIEWS and image_key not in equirect_seen:
                equirect_seen.add(image_key)
                equirect_paths.append(image_path)

        return MetashapePreviewTargets(
            image_paths=tuple(image_paths),
            equirect_paths=tuple(equirect_paths),
            action_counts=dict(action_counts),
            warnings=plan.warnings,
        )

    image_lookup, lookup_warnings = _build_image_lookup(scene, images_root)

    image_paths: list[Path] = []
    equirect_paths: list[Path] = []
    image_seen: set[str] = set()
    equirect_seen: set[str] = set()
    action_counts: Counter[str] = Counter()
    warnings: list[str] = list(lookup_warnings)

    for camera in model.cameras:
        sensor = model.sensor_for_camera(camera)
        action = _preview_action_for_camera_model(sensor.camera_model)
        image_path = _resolve_camera_image(camera, image_lookup)
        if image_path is None:
            action_counts[EXPORT_ACTION_SKIP] += 1
            warnings.append(f"Camera image not found in scene images: {camera.label or camera.camera_id}")
            continue
        action_counts[action] += 1
        if action == EXPORT_ACTION_SKIP:
            warnings.append(f"Unsupported Metashape camera model skipped: {sensor.camera_model} ({camera.label or camera.camera_id})")
            continue
        image_key = _path_key(image_path)
        if image_key not in image_seen:
            image_seen.add(image_key)
            image_paths.append(image_path)
        if action == EXPORT_ACTION_EXPAND_ERP_TO_VIEWS and image_key not in equirect_seen:
            equirect_seen.add(image_key)
            equirect_paths.append(image_path)

    return MetashapePreviewTargets(
        image_paths=tuple(image_paths),
        equirect_paths=tuple(equirect_paths),
        action_counts=dict(action_counts),
        warnings=tuple(warnings),
    )


def metashape_output_count_for_actions(
    action_counts: Mapping[str, int],
    *,
    enabled_view_count: int,
    direct_output: bool = False,
) -> int:
    expanded = max(0, int(action_counts.get(EXPORT_ACTION_EXPAND_ERP_TO_VIEWS, 0)))
    linked = max(0, int(action_counts.get(EXPORT_ACTION_LINK_PINHOLE, 0)))
    undistorted = max(0, int(action_counts.get(EXPORT_ACTION_UNDISTORT_FRAME_TO_PINHOLE, 0)))
    if direct_output:
        return expanded + linked + undistorted
    return expanded * max(0, int(enabled_view_count)) + linked + undistorted


def _path_key(path: Path) -> str:
    return str(path.absolute()).replace("\\", "/").casefold()


def _preview_action_for_camera_model(camera_model: str) -> str:
    if camera_model == CAMERA_MODEL_EQUIRECTANGULAR:
        return EXPORT_ACTION_EXPAND_ERP_TO_VIEWS
    if camera_model == CAMERA_MODEL_PINHOLE:
        return EXPORT_ACTION_LINK_PINHOLE
    if camera_model == CAMERA_MODEL_OPENCV:
        return EXPORT_ACTION_UNDISTORT_FRAME_TO_PINHOLE
    return EXPORT_ACTION_SKIP


def _build_image_lookup(scene: Path, images_root: Path) -> tuple[dict[str, Path], tuple[str, ...]]:
    grouped: dict[str, list[Path]] = {}
    for path in _iter_image_files(images_root):
        for key in _image_label_keys(scene, images_root, path):
            grouped.setdefault(key.casefold(), []).append(path)

    lookup: dict[str, Path] = {}
    warnings: list[str] = []
    for key, paths in grouped.items():
        unique = {_path_key(path): path for path in paths}
        if len(unique) == 1:
            lookup[key] = next(iter(unique.values()))
        else:
            warnings.append(f"Ambiguous image reference ignored: {key}")
    return lookup, tuple(warnings)


def _iter_image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
        key=lambda path: str(path).lower(),
    )


def _image_label_keys(scene: Path, images_root: Path, path: Path) -> set[str]:
    rel = ""
    keys = {path.name, path.stem}
    try:
        rel = path.relative_to(scene).as_posix()
        keys.add(rel)
        keys.add(Path(rel).name)
        keys.add(Path(rel).stem)
    except ValueError:
        pass
    try:
        rel_to_images = path.relative_to(images_root).as_posix()
        keys.add(rel_to_images)
        keys.add(Path(rel_to_images).name)
        keys.add(Path(rel_to_images).stem)
        if images_root.name:
            root_rel = (Path(images_root.name) / rel_to_images).as_posix()
            keys.add(root_rel)
            keys.add(Path(root_rel).name)
            keys.add(Path(root_rel).stem)
    except ValueError:
        pass
    if rel:
        parts = Path(rel).parts
        if parts and parts[0].casefold() == "images" and len(parts) > 1:
            without_images = Path(*parts[1:]).as_posix()
            keys.add(without_images)
            keys.add(Path(without_images).name)
            keys.add(Path(without_images).stem)
    return {key for key in keys if key}


def _resolve_camera_image(camera: MetashapeCamera, lookup: dict[str, Path]) -> Path | None:
    label = camera.label.replace("\\", "/").strip()
    if not label:
        return None
    candidates = [
        label.casefold(),
        Path(label).name.casefold(),
        Path(label).stem.casefold(),
    ]
    for key in candidates:
        image = lookup.get(key)
        if image is not None:
            return image
    return None
