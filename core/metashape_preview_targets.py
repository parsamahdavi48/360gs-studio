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
from core.metashape_model import parse_metashape_model
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
) -> MetashapePreviewTargets:
    scene = Path(scene_dir)
    images_root = Path(images_dir)
    masks_root = Path(masks_dir) if masks_dir else scene / "masks"
    model = parse_metashape_model(xml_path)
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
    try:
        return str(path.resolve()).casefold()
    except OSError:
        return str(path.absolute()).casefold()
