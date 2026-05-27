from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from core.realityscan_layout import mask_lookup_candidates
from core.realityscan_to_transforms import (
    REALITYSCAN_MASK_DIR_NAMES,
    RealityScanCameraRow,
    realityscan_image_asset_relative_path,
    related_realityscan_asset_roots,
    resolve_image_path,
    row_has_distortion,
)

ACTION_LINK_OR_COPY_PINHOLE = "link_or_copy_pinhole"
ACTION_UNDISTORT_TO_PINHOLE = "undistort_to_pinhole"
ACTION_SKIP_MISSING_IMAGE = "skip_missing_image"


@dataclass(frozen=True, slots=True)
class RealityScanDatasetPlanItem:
    image_name: str
    action: str
    image_path: Path
    mask_path: Path | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class RealityScanDatasetPlan:
    items: tuple[RealityScanDatasetPlanItem, ...]
    issues: tuple[str, ...] = ()

    @property
    def action_counts(self) -> dict[str, int]:
        return dict(Counter(item.action for item in self.items))


def build_realityscan_lfs_dataset_plan(
    rows: list[RealityScanCameraRow],
    images_dir: Path,
    masks_dir: Path,
    *,
    pre_undistort_distorted_images: bool,
    skip_missing_images: bool,
) -> RealityScanDatasetPlan:
    items: list[RealityScanDatasetPlanItem] = []
    issues: list[str] = []
    for row in rows:
        image_path = resolve_image_path(images_dir, row.name)
        if not image_path.is_file():
            if skip_missing_images:
                items.append(
                    RealityScanDatasetPlanItem(
                        image_name=row.name,
                        action=ACTION_SKIP_MISSING_IMAGE,
                        image_path=image_path,
                        reason="missing_image",
                    )
                )
                continue
            issues.append(f"Image referenced by RealityScan CSV was not found: {image_path}")
            continue
        action = (
            ACTION_UNDISTORT_TO_PINHOLE
            if pre_undistort_distorted_images and row_has_distortion(row)
            else ACTION_LINK_OR_COPY_PINHOLE
        )
        items.append(
            RealityScanDatasetPlanItem(
                image_name=row.name,
                action=action,
                image_path=image_path,
                mask_path=find_matching_mask(masks_dir, image_name_for_dataset(image_path, images_dir)),
            )
        )
    return RealityScanDatasetPlan(items=tuple(items), issues=tuple(issues))


def image_name_for_dataset(image_path: Path, images_dir: Path) -> str:
    asset_rel = realityscan_image_asset_relative_path(image_path, images_dir)
    if asset_rel is not None:
        return asset_rel.as_posix()
    try:
        rel = image_path.resolve().relative_to(images_dir.resolve())
    except ValueError:
        rel = Path(image_path.name)
    return rel.as_posix()


def find_matching_mask(masks_dir: Path, image_name: str) -> Path | None:
    roots = tuple(root for root in related_realityscan_asset_roots(masks_dir, REALITYSCAN_MASK_DIR_NAMES) if root.is_dir())
    if not roots:
        return None
    candidates = mask_lookup_candidates(image_name)

    seen: set[str] = set()
    for rel in candidates:
        key = rel.as_posix().lower()
        if key in seen:
            continue
        seen.add(key)
        for root in roots:
            candidate = root / rel
            if candidate.is_file():
                return candidate
    return None
