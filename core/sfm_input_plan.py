from __future__ import annotations

from dataclasses import dataclass

from core.projection_contract import PROJECTION_EQUIRECTANGULAR, PROJECTION_NORMAL
from core.scene_inventory import (
    SceneImage,
    SceneInventory,
)
from core.sfm_preflight import PreflightIssue, preflight_spheresfm

SFM_ACTION_EXPAND_ERP_TO_RIG_VIEWS = "expand_erp_to_rig_views"
SFM_ACTION_LINK_OR_COPY_NORMAL_IMAGE = "link_or_copy_normal_image"
SFM_ACTION_SKIP = "skip"


@dataclass(frozen=True, slots=True)
class SfmInputPlanItem:
    action: str
    image_rel_path: str
    projection: str
    mask_rel_path: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class SfmInputPlan:
    route: str
    items: tuple[SfmInputPlanItem, ...]
    issues: tuple[PreflightIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues

    def items_for_action(self, action: str) -> tuple[SfmInputPlanItem, ...]:
        return tuple(item for item in self.items if item.action == action)


def build_colmap_mixed_sfm_input_plan(inventory: SceneInventory) -> SfmInputPlan:
    items: list[SfmInputPlanItem] = []
    issues: list[PreflightIssue] = []
    for image in inventory.images:
        if image.projection == PROJECTION_EQUIRECTANGULAR:
            items.append(_item(SFM_ACTION_EXPAND_ERP_TO_RIG_VIEWS, image))
        elif image.projection == PROJECTION_NORMAL:
            items.append(_item(SFM_ACTION_LINK_OR_COPY_NORMAL_IMAGE, image))
        else:
            items.append(_item(SFM_ACTION_SKIP, image, reason="unknown_projection"))
            issues.append(
                PreflightIssue(
                    "unknown_projection",
                    f"COLMAP mixed SfM needs a known image type: {image.rel_path}",
                )
            )
    if not items:
        issues.append(PreflightIssue("no_images", "COLMAP mixed SfM requires images in images/."))
    return SfmInputPlan(route="colmap_mixed", items=tuple(items), issues=tuple(issues))


def build_spheresfm_input_plan(inventory: SceneInventory) -> SfmInputPlan:
    result = preflight_spheresfm(inventory)
    items = tuple(_item("run_spheresfm_source_image", image) for image in inventory.images)
    return SfmInputPlan(route="spheresfm", items=items, issues=result.issues)


def _item(action: str, image: SceneImage, *, reason: str = "") -> SfmInputPlanItem:
    mask_rel = image.mask.rel_path if image.mask is not None and image.mask.exists else ""
    return SfmInputPlanItem(
        action=action,
        image_rel_path=image.rel_path,
        projection=image.projection,
        mask_rel_path=mask_rel,
        reason=reason,
    )
