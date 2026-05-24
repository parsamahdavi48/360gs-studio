from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.scene_inventory import build_scene_inventory
from core.sfm_input_plan import (
    SFM_ACTION_EXPAND_ERP_TO_RIG_VIEWS,
    SFM_ACTION_LINK_OR_COPY_NORMAL_IMAGE,
    build_colmap_mixed_sfm_input_plan,
    build_spheresfm_input_plan,
)


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 130, 140)).save(path)


def _write_mask(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=255).save(path)


def test_colmap_mixed_plan_splits_erp_and_normal_images(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "pano.jpg", (64, 32))
    _write_image(tmp_path / "images" / "normal.jpg", (40, 30))
    _write_mask(tmp_path / "masks" / "normal.png", (40, 30))

    inventory = build_scene_inventory(tmp_path)
    plan = build_colmap_mixed_sfm_input_plan(inventory)

    assert plan.ok
    assert [item.action for item in plan.items] == [
        SFM_ACTION_LINK_OR_COPY_NORMAL_IMAGE,
        SFM_ACTION_EXPAND_ERP_TO_RIG_VIEWS,
    ]
    normal_items = plan.items_for_action(SFM_ACTION_LINK_OR_COPY_NORMAL_IMAGE)
    assert normal_items[0].image_rel_path == "images/normal.jpg"
    assert normal_items[0].mask_rel_path == "masks/normal.png"


def test_spheresfm_plan_reuses_preflight_rules(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "pano.jpg", (64, 32))
    _write_image(tmp_path / "images" / "normal.jpg", (40, 30))

    inventory = build_scene_inventory(tmp_path)
    plan = build_spheresfm_input_plan(inventory)

    assert not plan.ok
    assert [issue.code for issue in plan.issues] == [
        "requires_equirectangular_only",
        "requires_single_resolution",
    ]
