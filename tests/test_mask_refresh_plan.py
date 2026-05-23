from __future__ import annotations

from pathlib import Path

from core.mask_refresh_plan import MASK_SCOPE_ALL, MASK_SCOPE_MISSING, MASK_SCOPE_STALE, build_mask_refresh_plan
from core.scene_project import write_mask_item


def test_mask_refresh_plan_missing_scope_targets_only_missing_masks(tmp_path: Path) -> None:
    scene = tmp_path
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir()
    masks.mkdir()
    existing = images / "existing.jpg"
    missing = images / "missing.jpg"
    existing.write_bytes(b"existing")
    missing.write_bytes(b"missing")
    (masks / "existing.png").write_bytes(b"mask")

    plan = build_mask_refresh_plan(
        scene_dir=scene,
        image_paths=[existing, missing],
        mask_path_for_image=lambda image: masks / f"{image.stem}.png",
        settings={"quality": "high"},
        scope=MASK_SCOPE_MISSING,
    )

    assert plan.targets == (missing,)
    assert plan.current == (existing,)
    assert plan.protected == ()


def test_mask_refresh_plan_stale_scope_protects_untracked_or_edited_masks(tmp_path: Path) -> None:
    scene = tmp_path
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir()
    masks.mkdir()
    stale = images / "stale.jpg"
    edited = images / "edited.jpg"
    unknown = images / "unknown.jpg"
    current = images / "current.jpg"
    settings = {"quality": "high"}
    for image in (stale, edited, unknown, current):
        image.write_bytes(b"image")
        (masks / f"{image.stem}.png").write_bytes(b"mask")
    write_mask_item(
        scene,
        image_path=stale,
        mask_path=masks / "stale.png",
        settings={"quality": "old"},
        run_id="old",
        stats={},
    )
    write_mask_item(
        scene,
        image_path=edited,
        mask_path=masks / "edited.png",
        settings=settings,
        run_id="edited",
        stats={},
    )
    write_mask_item(
        scene,
        image_path=current,
        mask_path=masks / "current.png",
        settings=settings,
        run_id="current",
        stats={},
    )
    (masks / "edited.png").write_bytes(b"manual edit")

    plan = build_mask_refresh_plan(
        scene_dir=scene,
        image_paths=[stale, edited, unknown, current],
        mask_path_for_image=lambda image: masks / f"{image.stem}.png",
        settings=settings,
        scope=MASK_SCOPE_STALE,
    )

    assert plan.targets == (stale,)
    assert plan.protected == (edited, unknown)
    assert plan.current == (current,)


def test_mask_refresh_plan_all_scope_overwrites_everything(tmp_path: Path) -> None:
    scene = tmp_path
    image = scene / "images" / "frame.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")

    plan = build_mask_refresh_plan(
        scene_dir=scene,
        image_paths=[image],
        mask_path_for_image=lambda path: scene / "masks" / f"{path.stem}.png",
        settings={},
        scope=MASK_SCOPE_ALL,
    )

    assert plan.targets == (image,)
    assert plan.current == ()
    assert plan.protected == ()
