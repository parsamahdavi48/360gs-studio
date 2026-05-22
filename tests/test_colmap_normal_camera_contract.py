from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.colmap_normal_camera_contract import (
    COLMAP_NORMAL_CAMERA_MODEL,
    normal_camera_groups_for_images,
)
from core.scene_inventory import build_scene_inventory


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 130, 140)).save(path)


def test_normal_camera_groups_are_source_and_resolution_specific(tmp_path: Path) -> None:
    scene = tmp_path
    _write_image(scene / "images" / "a" / "frame_0001.jpg", (40, 30))
    _write_image(scene / "images" / "a" / "frame_0002.jpg", (40, 30))
    _write_image(scene / "images" / "b" / "frame_0001.jpg", (80, 60))

    inventory = build_scene_inventory(scene)
    groups = normal_camera_groups_for_images(list(inventory.normal_images()))

    assert [(group.image_dir, group.image_count) for group in groups] == [
        ("normal/unknown_40x30", 2),
        ("normal/unknown_80x60", 1),
    ]
    assert {group.camera_model for group in groups} == {COLMAP_NORMAL_CAMERA_MODEL}
