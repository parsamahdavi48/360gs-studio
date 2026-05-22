from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.colmap_normal_camera_contract import (
    COLMAP_NORMAL_CAMERA_MODEL,
    normal_camera_groups_for_images,
)
from core.scene_inventory import build_scene_inventory
from core.scene_layout import source_image_sets_path


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
        ("normal/unknown_40x30_simple_radial", 2),
        ("normal/unknown_80x60_simple_radial", 1),
    ]
    assert {group.camera_model for group in groups} == {COLMAP_NORMAL_CAMERA_MODEL}


def test_normal_camera_group_uses_explicit_camera_metadata(tmp_path: Path) -> None:
    scene = tmp_path
    _write_image(scene / "images" / "normal.jpg", (40, 30))
    source_sets = source_image_sets_path(scene)
    source_sets.parent.mkdir(parents=True, exist_ok=True)
    source_sets.write_text(
        """
        {
          "version": 1,
          "image_sets": [
            {
              "id": "cam_a",
              "source_type": "image_sequence",
              "projection": "normal",
              "files": [
                {
                  "scene_path": "images/normal.jpg",
                  "camera": {
                    "model": "PINHOLE",
                    "params": [20.0, 21.0, 19.5, 14.5],
                    "source": "manual"
                  }
                }
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    inventory = build_scene_inventory(scene)

    group = normal_camera_groups_for_images(list(inventory.normal_images()))[0]

    assert group.camera_model == "PINHOLE"
    assert group.camera_params == (20.0, 21.0, 19.5, 14.5)
    assert group.camera_source == "manual"
    assert group.image_dir == "normal/cam_a_40x30_pinhole_20_21_19p5_14p5"
