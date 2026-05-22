from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from core.normal_camera_metadata import save_normal_camera_default
from core.scene_inventory import (
    PROJECTION_EQUIRECTANGULAR,
    PROJECTION_NORMAL,
    PROJECTION_UNKNOWN,
    build_scene_inventory,
)
from core.scene_layout import selected_frames_path, source_image_sets_path
from core.scene_project import scene_image_projection_map


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 130, 140)).save(path)


def _write_mask(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=255).save(path)


def test_scene_inventory_detects_projection_sizes_and_masks(tmp_path: Path) -> None:
    scene = tmp_path
    _write_image(scene / "images" / "pano.jpg", (64, 32))
    _write_image(scene / "images" / "normal.jpg", (40, 30))
    _write_mask(scene / "masks" / "pano.png", (64, 32))
    _write_mask(scene / "masks" / "normal.png", (40, 30))

    inventory = build_scene_inventory(scene)

    assert inventory.image_count == 2
    assert inventory.projection_counts[PROJECTION_EQUIRECTANGULAR] == 1
    assert inventory.projection_counts[PROJECTION_NORMAL] == 1
    assert inventory.image_sizes == {(64, 32), (40, 30)}
    assert inventory.missing_masks == ()
    assert inventory.mismatched_masks == ()


def test_scene_inventory_reports_mismatched_masks(tmp_path: Path) -> None:
    scene = tmp_path
    _write_image(scene / "images" / "frame.jpg", (64, 32))
    _write_mask(scene / "masks" / "frame.png", (32, 16))

    inventory = build_scene_inventory(scene)

    assert [image.rel_path for image in inventory.mismatched_masks] == ["images/frame.jpg"]
    assert inventory.images[0].mask is not None
    assert inventory.images[0].mask.matches_image_size is False


def test_scene_inventory_reads_selected_frame_source_metadata(tmp_path: Path) -> None:
    scene = tmp_path
    image = scene / "images" / "seq_0001.jpg"
    _write_image(image, (64, 32))
    csv_path = selected_frames_path(scene)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seq", "final_index", "output_file", "source_type", "source_session"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "seq": "1",
                "final_index": "7",
                "output_file": "images/seq_0001.jpg",
                "source_type": "image_sequence",
                "source_session": "import_a",
            }
        )

    inventory = build_scene_inventory(scene)

    assert inventory.images[0].source_kind == "image_sequence"
    assert inventory.images[0].source_id == "import_a"
    assert inventory.images[0].sequence_index == 7


def test_scene_projection_map_keeps_unreadable_images_unknown(tmp_path: Path) -> None:
    scene = tmp_path
    image = scene / "images" / "broken.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not an image")

    projection_map = scene_image_projection_map(scene, [image])

    assert projection_map["images/broken.jpg"] == PROJECTION_UNKNOWN


def test_scene_inventory_accepts_explicit_external_image_root(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    source_images = tmp_path / "source_images"
    source_masks = tmp_path / "source_masks"
    _write_image(source_images / "normal.jpg", (40, 30))
    _write_mask(source_masks / "normal.png", (40, 30))

    inventory = build_scene_inventory(scene, images_dir=source_images, masks_dir=source_masks)

    assert inventory.images_dir == source_images
    assert inventory.images[0].rel_path == "normal.jpg"
    assert inventory.images[0].projection == PROJECTION_NORMAL
    assert inventory.images[0].mask is not None


def test_scene_inventory_reads_source_image_camera_metadata(tmp_path: Path) -> None:
    scene = tmp_path
    _write_image(scene / "images" / "normal.jpg", (40, 30))
    path = source_image_sets_path(scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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

    assert inventory.images[0].camera_model == "PINHOLE"
    assert inventory.images[0].camera_params == (20.0, 21.0, 19.5, 14.5)
    assert inventory.images[0].camera_source == "manual"


def test_scene_inventory_applies_normal_camera_default_to_unannotated_normal_images(tmp_path: Path) -> None:
    scene = tmp_path
    _write_image(scene / "images" / "normal.jpg", (40, 30))
    _write_image(scene / "images" / "pano.jpg", (64, 32))
    save_normal_camera_default(
        scene,
        camera_model="PINHOLE",
        camera_params=(20.0, 21.0, 19.5, 14.5),
        camera_source="test_default",
    )

    inventory = build_scene_inventory(scene)
    images = {image.path.name: image for image in inventory.images}

    assert images["normal.jpg"].camera_model == "PINHOLE"
    assert images["normal.jpg"].camera_params == (20.0, 21.0, 19.5, 14.5)
    assert images["normal.jpg"].camera_source == "test_default"
    assert images["pano.jpg"].camera_model == ""
