from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from core.normal_camera_metadata import (
    load_normal_camera_default,
    save_normal_camera_default,
    save_normal_camera_group_default,
)
from core.scene_inventory import (
    PROJECTION_EQUIRECTANGULAR,
    PROJECTION_NORMAL,
    PROJECTION_UNKNOWN,
    build_scene_image_label_path_lookup,
    build_scene_image_label_path_lookup_with_warnings,
    build_scene_inventory,
    resolve_scene_image_label,
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


def test_scene_inventory_reuses_cache_until_scene_files_change(tmp_path: Path) -> None:
    scene = tmp_path
    _write_image(scene / "images" / "frame.jpg", (64, 32))

    first = build_scene_inventory(scene)
    second = build_scene_inventory(scene)

    assert second is first
    assert first.missing_masks

    _write_mask(scene / "masks" / "frame.png", (64, 32))

    third = build_scene_inventory(scene)

    assert third is not second
    assert third.missing_masks == ()


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


def test_scene_inventory_groups_images_by_registered_source(tmp_path: Path) -> None:
    scene = tmp_path
    _write_image(scene / "images" / "a" / "erp.jpg", (64, 32))
    _write_image(scene / "images" / "a" / "normal.jpg", (40, 30))
    _write_image(scene / "images" / "b" / "normal.jpg", (80, 60))
    path = source_image_sets_path(scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
        {
          "version": 1,
          "image_sets": [
            {
              "id": "set_a",
              "source_type": "image_sequence",
              "files": [
                {"scene_path": "images/a/erp.jpg", "projection": "equirectangular"},
                {"scene_path": "images/a/normal.jpg", "projection": "normal"}
              ]
            },
            {
              "id": "set_b",
              "source_type": "video",
              "files": [
                {"scene_path": "images/b/normal.jpg", "projection": "normal"}
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    inventory = build_scene_inventory(scene)
    groups = {(group.source_kind, group.source_id): group for group in inventory.source_groups()}

    assert groups[("image_sequence", "set_a")].image_count == 2
    assert groups[("image_sequence", "set_a")].projection_counts[PROJECTION_EQUIRECTANGULAR] == 1
    assert groups[("image_sequence", "set_a")].projection_counts[PROJECTION_NORMAL] == 1
    assert groups[("video", "set_b")].image_sizes == {(80, 60)}
    assert inventory.source_group("missing", "x") is None


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


def test_scene_image_label_lookup_accepts_external_root_relative_labels(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    source_images = tmp_path / "source_images"
    image = source_images / "cam_a" / "Frame_0001.webp"
    _write_image(image, (40, 30))

    lookup = build_scene_image_label_path_lookup(scene, images_dir=source_images)

    assert resolve_scene_image_label("cam_a/Frame_0001.webp", lookup) == image
    assert resolve_scene_image_label("source_images/cam_a/Frame_0001.webp", lookup) == image
    assert resolve_scene_image_label("Frame_0001", lookup) == image


def test_scene_image_label_lookup_ignores_ambiguous_external_basenames(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    source_images = tmp_path / "source_images"
    image_a = source_images / "cam_a" / "Frame_0001.webp"
    image_b = source_images / "cam_b" / "Frame_0001.webp"
    _write_image(image_a, (40, 30))
    _write_image(image_b, (40, 30))

    lookup, warnings = build_scene_image_label_path_lookup_with_warnings(scene, images_dir=source_images)

    assert resolve_scene_image_label("Frame_0001.webp", lookup) is None
    assert resolve_scene_image_label("Frame_0001", lookup) is None
    assert resolve_scene_image_label("cam_a/Frame_0001.webp", lookup) == image_a
    assert resolve_scene_image_label("source_images/cam_b/Frame_0001.webp", lookup) == image_b
    assert any("frame_0001.webp" in warning for warning in warnings)


def test_scene_image_label_lookup_resolves_nested_and_extensionless_labels(tmp_path: Path) -> None:
    scene = tmp_path
    image = scene / "images" / "cam_a" / "Frame_0001.webp"
    _write_image(image, (40, 30))

    lookup = build_scene_image_label_path_lookup(scene)

    assert resolve_scene_image_label("Frame_0001.webp", lookup) == image
    assert resolve_scene_image_label("frame_0001", lookup) == image
    assert resolve_scene_image_label("cam_a/Frame_0001.webp", lookup) == image
    assert resolve_scene_image_label("images/cam_a/Frame_0001.webp", lookup) == image
    assert resolve_scene_image_label("missing", lookup) is None


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


def test_scene_inventory_applies_normal_camera_group_default_before_scene_default(tmp_path: Path) -> None:
    scene = tmp_path
    _write_image(scene / "images" / "a.jpg", (40, 30))
    _write_image(scene / "images" / "b.jpg", (80, 60))
    save_normal_camera_default(
        scene,
        camera_model="SIMPLE_RADIAL",
        camera_params=(70.0, 39.5, 29.5, 0.01),
        camera_source="scene_default",
    )
    save_normal_camera_group_default(
        scene,
        source_kind="unknown",
        source_id="",
        width=40,
        height=30,
        camera_model="PINHOLE",
        camera_params=(20.0, 21.0, 19.5, 14.5),
        camera_source="group_default",
    )

    inventory = build_scene_inventory(scene)
    images = {image.path.name: image for image in inventory.images}

    assert images["a.jpg"].camera_model == "PINHOLE"
    assert images["a.jpg"].camera_params == (20.0, 21.0, 19.5, 14.5)
    assert images["a.jpg"].camera_source == "group_default"
    assert images["b.jpg"].camera_model == "SIMPLE_RADIAL"
    assert images["b.jpg"].camera_source == "scene_default"


def test_clearing_scene_normal_camera_default_keeps_group_defaults(tmp_path: Path) -> None:
    scene = tmp_path
    save_normal_camera_default(
        scene,
        camera_model="SIMPLE_RADIAL",
        camera_params=(70.0, 39.5, 29.5, 0.01),
        camera_source="scene_default",
    )
    save_normal_camera_group_default(
        scene,
        source_kind="unknown",
        source_id="",
        width=40,
        height=30,
        camera_model="PINHOLE",
        camera_params=(20.0, 21.0, 19.5, 14.5),
        camera_source="group_default",
    )

    save_normal_camera_default(scene, camera_model="")

    assert not load_normal_camera_default(scene).enabled
    _write_image(scene / "images" / "a.jpg", (40, 30))
    inventory = build_scene_inventory(scene)
    assert inventory.images[0].camera_model == "PINHOLE"
