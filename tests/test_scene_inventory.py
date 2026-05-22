from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from core.scene_inventory import (
    PROJECTION_EQUIRECTANGULAR,
    PROJECTION_NORMAL,
    build_scene_inventory,
)
from core.scene_layout import selected_frames_path


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
