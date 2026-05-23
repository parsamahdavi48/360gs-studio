from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from core.mask_source_scope import build_mask_source_options, filter_images_by_source, source_scope_key
from core.scene_inventory import build_scene_inventory
from core.scene_layout import selected_frames_path, source_image_sets_path


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 8), color=(120, 130, 140)).save(path)


def test_mask_source_options_use_selected_frame_video_labels(tmp_path: Path) -> None:
    scene = tmp_path
    _write_image(scene / "images" / "a_0001.jpg")
    _write_image(scene / "images" / "b_0001.jpg")
    csv_path = selected_frames_path(scene)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["output_file", "source_session", "source_video", "decision", "status"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "output_file": "images/a_0001.jpg",
                "source_session": "session_a",
                "source_video": "D:/source/walk_a.mp4",
                "decision": "keep",
                "status": "ok",
            }
        )
        writer.writerow(
            {
                "output_file": "images/b_0001.jpg",
                "source_session": "session_b",
                "source_video": "D:/source/walk_b.mp4",
                "decision": "keep",
                "status": "ok",
            }
        )

    inventory = build_scene_inventory(scene)
    options = build_mask_source_options(scene, inventory.images)

    assert [(option.key, option.label, option.image_count) for option in options] == [
        (source_scope_key("video_extract", "session_a"), "walk_a.mp4", 1),
        (source_scope_key("video_extract", "session_b"), "walk_b.mp4", 1),
    ]
    filtered = filter_images_by_source(inventory.images, source_scope_key("video_extract", "session_b"))
    assert [image.rel_path for image in filtered] == ["images/b_0001.jpg"]


def test_mask_source_options_use_image_set_source_folder_labels(tmp_path: Path) -> None:
    scene = tmp_path
    _write_image(scene / "images" / "still_0001.jpg")
    source_image_sets_path(scene).parent.mkdir(parents=True, exist_ok=True)
    source_image_sets_path(scene).write_text(
        json.dumps(
            {
                "version": 1,
                "image_sets": [
                    {
                        "id": "imageset_stills",
                        "source_type": "image_sequence",
                        "source_dir": "D:/source/still_take",
                        "files": [{"scene_path": "images/still_0001.jpg"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    inventory = build_scene_inventory(scene)
    options = build_mask_source_options(scene, inventory.images)

    assert [(option.key, option.label, option.image_count) for option in options] == [
        (source_scope_key("image_sequence", "imageset_stills"), "still_take", 1)
    ]
