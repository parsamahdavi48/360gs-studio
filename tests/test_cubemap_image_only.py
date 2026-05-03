import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def test_image_only_export_writes_view_images_without_transforms(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images = scene / "images"
    images.mkdir(parents=True)
    cv2.imwrite(str(images / "frame_0001.png"), np.zeros((32, 64, 3), dtype=np.uint8))

    views_path = tmp_path / "views.json"
    views_path.write_text(
        json.dumps({"views": [{"name": "front", "yaw": 0.0, "pitch": 0.0}]}),
        encoding="utf-8",
    )
    output = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "cubemap_transforms_json.py",
            str(scene),
            str(output),
            "--image-only",
            "--views-json",
            str(views_path),
            "--output_scale",
            "0.5",
            "--yaw-offset-per-frame",
            "0",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "images" / "frame_0001_front.png").is_file()
    assert not (output / "transforms.json").exists()

    settings = json.loads((output / "view_export_settings.json").read_text(encoding="utf-8"))
    assert settings["export_type"] == "image_only"
    assert settings["camera_model"] == "SIMPLE_PINHOLE"
    assert settings["output_size"] == {"w": 16, "h": 16}
    assert settings["source_images"] == ["frame_0001.png"]
    assert settings["export_images"] is True
    assert settings["export_masks"] is True


def test_image_only_export_can_write_masks_without_images(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    cv2.imwrite(str(images / "frame_0001.png"), np.zeros((32, 64, 3), dtype=np.uint8))
    cv2.imwrite(str(masks / "frame_0001.png"), np.zeros((32, 64), dtype=np.uint8))

    views_path = tmp_path / "views.json"
    views_path.write_text(
        json.dumps({"views": [{"name": "front", "yaw": 0.0, "pitch": 0.0}]}),
        encoding="utf-8",
    )
    output = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "cubemap_transforms_json.py",
            str(scene),
            str(output),
            "--image-only",
            "--skip-images",
            "--views-json",
            str(views_path),
            "--output_scale",
            "0.5",
            "--yaw-offset-per-frame",
            "0",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not (output / "images").exists()
    assert (output / "masks" / "frame_0001_front.png").is_file()
    settings = json.loads((output / "view_export_settings.json").read_text(encoding="utf-8"))
    assert settings["export_images"] is False
    assert settings["export_masks"] is True


def test_colmap_rig_export_writes_camera_folders_masks_and_rig_config(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    cv2.imwrite(str(images / "frame_0001.png"), np.zeros((32, 64, 3), dtype=np.uint8))
    cv2.imwrite(str(masks / "frame_0001.png"), np.full((32, 64), 255, dtype=np.uint8))

    views_path = tmp_path / "views.json"
    views_path.write_text(
        json.dumps(
            {
                "views": [
                    {"name": "front", "yaw": 0.0, "pitch": 0.0},
                    {"name": "right", "yaw": 90.0, "pitch": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "cubemap_transforms_json.py",
            str(scene),
            str(output),
            "--image-only",
            "--colmap-rig",
            "--views-json",
            str(views_path),
            "--output_scale",
            "0.5",
            "--yaw-offset-per-frame",
            "30",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    rig_root = output / "colmap_rig"
    assert (rig_root / "images" / "rig1" / "cam01" / "frame_00001.png").is_file()
    assert (rig_root / "images" / "rig1" / "cam02" / "frame_00001.png").is_file()
    assert (rig_root / "masks" / "rig1" / "cam01" / "frame_00001.png.png").is_file()
    assert (rig_root / "masks" / "rig1" / "cam02" / "frame_00001.png.png").is_file()

    rig_config = json.loads((rig_root / "rig_config.json").read_text(encoding="utf-8"))
    cameras = rig_config[0]["cameras"]
    assert len(cameras) == 2
    assert cameras[0]["image_prefix"] == "rig1/cam01/"
    assert cameras[0]["ref_sensor"] is True
    assert cameras[1]["image_prefix"] == "rig1/cam02/"
    assert cameras[1]["cam_from_rig_translation"] == [0.0, 0.0, 0.0]

    settings = json.loads((rig_root / "view_export_settings.json").read_text(encoding="utf-8"))
    assert settings["export_type"] == "colmap_rig"
    assert settings["yaw_offset_per_frame"] == 0.0
