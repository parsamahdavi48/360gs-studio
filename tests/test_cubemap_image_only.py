import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from core.transforms_to_colmap import read_ply_points


def _write_ascii_ply(path: Path, points: list[tuple[float, float, float]]) -> None:
    rows = "\n".join(f"{x} {y} {z}" for x, y, z in points)
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        f"{rows}\n",
        encoding="ascii",
    )


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
            "-m",
            "core.cubemap_transforms_json",
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

    assert not (output / "view_export_settings.json").exists()


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
            "-m",
            "core.cubemap_transforms_json",
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
    assert not (output / "view_export_settings.json").exists()


def test_image_only_export_fails_when_mask_worker_fails(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    cv2.imwrite(str(images / "frame_0001.png"), np.zeros((32, 64, 3), dtype=np.uint8))
    (masks / "frame_0001.png").write_bytes(b"not an image")

    views_path = tmp_path / "views.json"
    views_path.write_text(
        json.dumps({"views": [{"name": "front", "yaw": 0.0, "pitch": 0.0}]}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.cubemap_transforms_json",
            str(scene),
            str(tmp_path / "out"),
            "--image-only",
            "--views-json",
            str(views_path),
            "--output_scale",
            "0.5",
            "--workers",
            "1",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Cubemap conversion failed: 1 worker(s) failed" in output


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
            "-m",
            "core.cubemap_transforms_json",
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

    assert not (rig_root / "view_export_settings.json").exists()


def test_lfs_final_orientation_writes_oriented_transforms_and_pointcloud(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images = scene / "images"
    images.mkdir(parents=True)
    cv2.imwrite(str(images / "frame_0001.png"), np.zeros((32, 64, 3), dtype=np.uint8))
    _write_ascii_ply(scene / "pointcloud.ply", [(1.0, 2.0, 3.0)])
    (scene / "transforms.json").write_text(
        json.dumps(
            {
                "camera_model": "EQUIRECTANGULAR",
                "ply_file_path": "pointcloud.ply",
                "frames": [{"file_path": "images/frame_0001.png", "transform_matrix": np.eye(4).tolist()}],
            }
        ),
        encoding="utf-8",
    )
    views_path = tmp_path / "views.json"
    views_path.write_text(
        json.dumps({"views": [{"name": "front", "yaw": 0.0, "pitch": 0.0}]}),
        encoding="utf-8",
    )
    output = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.cubemap_transforms_json",
            str(scene),
            str(output),
            "--views-json",
            str(views_path),
            "--output_scale",
            "0.5",
            "--yaw-offset-per-frame",
            "0",
            "--no_transform",
            "--final-orientation",
            "lichtfeld",
            "--skip-images",
            "--skip-masks",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
    assert data["camera_model"] == "PINHOLE"
    expected = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(np.array(data["frames"][0]["transform_matrix"]), expected)
    assert data["ply_file_path"] == "pointcloud.ply"
    assert data["postprocess"]["final_orientation"] == "lichtfeld"
    assert data["postprocess"]["final_orientation_stage"] == "cubemap_cli"
    points, _colors = read_ply_points(output / "pointcloud.ply")
    assert np.allclose(points[0], [3.0, -2.0, 1.0])


def test_colmap_rig_export_fails_when_mask_worker_fails(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    cv2.imwrite(str(images / "frame_0001.png"), np.zeros((32, 64, 3), dtype=np.uint8))
    (masks / "frame_0001.png").write_bytes(b"not an image")

    views_path = tmp_path / "views.json"
    views_path.write_text(
        json.dumps({"views": [{"name": "front", "yaw": 0.0, "pitch": 0.0}]}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.cubemap_transforms_json",
            str(scene),
            str(tmp_path / "out"),
            "--image-only",
            "--colmap-rig",
            "--views-json",
            str(views_path),
            "--output_scale",
            "0.5",
            "--workers",
            "1",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "COLMAP rig image conversion failed: 1 worker(s) failed" in output
