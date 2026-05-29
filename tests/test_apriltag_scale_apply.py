from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import core.apriltag_scale_apply as scale_apply
from core.apriltag_scale_apply import apply_scene_output_scale, validate_scale_output_dataset
from core.dataset_writer_colmap import ColmapCamera, ColmapImage, quaternion_from_matrix, write_colmap_text_dataset
from core.scene_layout import step4_export_settings_path


def _write_dataset(scene: Path, *, with_ply_file_path: bool = True) -> Path:
    output = scene / "output"
    images = output / "images"
    images.mkdir(parents=True)
    (images / "a.png").write_bytes(b"image-a")
    (images / "b.png").write_bytes(b"image-b")
    data = {
        "camera_model": "SIMPLE_PINHOLE",
        "w": 10,
        "h": 10,
        "fl_x": 5.0,
        "fl_y": 5.0,
        "cx": 4.5,
        "cy": 4.5,
        "frames": [
            {
                "file_path": "images/a.png",
                "transform_matrix": [[1, 0, 0, 2], [0, 1, 0, 4], [0, 0, 1, 6], [0, 0, 0, 1]],
            },
            {
                "file_path": "images/b.png",
                "transform_matrix": [[1, 0, 0, -2], [0, 1, 0, -4], [0, 0, 1, -6], [0, 0, 0, 1]],
            },
        ],
    }
    if with_ply_file_path:
        data["ply_file_path"] = "pointcloud.ply"
    transforms = output / "transforms.json"
    transforms.write_text(json.dumps(data), encoding="utf-8")
    (output / "pointcloud.ply").write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "1 2 3 10 20 30",
                "-1 -2 -3 40 50 60",
                "",
            ]
        ),
        encoding="ascii",
    )
    return transforms


def _write_colmap_dataset(scene: Path) -> Path:
    output = scene / "output" / "metashape_colmap"
    images = output / "images"
    images.mkdir(parents=True)
    (images / "a.png").write_bytes(b"image-a")
    (images / "b.png").write_bytes(b"image-b")
    cameras = [
        ColmapCamera(1, "PINHOLE", 10, 10, (5.0, 5.0, 4.5, 4.5)),
        ColmapCamera(2, "PINHOLE", 20, 10, (8.0, 8.0, 9.5, 4.5)),
    ]
    r_cw = np.eye(3)
    images_txt = [
        ColmapImage(1, quaternion_from_matrix(r_cw), np.array([0.0, 0.0, 0.0]), 1, "a.png"),
        ColmapImage(2, quaternion_from_matrix(r_cw), np.array([-2.0, 0.0, 0.0]), 2, "b.png"),
    ]
    write_colmap_text_dataset(output, cameras, images_txt)
    points = output / "sparse" / "0" / "points3D.txt"
    points.write_text(
        "\n".join(
            [
                "# points",
                "1 1 2 3 10 20 30 0.1",
                "2 -1 -2 -3 40 50 60 0.2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output


def test_validate_scale_output_dataset_checks_transforms_and_sample_images(tmp_path: Path) -> None:
    transforms = _write_dataset(tmp_path)

    dataset = validate_scale_output_dataset(tmp_path)

    assert dataset.transforms_json == transforms
    assert dataset.pointcloud_ply == tmp_path / "output" / "pointcloud.ply"
    assert dataset.frame_count == 2
    assert dataset.checked_image_count == 2


def test_validate_scale_output_dataset_prefers_configured_profile_json(tmp_path: Path) -> None:
    output = tmp_path / "output" / "metashape_cubemap"
    images = output / "images"
    images.mkdir(parents=True)
    (images / "a.png").write_bytes(b"image-a")
    base = {
        "camera_model": "SIMPLE_PINHOLE",
        "w": 10,
        "h": 10,
        "fl_x": 5.0,
        "fl_y": 5.0,
        "cx": 4.5,
        "cy": 4.5,
        "frames": [
            {
                "file_path": "images/a.png",
                "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            }
        ],
    }
    (output / "transforms_brush.json").write_text(json.dumps({**base, "profile_marker": "brush"}), encoding="utf-8")
    postshot = output / "transforms_postshot.json"
    postshot.write_text(json.dumps({**base, "profile_marker": "postshot"}), encoding="utf-8")
    settings = step4_export_settings_path(tmp_path)
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                "output_dir": str(output),
                "effective_profile": "postshot",
                "target_profile": "postshot",
                "output_files": {"transforms_json": "transforms_postshot.json"},
            }
        ),
        encoding="utf-8",
    )

    dataset = validate_scale_output_dataset(tmp_path, output_dir=output)

    assert dataset.transforms_json == postshot
    assert dataset.geometry_label == "transforms_postshot.json"


def test_validate_scale_output_dataset_rejects_equirectangular_output(tmp_path: Path) -> None:
    transforms = _write_dataset(tmp_path)
    data = json.loads(transforms.read_text(encoding="utf-8"))
    data["camera_model"] = "EQUIRECTANGULAR"
    transforms.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="projected Cubemap output"):
        validate_scale_output_dataset(tmp_path)


def test_apply_scene_output_scale_updates_transforms_and_pointcloud_with_backups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transforms = _write_dataset(tmp_path)
    monkeypatch.setattr(scale_apply, "_scale_pointcloud_with_open3d", lambda _path, _scale: None)

    result = apply_scene_output_scale(tmp_path, 0.5)

    data = json.loads(transforms.read_text(encoding="utf-8"))
    assert data["frames"][0]["transform_matrix"][0][3] == pytest.approx(1.0)
    assert data["frames"][0]["transform_matrix"][1][3] == pytest.approx(2.0)
    assert data["frames"][0]["transform_matrix"][2][3] == pytest.approx(3.0)
    assert data["frames"][1]["transform_matrix"][0][3] == pytest.approx(-1.0)
    assert result.transforms_backup.is_file()
    assert result.pointcloud_backup is not None
    assert result.pointcloud_backup.is_file()
    assert result.transforms_backup.parent.parent == tmp_path / "output"
    assert result.transforms_backup.parent.name.startswith("apriltag_scale_backup_")
    assert result.pointcloud_backup.parent == result.transforms_backup.parent
    assert not (tmp_path / "_stechdrive" / "step4" / "apriltag_scale_backups").exists()
    assert result.frames_scaled == 2
    assert result.points_scaled == 2

    ply_text = (tmp_path / "output" / "pointcloud.ply").read_text(encoding="ascii")
    assert "0.5 1 1.5 10 20 30" in ply_text
    assert "-0.5 -1 -1.5 40 50 60" in ply_text


def test_apply_scene_output_scale_fails_if_declared_pointcloud_is_missing(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    (tmp_path / "output" / "pointcloud.ply").unlink()

    with pytest.raises(ValueError, match="Point cloud"):
        apply_scene_output_scale(tmp_path, 0.5)


def test_validate_scale_output_dataset_accepts_colmap_dataset(tmp_path: Path) -> None:
    output = _write_colmap_dataset(tmp_path)

    dataset = validate_scale_output_dataset(tmp_path, output_dir=output)

    assert dataset.kind == "colmap"
    assert dataset.root == output
    assert dataset.sparse_dir == output / "sparse" / "0"
    assert dataset.images_dir == output / "images"
    assert dataset.estimation_input == output
    assert dataset.frame_count == 2
    assert dataset.checked_image_count == 2


def test_apply_scene_output_scale_updates_colmap_text_dataset_with_backups(
    tmp_path: Path,
) -> None:
    output = _write_colmap_dataset(tmp_path)
    sparse = output / "sparse" / "0"

    result = apply_scene_output_scale(tmp_path, 0.5, output_dir=output)

    images_lines = [
        line
        for line in (sparse / "images.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    second = images_lines[1].split()
    assert float(second[5]) == pytest.approx(-1.0)
    assert result.kind == "colmap"
    assert result.geometry_label == "COLMAP images.txt"
    assert result.transforms_backup.is_file()
    assert result.pointcloud_backup is not None
    assert result.pointcloud_backup.is_file()
    assert result.frames_scaled == 2
    assert result.points_scaled == 2
    points_text = (sparse / "points3D.txt").read_text(encoding="utf-8")
    assert "1 0.5 1 1.5 10 20 30 0.1" in points_text
    assert "2 -0.5 -1 -1.5 40 50 60 0.2" in points_text
