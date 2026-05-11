from __future__ import annotations

import json
from pathlib import Path

import pytest

import core.apriltag_scale_apply as scale_apply
from core.apriltag_scale_apply import apply_scene_output_scale, validate_scale_output_dataset


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


def test_validate_scale_output_dataset_checks_transforms_and_sample_images(tmp_path: Path) -> None:
    transforms = _write_dataset(tmp_path)

    dataset = validate_scale_output_dataset(tmp_path)

    assert dataset.transforms_json == transforms
    assert dataset.pointcloud_ply == tmp_path / "output" / "pointcloud.ply"
    assert dataset.frame_count == 2
    assert dataset.checked_image_count == 2


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
