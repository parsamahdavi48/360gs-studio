from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.pointcloud_io import load_point_cloud_sample
from core.realityscan_to_transforms import (
    REALITYSCAN_PLY_TO_LICHTFELD_FILE_MATRIX,
    TARGET_PROFILE_LICHTFELD,
    TARGET_PROFILE_REALITYSCAN,
    convert,
    pointcloud_target_profile_matrix,
    realityscan_rotation_matrix,
    target_profile_matrix,
    transform_points,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "#name",
        "x",
        "y",
        "alt",
        "yaw",
        "pitch",
        "roll",
        "f_35mm",
        "px_norm",
        "py_norm",
        "k1",
        "k2",
        "k3",
        "k4",
        "t1",
        "t2",
    ]
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(str(row.get(field, 0)) for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (16, 32, 48)).save(path)


def test_realityscan_rotation_identity() -> None:
    assert np.allclose(realityscan_rotation_matrix(0.0, 0.0, 0.0), np.eye(3))


def test_target_profile_matrix_maps_realityscan_to_lichtfeld() -> None:
    point = np.array([[1.0, 2.0, 3.0]])
    mapped = transform_points(point, target_profile_matrix(TARGET_PROFILE_LICHTFELD))
    assert np.allclose(mapped[0], [-1.0, 3.0, 2.0])
    assert np.allclose(target_profile_matrix(TARGET_PROFILE_REALITYSCAN), np.eye(4))


def test_convert_keeps_mixed_cubemap_and_normal_images(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    masks_dir = images_dir / "_mask"
    write_image(images_dir / "_geometry" / "cube_px.jpg", (100, 100))
    write_image(images_dir / "_geometry" / "normal.png", (200, 100))
    masks_dir.mkdir(parents=True)
    Image.new("L", (100, 100), 255).save(masks_dir / "cube_px.png")

    source = {
        "camera_model": "PINHOLE",
        "w": 100,
        "h": 100,
        "fl_x": 50.0,
        "fl_y": 50.0,
        "cx": 49.5,
        "cy": 49.5,
        "frames": [
            {
                "file_path": "images/cube_px.jpg",
                "source_file_path": "images/source.jpg",
                "source_image_index": 7,
                "view_name": "px",
                "view_index": 1,
                "yaw_offset_deg": 90.0,
                "transform_matrix": np.eye(4).tolist(),
            }
        ],
    }
    (tmp_path / "transforms.json").write_text(json.dumps(source), encoding="utf-8")
    write_csv(
        tmp_path / "rs.csv",
        [
            {"#name": "cube_px.jpg", "x": 1, "y": 2, "alt": 3, "f_35mm": 18},
            {
                "#name": "normal.png",
                "x": 4,
                "y": 5,
                "alt": 6,
                "f_35mm": 36,
                "px_norm": 0.1,
                "py_norm": -0.2,
                "k1": 0.01,
                "k2": -0.02,
                "k3": 0.03,
                "k4": 0.04,
                "t1": 0.001,
                "t2": -0.002,
            },
        ],
    )

    result = convert(
        tmp_path / "rs.csv",
        tmp_path,
        images_dir=images_dir,
        masks_dir=masks_dir,
        target_profile=TARGET_PROFILE_LICHTFELD,
    )

    assert result["num_frames"] == 2
    assert result["num_opencv_frames"] == 1
    data = json.loads((tmp_path / "transforms.json").read_text(encoding="utf-8"))
    assert data["camera_model"] == "PINHOLE"
    assert data["w"] == 100
    assert data["source"]["camera_group_count"] == 2
    assert len(data["frames"]) == 2

    cube, normal = data["frames"]
    assert cube["file_path"] == "images/_geometry/cube_px.jpg"
    assert cube["camera_model"] == "PINHOLE"
    assert cube["fl_x"] == 50.0
    assert cube["source_image_index"] == 7
    assert cube["mask_path"] == "images/_mask/cube_px.png"
    assert np.allclose(np.array(cube["transform_matrix"])[:3, 3], [-1.0, 3.0, 2.0])

    assert normal["file_path"] == "images/_geometry/normal.png"
    assert normal["camera_model"] == "OPENCV"
    assert normal["w"] == 200
    assert normal["h"] == 100
    assert normal["fl_x"] == pytest.approx(100.0)
    assert normal["cx"] == pytest.approx(109.5)
    assert normal["cy"] == pytest.approx(29.5)
    assert normal["k1"] == pytest.approx(0.01)
    assert normal["p1"] == pytest.approx(0.001)
    assert np.allclose(np.array(normal["transform_matrix"])[:3, 3], [-4.0, 6.0, 5.0])


def test_convert_resolves_sibling_extra_images_without_rewriting_path_to_images(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    extra_images_dir = tmp_path / "extra_images"
    write_image(images_dir / "_geometry" / "cube_px.jpg", (100, 100))
    write_image(extra_images_dir / "_geometry" / "normal.jpg", (200, 100))
    write_csv(
        tmp_path / "rs.csv",
        [
            {"#name": "cube_px.jpg", "f_35mm": 18},
            {"#name": "normal.jpg", "f_35mm": 36},
        ],
    )

    convert(
        tmp_path / "rs.csv",
        tmp_path / "out",
        images_dir=images_dir,
        target_profile=TARGET_PROFILE_REALITYSCAN,
    )

    data = json.loads((tmp_path / "out" / "transforms.json").read_text(encoding="utf-8"))
    assert [frame["file_path"] for frame in data["frames"]] == [
        "images/_geometry/cube_px.jpg",
        "extra_images/_geometry/normal.jpg",
    ]


def test_convert_writes_relative_paths_to_external_output(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images_dir = scene / "images"
    write_image(images_dir / "normal.png", (80, 40))
    write_csv(scene / "rs.csv", [{"#name": "normal.png", "f_35mm": 18}])

    output = tmp_path / "out"
    convert(
        scene / "rs.csv",
        output,
        images_dir=images_dir,
        image_path_mode="relative-to-output",
        target_profile=TARGET_PROFILE_REALITYSCAN,
    )
    data = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
    assert data["frames"][0]["file_path"] == "../scene/images/normal.png"


def test_write_transformed_binary_ply(tmp_path: Path) -> None:
    source = tmp_path / "points.ply"
    source.write_bytes(
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 2\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"property uchar red\n"
        b"property uchar green\n"
        b"property uchar blue\n"
        b"end_header\n"
        + struct.pack("<fffBBB", 1.0, 2.0, 3.0, 255, 0, 0)
        + struct.pack("<fffBBB", -4.0, 5.0, -6.0, 0, 255, 0)
    )
    images_dir = tmp_path / "images"
    write_image(images_dir / "normal.png", (80, 40))
    write_csv(tmp_path / "rs.csv", [{"#name": "normal.png", "f_35mm": 18}])

    convert(
        tmp_path / "rs.csv",
        tmp_path / "out",
        images_dir=images_dir,
        ply_path=source,
        target_profile=TARGET_PROFILE_LICHTFELD,
    )

    sample = load_point_cloud_sample(tmp_path / "out" / "pointcloud.ply")
    expected = transform_points(
        np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]]),
        REALITYSCAN_PLY_TO_LICHTFELD_FILE_MATRIX,
    )
    assert np.allclose(sample.points, expected)
    assert sample.colors is not None
    assert sample.colors.tolist() == [[255, 0, 0], [0, 255, 0]]


def test_lichtfeld_pointcloud_matrix_matches_loader_camera_frame() -> None:
    raw_point = np.array([[1.0, 2.0, 3.0]])
    lfs_file_point = transform_points(raw_point, pointcloud_target_profile_matrix(TARGET_PROFILE_LICHTFELD))
    lfs_internal_point = lfs_file_point.copy()
    lfs_internal_point[:, 1:3] *= -1.0
    assert np.allclose(lfs_internal_point[0], [1.0, -3.0, 2.0])
