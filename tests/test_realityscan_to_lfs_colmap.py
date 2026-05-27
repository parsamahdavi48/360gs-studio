from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from core.realityscan_to_lfs_colmap import (
    build_colmap_records,
    convert,
    lichtfeld_colmap_pointcloud_matrix,
)
from core.realityscan_to_transforms import read_realityscan_csv, transform_points


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
    Image.new("RGB", size, (64, 96, 128)).save(path)


def camera_center_from_colmap(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    r_cw = np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )
    return -r_cw.T @ tvec


def read_points3d_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points: list[list[float]] = []
    colors: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        points.append([float(parts[1]), float(parts[2]), float(parts[3])])
        colors.append([int(parts[4]), int(parts[5]), int(parts[6])])
    return np.asarray(points, dtype=np.float64), np.asarray(colors, dtype=np.uint8)


def test_build_colmap_records_preserves_mixed_image_names_and_camera_groups(tmp_path: Path) -> None:
    images = tmp_path / "images"
    write_image(images / "cube_px.jpg", (100, 100))
    write_image(images / "normal.png", (200, 100))
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

    rows = read_realityscan_csv(tmp_path / "rs.csv")
    cameras, colmap_images, missing = build_colmap_records(rows, images)

    assert missing == 0
    assert [image.name for image in colmap_images] == ["cube_px.jpg", "normal.png"]
    assert [camera.model for camera in cameras] == ["SIMPLE_PINHOLE", "FULL_OPENCV"]
    assert cameras[0].params == pytest.approx((50.0, 49.5, 49.5))
    assert cameras[1].params == pytest.approx((100.0, 100.0, 109.5, 29.5, 0.01, -0.02, 0.001, -0.002, 0.03, 0.04, 0.0, 0.0))

    first = colmap_images[0]
    assert first.camera_id == 1
    assert camera_center_from_colmap(first.qvec, first.tvec) == pytest.approx([1.0, -3.0, 2.0])


def test_build_colmap_records_can_keep_realityscan_world_without_lfs_rotation(tmp_path: Path) -> None:
    images = tmp_path / "images"
    write_image(images / "a.jpg", (100, 100))
    write_csv(tmp_path / "rs.csv", [{"#name": "a.jpg", "x": 1, "y": 2, "alt": 3, "f_35mm": 18}])

    rows = read_realityscan_csv(tmp_path / "rs.csv")
    _, colmap_images, _ = build_colmap_records(rows, images, camera_rotation_x_deg=0.0)

    first = colmap_images[0]
    assert camera_center_from_colmap(first.qvec, first.tvec) == pytest.approx([1.0, 2.0, 3.0])


def test_convert_writes_lfs_colmap_sparse_and_rotated_points3d_txt(tmp_path: Path) -> None:
    images = tmp_path / "images"
    write_image(images / "a.jpg", (80, 80))
    write_csv(tmp_path / "rs.csv", [{"#name": "a.jpg", "f_35mm": 18}])
    source_ply = tmp_path / "rs.ply"
    source_ply.write_bytes(
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

    result = convert(tmp_path / "rs.csv", tmp_path, images_dir=images, ply_path=source_ply)

    sparse = tmp_path / "sparse" / "0"
    assert result["sparse_dir"] == str(sparse)
    assert (sparse / "cameras.txt").is_file()
    assert (sparse / "images.txt").is_file()
    assert (sparse / "points3D.txt").is_file()
    assert not (sparse / "points3D.ply").exists()
    assert "1.0 0.0 0.0" not in (sparse / "cameras.txt").read_text(encoding="utf-8")
    assert (sparse / "images.txt").read_text(encoding="utf-8").splitlines()[4].endswith(" 1 a.jpg")

    points, colors = read_points3d_txt(sparse / "points3D.txt")
    expected = transform_points(
        np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]]),
        lichtfeld_colmap_pointcloud_matrix(),
    )
    assert np.allclose(points, expected)
    assert colors.tolist() == [[255, 0, 0], [0, 255, 0]]


def test_convert_rejects_lfs_json_marker_in_dataset_root(tmp_path: Path) -> None:
    images = tmp_path / "images"
    write_image(images / "a.jpg", (80, 80))
    write_csv(tmp_path / "rs.csv", [{"#name": "a.jpg", "f_35mm": 18}])
    (tmp_path / "transforms.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="transforms dataset marker"):
        convert(tmp_path / "rs.csv", tmp_path, images_dir=images)


def test_convert_dedicated_root_links_existing_images_and_masks(tmp_path: Path) -> None:
    source = tmp_path / "realityscan"
    images = source / "images"
    write_image(images / "_geometry" / "a.jpg", (80, 80))
    write_image(images / "_mask" / "a.png", (80, 80))
    write_csv(source / "rs.csv", [{"#name": "a.jpg", "f_35mm": 18}])
    (source / "transforms.json").write_text("{}", encoding="utf-8")

    output = source / "lfs_colmap"
    result = convert(source / "rs.csv", output)

    assert result["output_dir"] == str(output)
    assert (output / "images" / "a.jpg").is_file()
    assert (output / "masks" / "a.png").is_file()
    assert result["linked_assets"] == []
    assert result["asset_stats"]["linked_images"] == 1
    assert result["asset_stats"]["linked_masks"] == 1
    assert (output / "sparse" / "0" / "images.txt").read_text(encoding="utf-8").splitlines()[4].endswith(" 1 a.jpg")


def test_convert_merges_realityscan_images_and_extra_images_into_dataset_images(tmp_path: Path) -> None:
    source = tmp_path / "realityscan"
    write_image(source / "images" / "_geometry" / "cube_px.jpg", (64, 64))
    write_image(source / "extra_images" / "_geometry" / "extra_normal.jpg", (80, 60))
    write_image(source / "images" / "_mask" / "cube_px.png", (64, 64))
    write_image(source / "extra_images" / "_mask" / "extra_normal.png", (80, 60))
    write_csv(
        source / "rs.csv",
        [
            {"#name": "cube_px.jpg", "f_35mm": 18},
            {"#name": "extra_normal.jpg", "f_35mm": 24},
        ],
    )

    output = source / "lfs_colmap"
    result = convert(source / "rs.csv", output)

    assert result["num_images"] == 2
    assert (output / "images" / "cube_px.jpg").is_file()
    assert (output / "images" / "extra_normal.jpg").is_file()
    assert (output / "masks" / "cube_px.png").is_file()
    assert (output / "masks" / "extra_normal.png").is_file()
    image_lines = (output / "sparse" / "0" / "images.txt").read_text(encoding="utf-8").splitlines()
    assert image_lines[4].endswith(" 1 cube_px.jpg")
    assert image_lines[6].endswith(" 2 extra_normal.jpg")
    assert result["asset_stats"]["linked_images"] == 2
    assert result["asset_stats"]["linked_masks"] == 2


def test_convert_reports_row_based_progress(tmp_path: Path) -> None:
    source = tmp_path / "realityscan"
    write_image(source / "images" / "a.jpg", (64, 64))
    write_image(source / "images" / "b.jpg", (80, 60))
    write_csv(
        source / "rs.csv",
        [
            {"#name": "a.jpg", "f_35mm": 18},
            {"#name": "b.jpg", "f_35mm": 24},
        ],
    )
    progress: list[tuple[int, int]] = []

    convert(
        source / "rs.csv",
        source / "lfs_colmap",
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert progress == [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]


def test_convert_accepts_prefixed_extra_image_csv_names(tmp_path: Path) -> None:
    source = tmp_path / "realityscan"
    write_image(source / "images" / "_geometry" / "cube_px.jpg", (64, 64))
    write_image(source / "extra_images" / "_geometry" / "normal.jpg", (80, 60))
    write_csv(
        source / "rs.csv",
        [
            {"#name": "images/_geometry/cube_px.jpg", "f_35mm": 18},
            {"#name": "extra_images/_geometry/normal.jpg", "f_35mm": 24},
        ],
    )

    output = source / "lfs_colmap"
    result = convert(source / "rs.csv", output)

    assert result["num_images"] == 2
    assert (output / "images" / "cube_px.jpg").is_file()
    assert (output / "images" / "normal.jpg").is_file()


def test_convert_replaces_stale_materialized_dataset_assets(tmp_path: Path) -> None:
    source = tmp_path / "realityscan"
    write_image(source / "images" / "current.jpg", (64, 64))
    write_image(source / "masks" / "current.png", (64, 64))
    write_csv(source / "rs.csv", [{"#name": "current.jpg", "f_35mm": 18}])
    output = source / "lfs_colmap"
    write_image(output / "images" / "stale.jpg", (32, 32))
    write_image(output / "masks" / "stale.png", (32, 32))

    convert(source / "rs.csv", output)

    assert (output / "images" / "current.jpg").is_file()
    assert (output / "masks" / "current.png").is_file()
    assert not (output / "images" / "stale.jpg").exists()
    assert not (output / "masks" / "stale.png").exists()


def test_convert_can_pre_undistort_distorted_rows_as_pinhole_assets(tmp_path: Path) -> None:
    source = tmp_path / "realityscan"
    images = source / "images"
    masks = source / "masks"
    write_image(images / "distorted.png", (80, 60))
    write_image(images / "cube_px.jpg", (64, 64))
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[:, :40] = 255
    masks.mkdir(parents=True)
    assert cv2.imwrite(str(masks / "distorted.png"), mask)
    write_image(masks / "cube_px.png", (64, 64))
    write_csv(
        source / "rs.csv",
        [
            {"#name": "distorted.png", "f_35mm": 18, "k1": 0.08, "k2": -0.01, "t1": 0.001},
            {"#name": "cube_px.jpg", "f_35mm": 18},
        ],
    )

    output = source / "lfs_colmap_undistorted"
    result = convert(
        source / "rs.csv",
        output,
        pre_undistort_distorted_images=True,
    )

    cameras_text = (output / "sparse" / "0" / "cameras.txt").read_text(encoding="utf-8")
    assert " OPENCV " not in cameras_text
    assert " PINHOLE " in cameras_text
    assert " SIMPLE_PINHOLE " in cameras_text
    assert (output / "images" / "distorted.png").is_file()
    assert (output / "images" / "cube_px.jpg").is_file()
    assert (output / "masks" / "distorted.png").is_file()
    assert (output / "masks" / "cube_px.png").is_file()
    assert result["asset_stats"]["undistorted_images"] == 1
    assert result["asset_stats"]["undistorted_masks"] == 1
    assert result["asset_stats"]["linked_images"] == 1
    assert result["undistort_alpha"] == 1.0


def test_pre_undistort_alpha_generates_valid_masks_without_source_masks(tmp_path: Path) -> None:
    source = tmp_path / "realityscan"
    images = source / "images"
    write_image(images / "distorted.png", (80, 60))
    write_image(images / "cube_px.jpg", (64, 64))
    write_csv(
        source / "rs.csv",
        [
            {"#name": "distorted.png", "f_35mm": 18, "k1": 0.08, "k2": -0.01, "t1": 0.001},
            {"#name": "cube_px.jpg", "f_35mm": 18},
        ],
    )

    output = source / "lfs_colmap_undistorted"
    result = convert(
        source / "rs.csv",
        output,
        pre_undistort_distorted_images=True,
        undistort_alpha=1.0,
    )

    distorted_mask = cv2.imread(str(output / "masks" / "distorted.png"), cv2.IMREAD_GRAYSCALE)
    linked_mask = cv2.imread(str(output / "masks" / "cube_px.png"), cv2.IMREAD_GRAYSCALE)
    assert distorted_mask is not None
    assert linked_mask is not None
    assert distorted_mask.shape == (60, 80)
    assert linked_mask.shape == (64, 64)
    assert np.any(distorted_mask == 0)
    assert np.any(distorted_mask == 255)
    assert np.all(linked_mask == 255)
    assert result["asset_stats"]["generated_valid_masks"] == 2
    assert result["masks_dir"] == str(output / "masks")


def test_pre_undistort_alpha_remaps_source_mask_and_excludes_outside_pixels(tmp_path: Path) -> None:
    source = tmp_path / "realityscan"
    images = source / "images"
    masks = source / "masks"
    write_image(images / "distorted.png", (80, 60))
    masks.mkdir(parents=True)
    assert cv2.imwrite(str(masks / "distorted.png"), np.full((60, 80), 255, dtype=np.uint8))
    write_csv(
        source / "rs.csv",
        [{"#name": "distorted.png", "f_35mm": 18, "k1": 0.08, "k2": -0.01, "t1": 0.001}],
    )

    output = source / "lfs_colmap_undistorted"
    result = convert(
        source / "rs.csv",
        output,
        pre_undistort_distorted_images=True,
        undistort_alpha=1.0,
    )

    remapped_mask = cv2.imread(str(output / "masks" / "distorted.png"), cv2.IMREAD_GRAYSCALE)
    assert remapped_mask is not None
    assert np.any(remapped_mask == 0)
    assert np.any(remapped_mask == 255)
    assert result["asset_stats"]["undistorted_masks"] == 1
    assert result["asset_stats"]["generated_valid_masks"] == 0


def test_pre_undistort_handles_multiple_distorted_image_sizes(tmp_path: Path) -> None:
    source = tmp_path / "realityscan"
    images = source / "images"
    write_image(images / "normal_a.png", (80, 60))
    write_image(images / "normal_b.png", (120, 90))
    write_csv(
        source / "rs.csv",
        [
            {"#name": "normal_a.png", "f_35mm": 18, "k1": 0.08, "k2": -0.01},
            {"#name": "normal_b.png", "f_35mm": 18, "k1": -0.03, "t1": 0.001},
        ],
    )

    output = source / "lfs_colmap_undistorted"
    result = convert(
        source / "rs.csv",
        output,
        pre_undistort_distorted_images=True,
        undistort_alpha=1.0,
    )

    image_a = cv2.imread(str(output / "images" / "normal_a.png"), cv2.IMREAD_UNCHANGED)
    image_b = cv2.imread(str(output / "images" / "normal_b.png"), cv2.IMREAD_UNCHANGED)
    mask_a = cv2.imread(str(output / "masks" / "normal_a.png"), cv2.IMREAD_GRAYSCALE)
    mask_b = cv2.imread(str(output / "masks" / "normal_b.png"), cv2.IMREAD_GRAYSCALE)
    assert image_a is not None and image_a.shape[:2] == (60, 80)
    assert image_b is not None and image_b.shape[:2] == (90, 120)
    assert mask_a is not None and mask_a.shape == (60, 80)
    assert mask_b is not None and mask_b.shape == (90, 120)

    cameras_text = (output / "sparse" / "0" / "cameras.txt").read_text(encoding="utf-8")
    assert " PINHOLE 80 60 " in cameras_text
    assert " PINHOLE 120 90 " in cameras_text
    assert result["num_cameras"] == 2
    assert result["asset_stats"]["undistorted_images"] == 2
    assert result["asset_stats"]["generated_valid_masks"] == 2


def test_pre_undistort_rejects_mask_size_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "realityscan"
    images = source / "images"
    masks = source / "masks"
    write_image(images / "normal.png", (80, 60))
    masks.mkdir(parents=True)
    assert cv2.imwrite(str(masks / "normal.png"), np.full((32, 32), 255, dtype=np.uint8))
    write_csv(
        source / "rs.csv",
        [{"#name": "normal.png", "f_35mm": 18, "k1": 0.08}],
    )

    with pytest.raises(ValueError, match="Mask size must match"):
        convert(
            source / "rs.csv",
            source / "lfs_colmap_undistorted",
            pre_undistort_distorted_images=True,
            undistort_alpha=1.0,
        )
