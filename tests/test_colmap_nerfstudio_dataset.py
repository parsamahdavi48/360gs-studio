from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.colmap_nerfstudio_dataset import export_colmap_nerfstudio_dataset


def _write_sparse_text_model(
    sparse: Path,
    *,
    cameras: str | None = None,
    images: str | None = None,
) -> None:
    sparse.mkdir(parents=True, exist_ok=True)
    (sparse / "cameras.txt").write_text(
        cameras
        or "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "1 PINHOLE 10 12 5 6 4 5\n",
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        images
        or "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "1 1 0 0 0 0 0 0 1 rig1/cam01/frame_00001.jpg\n"
        "\n"
        "2 1 0 0 0 1 2 3 1 rig1/cam02/frame_00001.jpg\n"
        "\n",
        encoding="utf-8",
    )
    (sparse / "points3D.txt").write_text(
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"
        "10 1 2 3 4 5 6 0.1\n",
        encoding="utf-8",
    )


def _write_registered_images(root: Path) -> None:
    for rel in ("rig1/cam01/frame_00001.jpg", "rig1/cam02/frame_00001.jpg"):
        path = root / "images" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")


def _write_registered_masks(root: Path) -> None:
    for rel in ("rig1/cam01/frame_00001.jpg.png", "rig1/cam02/frame_00001.jpg.png"):
        path = root / "masks" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mask")


def test_export_colmap_rig_result_to_nerfstudio_json_ply_with_aligned_points(tmp_path: Path) -> None:
    colmap_root = tmp_path / "scene" / "output" / "colmap_rig"
    _write_sparse_text_model(colmap_root / "sparse" / "0")
    _write_registered_images(colmap_root)
    _write_registered_masks(colmap_root)

    result = export_colmap_nerfstudio_dataset(
        colmap_root=colmap_root,
        output_dir=tmp_path / "scene" / "output" / "colmap_nerfstudio",
    )

    data = json.loads(result.transforms_json.read_text(encoding="utf-8"))
    assert result.image_count == 2
    assert result.point_count == 1
    assert result.mask_count == 2
    assert data["camera_model"] == "OPENCV"
    assert data["applied_transform"] == [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0]]
    assert data["ply_file_path"] == "pointcloud.ply"

    frame = data["frames"][0]
    assert frame["file_path"] == "images/rig1/cam01/frame_00001.jpg"
    assert frame["mask_path"] == "masks/rig1/cam01/frame_00001.jpg.png"
    assert frame["w"] == 10
    assert frame["h"] == 12
    assert frame["fl_x"] == 5
    assert frame["fl_y"] == 6
    assert frame["cx"] == 4
    assert frame["cy"] == 5
    assert frame["k1"] == 0.0
    expected_identity_pose = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(np.array(frame["transform_matrix"]), expected_identity_pose)

    assert (result.output_dir / "images" / "rig1" / "cam01" / "frame_00001.jpg").is_file()
    assert (result.output_dir / "masks" / "rig1" / "cam01" / "frame_00001.jpg.png").is_file()
    assert "1 3 -2 4 5 6" in result.pointcloud.read_text(encoding="ascii")


def test_export_colmap_nerfstudio_rejects_mixed_projection_families(tmp_path: Path) -> None:
    colmap_root = tmp_path / "colmap_rig"
    _write_sparse_text_model(
        colmap_root / "sparse" / "0",
        cameras=(
            "1 PINHOLE 10 12 5 6 4 5\n"
            "2 OPENCV_FISHEYE 10 12 5 6 4 5 0.1 0.2 0.3 0.4\n"
        ),
        images=(
            "1 1 0 0 0 0 0 0 1 rig1/cam01/frame_00001.jpg\n\n"
            "2 1 0 0 0 0 0 0 2 rig1/cam02/frame_00001.jpg\n\n"
        ),
    )
    _write_registered_images(colmap_root)

    with pytest.raises(ValueError, match="mix incompatible models"):
        export_colmap_nerfstudio_dataset(
            colmap_root=colmap_root,
            output_dir=tmp_path / "out",
        )


def test_export_colmap_nerfstudio_rejects_partial_source_masks(tmp_path: Path) -> None:
    colmap_root = tmp_path / "colmap_rig"
    _write_sparse_text_model(colmap_root / "sparse" / "0")
    _write_registered_images(colmap_root)
    mask = colmap_root / "masks" / "rig1" / "cam01" / "frame_00001.jpg.png"
    mask.parent.mkdir(parents=True, exist_ok=True)
    mask.write_bytes(b"mask")

    with pytest.raises(ValueError, match="masks are incomplete"):
        export_colmap_nerfstudio_dataset(
            colmap_root=colmap_root,
            output_dir=tmp_path / "out",
        )
