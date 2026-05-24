import json
from pathlib import Path

import numpy as np

from core.spheresfm_to_transforms import convert, read_model


def _write_sparse_text_model(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "cameras.txt").write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "1 SPHERE 64 32 1 32 16\n",
        encoding="utf-8",
    )
    (path / "images.txt").write_text(
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "1 1 0 0 0 0 0 0 1 frame_0001.jpg\n"
        "\n"
        "2 1 0 0 0 1 2 3 1 nested/frame_0002.jpg\n"
        "\n",
        encoding="utf-8",
    )
    (path / "points3D.txt").write_text(
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"
        "10 1 2 3 4 5 6 0.1 1 1\n",
        encoding="utf-8",
    )


def test_spheresfm_reader_understands_sphere_camera_text_model(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse" / "0"
    _write_sparse_text_model(sparse)

    cameras, images, points, resolved = read_model(tmp_path / "sparse")

    assert resolved == sparse
    assert cameras[1].model == "SPHERE"
    assert cameras[1].params == (1.0, 32.0, 16.0)
    assert images[2].name == "nested/frame_0002.jpg"
    assert points[10].rgb == (4, 5, 6)


def test_spheresfm_convert_writes_equirect_transforms_and_pointcloud(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse" / "0"
    images = tmp_path / "images"
    (images / "nested").mkdir(parents=True)
    (images / "frame_0001.jpg").write_bytes(b"jpg")
    (images / "nested" / "frame_0002.jpg").write_bytes(b"jpg")
    _write_sparse_text_model(sparse)

    result = convert(
        tmp_path / "sparse",
        tmp_path / "out",
        images,
        image_path_mode="relative",
    )

    data = json.loads((tmp_path / "out" / "transforms.json").read_text(encoding="utf-8"))
    assert result["num_images"] == 2
    assert result["num_points"] == 1
    assert data["camera_model"] == "EQUIRECTANGULAR"
    assert data["w"] == 64
    assert data["h"] == 32
    assert data["frames"][0]["file_path"] == "frame_0001.jpg"
    assert data["frames"][1]["file_path"] == "nested/frame_0002.jpg"
    assert np.array(data["frames"][0]["transform_matrix"]).tolist() == np.diag([1, -1, -1, 1]).tolist()
    assert data["ply_file_path"] == "pointcloud.ply"
    assert "1 2 3 4 5 6" in (tmp_path / "out" / "pointcloud.ply").read_text(encoding="ascii")


def test_spheresfm_convert_can_write_paths_relative_to_output(tmp_path: Path) -> None:
    sparse = tmp_path / "scene" / "output" / "spheresfm" / "sparse" / "0"
    images = tmp_path / "scene" / "images"
    images.mkdir(parents=True)
    (images / "frame_0001.jpg").write_bytes(b"jpg")
    _write_sparse_text_model(sparse)

    convert(
        sparse,
        tmp_path / "scene" / "output" / "spheresfm" / "3dgut",
        images,
        image_path_mode="relative-to-output",
    )

    data = json.loads(
        (tmp_path / "scene" / "output" / "spheresfm" / "3dgut" / "transforms.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["frames"][0]["file_path"].replace("\\", "/") == "../../../images/frame_0001.jpg"
