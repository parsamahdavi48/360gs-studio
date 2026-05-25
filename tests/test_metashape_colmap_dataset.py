from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.metashape_colmap_dataset import (
    dataset_world_transform,
    export_metashape_colmap_dataset,
    metashape_model_requires_mixed_colmap_writer,
)
from core.metashape_coordinates import metashape_camera_matrix_to_output_world, metashape_pointcloud_matrix
from core.transforms_to_colmap import c2w_to_w2c

_IDENTITY = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"


def _write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _write_mask(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=255).save(path)


def _write_ply(path: Path) -> None:
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        "1 2 3\n",
        encoding="ascii",
    )


def _write_mixed_xml(path: Path) -> None:
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <sensors>
      <sensor id="0" type="spherical">
        <resolution width="64" height="32" />
      </sensor>
      <sensor id="1" type="frame">
        <resolution width="40" height="30" />
        <calibration><f>35</f><cx>0</cx><cy>0</cy></calibration>
      </sensor>
      <sensor id="2" type="frame">
        <resolution width="40" height="30" />
        <calibration><f>35</f><cx>0</cx><cy>0</cy><k1>0.05</k1></calibration>
      </sensor>
    </sensors>
    <cameras>
      <camera id="0" label="pano.jpg" sensor_id="0"><transform>{_IDENTITY}</transform></camera>
      <camera id="1" label="frame.jpg" sensor_id="1"><transform>{_IDENTITY}</transform></camera>
      <camera id="2" label="distorted.jpg" sensor_id="2"><transform>{_IDENTITY}</transform></camera>
    </cameras>
  </chunk>
</document>
""",
        encoding="utf-8",
    )


def _write_single_pinhole_xml(path: Path) -> None:
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <sensors>
      <sensor id="0" type="frame">
        <resolution width="40" height="30" />
        <calibration><f>35</f><cx>0</cx><cy>0</cy></calibration>
      </sensor>
    </sensors>
    <cameras>
      <camera id="0" label="frame.jpg" sensor_id="0"><transform>{_IDENTITY}</transform></camera>
    </cameras>
  </chunk>
</document>
""",
        encoding="utf-8",
    )


def test_export_metashape_colmap_dataset_expands_only_erp_and_undistorts_distorted(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_image(scene / "images" / "pano.jpg", (64, 32), (80, 120, 160))
    _write_image(scene / "images" / "frame.jpg", (40, 30), (10, 20, 30))
    _write_image(scene / "images" / "distorted.jpg", (40, 30), (40, 50, 60))
    _write_mask(scene / "masks" / "pano.png", (64, 32))
    _write_mask(scene / "masks" / "frame.png", (40, 30))
    xml = scene / "metashape.xml"
    ply = scene / "metashape.ply"
    _write_mixed_xml(xml)
    _write_ply(ply)

    result = export_metashape_colmap_dataset(
        scene_dir=scene,
        images_dir=scene / "images",
        masks_dir=scene / "masks",
        xml_path=xml,
        ply_path=ply,
        output_dir=scene / "output" / "metashape_colmap",
        views=[{"name": "pz", "yaw": 0.0, "pitch": 0.0}, {"name": "px", "yaw": 90.0, "pitch": 0.0}],
        output_scale=0.5,
        output_format="jpg",
    )

    output = scene / "output" / "metashape_colmap"
    assert result.image_count == 4
    assert (output / "images" / "pano_pz.jpg").is_file()
    assert (output / "images" / "pano_px.jpg").is_file()
    assert (output / "images" / "frame.jpg").is_file()
    assert (output / "images" / "distorted_undistorted.jpg").is_file()
    assert (output / "masks" / "pano_pz.png").is_file()
    assert (output / "masks" / "distorted_undistorted.png").is_file()
    assert (output / "sparse" / "0" / "cameras.txt").is_file()
    assert (output / "sparse" / "0" / "images.txt").read_text(encoding="utf-8").count("\n\n") == 4
    assert "# Number of points: 1" in (output / "sparse" / "0" / "points3D.txt").read_text(encoding="utf-8")
    assert (output / "sparse" / "0" / "points3D.ply").is_file()


def test_export_metashape_colmap_dataset_uses_explicit_image_root(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    source_images = tmp_path / "source_images"
    source_masks = tmp_path / "source_masks"
    _write_image(source_images / "frame.jpg", (40, 30), (10, 20, 30))
    _write_mask(source_masks / "frame.png", (40, 30))
    xml = scene / "metashape.xml"
    scene.mkdir()
    _write_single_pinhole_xml(xml)

    result = export_metashape_colmap_dataset(
        scene_dir=scene,
        images_dir=source_images,
        masks_dir=source_masks,
        xml_path=xml,
        ply_path=None,
        output_dir=scene / "output" / "metashape_colmap",
        views=[{"name": "pz", "yaw": 0.0, "pitch": 0.0}],
        output_scale=1.0,
        output_format="jpg",
    )

    output = scene / "output" / "metashape_colmap"
    assert result.image_count == 1
    assert (output / "images" / "frame.jpg").is_file()
    assert (output / "masks" / "frame.png").is_file()


def test_metashape_model_requires_mixed_writer_for_distortion(tmp_path: Path) -> None:
    xml = tmp_path / "mixed.xml"
    _write_mixed_xml(xml)

    assert metashape_model_requires_mixed_colmap_writer(xml)


def test_metashape_camera_transform_uses_pointcloud_basis_once() -> None:
    expected = metashape_pointcloud_matrix()
    expected[:, 1:3] *= -1.0
    expected = np.diag([-1.0, 1.0, -1.0, 1.0]) @ expected

    actual = metashape_camera_matrix_to_output_world(np.eye(4))

    assert np.allclose(actual, expected)


def test_metashape_colmap_camera_transform_matches_coordinate_contract() -> None:
    transform = np.array(
        [
            [0.96, -0.02, 0.28, 1.25],
            [0.10, 0.97, -0.21, -2.5],
            [-0.26, 0.23, 0.94, 3.75],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    expected = np.array(
        [
            [0.26, 0.23, 0.94, -3.75],
            [-0.10, 0.97, -0.21, 2.50],
            [-0.96, -0.02, 0.28, -1.25],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    assert np.allclose(metashape_camera_matrix_to_output_world(transform), expected)


def test_metashape_colmap_w2c_matches_transforms_to_colmap(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_image(scene / "images" / "pano.jpg", (64, 32), (80, 120, 160))
    xml = scene / "metashape.xml"
    _write_mixed_xml(xml)

    result = export_metashape_colmap_dataset(
        scene_dir=scene,
        images_dir=scene / "images",
        masks_dir=None,
        xml_path=xml,
        ply_path=None,
        output_dir=scene / "output" / "metashape_colmap",
        views=[{"name": "pz", "yaw": 0.0, "pitch": 0.0}],
        output_scale=0.5,
        output_format="jpg",
    )
    _ = result
    line = next(
        row
        for row in (scene / "output" / "metashape_colmap" / "sparse" / "0" / "images.txt").read_text().splitlines()
        if row and not row.startswith("#")
    )
    parts = line.split()
    qvec = np.array([float(value) for value in parts[1:5]])
    tvec = np.array([float(value) for value in parts[5:8]])
    c2w = metashape_camera_matrix_to_output_world(np.eye(4))
    expected_r, expected_t = c2w_to_w2c(c2w)

    # Compare translation directly and rotation via quaternion-generated matrix.
    assert np.allclose(tvec, expected_t)
    assert np.allclose(_rotation_from_quaternion(qvec), expected_r)


def test_metashape_colmap_dataset_world_transform_matches_nerf_route() -> None:
    expected = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    assert np.allclose(dataset_world_transform("none", "lichtfeld"), expected)


def test_export_metashape_colmap_dataset_rejects_empty_conversion(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images = scene / "images"
    images.mkdir(parents=True)
    xml = scene / "metashape.xml"
    _write_single_pinhole_xml(xml)

    with pytest.raises(ValueError, match="No Metashape cameras were converted"):
        export_metashape_colmap_dataset(
            scene_dir=scene,
            images_dir=images,
            masks_dir=None,
            xml_path=xml,
            ply_path=None,
            output_dir=scene / "output" / "metashape_colmap",
            views=[{"name": "pz", "yaw": 0.0, "pitch": 0.0}],
            output_scale=1.0,
            output_format="jpg",
        )


def _rotation_from_quaternion(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )
