from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.metashape_coordinates import metashape_camera_matrix_to_output_world
from core.metashape_nerf_dataset import export_metashape_nerf_dataset, metashape_model_requires_mixed_nerf_writer

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


def _write_single_spherical_xml(path: Path) -> None:
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <sensors>
      <sensor id="0" type="spherical">
        <resolution width="64" height="32" />
      </sensor>
    </sensors>
    <cameras>
      <camera id="0" label="pano.jpg" sensor_id="0"><transform>{_IDENTITY}</transform></camera>
    </cameras>
  </chunk>
</document>
""",
        encoding="utf-8",
    )


def test_export_metashape_nerf_dataset_expands_links_and_undistorts(tmp_path: Path) -> None:
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

    output = scene / "output" / "metashape_cubemap"
    result = export_metashape_nerf_dataset(
        scene_dir=scene,
        images_dir=scene / "images",
        masks_dir=scene / "masks",
        xml_path=xml,
        ply_path=ply,
        output_dir=output,
        views=[{"name": "pz", "yaw": 0.0, "pitch": 0.0}, {"name": "px", "yaw": 90.0, "pitch": 0.0}],
        output_scale=0.5,
        output_format="jpg",
        axis_transform="none",
        final_orientation="none",
    )

    data = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
    frames = data["frames"]
    assert result.frame_count == 4
    assert len(frames) == 4
    assert data["source"]["per_frame_intrinsics"] is True
    assert data["source"]["per_frame_camera_model"] is True
    assert result.action_counts == {
        "expand_erp_to_views": 1,
        "link_pinhole": 1,
        "undistort_frame_to_pinhole": 1,
    }
    assert (output / "images" / "pano_pz.jpg").is_file()
    assert (output / "images" / "pano_px.jpg").is_file()
    assert (output / "images" / "frame.jpg").is_file()
    assert (output / "images" / "distorted_undistorted.jpg").is_file()
    assert (output / "masks" / "pano_pz.png").is_file()
    assert (output / "masks" / "frame.png").is_file()
    assert (output / "masks" / "distorted_undistorted.png").is_file()
    assert (output / "pointcloud.ply").is_file()
    assert all(frame["camera_model"] == "PINHOLE" for frame in frames)
    assert {frame["file_path"] for frame in frames} == {
        "images/pano_pz.jpg",
        "images/pano_px.jpg",
        "images/frame.jpg",
        "images/distorted_undistorted.jpg",
    }


def test_metashape_nerf_camera_transform_matches_coordinate_contract() -> None:
    transform = np.array(
        [
            [0.36, -0.48, 0.80, 1.25],
            [0.80, 0.60, 0.00, -2.5],
            [-0.48, 0.64, 0.60, 3.75],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    expected = np.array(
        [
            [0.48, 0.64, 0.60, -3.75],
            [-0.80, 0.60, 0.00, 2.50],
            [-0.36, -0.48, 0.80, -1.25],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    assert np.allclose(metashape_camera_matrix_to_output_world(transform), expected)


def test_export_metashape_nerf_dataset_blocks_multicamera_lichtfeld_target(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_image(scene / "images" / "pano.jpg", (64, 32), (80, 120, 160))
    _write_image(scene / "images" / "frame.jpg", (40, 30), (10, 20, 30))
    xml = scene / "metashape.xml"
    _write_mixed_xml(xml)

    with pytest.raises(ValueError, match="LichtFeld"):
        export_metashape_nerf_dataset(
            scene_dir=scene,
            images_dir=scene / "images",
            masks_dir=None,
            xml_path=xml,
            ply_path=None,
            output_dir=scene / "output" / "metashape_cubemap",
            views=[{"name": "pz", "yaw": 0.0, "pitch": 0.0}],
            output_scale=0.5,
            output_format="jpg",
            axis_transform="none",
            final_orientation="lichtfeld",
        )

    assert not (scene / "output" / "metashape_cubemap" / "transforms.json").exists()


def test_metashape_nerf_writer_detection_flags_models_requiring_projected_output(tmp_path: Path) -> None:
    simple = tmp_path / "simple.xml"
    mixed = tmp_path / "mixed.xml"
    _write_single_spherical_xml(simple)
    _write_mixed_xml(mixed)

    assert not metashape_model_requires_mixed_nerf_writer(simple)
    assert metashape_model_requires_mixed_nerf_writer(mixed)
