from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.metashape_coordinates import metashape_pointcloud_file_matrix
from core.metashape_preprocess import export_metashape_equirectangular_dataset
from metashape_360_lfs import convert_metashape_to_lichtfeld
from scripts.run_workflow_job import _run_metashape_preprocess
from vendor.metashape_360_lfs.metashape_360_lfs import transform_camera_matrix as legacy_transform_camera_matrix

_IDENTITY = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
_ROTATED = "0.36 -0.48 0.80 1.25 0.80 0.60 0.00 -2.5 -0.48 0.64 0.60 3.75 0 0 0 1"


def _write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _write_ply(path: Path) -> None:
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
        "1 2 3 10 20 30\n",
        encoding="ascii",
    )


def _write_spherical_xml(path: Path, *, include_frame_sensor: bool = False) -> None:
    extra_sensor = ""
    extra_camera = ""
    if include_frame_sensor:
        extra_sensor = """
      <sensor id="2" type="frame">
        <resolution width="40" height="30" />
        <calibration><f>35</f><cx>0</cx><cy>0</cy></calibration>
      </sensor>"""
        extra_camera = f"""
      <camera id="2" label="frame.jpg" sensor_id="2"><transform>{_IDENTITY}</transform></camera>"""
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <sensors>
      <sensor id="0" type="spherical">
        <resolution width="64" height="32" />
      </sensor>
      <sensor id="1" type="spherical">
        <resolution width="80" height="40" />
      </sensor>{extra_sensor}
    </sensors>
    <cameras>
      <camera id="0" label="wide/pano64.jpg" sensor_id="0"><transform>{_ROTATED}</transform></camera>
      <camera id="1" label="pano80.jpg" sensor_id="1"><transform>{_IDENTITY}</transform></camera>{extra_camera}
    </cameras>
  </chunk>
</document>
""",
        encoding="utf-8",
    )


def test_export_metashape_preprocess_supports_multiple_spherical_resolutions(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images = scene / "images"
    output = scene / "output" / "metashape_work"
    _write_image(images / "wide" / "pano64.jpg", (64, 32), (80, 120, 160))
    _write_image(images / "pano80.jpg", (80, 40), (10, 20, 30))
    xml = scene / "cameras.xml"
    ply = scene / "points.ply"
    _write_spherical_xml(xml)
    _write_ply(ply)

    result = export_metashape_equirectangular_dataset(
        images_dir=images,
        xml_path=xml,
        output_dir=output,
        ply_path=ply,
        scale=2.0,
    )

    data = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
    assert result.num_frames == 2
    assert result.num_skipped == 0
    assert data["camera_model"] == "EQUIRECTANGULAR"
    assert data["applied_transform"] == metashape_pointcloud_file_matrix(scale=1.0)[:3, :].tolist()
    assert {frame["file_path"] for frame in data["frames"]} == {
        "images/wide/pano64.jpg",
        "images/pano80.jpg",
    }
    assert {(frame["w"], frame["h"]) for frame in data["frames"]} == {(64, 32), (80, 40)}
    assert (output / "pointcloud.ply").is_file()
    pointcloud_text = (output / "pointcloud.ply").read_text(encoding="ascii")
    assert "property uchar red" in pointcloud_text
    assert "10 20 30" in pointcloud_text


def test_metashape_preprocess_camera_transform_matches_vendor_reference(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images = scene / "images"
    output = scene / "output"
    _write_image(images / "wide" / "pano64.jpg", (64, 32), (80, 120, 160))
    _write_image(images / "pano80.jpg", (80, 40), (10, 20, 30))
    xml = scene / "cameras.xml"
    _write_spherical_xml(xml)

    export_metashape_equirectangular_dataset(images_dir=images, xml_path=xml, output_dir=output)

    data = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
    first = next(frame for frame in data["frames"] if frame["file_path"] == "images/wide/pano64.jpg")
    expected = legacy_transform_camera_matrix(
        np.array([float(value) for value in _ROTATED.split()], dtype=np.float64).reshape((4, 4)),
        fix_upside_down=True,
    )
    assert np.allclose(np.array(first["transform_matrix"], dtype=np.float64), expected)


def test_workflow_job_metashape_preprocess_uses_core_writer(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images = scene / "images"
    output = scene / "output"
    _write_image(images / "wide" / "pano64.jpg", (64, 32), (80, 120, 160))
    _write_image(images / "pano80.jpg", (80, 40), (10, 20, 30))
    xml = scene / "cameras.xml"
    _write_spherical_xml(xml)

    _run_metashape_preprocess(
        {
            "images_dir": str(images),
            "xml_path": str(xml),
            "output_dir": str(output),
            "scale": 1.0,
            "use_ply": False,
            "ply_path": "",
            "no_fix_rotation": False,
        }
    )

    manifest = json.loads((output / "stechdrive_metashape_preprocess.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "metashape_preprocess"
    assert manifest["source_kind"] == "metashape_xml_ply"
    assert manifest["frames"] == 2
    assert manifest["camera_model"] == "EQUIRECTANGULAR"


def test_metashape_360_lfs_compat_wrapper_uses_core_writer(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images = scene / "images"
    output = scene / "output"
    _write_image(images / "wide" / "pano64.jpg", (64, 32), (80, 120, 160))
    _write_image(images / "pano80.jpg", (80, 40), (10, 20, 30))
    xml = scene / "cameras.xml"
    _write_spherical_xml(xml)

    result = convert_metashape_to_lichtfeld(images_dir=images, xml_path=xml, output_dir=output, verbose=False)

    assert result["num_frames"] == 2
    assert result["camera_model"] == "EQUIRECTANGULAR"
    manifest = json.loads((output / "stechdrive_metashape_preprocess.json").read_text(encoding="utf-8"))
    assert manifest["frames"] == 2


def test_metashape_preprocess_rejects_mixed_camera_models(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images = scene / "images"
    _write_image(images / "wide" / "pano64.jpg", (64, 32), (80, 120, 160))
    _write_image(images / "pano80.jpg", (80, 40), (10, 20, 30))
    _write_image(images / "frame.jpg", (40, 30), (1, 2, 3))
    xml = scene / "cameras.xml"
    _write_spherical_xml(xml, include_frame_sensor=True)

    with pytest.raises(ValueError, match="equirectangular cameras only"):
        export_metashape_equirectangular_dataset(images_dir=images, xml_path=xml, output_dir=scene / "output")
