import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from core.realityscan_xmp import REALITYSCAN_XMP_NAMESPACE, write_realityscan_xmp_sidecars
from core.transforms_to_colmap import convert


def _write_mixed_equirect_scene(scene: Path) -> Path:
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    cv2.imwrite(str(images / "large.png"), np.full((32, 64, 3), 96, dtype=np.uint8))
    cv2.imwrite(str(images / "small.png"), np.full((16, 32, 3), 160, dtype=np.uint8))
    cv2.imwrite(str(masks / "large.png"), np.full((32, 64), 255, dtype=np.uint8))
    cv2.imwrite(str(masks / "small.png"), np.full((16, 32), 255, dtype=np.uint8))
    transforms = {
        "camera_model": "EQUIRECTANGULAR",
        "frames": [
            {
                "file_path": "images/large.png",
                "w": 64,
                "h": 32,
                "fl_x": 32.0,
                "fl_y": 32.0,
                "cx": 32.0,
                "cy": 16.0,
                "transform_matrix": np.eye(4).tolist(),
            },
            {
                "file_path": "images/small.png",
                "w": 32,
                "h": 16,
                "fl_x": 16.0,
                "fl_y": 16.0,
                "cx": 16.0,
                "cy": 8.0,
                "transform_matrix": np.eye(4).tolist(),
            },
        ],
    }
    path = scene / "transforms.json"
    path.write_text(json.dumps(transforms), encoding="utf-8")
    return path


def test_cubemap_conversion_preserves_mixed_erp_face_sizes(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_mixed_equirect_scene(scene)
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
            "--workers",
            "1",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert Image.open(output / "images" / "large_front.png").size == (16, 16)
    assert Image.open(output / "images" / "small_front.png").size == (8, 8)
    assert Image.open(output / "masks" / "large_front.png").size == (16, 16)
    assert Image.open(output / "masks" / "small_front.png").size == (8, 8)

    data = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
    by_source = {frame["source_file_path"]: frame for frame in data["frames"]}
    assert by_source["images/large.png"]["w"] == 16
    assert by_source["images/large.png"]["fl_x"] == pytest.approx(8.0)
    assert by_source["images/small.png"]["w"] == 8
    assert by_source["images/small.png"]["fl_x"] == pytest.approx(4.0)
    assert data["mixed_camera_intrinsics"] is True


def _xcr_attr(element: ET.Element, name: str) -> str:
    return element.attrib[f"{{{REALITYSCAN_XMP_NAMESPACE}}}{name}"]


def test_realityscan_xmp_groups_mixed_frame_intrinsics(tmp_path: Path) -> None:
    output = tmp_path / "realityscan"
    images = output / "images"
    images.mkdir(parents=True)
    (images / "large_front.jpg").write_bytes(b"large")
    (images / "small_front.jpg").write_bytes(b"small")
    transform = np.eye(4).tolist()
    data = {
        "camera_model": "PINHOLE",
        "w": 16,
        "h": 16,
        "fl_x": 8.0,
        "fl_y": 8.0,
        "cx": 7.5,
        "cy": 7.5,
        "frames": [
            {
                "file_path": "images/large_front.jpg",
                "source_file_path": "images/large.png",
                "source_image_index": 0,
                "view_name": "front",
                "view_index": 0,
                "w": 16,
                "h": 16,
                "fl_x": 8.0,
                "fl_y": 8.0,
                "cx": 7.5,
                "cy": 7.5,
                "transform_matrix": transform,
            },
            {
                "file_path": "images/small_front.jpg",
                "source_file_path": "images/small.png",
                "source_image_index": 1,
                "view_name": "front",
                "view_index": 0,
                "w": 8,
                "h": 8,
                "fl_x": 4.0,
                "fl_y": 4.0,
                "cx": 3.5,
                "cy": 3.5,
                "transform_matrix": transform,
            },
        ],
    }
    (output / "transforms.json").write_text(json.dumps(data), encoding="utf-8")

    manifest = write_realityscan_xmp_sidecars(output)

    assert manifest["calibration_group_count"] == 2
    large_desc = ET.parse(images / "large_front.xmp").getroot().find(
        ".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"
    )
    small_desc = ET.parse(images / "small_front.xmp").getroot().find(
        ".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"
    )
    assert large_desc is not None
    assert small_desc is not None
    assert _xcr_attr(large_desc, "CalibrationGroup") == "0"
    assert _xcr_attr(small_desc, "CalibrationGroup") == "1"
    assert _xcr_attr(large_desc, "DistortionGroup") == "0"
    assert _xcr_attr(small_desc, "DistortionGroup") == "1"


def test_transforms_to_colmap_writes_multiple_cameras_for_frame_intrinsics(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    data = {
        "camera_model": "PINHOLE",
        "w": 16,
        "h": 16,
        "fl_x": 8.0,
        "fl_y": 8.0,
        "cx": 7.5,
        "cy": 7.5,
        "frames": [
            {
                "file_path": "images/large_front.jpg",
                "w": 16,
                "h": 16,
                "fl_x": 8.0,
                "fl_y": 8.0,
                "cx": 7.5,
                "cy": 7.5,
                "transform_matrix": np.eye(4).tolist(),
            },
            {
                "file_path": "images/small_front.jpg",
                "w": 8,
                "h": 8,
                "fl_x": 4.0,
                "fl_y": 4.0,
                "cx": 3.5,
                "cy": 3.5,
                "transform_matrix": np.eye(4).tolist(),
            },
        ],
    }
    (input_dir / "transforms.json").write_text(json.dumps(data), encoding="utf-8")

    result = convert(input_dir, "transforms.json", tmp_path / "colmap", None, "images/")

    cameras = [
        line
        for line in (tmp_path / "colmap" / "cameras.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    images = [
        line
        for line in (tmp_path / "colmap" / "images.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert result["camera_model"] == "mixed"
    assert result["num_cameras"] == 2
    assert len(cameras) == 2
    assert images[0].split()[8] == "1"
    assert images[1].split()[8] == "2"
