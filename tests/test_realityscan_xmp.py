import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from core.realityscan_xmp import (
    REALITYSCAN_XMP_NAMESPACE,
    c2w_to_xmp_rotation_position,
    write_realityscan_mask_layers,
    write_realityscan_xmp_sidecars,
)


def _write_transforms(output: Path) -> None:
    images = output / "images"
    masks = output / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    (images / "frame_0001_pz.jpg").write_bytes(b"image")
    (masks / "frame_0001_pz.png").write_bytes(b"mask")

    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    data = {
        "camera_model": "PINHOLE",
        "w": 100,
        "h": 100,
        "fl_x": 50.0,
        "fl_y": 50.0,
        "cx": 49.5,
        "cy": 49.5,
        "frames": [
            {
                "file_path": "images/frame_0001_pz.jpg",
                "source_file_path": "images/frame_0001.jpg",
                "source_image_index": 0,
                "view_name": "pz",
                "view_index": 4,
                "transform_matrix": transform.tolist(),
            }
        ],
    }
    (output / "transforms.json").write_text(json.dumps(data), encoding="utf-8")


def _xcr_attr(element: ET.Element, name: str) -> str:
    return element.attrib[f"{{{REALITYSCAN_XMP_NAMESPACE}}}{name}"]


def test_write_realityscan_xmp_sidecars(tmp_path: Path) -> None:
    output = tmp_path / "realityscan"
    _write_transforms(output)

    manifest = write_realityscan_xmp_sidecars(
        output,
        pose_prior="exact",
        calibration_prior="initial",
        rig_name="test-rig",
    )

    xmp = output / "images" / "frame_0001_pz.xmp"
    assert xmp.is_file()
    assert manifest["xmp_count"] == 1
    assert manifest["focal_length_35mm"] == 18.0

    root = ET.parse(xmp).getroot()
    desc = root.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
    assert desc is not None
    assert _xcr_attr(desc, "PosePrior") == "exact"
    assert _xcr_attr(desc, "CalibrationPrior") == "initial"
    assert _xcr_attr(desc, "Rig").startswith("{")
    assert _xcr_attr(desc, "RigInstance").startswith("{")
    assert _xcr_attr(desc, "RigPoseIndex") == "4"
    assert _xcr_attr(desc, "DistortionModel") == "division"
    assert _xcr_attr(desc, "FocalLength35mm") == "18"
    assert _xcr_attr(desc, "PrincipalPointU") == "0"
    assert _xcr_attr(desc, "PrincipalPointV") == "0"
    assert _xcr_attr(desc, "Rotation") == "1 0 0 0 1 0 0 0 1"

    position = desc.find("{http://www.capturingreality.com/ns/xcr/1.1#}Position")
    assert position is not None
    assert position.text == "1 2 3"


def test_write_realityscan_mask_layers(tmp_path: Path) -> None:
    output = tmp_path / "realityscan"
    _write_transforms(output)
    write_realityscan_xmp_sidecars(output)

    manifest = write_realityscan_mask_layers(output)

    layer = output / "images" / "frame_0001_pz.jpg.mask.png"
    assert layer.read_bytes() == b"mask"
    assert manifest["mask_layer_count"] == 1
    assert manifest["mask_layer_files"] == ["images/frame_0001_pz.jpg.mask.png"]


def test_c2w_to_xmp_rotation_position_uses_world_to_camera_rotation() -> None:
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform[:3, 3] = [4.0, 5.0, 6.0]

    rotation, position = c2w_to_xmp_rotation_position(transform)

    assert np.allclose(rotation, transform[:3, :3].T)
    assert np.allclose(position, [4.0, 5.0, 6.0])
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_mask_layer_overwrites_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "realityscan"
    _write_transforms(output)
    layer = output / "images" / "frame_0001_pz.jpg.mask.png"
    layer.write_bytes(b"old")

    write_realityscan_mask_layers(output)

    assert layer.read_bytes() == b"mask"
