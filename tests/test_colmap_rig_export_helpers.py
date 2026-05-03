import json
import math

import pytest

from colmap_rig_export import (
    build_rig_config,
    camera_name_for_index,
    cam_from_rig_rotation_quaternion,
    frame_filename,
    prepare_views_for_colmap,
    write_rig_config_json,
)


def test_prepare_views_for_colmap_sorts_and_names_cameras() -> None:
    views = [
        {"name": "top", "pitch": -90.0, "yaw": 0.0},
        {"name": "front", "pitch": 0.0, "yaw": 0.0},
        {"name": "left", "pitch": 0.0, "yaw": -90.0},
    ]

    prepared = prepare_views_for_colmap(views)

    assert [(v["name"], v["camera_name"]) for v in prepared] == [
        ("top", "cam01"),
        ("left", "cam02"),
        ("front", "cam03"),
    ]
    assert "camera_name" not in views[0]


def test_camera_name_and_frame_filename_padding() -> None:
    assert camera_name_for_index(1, 6) == "cam01"
    assert camera_name_for_index(1, 120) == "cam001"
    assert frame_filename(1, 9, ".jpg") == "frame_00001.jpg"
    assert frame_filename(100000, 100000, "png") == "frame_100000.png"


def test_rig_config_first_camera_is_reference() -> None:
    prepared = prepare_views_for_colmap(
        [
            {"name": "front", "pitch": 0.0, "yaw": 0.0, "fov": 90.0},
            {"name": "right", "pitch": 0.0, "yaw": 90.0, "fov": 90.0},
        ]
    )

    config = build_rig_config(prepared, (1024, 1024))

    cameras = config[0]["cameras"]
    assert cameras[0]["ref_sensor"] is True
    assert "cam_from_rig_rotation" not in cameras[0]
    assert cameras[1]["cam_from_rig_translation"] == [0.0, 0.0, 0.0]
    assert cameras[1]["camera_model_name"] == "PINHOLE"
    assert cameras[1]["image_prefix"] == "rig1/cam02/"


def test_cam_from_rig_quaternion_is_unit_length() -> None:
    q = cam_from_rig_rotation_quaternion(45.0, 30.0)
    assert math.sqrt(sum(v * v for v in q)) == pytest.approx(1.0)


def test_write_rig_config_json(tmp_path) -> None:
    prepared = prepare_views_for_colmap([{"name": "front", "pitch": 0.0, "yaw": 0.0, "fov": 90.0}])

    path = write_rig_config_json(tmp_path, prepared, (512, 512))

    assert path == tmp_path / "colmap_rig" / "rig_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0]["cameras"][0]["image_prefix"] == "rig1/cam01/"
