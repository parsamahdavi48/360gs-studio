from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication

from core.apriltag_cubemap import CubemapViewMetadata
from core.apriltag_geometry import PinholeFrame
from core.image_io import imwrite_unicode
from devtools.apriltag.case import AprilTagDevCase, save_case
from devtools.apriltag.cubemap_preview import CubemapFrameGroup
from devtools.apriltag.scene_viewer import (
    AprilTagSceneViewerWindow,
    camera_pose_from_perspective_params,
    metashape_image_ray_groups,
)
from gui.common.perspective_preview import PerspectiveParams


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    ry = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float64,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    return ry @ rx


def _write_cube6_case(root: Path) -> Path:
    source = root / "source"
    images = source / "images"
    images.mkdir(parents=True)
    faces = {
        "pz": (0.0, 0.0),
        "px": (90.0, 0.0),
        "nx": (-90.0, 0.0),
        "nz": (180.0, 0.0),
        "top": (0.0, 90.0),
        "bottom": (0.0, -90.0),
    }
    frames = []
    for index, (prefix, x) in enumerate((("cam_001", 0.0), ("cam_002", 2.0))):
        for face, view in faces.items():
            image = np.full((64, 64, 3), 20 + index * 60, dtype=np.uint8)
            assert imwrite_unicode(images / f"{prefix}_{face}.jpg", image)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = _rotation(*view).T
            transform[:3, 3] = (x, 0.0, 0.0)
            frames.append(
                {
                    "file_path": f"images/{prefix}_{face}.jpg",
                    "transform_matrix": transform.tolist(),
                }
            )
    transforms = source / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "camera_model": "SIMPLE_PINHOLE",
                "w": 64,
                "h": 64,
                "fl_x": 32.0,
                "fl_y": 32.0,
                "cx": 31.5,
                "cy": 31.5,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    pointcloud = source / "pointcloud.ply"
    pointcloud.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 3",
                "1 0 3",
                "0 1 3",
                "1 1 3",
            ]
        ),
        encoding="ascii",
    )
    case_dir = root / "case"
    case = AprilTagDevCase(
        name="case",
        case_dir=case_dir,
        input_mode="reference",
        source_transforms=transforms,
        source_pointcloud=pointcloud,
        coordinate_profile="custom",
    )
    save_case(case)
    return case_dir


def test_camera_pose_from_perspective_params_uses_fov_view_axes() -> None:
    frame = PinholeFrame(
        frame_id="a",
        file_path="a.jpg",
        image_path=Path("a.jpg"),
        width=64,
        height=64,
        fl_x=32.0,
        fl_y=32.0,
        cx=31.5,
        cy=31.5,
        transform_matrix=np.eye(4),
    )
    group = CubemapFrameGroup("cam", {"pz": frame})

    camera, right, up, forward = camera_pose_from_perspective_params(
        group,
        PerspectiveParams(yaw_deg=90.0, pitch_deg=0.0, fov_deg=90.0),
    )

    assert np.allclose(camera, np.zeros(3))
    assert np.allclose(right, np.array([0.0, 0.0, -1.0]), atol=1e-6)
    assert np.allclose(up, np.array([0.0, 1.0, 0.0]), atol=1e-6)
    assert np.allclose(forward, np.array([1.0, 0.0, 0.0]), atol=1e-6)


def test_metashape_image_ray_groups_use_xml_pose_and_view_metadata(tmp_path: Path) -> None:
    image = tmp_path / "frame_px.jpg"
    image.write_bytes(b"fake")
    transform = np.eye(4, dtype=np.float64)
    frame = PinholeFrame(
        frame_id="cam_px",
        file_path="images/cam_px.jpg",
        image_path=image,
        width=64,
        height=64,
        fl_x=32.0,
        fl_y=32.0,
        cx=31.5,
        cy=31.5,
        transform_matrix=transform,
    )
    group = CubemapFrameGroup(
        "cam",
        {
            "px": frame,
            "nx": frame,
            "pz": frame,
            "nz": frame,
            "top": frame,
            "bottom": frame,
        },
        group_index=1,
    )
    base = np.eye(4, dtype=np.float64)
    base[:3, :3] = _rotation(20.0, 0.0)
    base[:3, 3] = (1.0, 2.0, 3.0)
    xml = tmp_path / "metashape.xml"
    xml.write_text(
        "<document><chunk><cameras>"
        f"<camera label=\"cam\"><transform>{' '.join(str(v) for v in base.reshape(-1))}</transform></camera>"
        "</cameras></chunk></document>",
        encoding="utf-8",
    )
    metadata = CubemapViewMetadata(
        {
            "px": (45.0, 0.0),
            "nx": (-135.0, 0.0),
            "pz": (-45.0, 0.0),
            "nz": (135.0, 0.0),
            "top": (-45.0, 90.0),
            "bottom": (-45.0, -90.0),
        },
        yaw_offset_per_frame=30.0,
    )

    groups = metashape_image_ray_groups((group,), xml, metadata)

    assert groups is not None
    px = groups[0].frames_by_face["px"]
    assert np.allclose(px.camera_position_sfm, [1.0, 2.0, 3.0])
    expected_rotation = _rotation(20.0, 0.0) @ _rotation(75.0, 0.0)
    expected_forward = np.array([0.0, 0.0, 1.0]) @ expected_rotation.T
    actual_forward = np.array([0.0, 0.0, 1.0]) @ px.camera_to_world_rotation.T
    assert np.allclose(actual_forward, expected_forward, atol=1e-6)


def test_scene_viewer_loads_case_and_selects_camera(tmp_path: Path) -> None:
    _app()
    case_dir = _write_cube6_case(tmp_path)

    window = AprilTagSceneViewerWindow(initial_case=case_dir)

    assert window.camera_combo.count() == 2
    assert window.selected_world_group() is not None
    assert window.selected_world_group().name == "cam_001"
    window.select_camera_by_name("cam_002")
    assert window.selected_world_group() is not None
    assert window.selected_world_group().name == "cam_002"
    assert window._world_pointcloud is not None
    assert len(window._world_pointcloud.points) == 4
    assert window.right_stack.currentWidget() is window.point_view
    window.mode_combo.setCurrentIndex(window.mode_combo.findData("image"))
    assert window.right_stack.currentWidget() is window.image_view
