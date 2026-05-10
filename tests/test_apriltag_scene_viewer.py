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
    transform_group_for_world_display,
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


def _export_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
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
    return rx @ ry


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


def _write_generated_cube6_case(root: Path) -> Path:
    source = root / "source"
    images = source / "images"
    images.mkdir(parents=True)
    views = {
        "bottom": (-45.0, -90.0),
        "px": (45.0, 0.0),
        "nz": (135.0, 0.0),
        "nx": (-135.0, 0.0),
        "pz": (-45.0, 0.0),
        "top": (-45.0, 90.0),
    }
    positions = {
        "cam_001": (1.0, 0.0, 0.0),
        "cam_002": (0.0, 2.0, 0.0),
        "cam_003": (0.0, 0.0, 3.0),
    }
    frames = []
    for prefix, position in positions.items():
        for face, view in views.items():
            image = np.full((64, 64, 3), 30, dtype=np.uint8)
            assert imwrite_unicode(images / f"{prefix}_{face}.jpg", image)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = _export_rotation(*view).T
            transform[:3, 3] = position
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
    (source / "view_export_settings.json").write_text(
        json.dumps(
            {
                "views": [
                    {"name": name, "yaw": yaw, "pitch": pitch, "enabled": True}
                    for name, (yaw, pitch) in views.items()
                ],
                "yaw_offset_per_frame": 0.0,
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
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
                "0 1 0",
                "0 0 1",
            ]
        ),
        encoding="ascii",
    )
    cameras = []
    for prefix, position in positions.items():
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = position
        cameras.append(
            f"<camera label=\"{prefix}\"><transform>{' '.join(str(v) for v in transform.reshape(-1))}</transform></camera>"
        )
    xml = source / "metashape.xml"
    xml.write_text(f"<document><chunk><cameras>{''.join(cameras)}</cameras></chunk></document>", encoding="utf-8")
    case_dir = root / "case_generated"
    case = AprilTagDevCase(
        name="case_generated",
        case_dir=case_dir,
        input_mode="reference",
        source_transforms=transforms,
        source_pointcloud=pointcloud,
        source_metashape_xml=xml,
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


def test_scene_viewer_uses_generated_json_image_rays_for_world_faces(tmp_path: Path) -> None:
    _app()
    case_dir = _write_generated_cube6_case(tmp_path)

    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    group = window.selected_world_group()

    assert group is not None
    directions = {
        face: np.array([0.0, 0.0, 1.0]) @ frame.camera_to_world_rotation.T
        for face, frame in group.frames_by_face.items()
    }
    determinants = {
        face: float(np.linalg.det(frame.camera_to_world_rotation))
        for face, frame in group.frames_by_face.items()
    }
    assert all(value > 0.999 for value in determinants.values())
    diagonal = 2.0**-0.5
    assert np.allclose(directions["px"], [diagonal, 0.0, diagonal], atol=1e-6)
    assert np.allclose(directions["nz"], [diagonal, 0.0, -diagonal], atol=1e-6)
    assert np.allclose(directions["nx"], [-diagonal, 0.0, -diagonal], atol=1e-6)
    assert np.allclose(directions["pz"], [-diagonal, 0.0, diagonal], atol=1e-6)
    assert np.allclose(directions["top"], [0.0, -1.0, 0.0], atol=1e-6)
    assert np.allclose(directions["bottom"], [0.0, 1.0, 0.0], atol=1e-6)
    assert window.point_view._fixed_view_basis is not None
    right, up, forward = window.point_view._fixed_view_basis
    assert np.linalg.det(np.column_stack([right, up, -forward])) > 0.999
    assert np.allclose(forward, directions["pz"], atol=1e-6)
    assert "transforms.json normalized Cube6 image rays" in window.case_label.text()
    window.deleteLater()


def test_lichtfeld_image_ray_correction_preserves_cube6_cut_yaw_and_vertical_names(tmp_path: Path) -> None:
    views = {
        "bottom": (-45.0, -90.0),
        "px": (45.0, 0.0),
        "nz": (135.0, 0.0),
        "nx": (-135.0, 0.0),
        "pz": (-45.0, 0.0),
        "top": (-45.0, 90.0),
    }
    opengl_base = np.diag([1.0, -1.0, -1.0])
    frames: dict[str, PinholeFrame] = {}
    for face, (yaw, pitch) in views.items():
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = opengl_base @ _rotation(yaw, pitch)
        frames[face] = PinholeFrame(
            frame_id=face,
            file_path=f"images/frame_0001_{face}.jpg",
            image_path=tmp_path / f"frame_0001_{face}.jpg",
            width=64,
            height=64,
            fl_x=32.0,
            fl_y=32.0,
            cx=31.5,
            cy=31.5,
            transform_matrix=transform,
        )
    group = CubemapFrameGroup("frame_0001", frames)
    corrected = transform_group_for_world_display(
        group,
        None,
        image_ray_correction=np.diag([1.0, 1.0, -1.0, 1.0]),
        cubemap_view_params=CubemapViewMetadata(views),
    )

    image_local = np.diag([1.0, -1.0, 1.0])
    for face, (yaw, pitch) in views.items():
        actual = np.array([0.0, 0.0, 1.0]) @ corrected.frames_by_face[face].camera_to_world_rotation.T
        expected = np.array([0.0, 0.0, 1.0]) @ (image_local @ _rotation(yaw, pitch)).T
        assert np.allclose(actual, expected, atol=1e-6)

    original_top = np.array([0.0, 0.0, 1.0]) @ frames["top"].camera_to_world_rotation.T
    original_bottom = np.array([0.0, 0.0, 1.0]) @ frames["bottom"].camera_to_world_rotation.T
    corrected_top = np.array([0.0, 0.0, 1.0]) @ corrected.frames_by_face["top"].camera_to_world_rotation.T
    corrected_bottom = np.array([0.0, 0.0, 1.0]) @ corrected.frames_by_face["bottom"].camera_to_world_rotation.T
    assert np.allclose(corrected_top, original_top, atol=1e-6)
    assert np.allclose(corrected_bottom, original_bottom, atol=1e-6)


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
