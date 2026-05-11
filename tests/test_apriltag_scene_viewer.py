from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QAbstractSpinBox, QApplication

from core.apriltag_cubemap import CubemapViewMetadata, cubemap_view_params_for_group
from core.apriltag_detection import detect_apriltags
from core.apriltag_geometry import PinholeFrame, load_pinhole_frames
from core.image_io import imread_unicode, imwrite_unicode
from devtools.apriltag.case import AprilTagDevCase, load_case, save_case
from devtools.apriltag.cubemap_preview import (
    CubemapFrameGroup,
    axis_face_view_params,
    render_cubemap_axis_equirect,
    render_generated_cubemap_source_axis,
    render_source_equirect_axis,
    render_source_equirect_perspective,
    source_equirect_base_rotation,
)
from devtools.apriltag.scene_viewer import (
    AprilTagSceneViewerWindow,
    RAY_BASIS_BOTH,
    RAY_BASIS_IMAGE,
    RAY_BASIS_WORLD,
    RIGHT_VIEW_POINTCLOUD,
    RIGHT_VIEW_RECONSTRUCTED_CUBE6,
    RIGHT_VIEW_SOURCE_EQUIRECT,
    _resolve_source_equirect_paths,
    _source_equirect_preview_params,
    camera_pose_from_perspective_params,
    case_cubemap_view_metadata,
    closest_image_face_for_world_face,
    face_forward_ray,
    image_ray_display_matrix_for_profile,
    opposite_image_face_for_world_face,
    rotation_from_perspective_params,
    transform_group_for_world_display,
)
from devtools.apriltag.coordinates import world_display_matrix
from devtools.apriltag.synthetic import SyntheticAprilTagConfig, inject_synthetic_apriltag
from gui.common.drag_spinbox import DragDoubleSpinBox
from gui.common.perspective_preview import PerspectiveParams, equirect_to_perspective, params_from_drag
from gui.common.perspective_preview import normalize_yaw_deg


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


def _source_face_centers(metadata: CubemapViewMetadata, group_index: int) -> dict[str, np.ndarray]:
    params = cubemap_view_params_for_group(metadata, group_index)
    assert params is not None
    centers: dict[str, np.ndarray] = {}
    for face, (yaw, pitch) in params.items():
        ray = np.array([0.0, 0.0, 1.0], dtype=np.float64) @ _rotation(yaw, pitch).T
        centers[face] = ray / max(float(np.linalg.norm(ray)), 1e-12)
    return centers


def _closest_source_face(source_local_ray: np.ndarray, metadata: CubemapViewMetadata, group_index: int) -> str:
    ray = np.asarray(source_local_ray, dtype=np.float64)
    ray = ray / max(float(np.linalg.norm(ray)), 1e-12)
    centers = _source_face_centers(metadata, group_index)
    return max(centers, key=lambda face: float(ray @ centers[face]))


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


def _write_scale_validation_cube6_case(root: Path) -> Path:
    source = root / "source_validation"
    images = source / "images"
    images.mkdir(parents=True)
    size = 400
    faces = {
        "pz": (0.0, 0.0),
        "px": (90.0, 0.0),
        "nx": (-90.0, 0.0),
        "nz": (180.0, 0.0),
        "top": (0.0, 90.0),
        "bottom": (0.0, -90.0),
    }
    frames = []
    for prefix, x in (("cam_001", 0.0), ("cam_002", 0.5)):
        for face, view in faces.items():
            image = np.full((size, size, 3), 255, dtype=np.uint8)
            assert imwrite_unicode(images / f"{prefix}_{face}.png", image)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = _rotation(*view).T
            transform[:3, 3] = (x, 0.0, 0.0)
            frames.append(
                {
                    "file_path": f"images/{prefix}_{face}.png",
                    "transform_matrix": transform.tolist(),
                }
            )
    transforms = source / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "camera_model": "SIMPLE_PINHOLE",
                "w": size,
                "h": size,
                "fl_x": 240.0,
                "fl_y": 240.0,
                "cx": (size - 1) / 2.0,
                "cy": (size - 1) / 2.0,
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
                "0 0 2",
                "0.5 0 2",
                "0 0.5 2",
                "0.5 0.5 2",
            ]
        ),
        encoding="ascii",
    )
    case_dir = root / "case_validation"
    case = AprilTagDevCase(
        name="case_validation",
        case_dir=case_dir,
        input_mode="reference",
        source_transforms=transforms,
        source_pointcloud=pointcloud,
        default_tag_size_m=0.160,
        true_scale=0.25,
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


def test_scene_viewer_keeps_world_face_rays_separate_from_generated_image_rays(tmp_path: Path) -> None:
    _app()
    case_dir = _write_generated_cube6_case(tmp_path)

    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    world_group = window.selected_world_group()
    image_group = window.selected_image_ray_group()

    assert world_group is not None
    assert image_group is not None
    directions = {
        face: face_forward_ray(world_group, face)
        for face in world_group.frames_by_face
    }
    image_directions = {
        face: face_forward_ray(image_group, face)
        for face in image_group.frames_by_face
    }
    determinants = {
        face: float(np.linalg.det(frame.camera_to_world_rotation))
        for face, frame in world_group.frames_by_face.items()
    }
    assert all(value > 0.999 for value in determinants.values())
    diagonal = 2.0**-0.5
    assert np.allclose(directions["px"], [-diagonal, 0.0, diagonal], atol=1e-6)
    assert np.allclose(directions["nz"], [-diagonal, 0.0, -diagonal], atol=1e-6)
    assert np.allclose(directions["nx"], [diagonal, 0.0, -diagonal], atol=1e-6)
    assert np.allclose(directions["pz"], [diagonal, 0.0, diagonal], atol=1e-6)
    assert np.allclose(directions["top"], [0.0, 1.0, 0.0], atol=1e-6)
    assert np.allclose(directions["bottom"], [0.0, -1.0, 0.0], atol=1e-6)
    assert np.allclose(image_directions["px"], [diagonal, 0.0, diagonal], atol=1e-6)
    assert np.allclose(image_directions["nx"], [-diagonal, 0.0, -diagonal], atol=1e-6)
    assert not np.allclose(directions["pz"], image_directions["pz"], atol=1e-6)
    assert closest_image_face_for_world_face(world_group, image_group, "pz")[:2] == ("px", 0.0)
    assert opposite_image_face_for_world_face(world_group, image_group, "pz") == ("nx", 0.0)
    assert window.point_view._fixed_view_basis is not None
    right, up, forward = window.point_view._fixed_view_basis
    assert np.linalg.det(np.column_stack([right, up, -forward])) > 0.999
    assert np.allclose(forward, directions["pz"], atol=1e-6)
    assert "world rays=transforms.json face +Z" in window.case_label.text()
    assert "image rays=Cube6 export yaw/pitch" in window.case_label.text()
    assert "active basis=both (active=transforms.json face +Z)" in window.case_label.text()
    assert "active pz->image px" in window.case_label.text()
    assert "reverse=nx" in window.case_label.text()

    window.ray_basis_combo.setCurrentIndex(window.ray_basis_combo.findData("image"))
    assert window.selected_face_basis_group() is image_group
    assert window.point_view._fixed_view_basis is not None
    _right, _up, image_forward = window.point_view._fixed_view_basis
    assert np.allclose(image_forward, image_directions["pz"], atol=1e-6)
    assert "active basis=Cube6 image ray" in window.case_label.text()

    window.ray_basis_combo.setCurrentIndex(window.ray_basis_combo.findData("world"))
    assert window.selected_face_basis_group() is world_group
    assert window.point_view._fixed_view_basis is not None
    _right, _up, world_forward = window.point_view._fixed_view_basis
    assert np.allclose(world_forward, directions["pz"], atol=1e-6)
    assert "active basis=transforms.json face +Z" in window.case_label.text()
    window.deleteLater()


def test_world_display_does_not_mix_image_ray_correction_into_camera_poses(tmp_path: Path) -> None:
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
    world_matrix = np.diag([-1.0, 1.0, -1.0, 1.0])
    displayed = transform_group_for_world_display(
        group,
        world_matrix,
        image_ray_correction=np.diag([1.0, 1.0, -1.0, 1.0]),
        cubemap_view_params=CubemapViewMetadata(views),
    )

    for face, frame in frames.items():
        actual = displayed.frames_by_face[face].camera_to_world_rotation
        expected = world_matrix[:3, :3] @ frame.camera_to_world_rotation
        assert np.allclose(actual, expected, atol=1e-6)
        assert float(np.linalg.det(actual)) > 0.999


def test_world_display_pz_image_preview_is_not_mirrored_by_image_ray_correction(tmp_path: Path) -> None:
    size = 96
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
    pz_image = np.full((size, size, 3), 40, dtype=np.uint8)
    horizontal = np.linspace(0, 255, size, dtype=np.uint8)
    pz_image[:, :, 0] = horizontal[None, :]
    pz_image[:, :, 1] = np.arange(size, dtype=np.uint8)[:, None]
    pz_image[:, :, 2] = 255 - horizontal[None, :]
    for face, (yaw, pitch) in views.items():
        image = pz_image if face == "pz" else np.full((size, size, 3), 40, dtype=np.uint8)
        image_path = tmp_path / f"frame_000001_{face}.png"
        assert imwrite_unicode(image_path, image)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = opengl_base @ _rotation(yaw, pitch)
        frames[face] = PinholeFrame(
            frame_id=face,
            file_path=f"images/frame_000001_{face}.png",
            image_path=image_path,
            width=size,
            height=size,
            fl_x=size / 2.0,
            fl_y=size / 2.0,
            cx=(size - 1) / 2.0,
            cy=(size - 1) / 2.0,
            transform_matrix=transform,
        )
    group = transform_group_for_world_display(
        CubemapFrameGroup("frame_000001", frames),
        np.diag([-1.0, 1.0, -1.0, 1.0]),
        image_ray_correction=np.diag([1.0, 1.0, -1.0, 1.0]),
        cubemap_view_params=CubemapViewMetadata(views),
    )
    yaw, pitch, roll, fov = axis_face_view_params(group, "pz", fov_deg=90.0)

    equirect = render_cubemap_axis_equirect(group, output_width=256, output_height=128)
    rendered = equirect_to_perspective(
        equirect,
        PerspectiveParams(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, fov_deg=fov),
        output_size=size,
    )

    direct_error = float(np.mean(np.abs(rendered.astype(np.int16) - pz_image.astype(np.int16))))
    mirrored_error = float(np.mean(np.abs(rendered.astype(np.int16) - pz_image[:, ::-1].astype(np.int16))))
    assert direct_error < 8.0
    assert direct_error * 4.0 < mirrored_error


def test_lichtfeld_image_rays_are_not_world_display_transformed_twice() -> None:
    matrix = np.diag([-1.0, 1.0, -1.0, 1.0])

    assert image_ray_display_matrix_for_profile("lichtfeld_cube6", matrix) is None
    assert image_ray_display_matrix_for_profile("lichtfeld_cube6_pre_final_ply", matrix) is None
    assert np.allclose(image_ray_display_matrix_for_profile("custom", matrix), matrix)


def test_current_case_reports_world_to_image_ray_mapping() -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    world_group = window.selected_world_group()
    image_group = window.selected_image_render_group()
    raw_image_group = window.selected_image_ray_group()

    assert world_group is not None
    assert image_group is not None
    assert raw_image_group is not None
    assert world_group.name == "frame_000001"
    closest = closest_image_face_for_world_face(world_group, image_group, "pz")
    opposite = opposite_image_face_for_world_face(world_group, image_group, "pz")
    image_basis_closest = closest_image_face_for_world_face(image_group, image_group, "pz")
    raw_closest = closest_image_face_for_world_face(world_group, raw_image_group, "pz")
    assert closest is not None
    assert opposite is not None
    assert image_basis_closest is not None
    assert raw_closest is not None
    assert raw_closest[0] == "nx"
    assert closest[0] == "nx"
    assert closest[1] < 2.0
    assert opposite[0] == "px"
    assert opposite[1] < 2.0
    assert image_basis_closest[0] == "pz"
    assert image_basis_closest[1] < 1e-4
    assert "source equirect=frame_000001.jpg" in window.case_label.text()
    window.mode_combo.setCurrentIndex(window.mode_combo.findData(RIGHT_VIEW_SOURCE_EQUIRECT))
    assert "image preview=source equirect direct" in window.case_label.text()
    window.ray_basis_combo.setCurrentIndex(window.ray_basis_combo.findData("image"))
    assert "source equirect=frame_000001.jpg" in window.case_label.text()
    window.deleteLater()


def test_current_case_image_ray_group_matches_metashape_build_remap_rays() -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    case = load_case(case_dir)
    if case.source_metashape_xml is None or not case.source_metashape_xml.is_file():
        pytest.skip("local AprilTag comparison Metashape XML is not available")
    metadata = case_cubemap_view_metadata(case)
    if metadata is None:
        pytest.skip("local AprilTag comparison Cube6 metadata is not available")
    xml_root = ET.parse(case.source_metashape_xml).getroot()
    metashape_transform = None
    for camera in xml_root.findall(".//camera"):
        if Path(str(camera.attrib.get("label") or "")).stem == "frame_000001":
            text = camera.findtext("transform")
            if text:
                metashape_transform = np.asarray([float(value) for value in text.split()], dtype=np.float64).reshape(4, 4)
                break
    if metashape_transform is None:
        pytest.skip("frame_000001 is not present in the local Metashape XML")
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    world_group = window.selected_world_group()
    image_group = window.selected_image_ray_group()
    assert world_group is not None
    assert image_group is not None

    metashape_local_from_lfs_local = np.diag([1.0, -1.0, -1.0])
    expected: dict[str, np.ndarray] = {}
    for face in ("px", "nz", "nx", "pz"):
        yaw, pitch = metadata.view_params[face]
        ray = np.array([0.0, 0.0, 1.0]) @ _rotation(yaw, pitch).T
        ray = ray @ metashape_local_from_lfs_local.T @ metashape_transform[:3, :3].T
        ray = ray @ world_display_matrix(case.coordinate_profile)[:3, :3].T
        expected[face] = ray / max(float(np.linalg.norm(ray)), 1e-12)

    actual_image = {face: face_forward_ray(image_group, face) for face in expected}
    actual_world = {face: face_forward_ray(world_group, face) for face in expected}

    for face, expected_ray in expected.items():
        assert actual_image[face] is not None
        assert float(actual_image[face] @ expected_ray) > 0.999
        assert actual_world[face] is not None
        assert abs(float(actual_world[face] @ expected_ray)) < 0.1
    assert closest_image_face_for_world_face(world_group, image_group, "pz")[0] == "nx"
    assert opposite_image_face_for_world_face(world_group, image_group, "pz")[0] == "px"
    window.deleteLater()


def test_current_case_source_equirect_rotation_maps_json_faces_to_source_centers() -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    case = load_case(case_dir)
    metadata = case_cubemap_view_metadata(case)
    if metadata is None:
        pytest.skip("local AprilTag comparison Cube6 metadata is not available")
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    raw_group = window.selected_raw_group()
    world_group = window.selected_world_group()
    source_rotation = window._source_equirect_rotations.get("frame_000001")
    assert raw_group is not None
    assert world_group is not None
    assert source_rotation is not None

    code_chain_rotation = source_equirect_base_rotation(raw_group, cubemap_view_params=metadata)
    assert code_chain_rotation is not None
    source_local_from_lichtfeld_local = np.diag([1.0, -1.0, -1.0])
    assert np.allclose(source_rotation, code_chain_rotation @ source_local_from_lichtfeld_local, atol=1e-6)

    display_matrix = world_display_matrix(case.coordinate_profile)
    preview_to_sfm = np.linalg.inv(display_matrix[:3, :3]).T
    actual: dict[str, str] = {}
    for face in ("pz", "px", "nx", "nz", "top", "bottom"):
        ray = face_forward_ray(world_group, face)
        assert ray is not None
        source_local = ray @ preview_to_sfm @ source_rotation
        actual[face] = _closest_source_face(source_local, metadata, world_group.group_index)

    # These are assertions for the code-derived transform, not inputs used to
    # construct the rotation above.
    assert actual == {
        "pz": "nz",
        "px": "nx",
        "nx": "px",
        "nz": "pz",
        "top": "top",
        "bottom": "bottom",
    }
    window.deleteLater()


def test_current_case_source_equirect_center_pixels_match_expected_faces() -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    case = load_case(case_dir)
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    raw_group = window.selected_raw_group()
    world_group = window.selected_world_group()
    source_path = _resolve_source_equirect_paths(case, ("frame_000001",)).get("frame_000001")
    source_rotation = window._source_equirect_rotations.get("frame_000001")
    if source_path is None or source_rotation is None:
        pytest.skip("local AprilTag source equirect image is not available")
    assert raw_group is not None
    assert world_group is not None
    source = imread_unicode(source_path)
    if source is None:
        pytest.skip("local AprilTag source equirect image cannot be read")

    display_matrix = world_display_matrix(case.coordinate_profile)
    expected_mapping = {
        "pz": "nz",
        "px": "nx",
        "nx": "px",
        "nz": "pz",
        "top": "top",
        "bottom": "bottom",
    }
    for face, expected_face in expected_mapping.items():
        target_frame = raw_group.frames_by_face.get(expected_face)
        assert target_frame is not None
        target = imread_unicode(target_frame.image_path)
        assert target is not None
        yaw, pitch, roll, fov = axis_face_view_params(world_group, face, fov_deg=90.0)
        rendered = render_source_equirect_perspective(
            source,
            source_rotation,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            fov_deg=fov,
            output_size=513,
            sfm_to_preview_matrix=display_matrix,
        )
        target = cv2.resize(target, (513, 513), interpolation=cv2.INTER_AREA)
        center_error = float(
            np.mean(np.abs(rendered[256, 256].astype(np.int16) - target[256, 256].astype(np.int16)))
        )
        assert center_error < 15.0
    window.deleteLater()


def test_current_case_source_equirect_faces_match_expected_orientation() -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    case = load_case(case_dir)
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    raw_group = window.selected_raw_group()
    world_group = window.selected_world_group()
    source_path = _resolve_source_equirect_paths(case, ("frame_000001",)).get("frame_000001")
    source_rotation = window._source_equirect_rotations.get("frame_000001")
    if source_path is None or source_rotation is None:
        pytest.skip("local AprilTag source equirect image is not available")
    assert raw_group is not None
    assert world_group is not None
    source = imread_unicode(source_path)
    if source is None:
        pytest.skip("local AprilTag source equirect image cannot be read")

    display_matrix = world_display_matrix(case.coordinate_profile)
    expected_mapping = {
        "pz": "nz",
        "px": "nx",
        "nx": "px",
        "nz": "pz",
        "top": "top",
        "bottom": "bottom",
    }
    for face, expected_face in expected_mapping.items():
        target_frame = raw_group.frames_by_face.get(expected_face)
        assert target_frame is not None
        target = imread_unicode(target_frame.image_path)
        assert target is not None
        yaw, pitch, roll, fov = axis_face_view_params(world_group, face, fov_deg=90.0)
        params = _source_equirect_preview_params(
            PerspectiveParams(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, fov_deg=fov),
            face,
            RAY_BASIS_BOTH,
        )
        rendered = render_source_equirect_perspective(
            source,
            source_rotation,
            yaw_deg=params.yaw_deg,
            pitch_deg=params.pitch_deg,
            roll_deg=params.roll_deg,
            fov_deg=params.fov_deg,
            output_size=512,
            sfm_to_preview_matrix=display_matrix,
        )
        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_AREA)
        error = float(np.mean(np.abs(rendered.astype(np.int16) - target.astype(np.int16))))
        assert error < 8.0
    window.deleteLater()


def test_current_case_image_overlay_uses_synthetic_output_projection() -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    if "frame_000001" not in window._source_equirect_rotations:
        pytest.skip("local AprilTag source equirect rotation is not available")
    window.select_camera_by_name("frame_000001")
    window.ray_basis_combo.setCurrentIndex(window.ray_basis_combo.findData(RAY_BASIS_WORLD))
    window.set_active_face("pz")
    group = window.selected_world_group()
    assert group is not None
    camera, right, up, forward = camera_pose_from_perspective_params(group, window._params)
    window.set_tag_transform(
        center=tuple(camera + forward * 5.0 + right * 0.5 + up * 0.25),
        yaw_deg=window._params.yaw_deg,
        pitch_deg=window._params.pitch_deg,
        roll_deg=window._params.roll_deg,
        size_sfm=1.0,
    )

    overlays = window._tag_output_image_overlays(output_size=768)
    candidates, _total = window._synthetic_tag_candidates()
    selected = [candidate for candidate in candidates if candidate.frame.file_path.startswith("images/frame_000001_")]

    assert len(overlays) == 1
    assert len(selected) == 1
    assert selected[0].frame.file_path == "images/frame_000001_nz.jpg"
    expected = selected[0].projected_points * (768.0 / selected[0].frame.width)
    assert np.allclose(np.asarray(overlays[0].polygon), expected, atol=1e-4)
    world_overlay = window._tag_image_overlays(group, window._params, output_size=768)
    assert len(world_overlay) == 1
    assert np.allclose(np.asarray(overlays[0].polygon), np.asarray(world_overlay[0].polygon), atol=1.0)
    window.deleteLater()


def test_current_case_synthetic_output_reconstructs_at_world_overlay_position(tmp_path: Path) -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    case = load_case(case_dir)
    metadata = case_cubemap_view_metadata(case)
    if metadata is None:
        pytest.skip("local AprilTag comparison Cube6 metadata is not available")
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    if "frame_000001" not in window._source_equirect_rotations:
        pytest.skip("local AprilTag source equirect rotation is not available")
    window.select_camera_by_name("frame_000001")
    window.ray_basis_combo.setCurrentIndex(window.ray_basis_combo.findData(RAY_BASIS_WORLD))
    window.set_active_face("pz")
    group = window.selected_world_group()
    raw_group = window.selected_raw_group()
    assert group is not None
    assert raw_group is not None
    camera, right, up, forward = camera_pose_from_perspective_params(group, window._params)
    window.set_tag_transform(
        center=tuple(camera + forward * 5.0 + right * 0.5 + up * 0.25),
        yaw_deg=window._params.yaw_deg,
        pitch_deg=window._params.pitch_deg,
        roll_deg=window._params.roll_deg,
        size_sfm=1.0,
    )
    world_overlay = np.asarray(window._tag_image_overlays(group, window._params, output_size=768)[0].polygon)
    candidates, _total = window._synthetic_tag_candidates()
    selected = frozenset(
        candidate.frame.file_path
        for candidate in candidates
        if candidate.frame.file_path.startswith("images/frame_000001_")
    )
    assert selected == frozenset({"images/frame_000001_nz.jpg"})

    marker = tmp_path / "black_marker.png"
    assert imwrite_unicode(marker, np.zeros((128, 128), dtype=np.uint8))
    center_sfm, normal_sfm, up_sfm, true_scale = window._synthetic_tag_placement_sfm()
    output_dir = tmp_path / "tagged"
    report = inject_synthetic_apriltag(
        SyntheticAprilTagConfig(
            input_transforms=case.transforms_for_processing(),
            output_dir=output_dir,
            tag_image=marker,
            tag_size_m=float(case.default_tag_size_m),
            true_scale=true_scale,
            tag_center_sfm=center_sfm,
            tag_normal_sfm=normal_sfm,
            tag_up_sfm=up_sfm,
            frame_file_paths=selected,
            copy_unselected_frames=False,
            output_tagged_only=True,
            cubemap_view_params=metadata,
            frame_transform_overrides=window._synthetic_frame_transform_overrides(),
            write_normalized_transforms=True,
        )
    )
    assert report["frames_written"] == 1

    source_rotation = window._source_equirect_rotation_for_group(group)
    assert source_rotation is not None
    display_matrix = world_display_matrix(case.coordinate_profile)
    base_equirect = render_generated_cubemap_source_axis(
        raw_group,
        source_rotation,
        cubemap_view_params=metadata,
        output_width=1024,
        output_height=512,
        image_cache={},
        sfm_to_preview_matrix=display_matrix,
    )
    output_frames = {
        face: replace(frame, image_path=output_dir / frame.file_path)
        if (output_dir / frame.file_path).is_file()
        else frame
        for face, frame in raw_group.frames_by_face.items()
    }
    tagged_equirect = render_generated_cubemap_source_axis(
        CubemapFrameGroup(raw_group.name, output_frames, raw_group.group_index),
        source_rotation,
        cubemap_view_params=metadata,
        output_width=1024,
        output_height=512,
        image_cache={},
        sfm_to_preview_matrix=display_matrix,
    )
    yaw, pitch, roll, fov = axis_face_view_params(group, "pz", fov_deg=window._params.fov_deg)
    anchor = PerspectiveParams(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, fov_deg=fov)
    params = _source_equirect_preview_params(window._params, "pz", RAY_BASIS_WORLD, anchor)
    base_view = equirect_to_perspective(base_equirect, params, output_size=768)
    tagged_view = equirect_to_perspective(tagged_equirect, params, output_size=768)
    diff = np.max(np.abs(tagged_view.astype(np.int16) - base_view.astype(np.int16)), axis=2)
    ys, xs = np.where(diff > 30)

    assert len(xs) > 100
    assert np.allclose([float(xs.mean()), float(ys.mean())], world_overlay.mean(axis=0), atol=2.0)
    window.deleteLater()


def test_current_case_synthetic_floor_tag_uses_floor_pole_image() -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    if "frame_000012" not in {group.name for group in window._raw_groups}:
        pytest.skip("local AprilTag floor regression frame is not available")
    window.validation_distance_spin.setValue(4.7)
    window.validation_angle_spin.setValue(75.0)
    window.validation_min_area_spin.setValue(64.0)
    window.set_tag_transform(
        center=(0.0, -1.817, -0.9),
        yaw_deg=180.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_sfm=0.1733,
    )

    candidates, _total = window._synthetic_tag_candidates()
    selected = {
        candidate.frame.file_path
        for candidate in candidates
        if candidate.frame.file_path.startswith("images/frame_000012_")
    }

    assert "images/frame_000012_top.jpg" in selected
    assert "images/frame_000012_bottom.jpg" not in selected
    window.deleteLater()


def test_current_case_synthetic_output_tag_is_not_mirrored(tmp_path: Path) -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    if "frame_000012" not in {group.name for group in window._raw_groups}:
        pytest.skip("local AprilTag floor regression frame is not available")
    window.validation_distance_spin.setValue(4.7)
    window.validation_angle_spin.setValue(75.0)
    window.validation_min_area_spin.setValue(64.0)
    window.set_tag_transform(
        center=(0.0, -1.817, -0.9),
        yaw_deg=-180.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_sfm=0.1733,
    )

    candidates, _total = window._synthetic_tag_candidates()
    inside = [
        candidate
        for candidate in candidates
        if float(candidate.projected_points[:, 0].min()) > 8.0
        and float(candidate.projected_points[:, 1].min()) > 8.0
        and float(candidate.projected_points[:, 0].max()) < candidate.frame.width - 8.0
        and float(candidate.projected_points[:, 1].max()) < candidate.frame.height - 8.0
    ]
    if not inside:
        pytest.skip("local AprilTag mirrored-output regression frame is not visible")
    candidate = max(inside, key=lambda item: item.area_px)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11), 7, 240)
    marker_path = tmp_path / "tag.png"
    assert imwrite_unicode(marker_path, marker)

    center_sfm, normal_sfm, up_sfm, true_scale = window._synthetic_tag_placement_sfm()
    output_dir = tmp_path / "tagged"
    report = inject_synthetic_apriltag(
        SyntheticAprilTagConfig(
            input_transforms=window.case.transforms_for_processing(),
            output_dir=output_dir,
            tag_image=marker_path,
            tag_size_m=float(window.case.default_tag_size_m),
            true_scale=true_scale,
            tag_center_sfm=center_sfm,
            tag_normal_sfm=normal_sfm,
            tag_up_sfm=up_sfm,
            frame_file_paths=frozenset({candidate.frame.file_path}),
            copy_unselected_frames=False,
            output_tagged_only=True,
            cubemap_view_params=case_cubemap_view_metadata(window.case),
            frame_transform_overrides=window._synthetic_frame_transform_overrides(),
            write_normalized_transforms=True,
        )
    )
    assert report["frames_written"] == 1
    frame = load_pinhole_frames(output_dir / "transforms.json", normalize_cubemap=False)[0]
    image = cv2.imread(str(output_dir / frame.file_path), cv2.IMREAD_COLOR)
    detections = detect_apriltags(image, frame, tag_size_m=float(window.case.default_tag_size_m), family="tag36h11", tag_ids={7})

    assert len(detections) == 1
    assert detections[0].tag_id == 7
    assert not detect_apriltags(
        cv2.flip(image, 1),
        frame,
        tag_size_m=float(window.case.default_tag_size_m),
        family="tag36h11",
        tag_ids={7},
    )
    assert not detect_apriltags(
        cv2.flip(image, 0),
        frame,
        tag_size_m=float(window.case.default_tag_size_m),
        family="tag36h11",
        tag_ids={7},
    )
    window.deleteLater()


def test_current_case_image_overlay_respects_synthetic_front_side() -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    if "frame_000001" not in window._source_equirect_rotations:
        pytest.skip("local AprilTag source equirect rotation is not available")
    window.select_camera_by_name("frame_000001")
    window.ray_basis_combo.setCurrentIndex(window.ray_basis_combo.findData(RAY_BASIS_WORLD))
    window.set_active_face("pz")
    group = window.selected_world_group()
    assert group is not None
    camera, right, up, forward = camera_pose_from_perspective_params(group, window._params)
    center = camera + forward * 5.0 + right * 0.5 + up * 0.25

    window.set_tag_transform(
        center=tuple(center),
        yaw_deg=window._params.yaw_deg,
        pitch_deg=window._params.pitch_deg,
        roll_deg=window._params.roll_deg,
        size_sfm=1.0,
    )
    assert len(window._right_image_tag_overlays(group, use_output_projection=True, output_size=768)) == 1

    window.set_tag_transform(
        center=tuple(center),
        yaw_deg=window._params.yaw_deg + 180.0,
        pitch_deg=-window._params.pitch_deg,
        roll_deg=window._params.roll_deg,
        size_sfm=1.0,
    )
    assert window._tag_image_overlays(group, window._params, output_size=768)
    assert window._right_image_tag_overlays(group, use_output_projection=True, output_size=768) == []
    window.deleteLater()


def test_current_case_generated_cube6_reconstruction_matches_expected_orientation() -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    case = load_case(case_dir)
    metadata = case_cubemap_view_metadata(case)
    if metadata is None:
        pytest.skip("local AprilTag comparison Cube6 metadata is not available")
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    raw_group = window.selected_raw_group()
    world_group = window.selected_world_group()
    source_rotation = window._source_equirect_rotations.get("frame_000001")
    if source_rotation is None:
        pytest.skip("local AprilTag source equirect rotation is not available")
    assert raw_group is not None
    assert world_group is not None

    display_matrix = world_display_matrix(case.coordinate_profile)
    reconstructed = render_generated_cubemap_source_axis(
        raw_group,
        source_rotation,
        cubemap_view_params=metadata,
        output_width=1024,
        output_height=512,
        image_cache={},
        sfm_to_preview_matrix=display_matrix,
    )
    expected_mapping = {
        "pz": "nz",
        "px": "nx",
        "nx": "px",
        "nz": "pz",
        "top": "top",
        "bottom": "bottom",
    }
    for face, expected_face in expected_mapping.items():
        target_frame = raw_group.frames_by_face.get(expected_face)
        assert target_frame is not None
        target = imread_unicode(target_frame.image_path)
        assert target is not None
        yaw, pitch, roll, fov = axis_face_view_params(world_group, face, fov_deg=90.0)
        anchor = PerspectiveParams(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, fov_deg=fov)
        params = _source_equirect_preview_params(anchor, face, RAY_BASIS_BOTH)
        rendered = equirect_to_perspective(reconstructed, params, output_size=512)
        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_AREA)
        error = float(np.mean(np.abs(rendered.astype(np.int16) - target.astype(np.int16))))
        assert error < 10.0
    window.deleteLater()


def test_current_case_source_equirect_tangent_axes_match_expected_faces() -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    case = load_case(case_dir)
    metadata = case_cubemap_view_metadata(case)
    if metadata is None:
        pytest.skip("local AprilTag comparison Cube6 metadata is not available")
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    world_group = window.selected_world_group()
    source_rotation = window._source_equirect_rotations.get("frame_000001")
    if source_rotation is None:
        pytest.skip("local AprilTag source equirect image is not available")
    assert world_group is not None

    display_matrix = world_display_matrix(case.coordinate_profile)
    preview_to_sfm = np.linalg.inv(display_matrix[:3, :3]).T
    expected_mapping = {
        "pz": "nz",
        "px": "nx",
        "nx": "px",
        "nz": "pz",
        "top": "top",
        "bottom": "bottom",
    }
    for face, expected_face in expected_mapping.items():
        yaw, pitch, roll, fov = axis_face_view_params(world_group, face, fov_deg=90.0)
        params = PerspectiveParams(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, fov_deg=fov)
        display_params = _source_equirect_preview_params(params, face, RAY_BASIS_BOTH)
        display_rotation = rotation_from_perspective_params(display_params)

        source_right = np.array([1.0, 0.0, 0.0]) @ display_rotation.T @ preview_to_sfm @ source_rotation
        source_up = np.array([0.0, 1.0, 0.0]) @ display_rotation.T @ preview_to_sfm @ source_rotation
        source_forward = np.array([0.0, 0.0, 1.0]) @ display_rotation.T @ preview_to_sfm @ source_rotation

        expected_yaw, expected_pitch = metadata.view_params[expected_face]
        expected_rotation = _rotation(expected_yaw, expected_pitch)
        expected_right = np.array([1.0, 0.0, 0.0]) @ expected_rotation.T
        expected_up = np.array([0.0, 1.0, 0.0]) @ expected_rotation.T
        expected_forward = np.array([0.0, 0.0, 1.0]) @ expected_rotation.T

        assert float(source_right @ expected_right) > 0.999
        assert float(source_up @ expected_up) > 0.999
        assert float(source_forward @ expected_forward) > 0.999
    window.deleteLater()


def test_current_case_pointcloud_drag_uses_grab_style_camera_motion() -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    window.set_active_face("pz")
    window.mode_combo.setCurrentIndex(window.mode_combo.findData(RIGHT_VIEW_POINTCLOUD))

    start_params = window._params
    start_rotation = rotation_from_perspective_params(start_params)
    start_forward = np.array([0.0, 0.0, 1.0]) @ start_rotation.T
    start_right = np.array([1.0, 0.0, 0.0]) @ start_rotation.T
    start_up = np.array([0.0, 1.0, 0.0]) @ start_rotation.T

    window._params = start_params
    window._on_right_view_dragged(-10.0, 0.0)
    after_left_rotation = rotation_from_perspective_params(window._params)
    after_left_forward = np.array([0.0, 0.0, 1.0]) @ after_left_rotation.T
    left_movement = after_left_forward - start_forward
    left_movement /= max(float(np.linalg.norm(left_movement)), 1e-12)
    assert window._params == params_from_drag(start_params, -10.0, 0.0)
    assert window._params.yaw_deg > start_params.yaw_deg
    assert float(left_movement @ start_right) > 0.99

    window._params = start_params
    window._on_right_view_dragged(0.0, 10.0)
    after_down_rotation = rotation_from_perspective_params(window._params)
    after_down_forward = np.array([0.0, 0.0, 1.0]) @ after_down_rotation.T
    down_movement = after_down_forward - start_forward
    down_movement /= max(float(np.linalg.norm(down_movement)), 1e-12)
    assert window._params == params_from_drag(start_params, 0.0, 10.0)
    assert window._params.pitch_deg < start_params.pitch_deg
    assert float(down_movement @ start_up) > 0.99
    window.deleteLater()


def test_current_case_source_equirect_drag_uses_grab_style_display_motion() -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    case = load_case(case_dir)
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    source_rotation = window._source_equirect_rotations.get("frame_000001")
    if source_rotation is None:
        pytest.skip("local AprilTag source equirect image is not available")
    window.set_active_face("pz")

    start_params = window._params
    display_matrix = world_display_matrix(case.coordinate_profile)
    preview_to_sfm = np.linalg.inv(display_matrix[:3, :3]).T

    def source_basis(params: PerspectiveParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        display_params = _source_equirect_preview_params(params, "pz", RAY_BASIS_BOTH, start_params)
        rotation = rotation_from_perspective_params(display_params)
        forward = np.array([0.0, 0.0, 1.0]) @ rotation.T @ preview_to_sfm @ source_rotation
        right = np.array([1.0, 0.0, 0.0]) @ rotation.T @ preview_to_sfm @ source_rotation
        up = np.array([0.0, 1.0, 0.0]) @ rotation.T @ preview_to_sfm @ source_rotation
        forward /= max(float(np.linalg.norm(forward)), 1e-12)
        right /= max(float(np.linalg.norm(right)), 1e-12)
        up /= max(float(np.linalg.norm(up)), 1e-12)
        return forward, right, up

    start_forward, start_right, start_up = source_basis(start_params)

    for mode in (RIGHT_VIEW_SOURCE_EQUIRECT, RIGHT_VIEW_RECONSTRUCTED_CUBE6):
        window.mode_combo.setCurrentIndex(window.mode_combo.findData(mode))

        window._params = start_params
        window._on_right_view_dragged(-10.0, 0.0)
        after_left_forward, _right, _up = source_basis(window._params)
        left_movement = after_left_forward - start_forward
        left_movement /= max(float(np.linalg.norm(left_movement)), 1e-12)
        assert window._params == params_from_drag(start_params, 10.0, 0.0)
        assert float(left_movement @ start_right) > 0.99

        window._params = start_params
        window._on_right_view_dragged(0.0, 10.0)
        after_down_forward, _right, _up = source_basis(window._params)
        down_movement = after_down_forward - start_forward
        down_movement /= max(float(np.linalg.norm(down_movement)), 1e-12)
        assert window._params == params_from_drag(start_params, 0.0, 10.0)
        assert float(down_movement @ start_up) > 0.99
    window.deleteLater()


def test_current_case_source_equirect_preview_reflects_vertical_display_motion() -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    if "frame_000001" not in window._source_equirect_rotations:
        pytest.skip("local AprilTag source equirect image is not available")
    window.set_active_face("pz")

    anchor = window._params
    dragged = params_from_drag(anchor, -10.0, -10.0)
    displayed_anchor = _source_equirect_preview_params(anchor, "pz", RAY_BASIS_BOTH, anchor)
    displayed_dragged = _source_equirect_preview_params(dragged, "pz", RAY_BASIS_BOTH, anchor)

    assert dragged.yaw_deg > anchor.yaw_deg
    assert displayed_dragged.yaw_deg == dragged.yaw_deg
    assert dragged.pitch_deg > anchor.pitch_deg
    assert displayed_dragged.pitch_deg < displayed_anchor.pitch_deg
    assert displayed_dragged.pitch_deg == pytest.approx(2.0 * anchor.pitch_deg - dragged.pitch_deg)
    assert displayed_anchor.roll_deg == normalize_yaw_deg(anchor.roll_deg + 180.0)
    assert displayed_dragged.roll_deg == normalize_yaw_deg(dragged.roll_deg + 180.0)
    assert _source_equirect_preview_params(dragged, "top", RAY_BASIS_BOTH, anchor) == dragged
    assert _source_equirect_preview_params(dragged, "pz", RAY_BASIS_IMAGE, anchor) == dragged
    window.deleteLater()


def test_current_case_image_preview_uses_source_equirect_when_available() -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    world_group = window.selected_world_group()
    assert world_group is not None
    case = load_case(case_dir)
    source_paths = _resolve_source_equirect_paths(case, ("frame_000001",))
    source_path = source_paths.get("frame_000001")
    source_rotation = window._source_equirect_rotations.get("frame_000001")
    if source_path is None or source_rotation is None:
        pytest.skip("local AprilTag source equirect image is not available")

    window.mode_combo.setCurrentIndex(window.mode_combo.findData(RIGHT_VIEW_SOURCE_EQUIRECT))
    assert "source-equirect" in window._displayed_image_key
    assert str(source_path) in window._displayed_image_key

    raw_group = window.selected_raw_group()
    assert raw_group is not None
    expected_rotation = source_equirect_base_rotation(raw_group, cubemap_view_params=case_cubemap_view_metadata(case))
    assert expected_rotation is not None
    source_local_from_lichtfeld_local = np.diag([1.0, -1.0, -1.0])
    assert np.allclose(source_rotation, expected_rotation @ source_local_from_lichtfeld_local, atol=1e-6)

    source = imread_unicode(source_path)
    display_matrix = world_display_matrix(case.coordinate_profile)
    source_equirect = render_source_equirect_axis(
        source,
        source_rotation,
        output_width=1024,
        output_height=512,
        sfm_to_preview_matrix=display_matrix,
    )
    yaw, pitch, roll, fov = axis_face_view_params(world_group, "pz", fov_deg=90.0)
    rendered = equirect_to_perspective(
        source_equirect,
        PerspectiveParams(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, fov_deg=fov),
        output_size=512,
    )
    direct = render_source_equirect_perspective(
        source,
        source_rotation,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        fov_deg=fov,
        output_size=512,
        sfm_to_preview_matrix=display_matrix,
    )
    assert rendered.shape == (512, 512, 3)
    assert float(np.std(rendered)) > 1.0
    assert float(np.mean(np.abs(rendered.astype(np.int16) - direct.astype(np.int16)))) < 8.0
    window.deleteLater()


def test_current_case_image_preview_can_use_generated_cube6_reconstruction() -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    if "frame_000001" not in window._source_equirect_rotations:
        pytest.skip("local AprilTag source equirect rotation is not available")

    mode = window.mode_combo.findData(RIGHT_VIEW_RECONSTRUCTED_CUBE6)
    assert mode >= 0
    window.mode_combo.setCurrentIndex(mode)

    assert "cube6-reconstruct" in window._displayed_image_key
    assert "source-equirect" not in window._displayed_image_key
    assert "image preview=Cube6 reconstructed" in window.case_label.text()
    window.deleteLater()


def test_current_case_reconstructed_image_tag_overlay_tracks_view_drag() -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    if "frame_000001" not in window._source_equirect_rotations:
        pytest.skip("local AprilTag source equirect rotation is not available")
    window.select_camera_by_name("frame_000001")
    window.ray_basis_combo.setCurrentIndex(window.ray_basis_combo.findData(RAY_BASIS_WORLD))
    window.mode_combo.setCurrentIndex(window.mode_combo.findData(RIGHT_VIEW_RECONSTRUCTED_CUBE6))
    window.set_active_face("pz")
    group = window.selected_world_group()
    assert group is not None
    camera, right, up, forward = camera_pose_from_perspective_params(group, window._params)
    window.set_tag_transform(
        center=tuple(camera + forward * 5.0 + right * 0.5 + up * 0.25),
        yaw_deg=window._params.yaw_deg,
        pitch_deg=window._params.pitch_deg,
        roll_deg=window._params.roll_deg,
        size_sfm=1.0,
    )

    before = window._right_image_tag_overlays(group, use_output_projection=True, output_size=768)
    assert len(before) == 1
    assert np.allclose(
        np.asarray(before[0].polygon),
        np.asarray(window._tag_image_overlays(group, window._params, output_size=768)[0].polygon),
        atol=1e-4,
    )

    window._on_right_view_dragged(-24.0, 12.0)
    after = window._right_image_tag_overlays(group, use_output_projection=True, output_size=768)
    assert len(after) == 1
    expected = window._tag_image_overlays(group, window._params, output_size=768)
    assert len(expected) == 1
    assert np.allclose(np.asarray(after[0].polygon), np.asarray(expected[0].polygon), atol=1e-4)
    assert not np.allclose(np.asarray(after[0].polygon), np.asarray(before[0].polygon), atol=1.0)
    window.deleteLater()


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
    window.deleteLater()


def test_scene_viewer_tag_transform_controls_sync_world_and_point_views(tmp_path: Path) -> None:
    _app()
    case_dir = _write_cube6_case(tmp_path)

    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    assert window.tag_size_sfm_spin.value() == pytest.approx(0.64)
    for spin in (
        window.tag_x_spin,
        window.tag_y_spin,
        window.tag_z_spin,
        window.tag_yaw_spin,
        window.tag_pitch_spin,
        window.tag_roll_spin,
        window.tag_size_sfm_spin,
    ):
        assert isinstance(spin, DragDoubleSpinBox)
        assert spin.buttonSymbols() == QAbstractSpinBox.NoButtons

    window.set_tag_transform(
        center=(1.0, 2.0, 3.0),
        yaw_deg=90.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_sfm=1.25,
    )

    for view in (window.world_view, window.point_view):
        assert np.allclose(view._tag_center, [1.0, 2.0, 3.0])
        assert np.allclose(view._tag_normal, [-1.0, 0.0, 0.0], atol=1e-6)
        assert np.allclose(view._tag_up, [0.0, 1.0, 0.0], atol=1e-6)
        assert view._tag_size_m == pytest.approx(1.25)
        assert view._true_scale == pytest.approx(1.0)
    window.deleteLater()


def test_scene_viewer_synthetic_candidates_use_tag_front_side(tmp_path: Path) -> None:
    _app()
    case_dir = _write_cube6_case(tmp_path)

    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    window.validation_min_area_spin.setValue(1.0)
    window.validation_distance_spin.setValue(10.0)
    window.validation_angle_spin.setValue(80.0)
    window.set_tag_transform(
        center=(0.0, 0.0, 3.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_sfm=1.0,
    )

    candidates, total_frames = window._synthetic_tag_candidates()

    assert window.world_view._tag_validation_distance_sfm == pytest.approx(10.0)
    assert window.point_view._tag_validation_distance_sfm is None
    assert total_frames == 12
    paths = {candidate.frame.file_path for candidate in candidates}
    assert "images/cam_001_pz.jpg" in paths
    assert all(path.endswith("_pz.jpg") for path in paths)

    window.set_tag_transform(
        center=(0.0, 0.0, 3.0),
        yaw_deg=180.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_sfm=1.0,
    )
    back_candidates, _total_frames = window._synthetic_tag_candidates()
    assert back_candidates == ()
    window.deleteLater()


def test_scene_viewer_synthetic_scale_validation_writes_result(tmp_path: Path) -> None:
    _app()
    case_dir = _write_scale_validation_cube6_case(tmp_path)

    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    window.validation_min_area_spin.setValue(1.0)
    window.validation_distance_spin.setValue(10.0)
    window.validation_angle_spin.setValue(80.0)
    window.set_active_face("pz")
    window.set_tag_transform(
        center=(0.0, 0.0, 2.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_sfm=0.64,
    )

    loop = QEventLoop()
    window.validation_finished.connect(loop.quit)
    window.run_synthetic_scale_validation()

    assert window._validation_running
    assert not window.run_validation_button.isEnabled()
    assert window.run_validation_button.text() == "実行中..."
    QTimer.singleShot(10_000, loop.quit)
    loop.exec()

    assert not window._validation_running

    report_paths = sorted((case_dir / "runs").glob("viewer_synthetic_*/viewer_scale_validation_report.json"))
    assert len(report_paths) == 1
    report = json.loads(report_paths[0].read_text(encoding="utf-8"))
    output_dir = report_paths[0].parent
    output_transforms = json.loads((output_dir / "transforms.json").read_text(encoding="utf-8"))
    output_images = sorted((output_dir / "images").glob("*.png"))

    assert report["candidate_count"] == 2
    assert report["synthetic_report"]["frames_written"] == 2
    assert report["synthetic_report"]["frames_copied"] == 0
    assert report["synthetic_report"]["transforms_frame_count"] == 2
    assert report["loaded_frame_count"] == 2
    assert len(output_transforms["frames"]) == 2
    assert len(output_images) == 2
    assert report["observation_count"] >= 2
    assert report["estimate"]["scale"] == pytest.approx(0.25, rel=0.08)
    assert "scale=" in window.validation_status_label.text()
    assert window.run_validation_button.isEnabled()
    assert window.run_validation_button.text() == "合成→検出"
    log_text = window.log.toPlainText()
    assert "実行中 1/6: 候補フレームを選別中" in log_text
    assert "実行中 3/6: Cube6画像へタグを合成中" in log_text
    assert "実行中 4/6: AprilTagを検出中" in log_text
    assert "実行中 6/6: レポートを書き出し中" in log_text
    window.deleteLater()


def test_scene_viewer_projects_tag_overlay_into_image_view_params(tmp_path: Path) -> None:
    _app()
    case_dir = _write_cube6_case(tmp_path)

    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    window.set_active_face("pz")
    window.set_tag_transform(
        center=(0.0, 0.0, 3.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_sfm=1.0,
    )
    group = window.selected_world_group()
    assert group is not None

    overlays = window._tag_image_overlays(group, window._params, output_size=768)

    assert len(overlays) == 1
    points = np.asarray(overlays[0].polygon, dtype=np.float64)
    assert points.shape == (4, 2)
    assert np.all(np.isfinite(points))
    assert np.allclose(points.mean(axis=0), [383.5, 383.5], atol=1.0)
    assert 310.0 < float(points[:, 0].min()) < 330.0
    assert 437.0 < float(points[:, 0].max()) < 457.0
    assert 310.0 < float(points[:, 1].min()) < 330.0
    assert 437.0 < float(points[:, 1].max()) < 457.0
    window.deleteLater()


def test_scene_viewer_image_tag_overlay_matches_pointcloud_projection(tmp_path: Path) -> None:
    _app()
    case_dir = _write_cube6_case(tmp_path)

    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    window.set_active_face("pz")
    group = window.selected_world_group()
    assert group is not None
    camera, right, up, forward = camera_pose_from_perspective_params(group, window._params)
    window.set_tag_transform(
        center=tuple(camera + forward * 3.0 + right * 0.35 + up * 0.2),
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_sfm=0.5,
    )
    window.point_view.resize(768, 768)
    window.mode_combo.setCurrentIndex(window.mode_combo.findData(RIGHT_VIEW_POINTCLOUD))
    window._sync_views()
    point_projection, _depth = window.point_view._project(window._tag_corners_world_display())

    overlays = window._tag_image_overlays(group, window._params, output_size=768)

    assert len(overlays) == 1
    assert np.allclose(np.asarray(overlays[0].polygon), point_projection, atol=1e-4)
    window.deleteLater()
