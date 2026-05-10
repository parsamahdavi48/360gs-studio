from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from core.apriltag_cubemap import CubemapViewMetadata
from core.apriltag_geometry import PinholeFrame
from core.image_io import imread_unicode, imwrite_unicode
from devtools.apriltag.case import AprilTagDevCase, load_case, save_case
from devtools.apriltag.cubemap_preview import (
    CubemapFrameGroup,
    axis_face_view_params,
    render_cubemap_axis_equirect,
)
from devtools.apriltag.scene_viewer import (
    AprilTagSceneViewerWindow,
    camera_pose_from_perspective_params,
    case_cubemap_view_metadata,
    closest_image_face_for_world_face,
    face_forward_ray,
    image_ray_display_matrix_for_profile,
    opposite_image_face_for_world_face,
    transform_group_for_world_display,
)
from devtools.apriltag.coordinates import world_display_matrix
from gui.common.perspective_preview import PerspectiveParams, equirect_to_perspective


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
    assert closest[0] == "nz"
    assert closest[1] < 1e-4
    assert closest[2] == pytest.approx(180.0, abs=0.1)
    assert opposite[0] == "pz"
    assert opposite[1] < 1e-4
    assert image_basis_closest[0] == "pz"
    assert image_basis_closest[1] < 1e-4
    assert "active pz->image nz" in window.case_label.text()
    window.ray_basis_combo.setCurrentIndex(window.ray_basis_combo.findData("image"))
    assert "active pz->image pz" in window.case_label.text()
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


def test_current_case_json_face_image_preview_matches_expected_saved_faces() -> None:
    case_dir = Path("_compare/apriltag_test/cases/current")
    if not (case_dir / "case.json").is_file():
        pytest.skip("local AprilTag comparison case is not available")
    _app()
    window = AprilTagSceneViewerWindow(initial_case=case_dir)
    world_group = window.selected_world_group()
    image_group = window.selected_image_render_group()
    assert world_group is not None
    assert image_group is not None

    equirect = render_cubemap_axis_equirect(image_group, output_width=1024, output_height=512)
    image_root = case_dir / "input" / "images"
    expected = {
        "pz": "nz",
        "px": "nx",
        "nx": "px",
        "nz": "pz",
        "top": "top",
        "bottom": "bottom",
    }

    def error_to(rendered: np.ndarray, face: str) -> float:
        saved = imread_unicode(image_root / f"frame_000001_{face}.jpg")
        saved = cv2.resize(saved[:, :, :3], (512, 512), interpolation=cv2.INTER_AREA)
        return float(np.mean(np.abs(rendered.astype(np.int16) - saved.astype(np.int16))))

    for json_face, saved_face in expected.items():
        yaw, pitch, roll, fov = axis_face_view_params(world_group, json_face, fov_deg=90.0)
        rendered = equirect_to_perspective(
            equirect,
            PerspectiveParams(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, fov_deg=fov),
            output_size=512,
        )
        expected_error = error_to(rendered, saved_face)
        other_errors = [
            error_to(rendered, face)
            for face in expected.values()
            if face != saved_face
        ]
        assert expected_error < 12.0, json_face
        assert expected_error * 4.0 < min(other_errors), json_face
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
