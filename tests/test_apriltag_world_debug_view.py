from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from core.apriltag_geometry import PinholeFrame
from devtools.apriltag.coordinates import (
    combined_pointcloud_display_matrix,
    pointcloud_display_matrix,
    world_display_matrix,
)
from devtools.apriltag.cubemap_preview import CubemapFrameGroup
from devtools.apriltag.scene_viewer import (
    estimate_world_display_matrix_from_metashape,
    transform_group_for_world_display,
)
from devtools.apriltag.world_debug_view import (
    AprilTagWorldDebugView,
    PointCloudSample,
    load_point_cloud_sample,
    transform_point_cloud_sample,
)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _frame_at(name: str, position: tuple[float, float, float]) -> PinholeFrame:
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(position, dtype=float)
    return PinholeFrame(
        frame_id=f"{name}_pz",
        file_path=f"images/{name}_pz.png",
        image_path=Path(f"images/{name}_pz.png"),
        width=100,
        height=100,
        fl_x=50.0,
        fl_y=50.0,
        cx=49.5,
        cy=49.5,
        transform_matrix=transform,
    )


def _frame_with_rotation(name: str, face: str, rotation: np.ndarray) -> PinholeFrame:
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    return PinholeFrame(
        frame_id=f"{name}_{face}",
        file_path=f"images/{name}_{face}.png",
        image_path=Path(f"images/{name}_{face}.png"),
        width=100,
        height=100,
        fl_x=50.0,
        fl_y=50.0,
        cx=49.5,
        cy=49.5,
        transform_matrix=transform,
    )


def test_load_point_cloud_sample_ascii_ply(tmp_path: Path) -> None:
    ply = tmp_path / "points.ply"
    ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0 0 0 255 0 0",
                "1 0 0 0 255 0",
                "0 1 0 0 0 255",
                "0 0 1 255 255 255",
            ]
        ),
        encoding="ascii",
    )

    sample = load_point_cloud_sample(ply, max_points=2)

    assert sample.source_count == 4
    assert sample.points.shape == (2, 3)
    assert sample.colors is not None
    assert sample.colors.shape == (2, 3)


def test_transform_point_cloud_sample_applies_lichtfeld_display_matrix() -> None:
    sample = PointCloudSample(
        points=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        colors=None,
        source_count=1,
    )

    transformed = transform_point_cloud_sample(sample, pointcloud_display_matrix("lichtfeld_cube6"))

    assert np.allclose(transformed.points[0], [-1.0, 2.0, -3.0])


def test_transform_point_cloud_sample_aligns_pre_final_lichtfeld_pointcloud() -> None:
    sample = PointCloudSample(
        points=np.array([[3.0, -2.0, 1.0]], dtype=np.float32),
        colors=None,
        source_count=1,
    )

    transformed = transform_point_cloud_sample(sample, pointcloud_display_matrix("lichtfeld_cube6_pre_final_ply"))

    assert np.allclose(transformed.points[0], [-1.0, 2.0, -3.0])


def test_lichtfeld_world_display_matrix_restores_metashape_axes() -> None:
    sample = PointCloudSample(
        points=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        colors=None,
        source_count=1,
    )

    world_matrix = world_display_matrix("lichtfeld_cube6")
    final_pointcloud_matrix = combined_pointcloud_display_matrix("lichtfeld_cube6")
    transformed_world = transform_point_cloud_sample(sample, world_matrix)
    transformed_pointcloud = transform_point_cloud_sample(sample, final_pointcloud_matrix)

    assert transformed_world is not None
    assert np.allclose(transformed_world.points[0], [-1.0, 2.0, -3.0])
    assert np.allclose(transformed_pointcloud.points[0], [1.0, 2.0, 3.0])


def test_world_debug_view_accepts_scene_and_tag() -> None:
    _app()
    frame = PinholeFrame(
        frame_id="frame_0001_pz",
        file_path="images/frame_0001_pz.png",
        image_path=Path("images/frame_0001_pz.png"),
        width=100,
        height=100,
        fl_x=50.0,
        fl_y=50.0,
        cx=49.5,
        cy=49.5,
        transform_matrix=np.eye(4),
    )
    group = CubemapFrameGroup(name="frame_0001", frames_by_face={"pz": frame})
    view = AprilTagWorldDebugView()

    view.set_groups((group,))
    view.set_selected_group("frame_0001")
    view.set_preview_params(yaw_deg=0.0, pitch_deg=0.0, fov_deg=90.0)
    view.set_grid(step=1.0, extent=10.0)
    view.set_tag(
        center=np.array([0.0, 0.0, 5.0]),
        normal=np.array([0.0, 0.0, -1.0]),
        up=np.array([0.0, 1.0, 0.0]),
        tag_size_m=0.16,
        true_scale=0.25,
    )

    assert view.sizeHint().width() > 0
    view.deleteLater()


def test_world_debug_view_exposes_tag_front_direction_segment() -> None:
    _app()
    view = AprilTagWorldDebugView()
    view.set_grid(step=1.0, extent=10.0)
    view.set_tag(
        center=np.array([1.0, 2.0, 3.0]),
        normal=np.array([0.0, 0.0, -1.0]),
        up=np.array([0.0, 1.0, 0.0]),
        tag_size_m=0.16,
        true_scale=0.25,
    )

    segments = {label: (start, end, dashed) for label, start, end, _color, dashed in view._tag_orientation_segments()}
    front_start, front_end, front_dashed = segments["front"]
    back_start, back_end, back_dashed = segments["back"]
    up_start, up_end, up_dashed = segments["up"]

    assert np.allclose(front_start, [1.0, 2.0, 3.0])
    assert np.allclose((front_end - front_start) / np.linalg.norm(front_end - front_start), [0.0, 0.0, -1.0])
    assert front_dashed is False
    assert np.allclose((back_end - back_start) / np.linalg.norm(back_end - back_start), [0.0, 0.0, 1.0])
    assert back_dashed is True
    assert np.allclose((up_end - up_start) / np.linalg.norm(up_end - up_start), [0.0, 1.0, 0.0])
    assert up_dashed is False
    view.deleteLater()


def test_world_debug_view_exposes_tag_validation_distance_sphere() -> None:
    _app()
    view = AprilTagWorldDebugView()
    center = np.array([1.0, 2.0, 3.0], dtype=float)
    view.set_tag(
        center=center,
        normal=np.array([0.0, 0.0, -1.0]),
        up=np.array([0.0, 1.0, 0.0]),
        tag_size_m=0.16,
        true_scale=0.25,
    )
    view.set_tag_validation_distance(4.0)

    circles = dict(view._tag_validation_distance_circles(samples=32))

    assert set(circles) == {"XZ", "XY", "YZ"}
    for points in circles.values():
        assert points.shape == (32, 3)
        assert np.allclose(np.linalg.norm(points - center, axis=1), 4.0)
    assert np.allclose(circles["XZ"][:, 1], center[1])
    assert np.allclose(circles["XY"][:, 2], center[2])
    assert np.allclose(circles["YZ"][:, 0], center[0])

    view.set_tag_validation_distance(0.0)
    assert view._tag_validation_distance_circles() == ()
    view.deleteLater()


def test_world_debug_frustum_converts_camera_preview_axes_to_world_axes() -> None:
    _app()
    view = AprilTagWorldDebugView()

    view.set_preview_params(yaw_deg=180.0, pitch_deg=0.0, fov_deg=90.0)
    view.set_preview_to_world_matrix(np.diag([-1.0, 1.0, -1.0]))
    forward, corners = view._preview_frustum_rays_in_world()

    assert np.allclose(forward, [0.0, 0.0, 1.0])
    assert np.all(corners @ forward > 0.0)
    view.deleteLater()


def test_world_debug_view_selected_face_rays_follow_frame_rotations() -> None:
    _app()
    view = AprilTagWorldDebugView()
    group = CubemapFrameGroup(
        name="frame_0001",
        frames_by_face={
            "pz": _frame_with_rotation("frame_0001", "pz", np.eye(3)),
            "nz": _frame_with_rotation("frame_0001", "nz", np.diag([-1.0, 1.0, -1.0])),
        },
    )

    segments = view._selected_face_ray_segments(group)
    directions = {
        label: (end - start) / np.linalg.norm(end - start)
        for label, start, end, _color in segments
    }

    assert np.allclose(directions["pz"], [0.0, 0.0, 1.0])
    assert np.allclose(directions["nz"], [0.0, 0.0, -1.0])
    view.deleteLater()


def test_world_debug_view_face_direction_rays_use_frame_rotations_directly() -> None:
    _app()
    view = AprilTagWorldDebugView()
    group = CubemapFrameGroup(
        name="frame_0001",
        frames_by_face={
            "pz": _frame_with_rotation("frame_0001", "pz", np.eye(3)),
            "nz": _frame_with_rotation("frame_0001", "nz", np.diag([-1.0, 1.0, -1.0])),
        },
    )
    segments = view._selected_face_ray_segments(group)
    directions = {
        label: (end - start) / np.linalg.norm(end - start)
        for label, start, end, _color in segments
    }

    assert np.allclose(directions["pz"], [0.0, 0.0, 1.0])
    assert np.allclose(directions["nz"], [0.0, 0.0, -1.0])
    view.deleteLater()


def test_world_debug_view_keeps_image_rays_separate_from_world_face_rays() -> None:
    _app()
    view = AprilTagWorldDebugView()
    world_group = CubemapFrameGroup(
        name="frame_0001",
        frames_by_face={
            "pz": _frame_with_rotation("frame_0001", "pz", np.eye(3)),
        },
    )
    image_group = CubemapFrameGroup(
        name="frame_0001",
        frames_by_face={
            "pz": _frame_with_rotation("frame_0001", "pz", np.diag([-1.0, 1.0, -1.0])),
        },
    )

    view.set_groups((world_group,))
    view.set_image_ray_groups((image_group,))
    view.set_selected_group("frame_0001")
    view.set_face_ray_mode("image")
    world_segments = view._selected_face_ray_segments(view._selected_group())
    image_segments = view._selected_image_ray_segments(view._selected_image_ray_group())
    world_direction = (world_segments[0][2] - world_segments[0][1]) / np.linalg.norm(
        world_segments[0][2] - world_segments[0][1]
    )
    image_direction = (image_segments[0][2] - image_segments[0][1]) / np.linalg.norm(
        image_segments[0][2] - image_segments[0][1]
    )

    assert view._selected_group() is world_group
    assert view._selected_image_ray_group() is image_group
    assert view._face_ray_mode == "image"
    assert np.allclose(world_direction, [0.0, 0.0, 1.0])
    assert np.allclose(image_direction, [0.0, 0.0, -1.0])
    view.set_face_ray_mode("both")
    assert view._face_ray_mode == "both"
    view.deleteLater()


def test_world_display_group_rotates_camera_position_and_rotation() -> None:
    transform = np.eye(4)
    transform[:3, 3] = np.array([1.0, 2.0, 3.0])
    frame = PinholeFrame(
        frame_id="frame_0001_pz",
        file_path="images/frame_0001_pz.png",
        image_path=Path("images/frame_0001_pz.png"),
        width=100,
        height=100,
        fl_x=50.0,
        fl_y=50.0,
        cx=49.5,
        cy=49.5,
        transform_matrix=transform,
    )
    group = CubemapFrameGroup(name="frame_0001", frames_by_face={"pz": frame})

    transformed = transform_group_for_world_display(group, world_display_matrix("lichtfeld_cube6"))
    transformed_frame = transformed.frames_by_face["pz"]

    assert np.allclose(transformed_frame.camera_position_sfm, [-1.0, 2.0, -3.0])
    assert np.allclose(transformed_frame.camera_to_world_rotation, np.diag([-1.0, 1.0, -1.0]))


def test_world_display_matrix_can_be_estimated_from_metashape_xml(tmp_path: Path) -> None:
    xml = tmp_path / "metashape.xml"
    xml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<document><chunk><cameras>
  <camera id="0" label="frame_a"><transform>1 0 0 -1 0 1 0 2 0 0 1 -3 0 0 0 1</transform></camera>
  <camera id="1" label="frame_b"><transform>1 0 0 -4 0 1 0 5 0 0 1 -6 0 0 0 1</transform></camera>
  <camera id="2" label="frame_c"><transform>1 0 0 7 0 1 0 8 0 0 1 9 0 0 0 1</transform></camera>
</cameras></chunk></document>
""",
        encoding="utf-8",
    )
    groups = (
        CubemapFrameGroup(name="frame_a", frames_by_face={"pz": _frame_at("frame_a", (1.0, 2.0, 3.0))}),
        CubemapFrameGroup(name="frame_b", frames_by_face={"pz": _frame_at("frame_b", (4.0, 5.0, 6.0))}),
        CubemapFrameGroup(name="frame_c", frames_by_face={"pz": _frame_at("frame_c", (-7.0, 8.0, -9.0))}),
    )

    estimated = estimate_world_display_matrix_from_metashape(groups, xml)

    assert estimated is not None
    matrix, rmse, count = estimated
    assert count == 3
    assert rmse < 1e-9
    assert np.allclose(matrix[:3, :3], np.diag([-1.0, 1.0, -1.0]))


def test_world_debug_view_orientation_gizmo_marks_positive_axes() -> None:
    _app()
    view = AprilTagWorldDebugView()
    origin = QPointF(100.0, 100.0)
    view._view_yaw_deg = 0.0
    view._view_pitch_deg = 0.0

    points = view._orientation_axis_points(origin, 20.0)

    assert view._screen_basis_determinant() > 0.999
    assert points["X"].x() < origin.x()
    assert points["X"].y() == origin.y()
    assert points["Y"].x() == origin.x()
    assert points["Y"].y() < origin.y()
    assert points["Z"].x() == origin.x()
    assert points["Z"].y() == origin.y()
    view.deleteLater()


def test_world_debug_view_orientation_gizmo_uses_right_handed_view_basis() -> None:
    _app()
    view = AprilTagWorldDebugView()
    origin = QPointF(100.0, 100.0)
    view._view_yaw_deg = 90.0
    view._view_pitch_deg = 0.0

    points = view._orientation_axis_points(origin, 20.0)

    assert view._screen_basis_determinant() > 0.999
    assert points["Z"].x() > origin.x()
    assert points["Z"].y() == origin.y()
    assert points["Y"].x() == origin.x()
    assert points["Y"].y() < origin.y()
    assert abs(points["X"].x() - origin.x()) < 1e-12
    assert abs(points["X"].y() - origin.y()) < 1e-12
    view.deleteLater()


def test_world_debug_view_orientation_gizmo_matches_orthographic_projection() -> None:
    _app()
    view = AprilTagWorldDebugView()
    view.resize(400, 300)
    view._view_center = np.zeros(3)
    view._pixels_per_unit = 25.0
    view._view_yaw_deg = 33.0
    view._view_pitch_deg = -18.0
    origin = QPointF(view.width() * 0.5, view.height() * 0.5)

    gizmo_points = view._orientation_axis_points(origin, view._pixels_per_unit)
    axis_points = {
        label: axis
        for label, axis, _color in view._axis_definitions()
    }

    for label, axis in axis_points.items():
        projected, _depth = view._project(axis.reshape(1, 3))
        assert np.allclose([gizmo_points[label].x(), gizmo_points[label].y()], projected[0])
    view.deleteLater()


def test_world_debug_view_grid_bounds_cover_oblique_viewport_ground_projection() -> None:
    _app()
    view = AprilTagWorldDebugView()
    view.resize(800, 600)
    view._view_center = np.zeros(3)
    view._pixels_per_unit = 20.0
    view._view_yaw_deg = 35.0
    view._view_pitch_deg = -28.0

    fallback = view._fallback_grid_bounds()
    ground = view._screen_ground_points(margin_px=0.0)
    bounds = view._grid_bounds()

    assert ground.shape == (4, 3)
    assert bounds.x_min <= float(np.min(ground[:, 0])) <= float(np.max(ground[:, 0])) <= bounds.x_max
    assert bounds.z_min <= float(np.min(ground[:, 2])) <= float(np.max(ground[:, 2])) <= bounds.z_max
    assert (bounds.x_max - bounds.x_min) > (fallback[1] - fallback[0])
    assert (bounds.z_max - bounds.z_min) > (fallback[3] - fallback[2])
    assert ((bounds.x_max - bounds.x_min) / bounds.step + (bounds.z_max - bounds.z_min) / bounds.step) <= 180.0
    view.deleteLater()


def test_world_debug_view_grid_bounds_fall_back_near_horizon() -> None:
    _app()
    view = AprilTagWorldDebugView()
    view.resize(800, 600)
    view._view_center = np.zeros(3)
    view._pixels_per_unit = 20.0
    view._view_yaw_deg = 35.0
    view._view_pitch_deg = 0.0

    bounds = view._grid_bounds()

    assert np.isfinite(
        [
            bounds.x_min,
            bounds.x_max,
            bounds.z_min,
            bounds.z_max,
            bounds.step,
        ]
    ).all()
    assert (bounds.x_min, bounds.x_max, bounds.z_min, bounds.z_max) == view._fallback_grid_bounds()
    view.deleteLater()


def test_world_debug_view_default_gizmo_screen_vectors_are_projection_locked() -> None:
    _app()
    view = AprilTagWorldDebugView()
    origin = QPointF(100.0, 100.0)

    points = view._orientation_axis_points(origin, 20.0)

    assert points["X"].x() < origin.x()
    assert points["X"].y() < origin.y()
    assert abs(points["Y"].x() - origin.x()) < 1e-12
    assert points["Y"].y() < origin.y()
    assert points["Z"].x() > origin.x()
    assert points["Z"].y() < origin.y()
    view.deleteLater()


def test_world_debug_view_free_and_fixed_views_share_projection_contract() -> None:
    _app()
    free_view = AprilTagWorldDebugView()
    fixed_view = AprilTagWorldDebugView()
    for view in (free_view, fixed_view):
        view.resize(400, 300)
        view._view_center = np.zeros(3)
        view._pixels_per_unit = 25.0
    free_view._view_yaw_deg = 33.0
    free_view._view_pitch_deg = -18.0
    right, up, forward = free_view._view_basis()
    fixed_view.set_fixed_view(
        center=np.zeros(3),
        right=right,
        up=up,
        forward=forward,
        pixels_per_unit=25.0,
    )
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=float,
    )

    free_xy, free_depth = free_view._project(points)
    fixed_xy, fixed_depth = fixed_view._project(points)

    assert free_view._screen_basis_determinant() > 0.999
    assert fixed_view._screen_basis_determinant() > 0.999
    assert np.allclose(free_xy, fixed_xy)
    assert np.allclose(free_depth, fixed_depth)
    free_view.deleteLater()
    fixed_view.deleteLater()


def test_world_debug_view_fixed_view_uses_same_projection_path() -> None:
    _app()
    view = AprilTagWorldDebugView()
    view.resize(400, 300)

    view.set_fixed_view(
        center=np.array([0.0, 0.0, 0.0]),
        right=np.array([-1.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        pixels_per_unit=20.0,
    )
    projected, depth = view._project(np.array([[1.0, 2.0, 3.0]], dtype=float))

    assert np.allclose(projected[0], [view.width() * 0.5 - 20.0, view.height() * 0.5 - 40.0])
    assert np.allclose(depth[0], 3.0)
    view.clear_fixed_view()
    assert view._fixed_view_basis is None
    view.deleteLater()


def test_world_debug_view_fixed_perspective_view_uses_pinhole_projection() -> None:
    _app()
    view = AprilTagWorldDebugView()
    view.resize(400, 300)

    view.set_fixed_perspective_view(
        camera_position=np.array([0.0, 0.0, 0.0]),
        right=np.array([-1.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        fov_deg=90.0,
    )
    projected, depth = view._project(np.array([[1.0, 0.0, 2.0], [0.0, 0.0, -1.0]], dtype=float))

    focal = min(view.width(), view.height()) * 0.5
    assert np.allclose(projected[0], [view.width() * 0.5 - focal * 0.5, view.height() * 0.5])
    assert np.allclose(depth[0], 2.0)
    assert np.all(np.isnan(projected[1]))
    view.deleteLater()


def test_world_debug_view_fixed_perspective_view_uses_right_handed_screen_basis() -> None:
    _app()
    view = AprilTagWorldDebugView()

    view.set_fixed_perspective_view(
        camera_position=np.array([0.0, 0.0, 0.0]),
        right=np.array([0.0, 0.0, 1.0]),
        up=np.array([0.0, 1.0, 0.0]),
        forward=np.array([1.0, 0.0, 0.0]),
        fov_deg=90.0,
    )
    assert view._fixed_view_basis is not None
    right, up, forward = view._fixed_view_basis
    basis = np.column_stack([right, up, -forward])

    assert np.linalg.det(basis) > 0.999
    assert view._screen_basis_determinant() > 0.999
    assert np.allclose(forward, [1.0, 0.0, 0.0])
    assert float(up @ np.array([0.0, 1.0, 0.0])) > 0.999
    view.deleteLater()


def test_world_debug_view_fixed_view_drag_emits_camera_delta() -> None:
    _app()

    class MouseEventStub:
        def __init__(
            self,
            position: QPointF,
            button: Qt.MouseButton,
            buttons: Qt.MouseButton,
        ) -> None:
            self._position = position
            self._button = button
            self._buttons = buttons

        def position(self) -> QPointF:
            return self._position

        def button(self) -> Qt.MouseButton:
            return self._button

        def buttons(self) -> Qt.MouseButton:
            return self._buttons

    view = AprilTagWorldDebugView()
    view.resize(400, 300)
    view.set_fixed_view(
        center=np.array([0.0, 0.0, 0.0]),
        right=np.array([1.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        pixels_per_unit=20.0,
    )
    deltas: list[tuple[float, float]] = []
    view.fixed_view_dragged.connect(lambda dx, dy: deltas.append((dx, dy)))

    view.mousePressEvent(
        MouseEventStub(
            QPointF(100.0, 100.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
        )
    )
    view.mouseMoveEvent(
        MouseEventStub(
            QPointF(112.0, 93.0),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
        )
    )

    assert deltas == [(12.0, -7.0)]
    view.deleteLater()


def test_world_debug_view_camera_point_click_emits_group_name() -> None:
    _app()
    group_a = CubemapFrameGroup(name="frame_a", frames_by_face={"pz": _frame_at("frame_a", (-2.0, 0.0, 0.0))})
    group_b = CubemapFrameGroup(name="frame_b", frames_by_face={"pz": _frame_at("frame_b", (3.0, 0.0, 0.0))})
    view = AprilTagWorldDebugView()
    view.resize(500, 400)
    view.set_groups((group_a, group_b))
    view.set_selected_group("frame_a")
    clicked: list[str] = []
    view.camera_clicked.connect(clicked.append)
    xy, _depth = view._project(np.asarray([group_b.camera_position_sfm], dtype=float))
    pos = QPointF(float(xy[0, 0]), float(xy[0, 1]))

    assert view._camera_name_at_screen_pos(pos) == "frame_b"
    QTest.mouseClick(view, Qt.LeftButton, Qt.NoModifier, pos.toPoint())

    assert clicked == ["frame_b"]
    view.deleteLater()
