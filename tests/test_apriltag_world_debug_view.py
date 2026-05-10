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
from devtools.apriltag.cubemap_preview import CubemapFrameGroup, view_pixel_to_axis_world_ray_and_up
from devtools.apriltag.world_debug_view import (
    AprilTagWorldDebugView,
    PointCloudSample,
    load_point_cloud_sample,
    transform_point_cloud_sample,
)
from gui.common.perspective_preview import PerspectiveParams
from scripts.dev_apriltag_placer_gui import (
    GRID_X_AXIS_BGR,
    GRID_Z_AXIS_BGR,
    DevAprilTagPlacerWindow,
    _estimate_world_display_matrix_from_metashape,
    _transform_group_for_world_display,
    _transform_points_from_world_display,
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

    transformed = _transform_group_for_world_display(group, world_display_matrix("lichtfeld_cube6"))
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

    estimated = _estimate_world_display_matrix_from_metashape(groups, xml)

    assert estimated is not None
    matrix, rmse, count = estimated
    assert count == 3
    assert rmse < 1e-9
    assert np.allclose(matrix[:3, :3], np.diag([-1.0, 1.0, -1.0]))


def test_dev_placer_world_view_uses_display_axes_without_changing_raw_groups() -> None:
    _app()
    group = CubemapFrameGroup(name="frame_0001", frames_by_face={"pz": _frame_at("frame_0001", (1.0, 2.0, 3.0))})
    window = DevAprilTagPlacerWindow()
    window._cubemap_groups = (group,)
    window.center_editor.set_value((1.0, 2.0, 3.0))
    window.normal_editor.set_value((0.0, 0.0, 1.0))
    window.up_editor.set_value((0.0, 1.0, 0.0))

    window._sync_world_debug_view()

    displayed_group = window.world_debug_view._groups[0]
    assert np.allclose(group.camera_position_sfm, [1.0, 2.0, 3.0])
    assert np.allclose(displayed_group.camera_position_sfm, [-1.0, 2.0, -3.0])
    assert np.allclose(window.world_debug_view._tag_center, [-1.0, 2.0, -3.0])
    assert np.allclose(window.world_debug_view._tag_normal, [0.0, 0.0, -1.0])
    assert np.allclose(window.world_debug_view._preview_to_world_matrix, np.eye(3))
    window.deleteLater()


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


def test_dev_placer_selects_frame_group_by_world_view_name() -> None:
    _app()
    group_a = CubemapFrameGroup(name="frame_a", frames_by_face={"pz": _frame_at("frame_a", (-2.0, 0.0, 0.0))})
    group_b = CubemapFrameGroup(name="frame_b", frames_by_face={"pz": _frame_at("frame_b", (3.0, 0.0, 0.0))})
    window = DevAprilTagPlacerWindow()
    window._cubemap_groups = (group_a, group_b)
    window.frame_group_combo.blockSignals(True)
    for group in window._cubemap_groups:
        window.frame_group_combo.addItem(f"{group.name} ({len(group.frames)} faces)", group.name)
    window.frame_group_combo.setCurrentIndex(0)
    window._select_frame_group_by_name("frame_b")
    window.frame_group_combo.blockSignals(False)

    assert window.frame_group_combo.currentData() == "frame_b"
    window.deleteLater()


def test_dev_placer_pointcloud_mode_uses_second_world_debug_view() -> None:
    _app()
    group = CubemapFrameGroup(name="frame_0001", frames_by_face={"pz": _frame_at("frame_0001", (0.0, 0.0, 0.0))})
    window = DevAprilTagPlacerWindow()
    window._cubemap_groups = (group,)
    window.frame_group_combo.blockSignals(True)
    window.frame_group_combo.addItem("frame_0001 (1 faces)", "frame_0001")
    window.frame_group_combo.blockSignals(False)
    window.coordinate_profile_combo.setCurrentIndex(window.coordinate_profile_combo.findData("custom"))
    window.center_editor.set_value((0.0, 0.0, 5.0))
    window.pointcloud_preview_check.blockSignals(True)
    window.pointcloud_preview_check.setChecked(True)
    window.pointcloud_preview_check.blockSignals(False)
    window._scene_preview_params = PerspectiveParams(yaw_deg=0.0, pitch_deg=0.0, fov_deg=90.0)

    window._sync_world_debug_view()

    assert window.preview_stack.currentWidget() is window.camera_debug_view
    assert window.camera_debug_view._fixed_view_basis is not None
    assert window.camera_debug_view._fixed_projection == "perspective"
    xy, _depth = window.camera_debug_view._project(np.array([[1.0, 0.0, 2.5]], dtype=float))
    assert xy[0, 0] < window.camera_debug_view.width() * 0.5
    window.deleteLater()


def test_dev_placer_pointcloud_camera_pose_uses_loaded_face_layout() -> None:
    _app()
    group = CubemapFrameGroup(name="frame_0001", frames_by_face={"pz": _frame_at("frame_0001", (0.0, 0.0, 0.0))})
    window = DevAprilTagPlacerWindow()
    window.coordinate_profile_combo.setCurrentIndex(window.coordinate_profile_combo.findData("lichtfeld_cube6"))

    window._scene_preview_params = PerspectiveParams(yaw_deg=0.0, pitch_deg=0.0, fov_deg=90.0)
    _camera, right, up, forward = window._camera_debug_view_pose(group)

    assert np.allclose(right, [1.0, 0.0, 0.0])
    assert np.allclose(up, [0.0, 1.0, 0.0])
    assert np.allclose(forward, [0.0, 0.0, 1.0])

    window._scene_preview_params = PerspectiveParams(yaw_deg=90.0, pitch_deg=0.0, fov_deg=90.0)
    _camera, _right, _up, forward_px = window._camera_debug_view_pose(group)

    assert np.allclose(forward_px, [1.0, 0.0, 0.0], atol=1e-12)
    window.deleteLater()


def test_dev_placer_grid_only_preview_background() -> None:
    _app()
    window = DevAprilTagPlacerWindow()

    image = window._grid_only_equirect_preview()

    assert window.grid_only_preview_check.text() == "画像OFF"
    assert window.pointcloud_preview_check.text() == "点群ビュー"
    assert window.coordinate_profile_combo.currentData() == "lichtfeld_cube6"
    assert image.shape == (1024, 2048, 3)
    assert image.dtype == np.uint8
    assert image[512, 1024, 0] > image[16, 16, 0]
    window.deleteLater()


def test_dev_placer_pointcloud_perspective_preview_projects_world_points() -> None:
    _app()
    group = CubemapFrameGroup(name="frame_0001", frames_by_face={"pz": _frame_at("frame_0001", (0.0, 0.0, 0.0))})
    window = DevAprilTagPlacerWindow()
    window._cubemap_groups = (group,)
    window.coordinate_profile_combo.setCurrentIndex(window.coordinate_profile_combo.findData("custom"))
    window._world_pointcloud = PointCloudSample(
        points=np.array([[1.0, 1.0, 5.0]], dtype=np.float32),
        colors=np.array([[255, 0, 0]], dtype=np.uint8),
        source_count=1,
    )
    window._scene_preview_params = PerspectiveParams(yaw_deg=0.0, pitch_deg=0.0, fov_deg=90.0)

    image = window._pointcloud_perspective_preview_bgr(group)
    xy, _depth, valid = window._project_world_display_points_for_preview(group, window._world_pointcloud.points)
    x, y = np.rint(xy[0]).astype(int)

    assert image.shape == (window._scene_preview_size, window._scene_preview_size, 3)
    assert bool(valid[0])
    assert image[y, x, 2] > 200
    window.deleteLater()


def test_dev_placer_pointcloud_perspective_uses_selected_camera_rotation() -> None:
    _app()
    transform = np.eye(4)
    transform[:3, :3] = np.diag([-1.0, 1.0, -1.0])
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
    window = DevAprilTagPlacerWindow()
    window._cubemap_groups = (group,)
    window.coordinate_profile_combo.setCurrentIndex(window.coordinate_profile_combo.findData("custom"))
    window.pointcloud_preview_check.blockSignals(True)
    window.pointcloud_preview_check.setChecked(True)
    window.pointcloud_preview_check.blockSignals(False)
    window._scene_preview_params = PerspectiveParams(yaw_deg=0.0, pitch_deg=0.0, fov_deg=90.0)

    points = np.array([[0.0, 0.0, -5.0], [1.0, 0.0, -5.0], [-1.0, 0.0, -5.0]], dtype=np.float32)
    projected, _depth, valid = window._project_world_display_points_for_preview(group, points)
    center, plus_x, minus_x = projected

    assert np.all(valid)
    assert np.allclose(center, [(window._scene_preview_size - 1) / 2.0, (window._scene_preview_size - 1) / 2.0])
    assert plus_x[0] < center[0]
    assert minus_x[0] > center[0]
    window.deleteLater()


def test_dev_placer_pointcloud_preview_keeps_world_display_points() -> None:
    _app()
    window = DevAprilTagPlacerWindow()
    window.coordinate_profile_combo.setCurrentIndex(window.coordinate_profile_combo.findData("lichtfeld_cube6"))
    window.pointcloud_preview_check.blockSignals(True)
    window.pointcloud_preview_check.setChecked(True)
    window.pointcloud_preview_check.blockSignals(False)

    points = window._pointcloud_preview_points_for_projection(np.array([[1.0, 2.0, 3.0]], dtype=float))

    assert np.allclose(points, [[1.0, 2.0, 3.0]])
    window.deleteLater()


def test_dev_placer_pointcloud_preview_uses_lichtfeld_scene_rotation_for_view() -> None:
    _app()
    group = CubemapFrameGroup(name="frame_0001", frames_by_face={"pz": _frame_at("frame_0001", (0.0, 0.0, 0.0))})
    window = DevAprilTagPlacerWindow()
    window._cubemap_groups = (group,)
    window.coordinate_profile_combo.setCurrentIndex(window.coordinate_profile_combo.findData("lichtfeld_cube6"))
    window.pointcloud_preview_check.blockSignals(True)
    window.pointcloud_preview_check.setChecked(True)
    window.pointcloud_preview_check.blockSignals(False)
    window._scene_preview_params = PerspectiveParams(yaw_deg=0.0, pitch_deg=0.0, fov_deg=90.0)

    display_group = window._selected_world_display_group()
    assert display_group is not None
    projected, _depth, valid = window._project_world_display_points_for_preview(
        display_group,
        np.array([[0.0, 0.0, -5.0], [1.0, 0.0, -5.0], [0.0, 0.0, -6.0]], dtype=np.float32),
    )
    center, plus_x, farther_z = projected

    assert np.all(valid)
    assert plus_x[0] < center[0]
    assert abs(float(farther_z[0] - center[0])) < 1.0
    window.deleteLater()


def test_dev_placer_camera_axis_gizmo_marks_positive_axes() -> None:
    _app()
    window = DevAprilTagPlacerWindow()

    overlays = window._camera_axis_gizmo_overlays()
    by_label = {overlay.label: overlay for overlay in overlays if overlay.label}
    origin = np.asarray(overlays[0].polyline[0], dtype=float)
    plus_x = np.asarray(by_label["+X"].polyline[-1], dtype=float)
    plus_y = np.asarray(by_label["+Y"].polyline[-1], dtype=float)
    plus_z = np.asarray(by_label["+Z"].polyline[-1], dtype=float)

    assert {"+X", "+Y", "+Z"}.issubset(by_label)
    assert plus_x[0] > origin[0]
    assert plus_y[1] < origin[1]
    assert np.linalg.norm(plus_z - origin) < 1.0
    window.deleteLater()


def test_dev_placer_pointcloud_axis_gizmo_uses_selected_camera_rotation() -> None:
    _app()
    transform = np.eye(4)
    transform[:3, :3] = np.diag([-1.0, 1.0, -1.0])
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
    window = DevAprilTagPlacerWindow()
    window._cubemap_groups = (group,)
    window.coordinate_profile_combo.setCurrentIndex(window.coordinate_profile_combo.findData("custom"))
    window.pointcloud_preview_check.blockSignals(True)
    window.pointcloud_preview_check.setChecked(True)
    window.pointcloud_preview_check.blockSignals(False)
    window._scene_preview_params = PerspectiveParams(yaw_deg=35.0, pitch_deg=0.0, fov_deg=90.0)

    overlays = window._camera_axis_gizmo_overlays()
    by_label = {overlay.label: overlay for overlay in overlays if overlay.label}
    origin = np.asarray(overlays[0].polyline[0], dtype=float)
    rotation = window._preview_camera_rotation_for_display_group(group)

    for label, axis in (
        ("+X", np.array([1.0, 0.0, 0.0], dtype=float)),
        ("+Z", np.array([0.0, 0.0, 1.0], dtype=float)),
    ):
        expected_view = axis @ rotation
        expected_screen = np.array([expected_view[0], -expected_view[1]], dtype=float)
        actual_screen = np.asarray(by_label[label].polyline[-1], dtype=float) - origin
        assert float(actual_screen @ expected_screen) > 0.0
    window.deleteLater()


def test_dev_placer_camera_grid_axes_include_origin() -> None:
    _app()
    transform = np.eye(4)
    transform[:3, 3] = np.array([1.0, 1.0, -5.0])
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
    window = DevAprilTagPlacerWindow()
    window._cubemap_groups = (group,)
    window.coordinate_profile_combo.setCurrentIndex(window.coordinate_profile_combo.findData("custom"))
    window.grid_step_spin.setValue(1.0)
    window.grid_extent_spin.setValue(12.0)
    window._scene_preview_params = PerspectiveParams(yaw_deg=0.0, pitch_deg=0.0, fov_deg=90.0)

    overlays = window._grid_preview_overlays()
    origin = next(overlay for overlay in overlays if overlay.label == "O").polyline[0]
    x_axis = [overlay for overlay in overlays if overlay.color_bgr == GRID_X_AXIS_BGR and overlay.polyline]
    z_axis = [overlay for overlay in overlays if overlay.color_bgr == GRID_Z_AXIS_BGR and overlay.polyline]

    assert any(any(np.allclose(point, origin) for point in overlay.polyline) for overlay in x_axis)
    assert any(any(np.allclose(point, origin) for point in overlay.polyline) for overlay in z_axis)
    window.deleteLater()


def test_dev_placer_camera_grid_axes_reach_distant_visible_origin() -> None:
    _app()
    transform = np.eye(4)
    transform[:3, 3] = np.array([0.0, 1.0, -50.0])
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
    window = DevAprilTagPlacerWindow()
    window._cubemap_groups = (group,)
    window.coordinate_profile_combo.setCurrentIndex(window.coordinate_profile_combo.findData("custom"))
    window.grid_step_spin.setValue(1.0)
    window.grid_extent_spin.setValue(12.0)
    window._scene_preview_params = PerspectiveParams(yaw_deg=0.0, pitch_deg=0.0, fov_deg=90.0)

    overlays = window._grid_preview_overlays()
    origin = next(overlay for overlay in overlays if overlay.label == "O").polyline[0]
    x_axis = [overlay for overlay in overlays if overlay.color_bgr == GRID_X_AXIS_BGR and overlay.polyline]
    z_axis = [overlay for overlay in overlays if overlay.color_bgr == GRID_Z_AXIS_BGR and overlay.polyline]

    assert any(any(np.allclose(point, origin) for point in overlay.polyline) for overlay in x_axis)
    assert any(any(np.allclose(point, origin) for point in overlay.polyline) for overlay in z_axis)
    window.deleteLater()


def test_dev_placer_camera_grid_projects_display_x_axis_direction() -> None:
    _app()
    display_camera = np.array([[10.0, 0.0, 5.0]], dtype=float)
    raw_camera = _transform_points_from_world_display(display_camera, world_display_matrix("lichtfeld_cube6"))[0]
    transform = np.eye(4)
    transform[:3, 3] = raw_camera
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
    window = DevAprilTagPlacerWindow()
    window._cubemap_groups = (group,)
    window._scene_preview_params = PerspectiveParams(yaw_deg=np.rad2deg(np.arctan2(-10.0, -5.0)), pitch_deg=0.0, fov_deg=90.0)

    projected = window._project_world_display_points_to_preview(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [-5.0, 0.0, 0.0],
            ],
            dtype=float,
        )
    )

    assert projected is not None
    origin, plus_x, minus_x = projected
    assert plus_x[0] < origin[0]
    assert minus_x[0] > origin[0]
    window.deleteLater()


def test_dev_placer_camera_grid_marks_positive_axis_sides() -> None:
    _app()
    transform = np.eye(4)
    transform[:3, 3] = np.array([0.0, 1.0, -8.0])
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
    window = DevAprilTagPlacerWindow()
    window._cubemap_groups = (group,)
    window.coordinate_profile_combo.setCurrentIndex(window.coordinate_profile_combo.findData("custom"))
    window.grid_step_spin.setValue(1.0)
    window.grid_extent_spin.setValue(12.0)
    window._scene_preview_params = PerspectiveParams(yaw_deg=0.0, pitch_deg=0.0, fov_deg=90.0)

    overlays = window._grid_preview_overlays()
    by_label = {overlay.label: overlay for overlay in overlays if overlay.label}

    assert "+X" in by_label
    assert "+Z" in by_label
    window.deleteLater()


def test_dev_placer_world_frustum_uses_display_camera_basis() -> None:
    _app()
    display_camera = np.array([[10.0, 0.0, 5.0]], dtype=float)
    raw_camera = _transform_points_from_world_display(display_camera, world_display_matrix("lichtfeld_cube6"))[0]
    transform = np.eye(4)
    transform[:3, 3] = raw_camera
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
    window = DevAprilTagPlacerWindow()
    window._cubemap_groups = (group,)
    window._scene_preview_params = PerspectiveParams(yaw_deg=30.0, pitch_deg=0.0, fov_deg=90.0)

    window._sync_world_debug_view()
    frustum_forward, _corners = window.world_debug_view._preview_frustum_rays_in_world()
    display_group = window._selected_world_display_group()
    assert display_group is not None
    ray, _up, _face = view_pixel_to_axis_world_ray_and_up(
        display_group,
        x_px=(window._scene_preview_size - 1) / 2.0,
        y_px=(window._scene_preview_size - 1) / 2.0,
        output_size=window._scene_preview_size,
        yaw_deg=window._scene_preview_params.yaw_deg,
        pitch_deg=window._scene_preview_params.pitch_deg,
        fov_deg=window._scene_preview_params.fov_deg,
        roll_deg=window._scene_preview_params.roll_deg,
    )

    assert np.allclose(frustum_forward, ray)
    window.deleteLater()
