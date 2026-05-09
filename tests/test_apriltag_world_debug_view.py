from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from core.apriltag_geometry import PinholeFrame
from devtools.apriltag.coordinates import combined_pointcloud_display_matrix, pointcloud_display_matrix, world_display_matrix
from devtools.apriltag.cubemap_preview import CubemapFrameGroup
from devtools.apriltag.world_debug_view import (
    AprilTagWorldDebugView,
    PointCloudSample,
    load_point_cloud_sample,
    transform_point_cloud_sample,
)
from scripts.dev_apriltag_placer_gui import (
    GRID_X_AXIS_BGR,
    GRID_Z_AXIS_BGR,
    DevAprilTagPlacerWindow,
    _estimate_world_display_matrix_from_metashape,
    _transform_group_for_world_display,
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
    window.deleteLater()


def test_world_debug_view_orientation_gizmo_marks_positive_axes() -> None:
    _app()
    view = AprilTagWorldDebugView()
    origin = QPointF(100.0, 100.0)
    view._view_yaw_deg = 0.0
    view._view_pitch_deg = 0.0

    points = view._orientation_axis_points(origin, 20.0)

    assert points["X"].x() > origin.x()
    assert points["X"].y() == origin.y()
    assert points["Y"].x() == origin.x()
    assert points["Y"].y() < origin.y()
    assert points["Z"].x() == origin.x()
    assert points["Z"].y() == origin.y()
    view.deleteLater()


def test_world_debug_view_orientation_gizmo_matches_viewport_z_convention() -> None:
    _app()
    view = AprilTagWorldDebugView()
    origin = QPointF(100.0, 100.0)
    view._view_yaw_deg = 90.0
    view._view_pitch_deg = 0.0

    points = view._orientation_axis_points(origin, 20.0)

    assert points["Z"].x() > origin.x()
    assert points["Z"].y() == origin.y()
    assert points["Y"].x() == origin.x()
    assert points["Y"].y() < origin.y()
    assert abs(points["X"].x() - origin.x()) < 1e-12
    assert abs(points["X"].y() - origin.y()) < 1e-12
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


def test_dev_placer_grid_only_preview_background() -> None:
    _app()
    window = DevAprilTagPlacerWindow()

    image = window._grid_only_equirect_preview()

    assert window.grid_only_preview_check.text() == "画像OFF"
    assert window.coordinate_profile_combo.currentData() == "lichtfeld_cube6"
    assert image.shape == (1024, 2048, 3)
    assert image.dtype == np.uint8
    assert image[512, 1024, 0] > image[16, 16, 0]
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
    window.look_yaw_spin.setValue(0.0)
    window.look_pitch_spin.setValue(0.0)
    window.look_fov_spin.setValue(90.0)

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
    window.look_yaw_spin.setValue(0.0)
    window.look_pitch_spin.setValue(0.0)
    window.look_fov_spin.setValue(90.0)

    overlays = window._grid_preview_overlays()
    origin = next(overlay for overlay in overlays if overlay.label == "O").polyline[0]
    x_axis = [overlay for overlay in overlays if overlay.color_bgr == GRID_X_AXIS_BGR and overlay.polyline]
    z_axis = [overlay for overlay in overlays if overlay.color_bgr == GRID_Z_AXIS_BGR and overlay.polyline]

    assert any(any(np.allclose(point, origin) for point in overlay.polyline) for overlay in x_axis)
    assert any(any(np.allclose(point, origin) for point in overlay.polyline) for overlay in z_axis)
    window.deleteLater()
