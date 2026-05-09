from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from core.apriltag_geometry import PinholeFrame
from devtools.apriltag.coordinates import pointcloud_display_matrix
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


def test_dev_placer_camera_texture_preview_flips_longitude_only() -> None:
    image = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)

    flipped = DevAprilTagPlacerWindow._camera_texture_for_preview(image)

    assert np.array_equal(flipped[:, 0], image[:, -1])
    assert np.array_equal(flipped[:, -1], image[:, 0])
    assert np.array_equal(flipped[0], image[0, ::-1])
    assert np.array_equal(flipped[1], image[1, ::-1])
    assert flipped.flags.c_contiguous


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
