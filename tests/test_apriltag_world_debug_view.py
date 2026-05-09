from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
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
from scripts.dev_apriltag_placer_gui import DevAprilTagPlacerWindow


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


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
