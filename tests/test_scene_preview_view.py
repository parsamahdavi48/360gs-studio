from __future__ import annotations

import builtins
import json
import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from core.apriltag_cubemap import cubemap_face_rotation
from core.scene_preview import (
    ScenePreviewCamera,
    ScenePreviewDataset,
    ScenePreviewPointCloud,
    load_transforms_preview_dataset,
)
from core.scene_preview_cubemap import cubemap_frame_groups_from_preview_cameras, face_view_params
from core.scene_preview_profiles import ScenePreviewDisplayTransform, step4_output_display_transform
from core.scene_preview_sources import ScenePreviewCandidate
from gui import i18n
from gui.common.perspective_image_view import PerspectiveImageView, _pixelated_texture_filter_for_zoom
from gui.common.perspective_preview import PerspectiveParams
from gui.scene_preview.camera_image_view import SceneCameraImageView, apply_mask_overlay
from gui.scene_preview.pointcloud_view import ScenePointCloudView
from gui.scene_preview.window import ScenePreviewWindow, _load_candidate


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _preview_camera(camera_id: str, label: str, position: tuple[float, float, float]) -> ScenePreviewCamera:
    return ScenePreviewCamera(
        camera_id=camera_id,
        label=label,
        image_path=Path(label),
        projection="pinhole",
        width=100,
        height=80,
        fl_x=50.0,
        fl_y=48.0,
        cx=49.5,
        cy=39.5,
        position=np.array(position, dtype=np.float64),
        right=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        up=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        forward=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        source={},
    )


def test_scene_pointcloud_view_upload_data_uses_all_points() -> None:
    _app()
    dataset = ScenePreviewDataset(
        source_kind="test",
        source_path=Path("scene"),
        cameras=(),
        pointcloud=ScenePreviewPointCloud(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
                dtype=np.float32,
            ),
            colors=np.array(
                [
                    [255, 0, 0],
                    [0, 128, 255],
                    [255, 255, 255],
                ],
                dtype=np.uint8,
            ),
        ),
    )
    view = ScenePointCloudView()
    view.set_dataset(dataset)

    data = view._pointcloud_gl_vertex_data()

    assert data.shape == (3, 7)
    assert np.allclose(data[:, :3], dataset.pointcloud.points)
    assert np.allclose(data[1, 3:6], [0.0, 128.0 / 255.0, 1.0])
    assert np.allclose(data[:, 6], [0.9, 0.9, 0.9])


def test_scene_pointcloud_view_can_pick_camera_at_screen_center() -> None:
    _app()
    camera = ScenePreviewCamera(
        camera_id="cam-1",
        label="frame_0001.jpg",
        image_path=None,
        projection="pinhole",
        width=100,
        height=100,
        fl_x=50.0,
        fl_y=50.0,
        cx=49.5,
        cy=49.5,
        position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        right=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        up=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        forward=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        source={},
    )
    dataset = ScenePreviewDataset(
        source_kind="test",
        source_path=Path("scene"),
        cameras=(camera,),
    )
    view = ScenePointCloudView()
    view.resize(400, 300)
    view.set_dataset(dataset)

    assert view._camera_id_at_screen_pos(QPointF(view.width() * 0.5, view.height() * 0.5)) == "cam-1"


def test_scene_camera_image_view_drag_updates_perspective_direction() -> None:
    _app()
    view = SceneCameraImageView()
    view._perspective_params = PerspectiveParams(yaw_deg=10.0, pitch_deg=5.0, fov_deg=90.0)

    view._on_look_dragged(10.0, -5.0)

    assert view._perspective_params is not None
    assert round(view._perspective_params.yaw_deg, 6) == 8.2
    assert round(view._perspective_params.pitch_deg, 6) == 5.9


def test_scene_camera_image_view_applies_existing_mask_overlay() -> None:
    image = np.full((2, 2, 3), 100, dtype=np.uint8)
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)

    overlaid = apply_mask_overlay(image, mask)

    assert overlaid[0, 0].tolist() == [55, 55, 169]
    assert overlaid[1, 1].tolist() == [55, 55, 169]
    assert overlaid[0, 1].tolist() == [100, 100, 100]
    assert image[0, 0].tolist() == [100, 100, 100]


def test_perspective_view_uses_crisp_texture_filter_when_zoomed() -> None:
    assert not _pixelated_texture_filter_for_zoom(1.0)
    assert _pixelated_texture_filter_for_zoom(1.2)


def test_perspective_view_resets_pan_zoom_when_source_changes() -> None:
    _app()
    view = PerspectiveImageView()
    assert view._gpu_view is not None
    view._gpu_view._zoom = 4.0
    view._gpu_view._pan = QPointF(50.0, -25.0)

    shown = view.set_perspective_image_bgr(
        np.full((32, 64, 3), 128, dtype=np.uint8),
        PerspectiveParams(),
    )

    assert shown
    assert view._gpu_view._zoom == 1.0
    assert view._gpu_view._pan == QPointF(0.0, 0.0)


def test_scene_preview_rebuilds_cubemap_camera_as_spherical_preview(tmp_path: Path) -> None:
    _app()
    transforms = _write_cubemap_scene(tmp_path)
    dataset = load_transforms_preview_dataset(transforms, image_root=tmp_path)
    window = ScenePreviewWindow()
    window._dataset = dataset
    window._build_cubemap_lookup(dataset)

    result = window._reconstructed_cubemap_preview(dataset.cameras[0])

    assert result is not None
    image, params, mask = result
    assert image.shape == (1024, 2048, 3)
    assert params.fov_deg == 90.0
    assert mask is None


def test_scene_preview_rebuilds_cubemap_from_colmap_like_preview_cameras(tmp_path: Path) -> None:
    _app()
    transforms = _write_cubemap_scene(tmp_path)
    source = load_transforms_preview_dataset(transforms, image_root=tmp_path)
    dataset = ScenePreviewDataset(
        source_kind="colmap",
        source_path=tmp_path / "sparse" / "0",
        cameras=source.cameras,
        image_root=source.image_root,
        mask_root=source.mask_root,
    )
    window = ScenePreviewWindow()
    window._set_dataset(dataset)

    result = window._reconstructed_cubemap_preview(dataset.cameras[0])

    assert result is not None
    image, params, mask = result
    assert image.shape == (1024, 2048, 3)
    assert params.fov_deg == 90.0
    assert mask is None


def test_scene_preview_mixed_colmap_keeps_normal_camera_as_static_image(tmp_path: Path) -> None:
    _app()
    transforms = _write_cubemap_scene(tmp_path)
    source = load_transforms_preview_dataset(transforms, image_root=tmp_path)
    normal_path = tmp_path / "images" / "normal.jpg"
    assert cv2.imwrite(str(normal_path), np.full((24, 32, 3), 120, dtype=np.uint8))
    normal = ScenePreviewCamera(
        camera_id="normal",
        label="normal.jpg",
        image_path=normal_path,
        projection="pinhole",
        width=32,
        height=24,
        fl_x=20.0,
        fl_y=20.0,
        cx=15.5,
        cy=11.5,
        position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        right=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        up=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        forward=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        source={},
    )
    dataset = ScenePreviewDataset(
        source_kind="colmap",
        source_path=tmp_path / "sparse" / "0",
        cameras=(*source.cameras, normal),
        image_root=source.image_root,
        mask_root=source.mask_root,
    )
    window = ScenePreviewWindow()
    window._set_dataset(dataset)

    window._select_camera_id("normal")

    assert window._selected_camera_id == "normal"
    assert window.camera_image_view._perspective_params is None


def test_scene_preview_summary_tracks_selected_camera(tmp_path: Path) -> None:
    _app()
    front = _preview_camera("front", "front.jpg", (1.0, 2.0, 3.0))
    side = _preview_camera("side", "side.jpg", (4.0, 5.0, 6.0))
    dataset = ScenePreviewDataset(
        source_kind="test",
        source_path=tmp_path / "dataset",
        cameras=(front, side),
        image_root=tmp_path,
    )
    window = ScenePreviewWindow()

    window._set_dataset(dataset)

    text = window.summary_text.toPlainText()
    assert i18n.t("SCENE_PREVIEW_SELECTED_CAMERA_SUMMARY") in text
    assert "front.jpg" in text
    assert "side.jpg" not in text
    assert f"{i18n.t('SCENE_PREVIEW_CAMERA_INDEX')}: 1 / 2" in text

    window._select_camera_id("side")

    text = window.summary_text.toPlainText()
    assert "side.jpg" in text
    assert "front.jpg" not in text
    assert f"{i18n.t('SCENE_PREVIEW_CAMERA_INDEX')}: 2 / 2" in text
    assert f"{i18n.t('SCENE_PREVIEW_POSITION')}: (4.0000, 5.0000, 6.0000)" in text


def test_scene_preview_rebuilds_cubemap_without_devtools_import(tmp_path: Path, monkeypatch) -> None:
    _app()
    transforms = _write_cubemap_scene(tmp_path)
    dataset = load_transforms_preview_dataset(transforms, image_root=tmp_path)
    window = ScenePreviewWindow()
    original_import = builtins.__import__

    def blocked_import(name: str, globals_: object = None, locals_: object = None, fromlist: object = (), level: int = 0) -> object:
        if name == "devtools" or name.startswith("devtools."):
            raise ModuleNotFoundError("devtools is not available in release builds")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    window._dataset = dataset
    window._build_cubemap_lookup(dataset)
    result = window._reconstructed_cubemap_preview(dataset.cameras[0])

    assert result is not None


def test_scene_preview_rebuilds_cubemap_masks_as_spherical_preview(tmp_path: Path) -> None:
    _app()
    transforms = _write_cubemap_scene(tmp_path)
    masks = tmp_path / "masks"
    masks.mkdir()
    assert cv2.imwrite(str(masks / "frame_0001_pz.png"), np.zeros((32, 32), dtype=np.uint8))
    dataset = load_transforms_preview_dataset(transforms, image_root=tmp_path, mask_root=masks)
    window = ScenePreviewWindow()
    window._dataset = dataset
    window._build_cubemap_lookup(dataset)

    result = window._reconstructed_cubemap_preview(dataset.cameras[0])

    assert result is not None
    _image, _params, mask = result
    assert mask is not None
    assert mask.shape == (1024, 2048)
    assert int(mask[512, 1024]) < 128


def test_scene_preview_does_not_show_single_cubemap_face_when_rebuild_fails(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    assert cv2.imwrite(str(images / "frame_0001_pz.png"), np.full((16, 16, 3), 90, dtype=np.uint8))
    transforms = tmp_path / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "camera_model": "SIMPLE_PINHOLE",
                "w": 16,
                "h": 16,
                "fl_x": 8.0,
                "fl_y": 8.0,
                "cx": 7.5,
                "cy": 7.5,
                "frames": [
                    {
                        "file_path": "images/frame_0001_pz.png",
                        "transform_matrix": np.eye(4, dtype=np.float64).tolist(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    window = ScenePreviewWindow()
    dataset = load_transforms_preview_dataset(transforms, image_root=tmp_path)
    window._set_dataset(dataset)

    assert window.camera_image_view._view._cpu_view.text() == i18n.t("SCENE_PREVIEW_CUBEMAP_REBUILD_FAILED")


def test_scene_preview_pointcloud_selection_keeps_clicked_cubemap_face(tmp_path: Path) -> None:
    _app()
    transforms = _write_cubemap_scene(tmp_path, face_order=("top", "px", "pz", "nx", "nz", "bottom"))
    dataset = load_transforms_preview_dataset(transforms, image_root=tmp_path)
    window = ScenePreviewWindow()
    window._set_dataset(dataset)
    top_camera = next(camera for camera in dataset.cameras if camera.label.endswith("_top.png"))
    front_camera = next(camera for camera in dataset.cameras if camera.label.endswith("_pz.png"))

    assert window._selected_camera_id == front_camera.camera_id
    assert window.camera_combo.currentData() == front_camera.camera_id
    assert window.pointcloud_view._selected_camera_id == front_camera.camera_id
    assert window.pointcloud_view._selected_view_ray_direction is not None
    assert np.allclose(window.pointcloud_view._selected_view_ray_direction, front_camera.forward)

    window._select_camera_from_pointcloud(top_camera.camera_id)

    assert window._selected_camera_id == top_camera.camera_id
    assert window.camera_combo.currentData() == top_camera.camera_id
    assert window.pointcloud_view._selected_camera_id == top_camera.camera_id
    assert window.camera_image_view._perspective_params is not None
    assert window.camera_image_view._perspective_params.yaw_deg == 0.0
    assert window.camera_image_view._perspective_params.pitch_deg == 90.0
    assert window.pointcloud_view._selected_view_ray_direction is not None
    assert np.allclose(window.pointcloud_view._selected_view_ray_direction, [0.0, -1.0, 0.0])


def test_scene_preview_updates_left_view_ray_when_right_view_rotates(tmp_path: Path) -> None:
    _app()
    transforms = _write_cubemap_scene(tmp_path)
    dataset = load_transforms_preview_dataset(transforms, image_root=tmp_path)
    window = ScenePreviewWindow()
    window._set_dataset(dataset)

    window.camera_image_view._on_look_dragged(10.0, 0.0)

    expected_yaw = np.deg2rad(-1.8)
    expected = np.array([np.sin(expected_yaw), 0.0, np.cos(expected_yaw)], dtype=np.float64)
    assert window.pointcloud_view._selected_view_ray_direction is not None
    assert np.allclose(window.pointcloud_view._selected_view_ray_direction, expected)

    window.camera_image_view._on_look_dragged(0.0, -10.0)

    expected_yaw = np.deg2rad(-1.8)
    expected_pitch = np.deg2rad(1.8)
    expected = np.array(
        [
            np.sin(expected_yaw) * np.cos(expected_pitch),
            -np.sin(expected_pitch),
            np.cos(expected_yaw) * np.cos(expected_pitch),
        ],
        dtype=np.float64,
    )
    assert window.pointcloud_view._selected_view_ray_direction is not None
    assert np.allclose(window.pointcloud_view._selected_view_ray_direction, expected)


def test_scene_preview_selected_cubemap_face_ray_matches_camera_forward_after_display_transform(
    tmp_path: Path,
) -> None:
    _app()
    transforms = _write_cubemap_scene(tmp_path, use_face_rotations=True, base_rotation=_yaw_rotation_matrix(35.0))
    display_transform = ScenePreviewDisplayTransform(
        profile="test",
        note="test display transform",
        camera_matrix=np.diag([1.0, -1.0, -1.0, 1.0]),
        pointcloud_matrix=np.diag([1.0, -1.0, -1.0, 1.0]),
    )
    candidate = ScenePreviewCandidate(
        kind="output",
        label="Transforms",
        path=transforms,
        image_root=tmp_path,
        display_transform=display_transform,
    )
    dataset = _load_candidate(candidate)
    window = ScenePreviewWindow()
    window._set_dataset(dataset)
    top_camera = next(camera for camera in dataset.cameras if camera.label.endswith("_top.png"))

    window._select_camera_from_pointcloud(top_camera.camera_id)

    assert window.pointcloud_view._selected_view_ray_direction is not None
    assert np.allclose(window.pointcloud_view._selected_view_ray_direction, top_camera.forward, atol=1e-8)


def test_scene_preview_cubemap_view_ray_uses_normalized_image_pose(tmp_path: Path) -> None:
    _app()
    transforms = _write_gui_cube6_export_scene(tmp_path, base_rotation=_yaw_rotation_matrix(35.0))
    dataset = load_transforms_preview_dataset(transforms, image_root=tmp_path)
    window = ScenePreviewWindow()
    window._set_dataset(dataset)
    pz_camera = next(camera for camera in dataset.cameras if camera.label.endswith("_pz.png"))
    raw_forward = pz_camera.forward.copy()

    window._select_camera_from_pointcloud(pz_camera.camera_id)

    match = window._cubemap_match_for_camera(pz_camera)
    assert match is not None
    group, face = match
    expected = group.frames_by_face[face].camera_to_world_rotation[:, 2]
    ray = window.pointcloud_view._selected_view_ray_direction
    assert ray is not None
    assert float(ray @ expected) > 0.999
    assert float(ray @ raw_forward) < -0.999


def test_scene_preview_cubemap_vertical_faces_use_stable_look_params(tmp_path: Path) -> None:
    transforms = _write_gui_cube6_export_scene(tmp_path)
    dataset = load_transforms_preview_dataset(transforms, image_root=tmp_path)
    group = cubemap_frame_groups_from_preview_cameras(dataset.cameras)[0]

    assert face_view_params(group, "py") == (0.0, -90.0, 90.0)
    assert face_view_params(group, "ny") == (0.0, 90.0, 90.0)


def test_scene_preview_cubemap_view_ray_uses_selected_face_pose_when_aligned(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    cameras = []
    for face in ("px", "nx", "pz", "nz", "py", "ny"):
        image_path = images / f"rs_0001_{face}.png"
        assert cv2.imwrite(str(image_path), np.full((32, 32, 3), 96, dtype=np.uint8))
        rotation = cubemap_face_rotation(face)
        assert rotation is not None
        if face == "pz":
            rotation = _yaw_rotation_matrix(25.0) @ rotation
        cameras.append(
            ScenePreviewCamera(
                camera_id=face,
                label=image_path.name,
                image_path=image_path,
                projection="pinhole",
                width=32,
                height=32,
                fl_x=16.0,
                fl_y=16.0,
                cx=15.5,
                cy=15.5,
                position=np.zeros(3, dtype=np.float64),
                right=rotation[:, 0],
                up=rotation[:, 1],
                forward=rotation[:, 2],
                source={},
            )
        )
    dataset = ScenePreviewDataset(
        source_kind="realityscan",
        source_path=tmp_path / "rs.csv",
        cameras=tuple(cameras),
        image_root=images,
    )
    window = ScenePreviewWindow()
    window._set_dataset(dataset)

    window._select_camera_from_pointcloud("pz")

    pz_camera = next(camera for camera in cameras if camera.camera_id == "pz")
    ray = window.pointcloud_view._selected_view_ray_direction
    assert ray is not None
    assert float(ray @ pz_camera.forward) > 0.999


def test_scene_preview_realityscan_normal_camera_view_ray_uses_photo_optical_axis(tmp_path: Path) -> None:
    _app()
    image_path = tmp_path / "extra_IMG_0316_00000.jpg"
    assert cv2.imwrite(str(image_path), np.full((24, 32, 3), 96, dtype=np.uint8))
    camera = ScenePreviewCamera(
        camera_id="normal",
        label=image_path.name,
        image_path=image_path,
        projection="pinhole",
        width=32,
        height=24,
        fl_x=20.0,
        fl_y=20.0,
        cx=15.5,
        cy=11.5,
        position=np.zeros(3, dtype=np.float64),
        right=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        up=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        forward=np.array([0.2, -0.1, 0.97], dtype=np.float64),
        source={"format": "realityscan_csv"},
    )
    dataset = ScenePreviewDataset(
        source_kind="realityscan",
        source_path=tmp_path / "rs.csv",
        cameras=(camera,),
        image_root=tmp_path,
        coordinate_note="realityscan_csv / realityscan",
    )
    window = ScenePreviewWindow()
    window._set_dataset(dataset)

    ray = window.pointcloud_view._selected_view_ray_direction

    expected = np.array([0.2, -0.1, 0.97], dtype=np.float64)
    expected /= np.linalg.norm(expected)
    assert ray is not None
    assert np.allclose(ray, expected)


def test_scene_preview_realityscan_cubemap_camera_view_ray_keeps_face_pose(tmp_path: Path) -> None:
    _app()
    transforms = _write_cubemap_scene(tmp_path, use_face_rotations=True)
    source = load_transforms_preview_dataset(transforms, image_root=tmp_path)
    dataset = ScenePreviewDataset(
        source_kind="realityscan",
        source_path=tmp_path / "rs.csv",
        cameras=source.cameras,
        image_root=source.image_root,
        coordinate_note="realityscan_csv / realityscan",
    )
    window = ScenePreviewWindow()
    window._set_dataset(dataset)
    pz_camera = next(camera for camera in dataset.cameras if camera.label.endswith("_pz.png"))

    window._select_camera_id(pz_camera.camera_id)

    ray = window.pointcloud_view._selected_view_ray_direction
    assert ray is not None
    assert float(ray @ pz_camera.forward) > 0.999


def test_load_step4_output_candidate_applies_display_transform(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "masks").mkdir()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [-1.0, 2.0, -3.0]
    (output / "transforms.json").write_text(
        json.dumps(
            {
                "camera_model": "SIMPLE_PINHOLE",
                "w": 32,
                "h": 32,
                "fl_x": 16.0,
                "fl_y": 16.0,
                "cx": 15.5,
                "cy": 15.5,
                "frames": [{"file_path": "frame.png", "transform_matrix": matrix.tolist()}],
            }
        ),
        encoding="utf-8",
    )
    (output / "pointcloud.ply").write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "1 2 3",
            ]
        ),
        encoding="ascii",
    )
    display_transform = step4_output_display_transform(
        {
            "effective_profile": "lichtfeld",
            "axis_transform": "none",
            "postprocess": {"final_orientation": "lichtfeld"},
        }
    )
    candidate = ScenePreviewCandidate(
        kind="output",
        label="Step 4 output",
        path=output / "transforms.json",
        image_root=output,
        mask_root=output / "masks",
        pointcloud_path=output / "pointcloud.ply",
        display_transform=display_transform,
    )

    dataset = _load_candidate(candidate)

    assert np.allclose(dataset.cameras[0].position, [1.0, 2.0, 3.0])
    assert dataset.mask_root == output / "masks"
    assert dataset.pointcloud is not None
    assert np.allclose(dataset.pointcloud.points[0], [1.0, 2.0, 3.0])


def test_load_colmap_candidate_applies_display_transform(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("1 PINHOLE 32 32 16 16 15.5 15.5\n", encoding="utf-8")
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame.png\n\n", encoding="utf-8")
    (sparse / "points3D.txt").write_text(
        "# Number of points: 1\n1 1.0 2.0 3.0 255 255 255 0\n",
        encoding="utf-8",
    )
    display_transform = ScenePreviewDisplayTransform(
        profile="test",
        note="colmap test",
        camera_matrix=np.diag([1.0, -1.0, -1.0, 1.0]),
        pointcloud_matrix=np.diag([1.0, -1.0, -1.0, 1.0]),
    )
    candidate = ScenePreviewCandidate(
        kind="colmap",
        label="COLMAP",
        path=sparse,
        display_transform=display_transform,
    )

    dataset = _load_candidate(candidate)

    assert dataset.coordinate_note == "colmap test"
    assert np.allclose(dataset.cameras[0].position, [0.0, 0.0, 0.0])
    assert dataset.pointcloud is not None
    assert np.allclose(dataset.pointcloud.points[0], [1.0, -2.0, -3.0])


def _write_cubemap_scene(
    root: Path,
    *,
    face_order: tuple[str, ...] = ("px", "nx", "pz", "nz", "top", "bottom"),
    use_face_rotations: bool = False,
    base_rotation: np.ndarray | None = None,
) -> Path:
    images = root / "images"
    images.mkdir()
    frames = []
    colors = {
        "px": (255, 0, 0),
        "nx": (0, 255, 0),
        "pz": (0, 0, 255),
        "nz": (255, 255, 0),
        "top": (255, 0, 255),
        "bottom": (0, 255, 255),
    }
    base = np.eye(3, dtype=np.float64) if base_rotation is None else np.asarray(base_rotation, dtype=np.float64)
    for face in face_order:
        color = colors[face]
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:, :] = color
        assert cv2.imwrite(str(images / f"frame_0001_{face}.png"), image)
        transform = np.eye(4, dtype=np.float64)
        if use_face_rotations:
            face_rotation = cubemap_face_rotation(face)
            assert face_rotation is not None
            transform[:3, :3] = base @ face_rotation
        frames.append(
            {
                "file_path": f"images/frame_0001_{face}.png",
                "transform_matrix": transform.tolist(),
            }
        )
    transforms = root / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "camera_model": "SIMPLE_PINHOLE",
                "w": 32,
                "h": 32,
                "fl_x": 16.0,
                "fl_y": 16.0,
                "cx": 15.5,
                "cy": 15.5,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    return transforms


def _write_gui_cube6_export_scene(root: Path, *, base_rotation: np.ndarray | None = None) -> Path:
    images = root / "images"
    images.mkdir()
    base = np.eye(3, dtype=np.float64) if base_rotation is None else np.asarray(base_rotation, dtype=np.float64)
    views = {
        "py": (-90.0, -90.0),
        "px": (0.0, 0.0),
        "nz": (90.0, 0.0),
        "nx": (180.0, 0.0),
        "pz": (-90.0, 0.0),
        "ny": (-90.0, 90.0),
    }
    frames = []
    for face, (yaw, pitch) in views.items():
        image = np.full((32, 32, 3), 120, dtype=np.uint8)
        assert cv2.imwrite(str(images / f"frame_0001_{face}.png"), image)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = base @ _export_rotation_matrix(yaw, pitch).T
        frames.append(
            {
                "file_path": f"images/frame_0001_{face}.png",
                "transform_matrix": transform.tolist(),
            }
        )
    transforms = root / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "camera_model": "SIMPLE_PINHOLE",
                "w": 32,
                "h": 32,
                "fl_x": 16.0,
                "fl_y": 16.0,
                "cx": 15.5,
                "cy": 15.5,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    return transforms


def _yaw_rotation_matrix(yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(float(yaw_deg))
    return np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float64,
    )


def _export_rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(float(yaw_deg))
    pitch = np.deg2rad(float(pitch_deg))
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
