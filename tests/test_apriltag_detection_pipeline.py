from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from core.apriltag_cubemap import (
    CUBEMAP_POSE_PRESET_AUTO,
    CUBEMAP_POSE_PRESET_BRUSH,
    CUBEMAP_POSE_PRESET_LICHTFELD,
    CUBEMAP_POSE_PRESET_POSTSHOT,
    CUBEMAP_POSE_PRESET_STANDARD,
    CubemapViewMetadata,
    cubemap_view_metadata_for_pose_preset,
    discover_cubemap_view_metadata,
)
from core.apriltag_detection import detect_apriltags
from core.apriltag_geometry import load_pinhole_frames, project_sfm_points
from core.apriltag_pipeline import run_apriltag_scale_estimation
from core.apriltag_projection import EquirectProjectionConfig, prepare_equirect_detection_dataset
from core.dataset_writer_colmap import ColmapCamera, ColmapImage, quaternion_from_matrix, write_colmap_text_dataset
from core.image_io import imwrite_unicode
from devtools.apriltag.synthetic import SyntheticAprilTagConfig, _warp_tag, inject_synthetic_apriltag


def _marker_image(marker_id: int = 7, size: int = 96) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    return cv2.aruco.generateImageMarker(dictionary, marker_id, size)


def _write_marker(path: Path, marker_id: int = 7, size: int = 240) -> None:
    assert imwrite_unicode(path, _marker_image(marker_id=marker_id, size=size))


def _write_tagged_scale_dataset(root: Path) -> Path:
    images = root / "images"
    images.mkdir(parents=True)
    marker = _marker_image(size=96)

    def write_frame(path: Path, *, center_x: int) -> None:
        image = np.full((400, 400, 3), 255, dtype=np.uint8)
        half = marker.shape[0] // 2
        y0 = 200 - half
        x0 = int(center_x) - half
        image[y0 : y0 + marker.shape[0], x0 : x0 + marker.shape[1]] = cv2.cvtColor(
            marker,
            cv2.COLOR_GRAY2BGR,
        )
        assert imwrite_unicode(path, image)

    write_frame(images / "a.png", center_x=200)
    write_frame(images / "b.png", center_x=140)

    transforms = root / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "camera_model": "SIMPLE_PINHOLE",
                "w": 400,
                "h": 400,
                "fl_x": 240.0,
                "fl_y": 240.0,
                "cx": 199.5,
                "cy": 199.5,
                "frames": [
                    {
                        "file_path": "images/a.png",
                        "transform_matrix": np.eye(4).tolist(),
                    },
                    {
                        "file_path": "images/b.png",
                        "transform_matrix": np.array(
                            [
                                [1.0, 0.0, 0.0, 2.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        ).tolist(),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return transforms


def _opengl_c2w_to_colmap_image(image_id: int, camera_id: int, name: str, c2w: np.ndarray) -> ColmapImage:
    converted = c2w.copy()
    converted[:3, 1:3] *= -1.0
    r_wc = converted[:3, :3]
    t_wc = converted[:3, 3]
    r_cw = r_wc.T
    t_cw = -r_cw @ t_wc
    return ColmapImage(
        image_id=image_id,
        qvec=quaternion_from_matrix(r_cw),
        tvec=t_cw,
        camera_id=camera_id,
        name=name,
    )


def _write_tagged_colmap_dataset(root: Path) -> Path:
    transforms = _write_tagged_scale_dataset(root)
    output = root / "colmap_dataset"
    images = output / "images"
    images.mkdir(parents=True)
    source_images = root / "images"
    for name in ("a.png", "b.png"):
        (images / name).write_bytes((source_images / name).read_bytes())
    data = json.loads(transforms.read_text(encoding="utf-8"))
    cameras = [
        ColmapCamera(1, "PINHOLE", 400, 400, (240.0, 240.0, 199.5, 199.5)),
        ColmapCamera(2, "OPENCV", 400, 400, (240.0, 240.0, 199.5, 199.5, 0.0, 0.0, 0.0, 0.0)),
    ]
    colmap_images = [
        _opengl_c2w_to_colmap_image(1, 1, "a.png", np.asarray(data["frames"][0]["transform_matrix"], dtype=float)),
        _opengl_c2w_to_colmap_image(2, 2, "b.png", np.asarray(data["frames"][1]["transform_matrix"], dtype=float)),
    ]
    write_colmap_text_dataset(output, cameras, colmap_images)
    return output


def _rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    ry = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=float,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=float,
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
        dtype=float,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=float,
    )
    return rx @ ry


def _write_two_frame_dataset(root: Path) -> Path:
    images = root / "images"
    images.mkdir(parents=True)
    for name in ("a.png", "b.png"):
        assert imwrite_unicode(images / name, np.full((400, 400, 3), 255, dtype=np.uint8))

    data = {
        "camera_model": "SIMPLE_PINHOLE",
        "w": 400,
        "h": 400,
        "fl_x": 240.0,
        "fl_y": 240.0,
        "cx": 199.5,
        "cy": 199.5,
        "frames": [
            {
                "file_path": "images/a.png",
                "transform_matrix": np.eye(4).tolist(),
            },
            {
                "file_path": "images/b.png",
                "transform_matrix": np.array(
                    [
                        [1.0, 0.0, 0.0, 2.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ).tolist(),
            },
        ],
    }
    transforms = root / "transforms.json"
    transforms.write_text(json.dumps(data), encoding="utf-8")
    return transforms


def _write_generated_cube6_dataset(root: Path) -> Path:
    images = root / "images"
    images.mkdir(parents=True)
    views = {
        "px": (90.0, 0.0),
        "nx": (-90.0, 0.0),
        "pz": (0.0, 0.0),
        "nz": (180.0, 0.0),
        "top": (0.0, 90.0),
        "bottom": (0.0, -90.0),
    }
    frames = []
    for face, view in views.items():
        assert imwrite_unicode(images / f"frame_0001_{face}.png", np.full((200, 200, 3), 255, dtype=np.uint8))
        transform = np.eye(4)
        transform[:3, :3] = _rotation(*view).T
        frames.append({"file_path": f"images/frame_0001_{face}.png", "transform_matrix": transform.tolist()})

    transforms = root / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "camera_model": "SIMPLE_PINHOLE",
                "w": 200,
                "h": 200,
                "fl_x": 100.0,
                "fl_y": 100.0,
                "cx": 99.5,
                "cy": 99.5,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    return transforms


def _write_generated_cube6_yaw_offset_dataset(root: Path) -> tuple[Path, CubemapViewMetadata]:
    images = root / "images"
    images.mkdir(parents=True)
    views = {
        "py": (-45.0, -90.0),
        "px": (45.0, 0.0),
        "nz": (135.0, 0.0),
        "nx": (-135.0, 0.0),
        "pz": (-45.0, 0.0),
        "ny": (-45.0, 90.0),
    }
    frames = []
    frame_groups = (
        ("frame_0001", (0.0, 0.0, 0.0)),
        ("frame_0002", (2.0, 0.0, 0.0)),
    )
    for group_index, (prefix, position) in enumerate(frame_groups):
        yaw_offset = group_index * 30.0
        for face, (yaw, pitch) in views.items():
            assert imwrite_unicode(
                images / f"{prefix}_{face}.png",
                np.full((400, 400, 3), 255, dtype=np.uint8),
            )
            transform = np.eye(4)
            transform[:3, :3] = _export_rotation(yaw + yaw_offset, pitch).T
            transform[:3, 3] = position
            frames.append({"file_path": f"images/{prefix}_{face}.png", "transform_matrix": transform.tolist()})

    transforms = root / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "camera_model": "SIMPLE_PINHOLE",
                "w": 400,
                "h": 400,
                "fl_x": 240.0,
                "fl_y": 240.0,
                "cx": 199.5,
                "cy": 199.5,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    return transforms, CubemapViewMetadata(views, yaw_offset_per_frame=30.0)


def _write_lichtfeld_export_settings(scene: Path, views: dict[str, tuple[float, float]]) -> None:
    settings = scene / "_stechdrive" / "step4" / "export_settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(
        json.dumps(
            {
                "effective_profile": "lichtfeld",
                "target_profile": "lichtfeld",
                "axis_transform": "none",
                "output_shape": "projected",
                "views_config_snapshot": {
                    "views": [
                        {"name": face, "yaw": yaw, "pitch": pitch, "enabled": True}
                        for face, (yaw, pitch) in views.items()
                    ]
                },
                "conversion": {"yaw_offset_per_frame": 30.0},
                "postprocess": {"lichtfeld_final_orientation_correction": True},
            }
        ),
        encoding="utf-8",
    )


def test_detect_apriltag_generated_marker_returns_metric_camera_vector(tmp_path: Path) -> None:
    transforms = _write_tagged_scale_dataset(tmp_path)
    frame = load_pinhole_frames(transforms)[0]
    image = cv2.imread(str(tmp_path / frame.file_path), cv2.IMREAD_COLOR)

    detections = detect_apriltags(image, frame, tag_size_m=0.8, family="tag36h11", tag_ids={7})

    assert len(detections) == 1
    detection = detections[0]
    assert detection.tag_id == 7
    assert detection.score > 0.0
    assert detection.reprojection_error_px < 1.0
    assert math.isclose(detection.camera_to_tag_m[2], 2.0, rel_tol=0.05)


def test_tagged_images_can_validate_scale_pipeline(tmp_path: Path) -> None:
    transforms = _write_tagged_scale_dataset(tmp_path)

    run = run_apriltag_scale_estimation(
        transforms,
        tag_size_m=0.8,
        family="tag36h11",
        tag_ids={7},
    )

    assert run.estimate.observation_count == 2
    assert run.estimate.pair_count == 1
    assert math.isclose(run.estimate.scale, 0.25, rel_tol=0.08)


def test_tagged_colmap_dataset_can_validate_scale_pipeline_with_mixed_cameras(tmp_path: Path) -> None:
    dataset = _write_tagged_colmap_dataset(tmp_path)

    run = run_apriltag_scale_estimation(
        dataset,
        tag_size_m=0.8,
        family="tag36h11",
        tag_ids={7},
    )

    assert run.frames[0].width == 400
    assert run.frames[1].distortion_coeffs is not None
    assert run.estimate.observation_count == 2
    assert run.estimate.pair_count == 1
    assert math.isclose(run.estimate.scale, 0.25, rel_tol=0.08)


def test_scale_pipeline_reports_progress_and_supports_parallel_detection(tmp_path: Path) -> None:
    transforms = _write_tagged_scale_dataset(tmp_path)
    progress: list[tuple[int, int]] = []
    logs: list[str] = []

    run = run_apriltag_scale_estimation(
        transforms,
        tag_size_m=0.8,
        family="tag36h11",
        tag_ids={7},
        workers=2,
        progress_callback=lambda done, total: progress.append((done, total)),
        log_callback=logs.append,
    )

    assert run.estimate.observation_count == 2
    assert progress[0] == (0, 2)
    assert progress[-1] == (2, 2)
    assert any("detection start" in line for line in logs)
    assert any("detection complete" in line for line in logs)
    assert run.timings_sec["total"] >= 0.0


def test_synthetic_injection_detects_metric_camera_vector(tmp_path: Path) -> None:
    transforms = _write_two_frame_dataset(tmp_path)
    marker = tmp_path / "tag.png"
    _write_marker(marker)
    output = tmp_path / "tagged"
    inject_synthetic_apriltag(
        SyntheticAprilTagConfig(
            input_transforms=transforms,
            output_dir=output,
            tag_image=marker,
            tag_size_m=0.8,
            true_scale=0.25,
            tag_center_sfm=np.array([0.0, 0.0, 8.0]),
            tag_normal_sfm=np.array([0.0, 0.0, -1.0]),
            tag_up_sfm=np.array([0.0, 1.0, 0.0]),
        )
    )

    frame = load_pinhole_frames(output / "transforms.json")[0]
    image = cv2.imread(str(output / frame.file_path), cv2.IMREAD_COLOR)
    detections = detect_apriltags(image, frame, tag_size_m=0.8, family="tag36h11", tag_ids={7})

    assert len(detections) == 1
    detection = detections[0]
    assert detection.tag_id == 7
    assert detection.score > 0.0
    assert detection.reprojection_error_px < 1.0
    assert math.isclose(detection.camera_to_tag_m[2], 2.0, rel_tol=0.05)


def test_synthetic_warp_keeps_reflected_raster_tag_upright() -> None:
    base = np.zeros((100, 100, 3), dtype=np.uint8)
    tag_rgba = np.zeros((20, 20, 4), dtype=np.uint8)
    tag_rgba[:10, :, :3] = 240
    tag_rgba[10:, :, :3] = 40
    tag_rgba[:, :, 3] = 255
    reflected_dst = np.array(
        [
            [80.0, 20.0],
            [20.0, 20.0],
            [20.0, 80.0],
            [80.0, 80.0],
        ],
        dtype=np.float32,
    )

    warped = _warp_tag(base, tag_rgba, reflected_dst)

    top_mean = float(np.mean(warped[25:35, 30:70, 0]))
    bottom_mean = float(np.mean(warped[65:75, 30:70, 0]))
    assert top_mean > bottom_mean


def test_synthetic_injection_can_validate_scale_pipeline(tmp_path: Path) -> None:
    transforms = _write_two_frame_dataset(tmp_path)
    marker = tmp_path / "tag.png"
    _write_marker(marker)
    output = tmp_path / "tagged"
    report = inject_synthetic_apriltag(
        SyntheticAprilTagConfig(
            input_transforms=transforms,
            output_dir=output,
            tag_image=marker,
            tag_size_m=0.8,
            true_scale=0.25,
            tag_center_sfm=np.array([0.0, 0.0, 8.0]),
            tag_normal_sfm=np.array([0.0, 0.0, -1.0]),
            tag_up_sfm=np.array([0.0, 1.0, 0.0]),
        )
    )

    assert report["frames_written"] == 2
    run = run_apriltag_scale_estimation(
        output / "transforms.json",
        tag_size_m=0.8,
        family="tag36h11",
        tag_ids={7},
    )

    assert run.estimate.observation_count == 2
    assert run.estimate.pair_count == 1
    assert math.isclose(run.estimate.scale, 0.25, rel_tol=0.08)


def test_synthetic_injection_uses_image_consistent_cube6_face_pose(tmp_path: Path) -> None:
    transforms = _write_generated_cube6_dataset(tmp_path)
    marker = tmp_path / "tag.png"
    _write_marker(marker, size=80)
    output = tmp_path / "tagged"

    report = inject_synthetic_apriltag(
        SyntheticAprilTagConfig(
            input_transforms=transforms,
            output_dir=output,
            tag_image=marker,
            tag_size_m=1.0,
            true_scale=1.0,
            tag_center_sfm=np.array([4.0, 0.0, 0.0]),
            tag_normal_sfm=np.array([-1.0, 0.0, 0.0]),
            tag_up_sfm=np.array([0.0, 1.0, 0.0]),
        )
    )

    assert report["frames_written"] == 1
    px = cv2.imread(str(output / "images" / "frame_0001_px.png"), cv2.IMREAD_COLOR)
    nx = cv2.imread(str(output / "images" / "frame_0001_nx.png"), cv2.IMREAD_COLOR)
    assert px is not None and nx is not None
    assert int(np.min(px)) < 64
    assert int(np.min(nx)) == 255


def test_synthetic_injection_can_write_metadata_normalized_cube6_transforms(tmp_path: Path) -> None:
    transforms, metadata = _write_generated_cube6_yaw_offset_dataset(tmp_path)
    marker = tmp_path / "tag.png"
    _write_marker(marker, size=80)
    output = tmp_path / "tagged"

    frames = {
        frame.file_path: frame
        for frame in load_pinhole_frames(transforms, cubemap_view_params=metadata)
    }
    selected_path = "images/frame_0002_px.png"
    selected = frames[selected_path]
    center = selected.camera_position_sfm + selected.camera_to_world_rotation[:, 2] * 8.0
    up = selected.camera_to_world_rotation[:, 1]
    normal = -selected.camera_to_world_rotation[:, 2]

    report = inject_synthetic_apriltag(
        SyntheticAprilTagConfig(
            input_transforms=transforms,
            output_dir=output,
            tag_image=marker,
            tag_size_m=0.8,
            true_scale=0.25,
            tag_center_sfm=center,
            tag_normal_sfm=normal,
            tag_up_sfm=up,
            frame_file_paths=frozenset({selected_path}),
            copy_unselected_frames=False,
            output_tagged_only=True,
            cubemap_view_params=metadata,
            write_normalized_transforms=True,
        )
    )

    assert report["frames_written"] == 1
    assert report["transforms_frame_count"] == 1
    assert report["tagged_frame_file_paths"] == [selected_path]

    output_frames = load_pinhole_frames(output / "transforms.json", normalize_cubemap=False)
    assert len(output_frames) == 1
    assert output_frames[0].file_path == selected_path
    assert np.allclose(output_frames[0].transform_matrix, selected.transform_matrix, atol=1e-8)

    image = cv2.imread(str(output / selected_path), cv2.IMREAD_COLOR)
    detections = detect_apriltags(image, output_frames[0], tag_size_m=0.8, family="tag36h11", tag_ids={7})
    assert len(detections) == 1
    assert detections[0].tag_id == 7
    assert detections[0].score > 0.0


def test_generated_cube6_transforms_are_normalized_for_projection(tmp_path: Path) -> None:
    transforms = _write_generated_cube6_dataset(tmp_path)
    frames = {Path(frame.file_path).stem.rsplit("_", 1)[-1]: frame for frame in load_pinhole_frames(transforms)}

    projected_px = project_sfm_points(frames["px"], np.array([[10.0, 0.0, 0.0]], dtype=float))
    projected_nx = project_sfm_points(frames["nx"], np.array([[10.0, 0.0, 0.0]], dtype=float))

    assert projected_px is not None
    assert np.allclose(projected_px[0], np.array([99.5, 99.5]))
    assert projected_nx is None


def test_lichtfeld_cube6_metadata_normalizes_to_saved_raster_pose(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    output = scene / "output"
    transforms, metadata = _write_generated_cube6_yaw_offset_dataset(output)
    _write_lichtfeld_export_settings(scene, metadata.view_params)

    discovered = discover_cubemap_view_metadata(transforms)
    assert discovered is not None
    assert discovered.image_pose_profile == "lichtfeld_cube6"

    frames = {frame.file_path: frame for frame in load_pinhole_frames(transforms)}
    source_local_from_lichtfeld = np.diag([1.0, -1.0, -1.0])
    raster_y_flip = np.diag([1.0, -1.0, 1.0])
    vertical_face_map = {"top": "bottom", "bottom": "top", "py": "ny", "ny": "py"}

    for group_index, prefix in enumerate(("frame_0001", "frame_0002")):
        yaw_offset = group_index * 30.0
        for face, (_yaw, _pitch) in metadata.view_params.items():
            raster_face = vertical_face_map.get(face, face)
            raster_yaw, raster_pitch = metadata.view_params[raster_face]
            expected = (
                source_local_from_lichtfeld
                @ _rotation(raster_yaw + yaw_offset, raster_pitch)
                @ raster_y_flip
            )
            frame = frames[f"images/{prefix}_{face}.png"]
            assert np.allclose(frame.camera_to_world_rotation, expected, atol=1e-8)


def test_conversion_pose_presets_provide_expected_cube6_fallbacks() -> None:
    assert cubemap_view_metadata_for_pose_preset(CUBEMAP_POSE_PRESET_AUTO) is None

    lichtfeld = cubemap_view_metadata_for_pose_preset(CUBEMAP_POSE_PRESET_LICHTFELD)
    assert lichtfeld is not None
    assert lichtfeld.view_params["px"] == (45.0, 0.0)
    assert lichtfeld.view_params["pz"] == (-45.0, 0.0)
    assert lichtfeld.view_params["py"] == (-45.0, -90.0)
    assert lichtfeld.view_params["ny"] == (-45.0, 90.0)
    assert lichtfeld.yaw_offset_per_frame == 30.0
    assert lichtfeld.image_pose_profile == "lichtfeld_cube6"

    for preset in (CUBEMAP_POSE_PRESET_POSTSHOT, CUBEMAP_POSE_PRESET_BRUSH):
        postshot_or_brush = cubemap_view_metadata_for_pose_preset(preset)
        assert postshot_or_brush is not None
        assert postshot_or_brush.view_params == lichtfeld.view_params
        assert postshot_or_brush.yaw_offset_per_frame == 30.0
        assert postshot_or_brush.image_pose_profile == ""

    standard = cubemap_view_metadata_for_pose_preset(CUBEMAP_POSE_PRESET_STANDARD)
    assert standard is not None
    assert standard.view_params["px"] == (90.0, 0.0)
    assert standard.view_params["pz"] == (0.0, 0.0)
    assert standard.view_params["py"] == (0.0, -90.0)
    assert standard.view_params["ny"] == (0.0, 90.0)
    assert standard.yaw_offset_per_frame == 0.0
    assert standard.image_pose_profile == ""


def test_equirect_detection_projection_writes_temporary_pinhole_dataset(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    assert imwrite_unicode(images / "a.png", np.full((32, 64, 3), 255, dtype=np.uint8))
    transforms = tmp_path / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "camera_model": "EQUIRECTANGULAR",
                "frames": [
                    {
                        "file_path": "images/a.png",
                        "transform_matrix": np.eye(4).tolist(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    projected = prepare_equirect_detection_dataset(
        EquirectProjectionConfig(
            transforms_json=transforms,
            output_dir=tmp_path / "apriltag_projection",
            output_scale=0.5,
            workers=1,
            remap_cache_limit=1,
        )
    )

    data = json.loads(projected.read_text(encoding="utf-8"))
    assert data["camera_model"] == "PINHOLE"
    assert len(data["frames"]) == 6
    assert (projected.parent / "images" / "a_px.png").is_file()


def test_estimate_apriltag_scale_cli_rejects_equirectangular_input(tmp_path: Path) -> None:
    transforms = tmp_path / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "camera_model": "EQUIRECTANGULAR",
                "frames": [
                    {
                        "file_path": "images/a.png",
                        "transform_matrix": np.eye(4).tolist(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.apriltag_scale_estimate",
            str(transforms),
            "--tag-size-m",
            "0.16",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "requires projected Cubemap output images" in result.stdout


def test_estimate_apriltag_scale_cli_passes_colmap_image_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import core.apriltag_scale_estimate as cli

    dataset = tmp_path / "colmap_dataset"
    image_root = tmp_path / "external_images"
    image_root.mkdir()
    (image_root / "a.png").write_bytes(b"image")
    write_colmap_text_dataset(
        dataset,
        [ColmapCamera(1, "PINHOLE", 8, 8, (4.0, 4.0, 3.5, 3.5))],
        [ColmapImage(1, quaternion_from_matrix(np.eye(3)), np.zeros(3), 1, "a.png")],
    )

    captured: dict[str, Path | None] = {}

    def fake_run(dataset_input: Path, **kwargs):
        captured["dataset_input"] = dataset_input
        captured["image_root"] = kwargs.get("image_root")
        return SimpleNamespace(
            estimate=SimpleNamespace(
                scale=1.0,
                observation_count=0,
                pair_count=0,
                inlier_count=0,
                rms_residual_m=0.0,
                median_pair_scale=0.0,
                mad_pair_scale=0.0,
            ),
            observations=(),
            frame_detections=(),
            timings_sec={},
        )

    monkeypatch.setattr(cli, "run_apriltag_scale_estimation", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "estimate_apriltag_scale.py",
            str(dataset),
            "--image-root",
            str(image_root),
            "--tag-size-m",
            "0.16",
        ],
    )

    assert cli.main() == 0
    assert captured["dataset_input"] == dataset
    assert captured["image_root"] == image_root
