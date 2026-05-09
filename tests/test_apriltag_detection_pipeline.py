from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from core.apriltag_detection import detect_apriltags
from core.apriltag_geometry import load_pinhole_frames
from core.apriltag_pipeline import run_apriltag_scale_estimation
from core.apriltag_projection import EquirectProjectionConfig, prepare_equirect_detection_dataset
from devtools.apriltag.synthetic import SyntheticAprilTagConfig, inject_synthetic_apriltag
from core.image_io import imwrite_unicode


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


def _write_marker(path: Path, marker_id: int = 7, size: int = 240) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    assert imwrite_unicode(path, marker)


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


def test_detect_apriltag_generated_marker_returns_metric_camera_vector(tmp_path: Path) -> None:
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
    assert data["camera_model"] == "SIMPLE_PINHOLE"
    assert len(data["frames"]) == 6
    assert (projected.parent / "images" / "a_px.png").is_file()
