from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from core.apriltag_detection import detect_apriltags
from core.apriltag_geometry import load_pinhole_frames
from core.apriltag_pipeline import run_apriltag_scale_estimation
from core.apriltag_synthetic import SyntheticAprilTagConfig, inject_synthetic_apriltag
from core.image_io import imwrite_unicode


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
