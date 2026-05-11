"""OpenCV-backed AprilTag detection and PnP observation conversion."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.apriltag_geometry import PinholeFrame
from core.apriltag_markers import available_families as _available_families
from core.apriltag_markers import dictionary_for_family
from core.apriltag_scale import TagObservation, score_tag_observation


@dataclass(frozen=True)
class AprilTagDetection:
    tag_id: int
    corners_px: np.ndarray
    center_px: tuple[float, float]
    area_px: float
    reprojection_error_px: float
    camera_to_tag_m: np.ndarray
    score: float


def available_families() -> tuple[str, ...]:
    return _available_families()


def _detector_for_family(family: str) -> cv2.aruco.ArucoDetector:
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(dictionary_for_family(family), parameters)


def _object_points(tag_size_m: float) -> np.ndarray:
    half = float(tag_size_m) / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def _reprojection_error(
    object_points: np.ndarray,
    corners_px: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
    projected = projected.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected - corners_px, axis=1)))


def detect_apriltags(
    image: np.ndarray,
    frame: PinholeFrame,
    *,
    tag_size_m: float,
    family: str = "tag36h11",
    tag_ids: set[int] | None = None,
) -> tuple[AprilTagDetection, ...]:
    """Detect AprilTags in one pinhole image and convert them to camera vectors."""
    if tag_size_m <= 0.0:
        raise ValueError("tag_size_m must be positive")
    gray = image if image.ndim == 2 else cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    detector = _detector_for_family(family)
    corners_list, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return ()

    object_points = _object_points(tag_size_m)
    detections: list[AprilTagDetection] = []
    for corners, marker_id in zip(corners_list, ids.reshape(-1), strict=True):
        tag_id = int(marker_id)
        if tag_ids is not None and tag_id not in tag_ids:
            continue
        corners_px = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        ok, rvec, tvec = cv2.solvePnP(object_points, corners_px, frame.camera_matrix, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(object_points, corners_px, frame.camera_matrix, None)
        if not ok:
            continue
        tvec = np.asarray(tvec, dtype=float).reshape(3)
        camera_to_tag_m = np.array([tvec[0], -tvec[1], tvec[2]], dtype=float)
        reprojection_error = _reprojection_error(object_points, corners_px, rvec, tvec.reshape(3, 1), frame.camera_matrix)
        area = float(abs(cv2.contourArea(corners_px)))
        center = tuple(float(v) for v in np.mean(corners_px, axis=0))
        score = score_tag_observation(
            tag_area_px=area,
            image_size=(frame.width, frame.height),
            center_px=center,
            reprojection_error_px=reprojection_error,
        )
        detections.append(
            AprilTagDetection(
                tag_id=tag_id,
                corners_px=corners_px,
                center_px=center,
                area_px=area,
                reprojection_error_px=reprojection_error,
                camera_to_tag_m=camera_to_tag_m,
                score=score,
            )
        )
    return tuple(detections)


def detection_to_observation(frame: PinholeFrame, detection: AprilTagDetection) -> TagObservation:
    return TagObservation(
        tag_id=detection.tag_id,
        frame_id=frame.frame_id,
        camera_position_sfm=frame.camera_position_sfm,
        camera_to_world_rotation=frame.camera_to_world_rotation,
        camera_to_tag_m=detection.camera_to_tag_m,
        score=detection.score,
    )
