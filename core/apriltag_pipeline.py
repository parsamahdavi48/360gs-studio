"""AprilTag scale-estimation pipeline wiring for CLI and GUI callers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.apriltag_detection import AprilTagDetection, detect_apriltags, detection_to_observation
from core.apriltag_geometry import PinholeFrame, load_pinhole_frames
from core.apriltag_scale import ScaleEstimate, TagObservation, estimate_scene_scale
from core.image_io import imread_unicode


@dataclass(frozen=True)
class FrameDetection:
    frame: PinholeFrame
    detections: tuple[AprilTagDetection, ...]


@dataclass(frozen=True)
class AprilTagScaleRun:
    frames: tuple[PinholeFrame, ...]
    frame_detections: tuple[FrameDetection, ...]
    observations: tuple[TagObservation, ...]
    estimate: ScaleEstimate


def collect_observations(
    transforms_json: Path,
    *,
    image_root: Path | None,
    tag_size_m: float,
    family: str,
    tag_ids: set[int] | None = None,
    min_score: float = 0.0,
) -> tuple[tuple[PinholeFrame, ...], tuple[FrameDetection, ...], tuple[TagObservation, ...]]:
    frames = load_pinhole_frames(transforms_json, image_root=image_root)
    frame_detections: list[FrameDetection] = []
    observations: list[TagObservation] = []
    for frame in frames:
        image = imread_unicode(frame.image_path)
        if image is None:
            frame_detections.append(FrameDetection(frame=frame, detections=()))
            continue
        detections = detect_apriltags(image, frame, tag_size_m=tag_size_m, family=family, tag_ids=tag_ids)
        accepted = tuple(d for d in detections if d.score >= min_score)
        frame_detections.append(FrameDetection(frame=frame, detections=accepted))
        observations.extend(detection_to_observation(frame, detection) for detection in accepted)
    return frames, tuple(frame_detections), tuple(observations)


def run_apriltag_scale_estimation(
    transforms_json: Path,
    *,
    image_root: Path | None = None,
    tag_size_m: float,
    family: str = "tag36h11",
    tag_ids: set[int] | None = None,
    min_score: float = 0.0,
    min_baseline_sfm: float = 1e-6,
) -> AprilTagScaleRun:
    frames, frame_detections, observations = collect_observations(
        transforms_json,
        image_root=image_root,
        tag_size_m=tag_size_m,
        family=family,
        tag_ids=tag_ids,
        min_score=min_score,
    )
    estimate = estimate_scene_scale(observations, min_baseline_sfm=min_baseline_sfm)
    return AprilTagScaleRun(
        frames=frames,
        frame_detections=frame_detections,
        observations=observations,
        estimate=estimate,
    )
