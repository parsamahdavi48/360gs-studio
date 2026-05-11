"""AprilTag scale-estimation pipeline wiring for CLI and GUI callers."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.apriltag_cubemap import CubemapViewMetadata
from core.apriltag_detection import AprilTagDetection, detect_apriltags, detection_to_observation
from core.apriltag_geometry import PinholeFrame, load_pinhole_frames
from core.apriltag_scale import ScaleEstimate, TagObservation, estimate_scene_scale
from core.image_io import imread_unicode

ProgressCallback = Callable[[int, int], None]
LogCallback = Callable[[str], None]


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
    timings_sec: Mapping[str, float]


def _resolve_detection_workers(workers: str | int | None, total: int) -> int:
    if total <= 1:
        return 1
    if workers is None or (isinstance(workers, str) and workers.strip().lower() in {"", "auto"}):
        cpu_count = os.cpu_count() or 1
        return max(1, min(total, cpu_count, 8))
    try:
        value = int(workers)
    except (TypeError, ValueError) as exc:
        raise ValueError("workers must be 'auto' or a positive integer") from exc
    if value <= 0:
        raise ValueError("workers must be 'auto' or a positive integer")
    return max(1, min(total, value))


def _detect_frame(
    frame: PinholeFrame,
    *,
    tag_size_m: float,
    family: str,
    tag_ids: set[int] | None,
    min_score: float,
) -> FrameDetection:
    image = imread_unicode(frame.image_path)
    if image is None:
        return FrameDetection(frame=frame, detections=())
    detections = detect_apriltags(image, frame, tag_size_m=tag_size_m, family=family, tag_ids=tag_ids)
    accepted = tuple(d for d in detections if d.score >= min_score)
    return FrameDetection(frame=frame, detections=accepted)


def collect_observations(
    transforms_json: Path,
    *,
    image_root: Path | None,
    tag_size_m: float,
    family: str,
    tag_ids: set[int] | None = None,
    min_score: float = 0.0,
    normalize_cubemap: bool = True,
    cubemap_view_params: CubemapViewMetadata | Mapping[str, tuple[float, float]] | None = None,
    workers: str | int | None = 1,
    progress_callback: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
) -> tuple[tuple[PinholeFrame, ...], tuple[FrameDetection, ...], tuple[TagObservation, ...]]:
    load_started = time.perf_counter()
    frames = load_pinhole_frames(
        transforms_json,
        image_root=image_root,
        normalize_cubemap=normalize_cubemap,
        cubemap_view_params=cubemap_view_params,
    )
    if log_callback is not None:
        log_callback(f"loaded {len(frames)} pinhole frames in {time.perf_counter() - load_started:.2f}s")

    total = len(frames)
    worker_count = _resolve_detection_workers(workers, total)
    if log_callback is not None:
        ids_text = "-" if tag_ids is None else ",".join(str(tag_id) for tag_id in sorted(tag_ids))
        log_callback(f"detection start: family={family}, tag_ids={ids_text}, workers={worker_count}")
    if progress_callback is not None:
        progress_callback(0, total)

    started = time.perf_counter()
    ordered: list[FrameDetection | None] = [None] * total
    detections_seen = 0
    frames_with_detections = 0
    completed = 0

    def record_result(index: int, frame_detection: FrameDetection) -> None:
        nonlocal completed, detections_seen, frames_with_detections
        ordered[index] = frame_detection
        completed += 1
        count = len(frame_detection.detections)
        detections_seen += count
        if count:
            frames_with_detections += 1
        if progress_callback is not None:
            progress_callback(completed, total)
        if log_callback is not None and (
            completed == 1 or completed == total or completed % max(1, min(50, total // 10 or 1)) == 0
        ):
            log_callback(
                "detection progress: "
                f"{completed}/{total} frames, "
                f"detected={detections_seen}, "
                f"frames_with_tags={frames_with_detections}"
            )

    if worker_count <= 1:
        for index, frame in enumerate(frames):
            record_result(
                index,
                _detect_frame(
                    frame,
                    tag_size_m=tag_size_m,
                    family=family,
                    tag_ids=tag_ids,
                    min_score=min_score,
                ),
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _detect_frame,
                    frame,
                    tag_size_m=tag_size_m,
                    family=family,
                    tag_ids=tag_ids,
                    min_score=min_score,
                ): index
                for index, frame in enumerate(frames)
            }
            for future in as_completed(futures):
                record_result(futures[future], future.result())

    frame_detections = tuple(item for item in ordered if item is not None)
    observations = tuple(
        detection_to_observation(item.frame, detection)
        for item in frame_detections
        for detection in item.detections
    )
    if log_callback is not None:
        elapsed = time.perf_counter() - started
        fps = (total / elapsed) if elapsed > 0.0 else 0.0
        log_callback(
            "detection complete: "
            f"{total} frames in {elapsed:.2f}s ({fps:.2f} fps), "
            f"accepted_observations={len(observations)}"
        )
    return frames, frame_detections, observations


def run_apriltag_scale_estimation(
    transforms_json: Path,
    *,
    image_root: Path | None = None,
    tag_size_m: float,
    family: str = "tag36h11",
    tag_ids: set[int] | None = None,
    min_score: float = 0.0,
    min_baseline_sfm: float = 1e-6,
    normalize_cubemap: bool = True,
    cubemap_view_params: CubemapViewMetadata | Mapping[str, tuple[float, float]] | None = None,
    workers: str | int | None = 1,
    progress_callback: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
) -> AprilTagScaleRun:
    total_started = time.perf_counter()
    detection_started = time.perf_counter()
    frames, frame_detections, observations = collect_observations(
        transforms_json,
        image_root=image_root,
        tag_size_m=tag_size_m,
        family=family,
        tag_ids=tag_ids,
        min_score=min_score,
        normalize_cubemap=normalize_cubemap,
        cubemap_view_params=cubemap_view_params,
        workers=workers,
        progress_callback=progress_callback,
        log_callback=log_callback,
    )
    detection_sec = time.perf_counter() - detection_started
    estimate_started = time.perf_counter()
    if log_callback is not None:
        log_callback(f"estimation start: observations={len(observations)}")
    estimate = estimate_scene_scale(observations, min_baseline_sfm=min_baseline_sfm)
    estimate_sec = time.perf_counter() - estimate_started
    total_sec = time.perf_counter() - total_started
    if log_callback is not None:
        log_callback(
            "estimation complete: "
            f"scale={estimate.scale:.9g}, "
            f"pairs={estimate.pair_count}, "
            f"inliers={estimate.inlier_count}, "
            f"elapsed={total_sec:.2f}s"
        )
    return AprilTagScaleRun(
        frames=frames,
        frame_detections=frame_detections,
        observations=observations,
        estimate=estimate,
        timings_sec={
            "detection": detection_sec,
            "estimation": estimate_sec,
            "total": total_sec,
        },
    )
