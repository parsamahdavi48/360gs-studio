#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from extract_sessions import (
    build_session_record,
    load_manifest,
    new_session_id,
    sanitize_filename_prefix,
    save_manifest,
    session_matches_video,
    video_identity,
)

try:
    import cv2
except Exception as e:  # pragma: no cover - environment-dependent import
    cv2 = None
    _CV2_IMPORT_ERROR = e
else:
    _CV2_IMPORT_ERROR = None

try:
    import numpy as np
except Exception as e:  # pragma: no cover - environment-dependent import
    np = None
    _NP_IMPORT_ERROR = e
else:
    _NP_IMPORT_ERROR = None


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    total_frames: int


@dataclass
class QualityMetrics:
    """Per-frame proxy metrics for SfM suitability.

    The final score is deliberately absolute-ish and bounded instead of a
    percentile rank. It is used only to choose a better representative inside
    an anchor window, not to drop frames automatically.
    """

    quality: float
    feature_count: int
    feature_spread: float
    sharpness: float
    contrast: float
    exposure_penalty: float


@dataclass
class PairMetrics:
    raw_change: float
    residual: float
    yaw_shift_px: int
    yaw_shift_deg: float


@dataclass
class PairTrackMetrics:
    track_count: int
    coverage: float
    confidence: float
    median_residual_motion: float


@dataclass(frozen=True)
class PairFrameRisk:
    blur_score: float
    sharpness_baseline: float | None
    sharpness_ratio: float | None
    motion_blur: bool
    low_texture: bool
    weak_match: bool


@dataclass(frozen=True)
class PairThresholdProfile:
    reference_interval_sec: float
    min_interval_sec: float
    max_interval_sec: float
    base_drop_threshold: float
    base_add_threshold: float
    exponent: float = 0.5


@dataclass(frozen=True)
class PairThresholds:
    drop: float
    add: float
    profile: str
    mode: str


PAIR_SHARPNESS_HISTORY = 7
PAIR_LOW_TEXTURE_SHARPNESS = 8.0
PAIR_MOTION_BLUR_BASELINE_MIN = 12.0
PAIR_MOTION_BLUR_RATIO = 0.35
PAIR_GATE_WIDTH_DEFAULT = 1280


PAIR_THRESHOLD_PROFILES: dict[str, PairThresholdProfile] = {
    # Slow handheld walking: 1 second is the reference cadence. The clamp range
    # is derived from the practical walking interval domain, not fixed literals.
    "walk": PairThresholdProfile(
        reference_interval_sec=1.0,
        min_interval_sec=0.35,
        max_interval_sec=2.5,
        base_drop_threshold=0.035,
        base_add_threshold=0.090,
    ),
    # Aerial 360 capture tends to have weaker residual parallax because most
    # features are farther away, so the reference residual thresholds are lower.
    "drone": PairThresholdProfile(
        reference_interval_sec=2.0,
        min_interval_sec=0.8,
        max_interval_sec=5.0,
        base_drop_threshold=0.025,
        base_add_threshold=0.065,
    ),
}


def parse_fraction(value: str) -> float:
    if not value:
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        if den_f == 0:
            return 0.0
        return float(num) / den_f
    return float(value)


def frame_index_digits(total_frames: int, frame_indices: Sequence[int] | None = None) -> int:
    """Return the zero-padding width needed for frame-index filenames."""
    max_index = total_frames - 1 if total_frames > 0 else -1
    if frame_indices:
        max_index = max(max_index, max(frame_indices))
    if max_index < 0:
        return 6
    return max(1, len(str(max_index)))


def frame_filename(filename_prefix: str, frame_index: int, image_ext: str, digits: int) -> str:
    return f"{filename_prefix}_{frame_index:0{max(1, digits)}d}.{image_ext}"


def ensure_python_deps() -> None:
    missing = []
    if cv2 is None:
        missing.append(f"opencv-python (cv2 import failed: {_CV2_IMPORT_ERROR})")
    if np is None:
        missing.append(f"numpy (import failed: {_NP_IMPORT_ERROR})")
    if missing:
        raise RuntimeError("Missing required Python modules: " + "; ".join(missing))


def run_cmd(cmd: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        check=False,
    )


def run_cmd_with_ffmpeg_progress(cmd: list[str], phase: str, total_items: int) -> subprocess.CompletedProcess:
    if total_items <= 0:
        return run_cmd(cmd, capture=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stderr is not None

    progress_step = max(1, total_items // 100)
    last_reported = -1
    observed_frame = 0
    stderr_lines: list[str] = []

    print(f"[progress] {phase} 0/{total_items} frames (0.0%)", flush=True)
    for raw in proc.stderr:
        line = raw.strip()
        if not line:
            continue
        if "=" not in line:
            stderr_lines.append(line)
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in {
            "frame",
            "fps",
            "stream_0_0_q",
            "bitrate",
            "total_size",
            "out_time_us",
            "out_time_ms",
            "out_time",
            "dup_frames",
            "drop_frames",
            "speed",
            "progress",
        }:
            stderr_lines.append(line)
        if key != "frame":
            continue
        try:
            frame_count = int(value)
        except ValueError:
            continue

        if frame_count < observed_frame:
            continue
        observed_frame = frame_count
        if observed_frame == 0:
            continue

        if observed_frame - last_reported >= progress_step or observed_frame >= total_items:
            shown = min(total_items, observed_frame)
            pct = min(100.0, (shown / float(total_items)) * 100.0)
            print(f"[progress] {phase} {shown}/{total_items} frames ({pct:.1f}%)", flush=True)
            last_reported = observed_frame

    proc.wait()
    if proc.returncode == 0 and last_reported < total_items:
        print(f"[progress] {phase} {total_items}/{total_items} frames (100.0%)", flush=True)

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout="",
        stderr="\n".join(stderr_lines),
    )


def _drain_binary_pipe(pipe: BinaryIO, chunks: list[bytes]) -> None:
    try:
        while True:
            data = pipe.read(8192)
            if not data:
                break
            chunks.append(data)
    except OSError as e:
        chunks.append(f"\n[stderr read error: {e}]".encode("utf-8", errors="replace"))


def ensure_binary(path: str, name: str) -> None:
    proc = run_cmd([path, "-version"], capture=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip() if proc.stderr else "not found"
        raise RuntimeError(f"Failed to execute {name}: {msg}")


def probe_video(video_path: Path, ffprobe_bin: str) -> VideoInfo:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    proc = run_cmd(cmd, capture=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")

    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No video stream found")

    stream = streams[0]
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    fps = parse_fraction(stream.get("avg_frame_rate", "0"))
    if fps <= 0:
        fps = parse_fraction(stream.get("r_frame_rate", "0"))

    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
    nb_frames_raw = stream.get("nb_frames")
    total_frames = int(nb_frames_raw) if nb_frames_raw and nb_frames_raw.isdigit() else 0

    if fps <= 0 and duration > 0 and total_frames > 0:
        fps = total_frames / duration
    if fps <= 0:
        raise RuntimeError("Could not determine FPS from video")
    if duration <= 0 and total_frames > 0:
        duration = total_frames / fps
    if total_frames <= 0 and duration > 0:
        total_frames = max(1, int(round(duration * fps)))

    return VideoInfo(width=width, height=height, fps=fps, duration=duration, total_frames=total_frames)


def scaled_dimensions(width: int, height: int, analysis_width: int) -> tuple[int, int]:
    if analysis_width <= 0 or analysis_width >= width:
        return width, height

    scaled_h = int(round(height * (analysis_width / float(width))))
    if scaled_h < 2:
        scaled_h = 2
    if scaled_h % 2 != 0:
        scaled_h += 1
    return analysis_width, scaled_h


def _bounded01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _feature_spread_score(corners: np.ndarray | None, width: int, height: int) -> float:
    if corners is None or len(corners) == 0 or width <= 0 or height <= 0:
        return 0.0

    pts = corners.reshape(-1, 2)
    grid_cols = 6
    grid_rows = 3
    xs = np.clip((pts[:, 0] / max(1.0, float(width))) * grid_cols, 0, grid_cols - 1).astype(np.int32)
    ys = np.clip((pts[:, 1] / max(1.0, float(height))) * grid_rows, 0, grid_rows - 1).astype(np.int32)
    cells = ys * grid_cols + xs
    counts = np.bincount(cells, minlength=grid_cols * grid_rows).astype(np.float64)
    occupied = float(np.count_nonzero(counts)) / float(grid_cols * grid_rows)

    total = float(np.sum(counts))
    if total <= 0.0:
        entropy = 0.0
    else:
        probs = counts[counts > 0] / total
        entropy = float(-np.sum(probs * np.log(probs)) / math.log(grid_cols * grid_rows))

    return _bounded01(0.65 * occupied + 0.35 * entropy)


def _sharpness_score_from_lap_var(lap_var: float) -> float:
    if lap_var <= 0.0:
        return 0.0
    # Log compression keeps the score stable across ordinary analysis sizes.
    return _bounded01(math.log1p(lap_var) / math.log1p(1200.0))


def compute_quality_metrics(frame: np.ndarray, lap_var: float, quality_mode: str) -> QualityMetrics:
    sharpness = _sharpness_score_from_lap_var(lap_var)
    if quality_mode == "sharpness":
        return QualityMetrics(
            quality=sharpness,
            feature_count=0,
            feature_spread=0.0,
            sharpness=sharpness,
            contrast=0.0,
            exposure_penalty=0.0,
        )

    height, width = frame.shape[:2]
    _, stddev = cv2.meanStdDev(frame)
    contrast = _bounded01(float(stddev[0][0]) / 56.0)
    bright = float(np.mean(frame >= 250))
    dark = float(np.mean(frame <= 5))
    exposure_penalty = _bounded01((bright + dark) * 2.0)

    min_dim = max(1, min(width, height))
    max_corners = max(200, min(1600, int(round((width * height) / 2500.0))))
    min_distance = max(4, int(round(min_dim / 160.0)))
    try:
        corners = cv2.goodFeaturesToTrack(
            frame,
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=min_distance,
            blockSize=5,
            useHarrisDetector=False,
        )
    except cv2.error:
        corners = None

    feature_count = int(0 if corners is None else len(corners))
    expected_features = max(80.0, float(max_corners) * 0.35)
    feature_score = _bounded01(feature_count / expected_features)
    spread = _feature_spread_score(corners, width, height)

    quality = (
        0.35 * feature_score
        + 0.30 * spread
        + 0.20 * sharpness
        + 0.15 * contrast
        - 0.25 * exposure_penalty
    )
    return QualityMetrics(
        quality=_bounded01(quality),
        feature_count=feature_count,
        feature_spread=spread,
        sharpness=sharpness,
        contrast=contrast,
        exposure_penalty=exposure_penalty,
    )


def compute_feature_motion_score(prev_frame: np.ndarray, frame: np.ndarray) -> float:
    """Sparse LK feature motion normalized by image diagonal."""
    if prev_frame.shape != frame.shape:
        return 0.0

    height, width = prev_frame.shape[:2]
    min_distance = max(6, min(width, height) // 80)
    try:
        points = cv2.goodFeaturesToTrack(
            prev_frame,
            maxCorners=320,
            qualityLevel=0.01,
            minDistance=min_distance,
            blockSize=7,
            useHarrisDetector=False,
        )
    except cv2.error:
        points = None
    if points is None or len(points) < 12:
        return 0.0

    try:
        next_points, status, _err = cv2.calcOpticalFlowPyrLK(
            prev_frame,
            frame,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
    except cv2.error:
        return 0.0
    if next_points is None or status is None:
        return 0.0

    valid = status.reshape(-1) == 1
    if int(np.count_nonzero(valid)) < 12:
        return 0.0

    src = points.reshape(-1, 2)[valid]
    dst = next_points.reshape(-1, 2)[valid]
    displacement = np.linalg.norm(dst - src, axis=1)
    diagonal = math.hypot(float(width), float(height))
    if displacement.size == 0 or diagonal <= 0.0:
        return 0.0
    return _bounded01(float(np.median(displacement)) / diagonal)


def _latitude_weights(height: int) -> np.ndarray:
    if height <= 0:
        return np.ones((1, 1), dtype=np.float32)
    y = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    lat = (y - 0.5) * math.pi
    weights = np.cos(lat).astype(np.float32)
    weights = np.maximum(weights, 0.05)
    return weights.reshape(height, 1)


def _weighted_mean_absdiff(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.float32)
    denom = float(np.sum(weights) * max(1, a.shape[1]) * 255.0)
    if denom <= 0.0:
        return 0.0
    return float(np.sum(diff * weights) / denom)


def _column_signal(frame: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted = frame.astype(np.float32) * weights
    signal = np.mean(weighted, axis=0)
    signal -= float(np.mean(signal))
    std = float(np.std(signal))
    if std > 1e-6:
        signal /= std
    return signal


def _estimate_yaw_shift_px(prev_frame: np.ndarray, frame: np.ndarray, weights: np.ndarray) -> int:
    if prev_frame.shape != frame.shape or frame.shape[1] <= 1:
        return 0

    prev_signal = _column_signal(prev_frame, weights)
    cur_signal = _column_signal(frame, weights)
    if float(np.std(prev_signal)) <= 1e-6 or float(np.std(cur_signal)) <= 1e-6:
        return 0

    corr = np.fft.irfft(
        np.fft.rfft(cur_signal) * np.conj(np.fft.rfft(prev_signal)),
        n=frame.shape[1],
    )
    peak = int(np.argmax(corr))
    if peak > frame.shape[1] // 2:
        peak -= frame.shape[1]
    return peak


def compute_pair_metrics(prev_frame: np.ndarray, frame: np.ndarray, weights: np.ndarray) -> PairMetrics:
    if prev_frame.shape != frame.shape:
        return PairMetrics(raw_change=0.0, residual=0.0, yaw_shift_px=0, yaw_shift_deg=0.0)

    raw_change = _weighted_mean_absdiff(prev_frame, frame, weights)
    estimated_shift = _estimate_yaw_shift_px(prev_frame, frame, weights)
    candidates = {0, estimated_shift, -estimated_shift}

    best_shift = 0
    best_residual = raw_change
    for shift in candidates:
        aligned = np.roll(frame, int(shift), axis=1)
        residual = _weighted_mean_absdiff(prev_frame, aligned, weights)
        if residual < best_residual:
            best_residual = residual
            best_shift = int(shift)

    yaw_deg = (float(best_shift) / float(max(1, frame.shape[1]))) * 360.0
    return PairMetrics(
        raw_change=raw_change,
        residual=best_residual,
        yaw_shift_px=best_shift,
        yaw_shift_deg=yaw_deg,
    )


def _feature_coverage(points: np.ndarray, width: int, height: int) -> float:
    if points.size == 0 or width <= 0 or height <= 0:
        return 0.0
    pts = points.reshape(-1, 2)
    grid_cols = 6
    grid_rows = 3
    xs = np.clip((pts[:, 0] / max(1.0, float(width))) * grid_cols, 0, grid_cols - 1).astype(np.int32)
    ys = np.clip((pts[:, 1] / max(1.0, float(height))) * grid_rows, 0, grid_rows - 1).astype(np.int32)
    cells = ys * grid_cols + xs
    return float(np.count_nonzero(np.bincount(cells, minlength=grid_cols * grid_rows))) / float(grid_cols * grid_rows)


def compute_pair_track_metrics(
    prev_frame: np.ndarray,
    frame: np.ndarray,
    yaw_shift_px: int,
) -> PairTrackMetrics:
    if prev_frame.shape != frame.shape:
        return PairTrackMetrics(0, 0.0, 0.0, 0.0)

    height, width = prev_frame.shape[:2]
    min_distance = max(8, min(width, height) // 64)
    try:
        points = cv2.goodFeaturesToTrack(
            prev_frame,
            maxCorners=240,
            qualityLevel=0.01,
            minDistance=min_distance,
            blockSize=7,
            useHarrisDetector=False,
        )
    except cv2.error:
        points = None
    if points is None or len(points) < 8:
        return PairTrackMetrics(0, 0.0, 0.0, 0.0)

    aligned = np.roll(frame, int(yaw_shift_px), axis=1)
    try:
        next_points, status, _err = cv2.calcOpticalFlowPyrLK(
            prev_frame,
            aligned,
            points,
            None,
            winSize=(19, 19),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 16, 0.03),
        )
    except cv2.error:
        return PairTrackMetrics(0, 0.0, 0.0, 0.0)
    if next_points is None or status is None:
        return PairTrackMetrics(0, 0.0, 0.0, 0.0)

    valid = status.reshape(-1) == 1
    track_count = int(np.count_nonzero(valid))
    if track_count <= 0:
        return PairTrackMetrics(0, 0.0, 0.0, 0.0)

    src = points.reshape(-1, 2)[valid]
    dst = next_points.reshape(-1, 2)[valid]
    displacement = np.linalg.norm(dst - src, axis=1)
    diagonal = max(1.0, math.hypot(float(width), float(height)))
    median_motion = float(np.median(displacement) / diagonal) if displacement.size else 0.0
    coverage = _feature_coverage(src, width, height)
    confidence = _bounded01(0.60 * min(track_count / 80.0, 1.0) + 0.40 * coverage)
    return PairTrackMetrics(
        track_count=track_count,
        coverage=coverage,
        confidence=confidence,
        median_residual_motion=median_motion,
    )


def compute_pair_blur_score(frame: np.ndarray) -> float:
    return float(cv2.Laplacian(frame, cv2.CV_64F).var())


def pair_gate_dimensions(analysis_w: int, analysis_h: int) -> tuple[int, int]:
    if analysis_w <= 0 or analysis_h <= 0:
        return analysis_w, analysis_h
    if analysis_w <= PAIR_GATE_WIDTH_DEFAULT:
        return analysis_w, analysis_h
    return scaled_dimensions(analysis_w, analysis_h, PAIR_GATE_WIDTH_DEFAULT)


def pair_gate_frame(frame: np.ndarray, gate_w: int, gate_h: int) -> np.ndarray:
    if frame.shape[1] == gate_w and frame.shape[0] == gate_h:
        return frame
    return cv2.resize(frame, (gate_w, gate_h), interpolation=cv2.INTER_AREA)


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def assess_pair_frame_risk(
    *,
    blur_score: float,
    sharpness_baseline: float | None,
    track: PairTrackMetrics | None,
    track_min_confidence: float,
    track_min_count: int,
) -> PairFrameRisk:
    """Classify candidate-only SfM risks without reviving the old quality score."""
    sharpness_ratio = None
    if sharpness_baseline is not None and sharpness_baseline > 1e-6:
        sharpness_ratio = blur_score / sharpness_baseline

    weak_match = False
    weak_for_blur = False
    low_texture_track = False
    if track is not None:
        min_count = max(0, int(track_min_count))
        min_confidence = max(0.0, float(track_min_confidence))
        weak_match = track.track_count < min_count or track.confidence < min_confidence
        weak_for_blur = (
            track.track_count < max(12, min_count // 2)
            or track.confidence < max(0.15, min_confidence * 0.75)
        )
        low_texture_track = (
            track.track_count < max(12, min_count)
            or track.coverage < 0.12
            or track.confidence < max(0.12, min_confidence * 0.5)
        )

    severe_sharpness_drop = sharpness_ratio is not None and sharpness_ratio <= min(0.12, PAIR_MOTION_BLUR_RATIO)
    motion_blur = (
        sharpness_baseline is not None
        and sharpness_baseline >= PAIR_MOTION_BLUR_BASELINE_MIN
        and sharpness_ratio is not None
        and sharpness_ratio <= PAIR_MOTION_BLUR_RATIO
        and (weak_for_blur or severe_sharpness_drop)
    )
    low_texture = (
        not motion_blur
        and blur_score <= PAIR_LOW_TEXTURE_SHARPNESS
        and (sharpness_baseline is None or sharpness_baseline <= PAIR_MOTION_BLUR_BASELINE_MIN)
        and (track is None or low_texture_track)
    )

    return PairFrameRisk(
        blur_score=blur_score,
        sharpness_baseline=sharpness_baseline,
        sharpness_ratio=sharpness_ratio,
        motion_blur=motion_blur,
        low_texture=low_texture,
        weak_match=weak_match,
    )


def analyze_video_window(
    video_path: Path,
    ffmpeg_bin: str,
    video_fps: float,
    src_w: int,
    src_h: int,
    analysis_width: int,
    start_sec: float | None = None,
    duration_sec: float | None = None,
    sample_fps: float = 0.0,
    progress_phase: str = "",
    progress_total_frames: int = 0,
    progress_step_frames: int = 0,
    quality_mode: str = "sfm",
    compute_feature_motion: bool = True,
) -> tuple[list[float], list[float], list[float], list[float], int, int, float]:
    out_w, out_h = scaled_dimensions(src_w, src_h, analysis_width)
    vf_parts = [f"scale={out_w}:{out_h}:flags=bilinear", "format=gray"]
    effective_fps = video_fps
    if sample_fps > 0:
        effective_fps = min(sample_fps, video_fps) if video_fps > 0 else sample_fps
        vf_parts.append(f"fps={effective_fps:.6f}")
    vf = ",".join(vf_parts)

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if start_sec is not None and start_sec > 0:
        cmd.extend(["-ss", f"{start_sec:.6f}"])
    cmd.extend(["-i", str(video_path)])
    if duration_sec is not None and duration_sec > 0:
        cmd.extend(["-t", f"{duration_sec:.6f}"])
    cmd.extend(
        [
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
    )

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    assert proc.stderr is not None
    stderr_chunks: list[bytes] = []
    stderr_thread = threading.Thread(
        target=_drain_binary_pipe,
        args=(proc.stderr, stderr_chunks),
        daemon=True,
    )
    stderr_thread.start()

    frame_size = out_w * out_h
    prev_frame: np.ndarray | None = None
    blur_scores: list[float] = []
    quality_scores: list[float] = []
    change_scores: list[float] = []
    feature_motion_scores: list[float] = []
    last_progress_report = 0

    def emit_progress(processed_frames: int, force: bool = False) -> None:
        nonlocal last_progress_report
        if not progress_phase:
            return
        if not force:
            if progress_step_frames <= 0:
                return
            if processed_frames - last_progress_report < progress_step_frames:
                return
        if progress_total_frames > 0:
            pct = min(100.0, (processed_frames / float(progress_total_frames)) * 100.0)
            print(
                f"[progress] {progress_phase} {processed_frames}/{progress_total_frames} frames ({pct:.1f}%)",
                flush=True,
            )
        else:
            print(f"[progress] {progress_phase} {processed_frames} frames", flush=True)
        last_progress_report = processed_frames

    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if not raw:
                break
            if len(raw) < frame_size:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((out_h, out_w))
            lap_var = float(cv2.Laplacian(frame, cv2.CV_64F).var())
            blur_scores.append(lap_var)
            metrics = compute_quality_metrics(frame, lap_var, quality_mode)
            quality_scores.append(metrics.quality)

            if prev_frame is None:
                change_scores.append(1.0)
                feature_motion_scores.append(0.0)
            else:
                diff = cv2.absdiff(frame, prev_frame)
                change_scores.append(float(np.mean(diff) / 255.0))
                if compute_feature_motion:
                    feature_motion_scores.append(compute_feature_motion_score(prev_frame, frame))
                else:
                    feature_motion_scores.append(0.0)

            prev_frame = frame
            processed = len(change_scores)
            if processed == 1:
                emit_progress(processed, force=True)
            else:
                emit_progress(processed)
    finally:
        ret = proc.wait()
        stderr_thread.join()
        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")

    if ret != 0:
        raise RuntimeError(f"ffmpeg analysis failed: {stderr_text.strip()}")
    if not blur_scores:
        raise RuntimeError("No frames decoded during analysis")
    emit_progress(len(blur_scores), force=True)

    return blur_scores, change_scores, quality_scores, feature_motion_scores, out_w, out_h, effective_fps


def analyze_video(
    video_path: Path,
    ffmpeg_bin: str,
    video_fps: float,
    src_w: int,
    src_h: int,
    analysis_width: int,
    progress_phase: str = "",
    progress_total_frames: int = 0,
    progress_step_frames: int = 0,
    quality_mode: str = "sfm",
    compute_feature_motion: bool = True,
) -> tuple[list[float], list[float], list[float], list[float], int, int]:
    blur_scores, change_scores, quality_scores, feature_motion_scores, out_w, out_h, _ = analyze_video_window(
        video_path=video_path,
        ffmpeg_bin=ffmpeg_bin,
        video_fps=video_fps,
        src_w=src_w,
        src_h=src_h,
        analysis_width=analysis_width,
        start_sec=None,
        duration_sec=None,
        sample_fps=0.0,
        progress_phase=progress_phase,
        progress_total_frames=progress_total_frames,
        progress_step_frames=progress_step_frames,
        quality_mode=quality_mode,
        compute_feature_motion=compute_feature_motion,
    )
    return blur_scores, change_scores, quality_scores, feature_motion_scores, out_w, out_h


def _scaled_pair_threshold(
    base_threshold: float,
    interval_sec: float,
    profile: PairThresholdProfile,
) -> float:
    reference = max(1e-6, profile.reference_interval_sec)
    interval = max(1e-6, float(interval_sec))
    raw = base_threshold * math.pow(interval / reference, profile.exponent)
    low = base_threshold * math.pow(profile.min_interval_sec / reference, profile.exponent)
    high = base_threshold * math.pow(profile.max_interval_sec / reference, profile.exponent)
    return max(low, min(high, raw))


def resolve_pair_thresholds(
    interval_sec: float,
    motion_profile: str = "walk",
    drop_threshold: float = -1.0,
    add_threshold: float = -1.0,
) -> PairThresholds:
    profile_name = motion_profile if motion_profile in PAIR_THRESHOLD_PROFILES else "walk"
    profile = PAIR_THRESHOLD_PROFILES[profile_name]

    auto_drop = _scaled_pair_threshold(profile.base_drop_threshold, interval_sec, profile)
    auto_add = _scaled_pair_threshold(profile.base_add_threshold, interval_sec, profile)
    drop = auto_drop if drop_threshold < 0.0 else float(drop_threshold)
    add = auto_add if add_threshold < 0.0 else float(add_threshold)
    if add <= drop:
        raise ValueError("--pair-add-threshold must be greater than --pair-drop-threshold")

    if drop_threshold < 0.0 and add_threshold < 0.0:
        mode = "auto"
    elif drop_threshold >= 0.0 and add_threshold >= 0.0:
        mode = "manual"
    else:
        mode = "mixed"
    return PairThresholds(drop=drop, add=add, profile=profile_name, mode=mode)


def _pair_status(tokens: Sequence[str]) -> str:
    filtered = [token for token in tokens if token and token != "ok"]
    return "+".join(filtered) if filtered else "ok"


def _pair_row(
    *,
    idx: int,
    prev_kept_idx: int | None,
    fps: float,
    status_tokens: Sequence[str],
    decision: str,
    reason: str,
    metrics: PairMetrics | None,
    track: PairTrackMetrics | None,
    risk: PairFrameRisk | None,
    analysis_w: int,
    gate_w: int,
    thresholds: PairThresholds | None = None,
) -> dict:
    gap_frames = 0 if prev_kept_idx is None else max(0, idx - prev_kept_idx)
    yaw_deg = "" if metrics is None else metrics.yaw_shift_deg
    status = _pair_status(status_tokens)
    risk_flags = [token for token in ("motion_blur", "low_texture", "weak_match") if token in status]
    if track is not None and track.track_count > 0 and track.confidence < 0.28 and "weak_match" not in risk_flags:
        risk_flags.append("weak_match")
    return {
        "original_index": idx,
        "final_index": idx,
        "change_score_original": None if metrics is None else metrics.residual,
        "change_score_final": None if metrics is None else metrics.residual,
        "blur_score_original": None if risk is None else risk.blur_score,
        "blur_score_final": None if risk is None else risk.blur_score,
        "sharpness_baseline": "" if risk is None or risk.sharpness_baseline is None else risk.sharpness_baseline,
        "sharpness_ratio": "" if risk is None or risk.sharpness_ratio is None else risk.sharpness_ratio,
        "quality_score_original": None,
        "quality_score_final": None,
        "status": status,
        "decision": decision,
        "analysis_pipeline": "pair",
        "selection_reason": reason,
        "review_required": "1" if status != "ok" else "0",
        "prev_kept_index": "" if prev_kept_idx is None else prev_kept_idx,
        "gap_sec": (gap_frames / fps) if fps > 0 else "",
        "yaw_shift_px": "" if metrics is None else metrics.yaw_shift_px,
        "yaw_shift_deg": yaw_deg,
        "residual_score": "" if metrics is None else metrics.residual,
        "raw_change_score": "" if metrics is None else metrics.raw_change,
        "track_count": "" if track is None else track.track_count,
        "track_coverage": "" if track is None else track.coverage,
        "match_confidence": "" if track is None else track.confidence,
        "risk_flags": ",".join(risk_flags),
        "analysis_width": analysis_w,
        "pair_gate_width": gate_w,
        "pair_motion_profile": "" if thresholds is None else thresholds.profile,
        "pair_threshold_mode": "" if thresholds is None else thresholds.mode,
        "pair_drop_threshold": "" if thresholds is None else thresholds.drop,
        "pair_add_threshold": "" if thresholds is None else thresholds.add,
    }


def analyze_pair_selection(
    video_path: Path,
    ffmpeg_bin: str,
    video_info: VideoInfo,
    analysis_width: int,
    interval_sec: float,
    fixed_smart: bool,
    min_gap_sec: float,
    max_gap_sec: float,
    drop_threshold: float,
    add_threshold: float,
    threshold_profile: str,
    threshold_mode: str,
    max_inserts_per_interval: int,
    track_min_confidence: float,
    track_min_count: int,
    progress_phase: str = "analyze",
) -> tuple[list[dict], int, int, int, int, int]:
    """Streaming pair analysis for GUI extraction.

    The metric is relative to the last kept frame: estimate equirectangular yaw
    as a horizontal roll, measure residual change after that alignment, and run
    sparse tracking only at actual keep/drop decision points.
    """
    if cv2 is None or np is None:
        ensure_python_deps()
    if interval_sec <= 0:
        raise ValueError("--interval-sec must be > 0")
    if min_gap_sec <= 0 or max_gap_sec <= 0:
        raise ValueError("--min-gap-sec and --max-gap-sec must be > 0")
    if max_gap_sec < min_gap_sec:
        raise ValueError("--max-gap-sec must be >= --min-gap-sec")

    out_w, out_h = scaled_dimensions(video_info.width, video_info.height, analysis_width)
    vf = f"scale={out_w}:{out_h}:flags=bilinear,format=gray"
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    assert proc.stderr is not None
    stderr_chunks: list[bytes] = []
    stderr_thread = threading.Thread(
        target=_drain_binary_pipe,
        args=(proc.stderr, stderr_chunks),
        daemon=True,
    )
    stderr_thread.start()

    frame_size = out_w * out_h
    gate_w, gate_h = pair_gate_dimensions(out_w, out_h)
    weights = _latitude_weights(gate_h)
    interval_frames = max(1, int(round(interval_sec * video_info.fps)))
    min_gap_frames = max(1, int(round(min_gap_sec * video_info.fps)))
    max_gap_frames = max(min_gap_frames, int(round(max_gap_sec * video_info.fps)))
    max_inserts = max(0, int(max_inserts_per_interval))
    thresholds = PairThresholds(
        drop=max(0.0, float(drop_threshold)),
        add=max(0.0, float(add_threshold)),
        profile=threshold_profile,
        mode=threshold_mode,
    )
    total_hint = video_info.total_frames
    progress_step = max(10, total_hint // 100) if total_hint > 0 else max(10, int(round(video_info.fps * 2.0)))

    rows: list[dict] = []
    seen_rows: set[int] = set()
    inserted_by_slot: dict[int, int] = {}
    kept_sharpness: list[float] = []
    last_keep_frame: np.ndarray | None = None
    last_keep_gate_frame: np.ndarray | None = None
    last_keep_idx: int | None = None
    last_frame: np.ndarray | None = None
    last_frame_idx = -1
    last_progress_report = 0

    def emit_progress(processed_frames: int, force: bool = False) -> None:
        nonlocal last_progress_report
        if not progress_phase:
            return
        if not force and processed_frames - last_progress_report < progress_step:
            return
        if total_hint > 0:
            pct = min(100.0, (processed_frames / float(total_hint)) * 100.0)
            print(
                f"[progress] {progress_phase} {processed_frames}/{total_hint} frames ({pct:.1f}%)",
                flush=True,
            )
        else:
            print(f"[progress] {progress_phase} {processed_frames} frames", flush=True)
        last_progress_report = processed_frames

    def append_row(row: dict) -> None:
        idx = int(row["final_index"])
        if idx in seen_rows:
            return
        rows.append(row)
        seen_rows.add(idx)

    def evaluate_row(
        *,
        idx: int,
        frame: np.ndarray,
        status_tokens: Sequence[str],
        decision: str,
        reason: str,
        metrics: PairMetrics,
        allow_motion_blur_drop: bool,
    ) -> dict:
        track_shift_px = int(round(metrics.yaw_shift_px * (frame.shape[1] / float(max(1, gate_w)))))
        track = compute_pair_track_metrics(last_keep_frame, frame, track_shift_px) if last_keep_frame is not None else None
        tokens = list(status_tokens)
        risk = assess_pair_frame_risk(
            blur_score=compute_pair_blur_score(frame),
            sharpness_baseline=_median_or_none(kept_sharpness),
            track=track,
            track_min_confidence=track_min_confidence,
            track_min_count=track_min_count,
        )
        if risk.motion_blur:
            tokens.append("motion_blur")
            if allow_motion_blur_drop and decision != "drop":
                decision = "drop"
                reason = "motion_blur"
        elif risk.low_texture:
            tokens.append("low_texture")
        if decision != "drop" and risk.weak_match:
            tokens.append("weak_match")
        return _pair_row(
            idx=idx,
            prev_kept_idx=last_keep_idx,
            fps=video_info.fps,
            status_tokens=tokens,
            decision=decision,
            reason=reason,
            metrics=metrics,
            track=track,
            risk=risk,
            analysis_w=out_w,
            gate_w=gate_w,
            thresholds=thresholds,
        )

    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if not raw:
                break
            if len(raw) < frame_size:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((out_h, out_w))
            idx = last_frame_idx + 1
            last_frame = frame
            last_frame_idx = idx

            if last_keep_frame is None:
                gate_frame = pair_gate_frame(frame, gate_w, gate_h)
                initial_risk = assess_pair_frame_risk(
                    blur_score=compute_pair_blur_score(frame),
                    sharpness_baseline=None,
                    track=None,
                    track_min_confidence=track_min_confidence,
                    track_min_count=track_min_count,
                )
                initial_tokens = ["low_texture"] if initial_risk.low_texture else []
                append_row(
                    _pair_row(
                        idx=idx,
                        prev_kept_idx=None,
                        fps=video_info.fps,
                        status_tokens=initial_tokens,
                        decision="keep",
                        reason="initial",
                        metrics=None,
                        track=None,
                        risk=initial_risk,
                        analysis_w=out_w,
                        gate_w=gate_w,
                        thresholds=thresholds,
                    )
                )
                kept_sharpness.append(initial_risk.blur_score)
                last_keep_frame = frame.copy()
                last_keep_gate_frame = gate_frame.copy()
                last_keep_idx = idx
                emit_progress(idx + 1, force=True)
                continue

            assert last_keep_idx is not None
            gap = idx - last_keep_idx
            if gap < min_gap_frames:
                emit_progress(idx + 1)
                continue

            slot = idx // interval_frames
            base_due = (idx % interval_frames) == 0
            max_due = gap >= max_gap_frames
            assert last_keep_gate_frame is not None
            gate_frame = pair_gate_frame(frame, gate_w, gate_h)
            metrics = compute_pair_metrics(last_keep_gate_frame, gate_frame, weights)
            high_novelty = thresholds.add <= 0.0 or metrics.residual >= thresholds.add
            low_redundancy = metrics.residual < thresholds.drop
            inserted = inserted_by_slot.get(slot, 0)

            row: dict | None = None
            keep_current = False
            status_tokens: list[str] = []
            reason = "ok"
            decision = "keep"
            insert_slot: int | None = None

            if fixed_smart and not base_due and high_novelty and inserted < max_inserts:
                keep_current = True
                status_tokens.append("novelty_added")
                reason = "novelty_added"
                insert_slot = slot
            elif base_due:
                if fixed_smart and low_redundancy and not max_due:
                    decision = "drop"
                    status_tokens.append("redundant_drop")
                    reason = "redundant_drop"
                else:
                    keep_current = True
                    reason = "fixed_interval"
            elif max_due:
                keep_current = True
                status_tokens.append("gap_forced")
                reason = "gap_forced"

            if max_due and keep_current and "gap_forced" not in status_tokens:
                status_tokens.append("gap_forced")
                reason = "gap_forced"

            if decision == "drop" or keep_current:
                row = evaluate_row(
                    idx=idx,
                    frame=frame,
                    status_tokens=status_tokens,
                    decision=decision,
                    reason=reason,
                    metrics=metrics,
                    allow_motion_blur_drop=not max_due,
                )
                append_row(row)
                if keep_current and row.get("decision") != "drop":
                    if insert_slot is not None:
                        inserted_by_slot[insert_slot] = inserted + 1
                    if "motion_blur" not in row.get("status", ""):
                        row_blur = row.get("blur_score_final")
                        if row_blur not in (None, ""):
                            kept_sharpness.append(float(row_blur))
                            del kept_sharpness[:-PAIR_SHARPNESS_HISTORY]
                    last_keep_frame = frame.copy()
                    last_keep_gate_frame = gate_frame.copy()
                    last_keep_idx = idx

            emit_progress(idx + 1)
    finally:
        ret = proc.wait()
        stderr_thread.join()
        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")

    if ret != 0:
        raise RuntimeError(f"ffmpeg pair analysis failed: {stderr_text.strip()}")
    if last_frame_idx < 0 or last_frame is None:
        raise RuntimeError("No frames decoded during pair analysis")

    if last_keep_idx is not None and last_frame_idx != last_keep_idx and last_frame_idx not in seen_rows:
        endpoint_gate_frame = pair_gate_frame(last_frame, gate_w, gate_h)
        metrics = (
            compute_pair_metrics(last_keep_gate_frame, endpoint_gate_frame, weights)
            if last_keep_gate_frame is not None
            else None
        )
        if metrics is not None:
            row = evaluate_row(
                idx=last_frame_idx,
                frame=last_frame,
                status_tokens=["gap_forced"],
                decision="keep",
                reason="endpoint",
                metrics=metrics,
                allow_motion_blur_drop=False,
            )
            append_row(row)
    emit_progress(last_frame_idx + 1, force=True)

    rows.sort(key=lambda row: int(row["final_index"]))
    return rows, out_w, out_h, min_gap_frames, max_gap_frames, last_frame_idx + 1


# ===========================================================================
# 解析キャッシュ: 動画メタ情報 + analysis_width が一致すれば再計算をスキップ
# ===========================================================================

CACHE_VERSION = 3
QUALITY_MODE = "sfm"


def cache_path_for(scene_dir: Path) -> Path:
    return scene_dir / "extract_cache.npz"


def video_signature(video_path: Path) -> tuple[int, int]:
    """動画ファイルの (size, mtime_ns) を返す。キャッシュ無効化判定用。"""
    st = video_path.stat()
    return int(st.st_size), int(st.st_mtime_ns)


def save_analysis_cache(
    cache_path: Path,
    video_path: Path,
    video_info: VideoInfo,
    analysis_w: int,
    analysis_h: int,
    blur_scores: list[float],
    change_scores: list[float],
    quality_scores: list[float],
    feature_motion_scores: list[float],
    quality_mode: str,
    feature_motion_computed: bool = True,
) -> None:
    if np is None:
        return
    size, mtime_ns = video_signature(video_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        version=np.int64(CACHE_VERSION),
        video_size=np.int64(size),
        video_mtime_ns=np.int64(mtime_ns),
        video_width=np.int32(video_info.width),
        video_height=np.int32(video_info.height),
        video_fps=np.float64(video_info.fps),
        video_duration=np.float64(video_info.duration),
        video_total_frames=np.int64(video_info.total_frames),
        analysis_width=np.int32(analysis_w),
        analysis_height=np.int32(analysis_h),
        quality_mode=np.asarray(quality_mode),
        feature_motion_computed=np.asarray(bool(feature_motion_computed)),
        blur_scores=np.asarray(blur_scores, dtype=np.float64),
        change_scores=np.asarray(change_scores, dtype=np.float64),
        quality_scores=np.asarray(quality_scores, dtype=np.float64),
        feature_motion_scores=np.asarray(feature_motion_scores, dtype=np.float64),
    )
    print(f"[cache] saved analysis cache: {cache_path}")


def _combined_motion_scores(
    change_scores: list[float],
    feature_motion_scores: list[float] | None = None,
) -> np.ndarray:
    """Combine image-difference motion and sparse feature displacement.

    ``change_scores`` catches broad brightness/texture changes. Sparse feature
    motion is closer to what SfM can use, but its natural scale is smaller, so
    it is weighted into the same rough range for cumulative thinning.
    """
    scores = np.asarray(change_scores, dtype=np.float64)
    if feature_motion_scores is None:
        return scores
    feature = np.asarray(feature_motion_scores, dtype=np.float64)
    if feature.shape != scores.shape:
        return scores
    return np.maximum(scores, feature * 4.0)


def _fixed_smart_motion_scores(
    change_scores: list[float],
    feature_motion_scores: list[float],
    change_threshold: float,
    feature_motion_threshold: float,
) -> np.ndarray:
    """Normalize motion by the same thresholds used for fixed-smart insertion."""
    change = np.asarray(change_scores, dtype=np.float64)
    scores = np.zeros_like(change)
    if change_threshold > 0.0:
        scores = np.maximum(scores, change / change_threshold)
    feature = np.asarray(feature_motion_scores, dtype=np.float64)
    if feature.shape == change.shape and feature_motion_threshold > 0.0:
        scores = np.maximum(scores, feature / feature_motion_threshold)
    return scores


def thin_stationary(
    rows: list[dict],
    change_scores: list[float],
    motion_threshold: float,
    keep_endpoints: bool = True,
    feature_motion_scores: list[float] | None = None,
    max_gap_frames: int | None = None,
) -> list[dict]:
    """累積モーションが閾値未満の連続採用フレームを間引く。

    立ち止まり区間（変化が小さい）でフレーム間隔が密集しすぎているのを削減する。
    歩行など変化が大きい区間では何も削らない。

    Args:
        rows: select_representative_frames の出力行（各 dict に "final_index" がある）。
        change_scores: 解析時の change_scores (per analyzed frame)。
        motion_threshold: 直前 kept フレームから次採用候補までの累積 change がこれ未満なら drop。
            0 以下なら間引きなし（全フレーム keep）。
        keep_endpoints: True なら各 stationary cluster の先頭・末尾は強制保持。
        feature_motion_scores: 疎な特徴点追跡から求めた正規化モーションスコア。
            指定された場合は change_scores と合成して、輝度変化だけでは拾いにくい
            SfM向けの視差を間引き判定に反映する。
        max_gap_frames: 直前 keep からこのフレーム数以上離れた候補は安全側で keep。

    Returns:
        各 row に "decision" と必要なら "status" を追加した同長のリスト。
        間引かれた row は decision="drop", status="thinned"。
        keep される row は decision="keep" を維持/設定。
    """
    if motion_threshold <= 0.0 or len(rows) < 2:
        for row in rows:
            row.setdefault("decision", "keep")
        return rows

    n = len(change_scores)
    out_rows = [dict(row) for row in rows]
    motion_scores = _combined_motion_scores(change_scores, feature_motion_scores)

    # 累積モーション計算: rows は final_index 順を仮定（代表選択は順番を保つ）
    last_kept_pos = 0
    out_rows[0]["decision"] = "keep"

    for pos in range(1, len(out_rows) - (1 if keep_endpoints else 0)):
        last_idx = int(out_rows[last_kept_pos]["final_index"])
        cur_idx = int(out_rows[pos]["final_index"])
        if cur_idx <= last_idx or last_idx < 0 or cur_idx >= n:
            # インデックス異常時は安全側で keep
            out_rows[pos]["decision"] = "keep"
            last_kept_pos = pos
            continue

        if max_gap_frames is not None and max_gap_frames > 0 and cur_idx - last_idx >= max_gap_frames:
            out_rows[pos]["decision"] = "keep"
            last_kept_pos = pos
            continue

        # last_idx (排他) から cur_idx (含む) までの motion_scores を累積
        cumulative = float(np.sum(motion_scores[last_idx + 1 : cur_idx + 1]))

        if cumulative >= motion_threshold:
            out_rows[pos]["decision"] = "keep"
            last_kept_pos = pos
        else:
            out_rows[pos]["decision"] = "drop"
            existing_status = out_rows[pos].get("status", "ok")
            if existing_status == "ok":
                out_rows[pos]["status"] = "thinned"
            else:
                out_rows[pos]["status"] = f"{existing_status}+thinned"

    # 末尾は強制保持（時間カバレッジ）
    if keep_endpoints and len(out_rows) >= 2:
        out_rows[-1]["decision"] = "keep"
    elif not keep_endpoints and len(out_rows) >= 2:
        # keep_endpoints=False なら末尾も判定対象
        last_idx = int(out_rows[last_kept_pos]["final_index"])
        cur_idx = int(out_rows[-1]["final_index"])
        if cur_idx > last_idx and last_idx >= 0 and cur_idx < n:
            if max_gap_frames is not None and max_gap_frames > 0 and cur_idx - last_idx >= max_gap_frames:
                cumulative = motion_threshold
            else:
                cumulative = float(np.sum(motion_scores[last_idx + 1 : cur_idx + 1]))
            if cumulative < motion_threshold:
                out_rows[-1]["decision"] = "drop"
                existing_status = out_rows[-1].get("status", "ok")
                if existing_status == "ok":
                    out_rows[-1]["status"] = "thinned"
                else:
                    out_rows[-1]["status"] = f"{existing_status}+thinned"
            else:
                out_rows[-1]["decision"] = "keep"
        else:
            out_rows[-1]["decision"] = "keep"

    return out_rows


def load_analysis_cache(
    cache_path: Path,
    video_path: Path,
    video_info: VideoInfo,
    analysis_width: int,
    quality_mode: str = "sfm",
    require_feature_motion: bool = False,
) -> tuple[list[float], list[float], list[float], list[float], int, int] | None:
    """キャッシュが有効なら解析スコア一式と (analysis_w, analysis_h) を返す。
    無効/不在/エラーなら None。"""
    if np is None or not cache_path.exists():
        return None
    try:
        with np.load(cache_path) as data:
            if int(data["version"]) != CACHE_VERSION:
                return None
            cur_size, cur_mtime_ns = video_signature(video_path)
            if int(data["video_size"]) != cur_size:
                return None
            if int(data["video_mtime_ns"]) != cur_mtime_ns:
                return None
            cached_aw = int(data["analysis_width"])
            # 解析幅が現在の指定と一致していること（同等の縮小寸法を生成するため厳密比較）
            cached_target_w, _ = scaled_dimensions(video_info.width, video_info.height, analysis_width)
            if cached_aw != cached_target_w:
                return None
            cached_quality_mode = str(data["quality_mode"].tolist())
            if cached_quality_mode != quality_mode:
                return None
            feature_motion_computed = True
            if "feature_motion_computed" in data.files:
                feature_motion_computed = bool(data["feature_motion_computed"].tolist())
            if require_feature_motion and not feature_motion_computed:
                return None
            blur = data["blur_scores"].tolist()
            change = data["change_scores"].tolist()
            quality = data["quality_scores"].tolist()
            feature_motion = data["feature_motion_scores"].tolist()
            ah = int(data["analysis_height"])
            if not (len(blur) == len(change) == len(quality) == len(feature_motion)):
                return None
            return blur, change, quality, feature_motion, cached_aw, ah
    except Exception as e:
        print(f"[cache] failed to load cache (will recompute): {e}")
        return None


def select_fixed(total_frames: int, fps: float, interval_sec: float) -> tuple[list[int], int]:
    if interval_sec <= 0:
        raise ValueError("--interval-sec must be > 0")

    step = max(1, int(round(interval_sec * fps)))
    indices = list(range(0, total_frames, step))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    return indices, step


def select_fixed_smart(
    base_indices: list[int],
    change_scores: list[float],
    feature_motion_scores: list[float],
    fps: float,
    min_gap_sec: float,
    max_gap_sec: float,
    change_threshold: float = 0.04,
    feature_motion_threshold: float = 0.012,
    max_inserts_per_interval: int = 2,
) -> tuple[list[int], set[int], set[int], int, int]:
    """Add high-motion anchors to fixed interval selection.

    The base cadence remains fixed. This adds extra anchors inside a fixed
    interval when either broad image difference or sparse feature displacement
    clearly indicates motion that may be useful for SfM, and marks low-motion
    base candidates for review/drop using the same normalized motion score.
    """
    if min_gap_sec <= 0 or max_gap_sec <= 0:
        raise ValueError("--min-gap-sec and --max-gap-sec must be > 0")
    if max_gap_sec < min_gap_sec:
        raise ValueError("--max-gap-sec must be >= --min-gap-sec")

    total = len(change_scores)
    if total <= 0:
        return [], set(), set(), 1, 1

    min_gap_frames = max(1, int(round(min_gap_sec * fps)))
    max_gap_frames = max(min_gap_frames, int(round(max_gap_sec * fps)))
    max_inserts = max(0, int(max_inserts_per_interval))

    base = sorted({idx for idx in base_indices if 0 <= idx < total})
    if not base:
        base = [0]
    if base[0] != 0:
        base.insert(0, 0)
    if base[-1] != total - 1:
        base.append(total - 1)

    if change_threshold <= 0.0 and feature_motion_threshold <= 0.0:
        return base, set(), set(), min_gap_frames, max_gap_frames

    has_feature_motion = len(feature_motion_scores) == total

    def event_score(idx: int) -> float:
        score = 0.0
        if change_threshold > 0.0:
            score = max(score, float(change_scores[idx]) / change_threshold)
        if has_feature_motion and feature_motion_threshold > 0.0:
            score = max(score, float(feature_motion_scores[idx]) / feature_motion_threshold)
        return score

    selected = set(base)
    added: set[int] = set()

    if max_inserts > 0:
        for left, right in zip(base, base[1:], strict=False):
            if right - left < min_gap_frames * 2:
                continue
            interval_selected = {left, right}
            for _ in range(max_inserts):
                best_idx: int | None = None
                best_score = 0.0
                for idx in range(left + min_gap_frames, right - min_gap_frames + 1):
                    if idx in interval_selected:
                        continue
                    if any(abs(idx - existing) < min_gap_frames for existing in interval_selected):
                        continue
                    score = event_score(idx)
                    if score > best_score:
                        best_idx = idx
                        best_score = score
                if best_idx is None or best_score < 1.0:
                    break
                selected.add(best_idx)
                added.add(best_idx)
                interval_selected.add(best_idx)

    selected_list = sorted(selected)
    thinned = _select_fixed_smart_thinned(
        selected_list,
        added,
        change_scores,
        feature_motion_scores,
        change_threshold,
        feature_motion_threshold,
        max_gap_frames,
    )
    return selected_list, added, thinned, min_gap_frames, max_gap_frames


def _select_fixed_smart_thinned(
    selected_indices: list[int],
    protected_indices: set[int],
    change_scores: list[float],
    feature_motion_scores: list[float],
    change_threshold: float,
    feature_motion_threshold: float,
    max_gap_frames: int,
) -> set[int]:
    """Mark low-motion fixed candidates while preserving smart-added anchors."""
    if len(selected_indices) < 3:
        return set()

    total = len(change_scores)
    motion_scores = _fixed_smart_motion_scores(
        change_scores,
        feature_motion_scores,
        change_threshold,
        feature_motion_threshold,
    )
    thinned: set[int] = set()
    last_kept = selected_indices[0]

    for idx in selected_indices[1:-1]:
        if idx < 0 or idx >= total or idx <= last_kept:
            last_kept = idx
            continue
        if idx in protected_indices:
            last_kept = idx
            continue
        if max_gap_frames > 0 and idx - last_kept >= max_gap_frames:
            last_kept = idx
            continue

        cumulative = float(np.sum(motion_scores[last_kept + 1 : idx + 1]))
        if cumulative >= 1.0:
            last_kept = idx
        else:
            thinned.add(idx)

    return thinned


def select_change(
    change_scores: list[float],
    fps: float,
    threshold: float,
    min_gap_sec: float,
    max_gap_sec: float,
) -> tuple[list[int], int, int]:
    if min_gap_sec <= 0 or max_gap_sec <= 0:
        raise ValueError("--min-gap-sec and --max-gap-sec must be > 0")
    if max_gap_sec < min_gap_sec:
        raise ValueError("--max-gap-sec must be >= --min-gap-sec")

    min_gap_frames = max(1, int(round(min_gap_sec * fps)))
    max_gap_frames = max(min_gap_frames, int(round(max_gap_sec * fps)))

    indices = [0]
    last = 0
    for i in range(1, len(change_scores)):
        gap = i - last
        if gap < min_gap_frames:
            continue
        if change_scores[i] >= threshold or gap >= max_gap_frames:
            indices.append(i)
            last = i

    if indices[-1] != len(change_scores) - 1:
        if len(change_scores) - 1 - indices[-1] >= max(1, min_gap_frames // 2):
            indices.append(len(change_scores) - 1)

    return indices, min_gap_frames, max_gap_frames


def estimate_count_range(duration_sec: float, min_gap_sec: float, max_gap_sec: float) -> tuple[int, int]:
    if duration_sec <= 0:
        return 1, 1
    if min_gap_sec <= 0 or max_gap_sec <= 0:
        raise ValueError("--min-gap-sec and --max-gap-sec must be > 0")
    if max_gap_sec < min_gap_sec:
        raise ValueError("--max-gap-sec must be >= --min-gap-sec")

    min_count = max(1, int(math.ceil(duration_sec / max_gap_sec)))
    max_count = max(min_count, int(math.ceil(duration_sec / min_gap_sec)))
    return min_count, max_count


def build_sample_windows(
    duration_sec: float,
    segment_sec: float,
    segment_count: int,
) -> list[tuple[float, float]]:
    if segment_sec <= 0:
        raise ValueError("--sample-segment-sec must be > 0")
    if segment_count <= 0:
        raise ValueError("--sample-segments must be > 0")
    if duration_sec <= 0:
        return []

    actual_segment = min(segment_sec, duration_sec)
    max_start = max(0.0, duration_sec - actual_segment)

    if segment_count == 1 or max_start <= 0:
        starts = [0.0]
    else:
        step = max_start / float(segment_count - 1)
        starts = [i * step for i in range(segment_count)]

    windows: list[tuple[float, float]] = []
    seen = set()
    for start in starts:
        clamped_start = max(0.0, min(start, max_start))
        key = int(round(clamped_start * 1000))
        if key in seen:
            continue
        seen.add(key)
        seg_dur = min(actual_segment, duration_sec - clamped_start)
        if seg_dur <= 0:
            continue
        windows.append((clamped_start, seg_dur))

    if not windows:
        windows.append((0.0, actual_segment))
    return windows


def estimate_change_sampled(
    video_path: Path,
    ffmpeg_bin: str,
    video_info: VideoInfo,
    analysis_width: int,
    threshold: float,
    min_gap_sec: float,
    max_gap_sec: float,
    sample_segments: int,
    sample_segment_sec: float,
    sample_fps: float,
) -> dict:
    windows = build_sample_windows(video_info.duration, sample_segment_sec, sample_segments)
    if not windows:
        raise RuntimeError("Could not build sample windows from video duration")

    weighted_rate_sum = 0.0
    weight_sum = 0.0
    used_segments = 0
    sampled_frames = 0
    sampled_duration = 0.0

    analysis_w = 0
    analysis_h = 0
    analysis_fps = min(sample_fps, video_info.fps) if sample_fps > 0 else video_info.fps
    min_gap_frames = max(1, int(round(min_gap_sec * analysis_fps)))

    for idx, (start_sec, seg_sec) in enumerate(windows, start=1):
        _, change_scores, _, _, out_w, out_h, seg_fps = analyze_video_window(
            video_path=video_path,
            ffmpeg_bin=ffmpeg_bin,
            video_fps=video_info.fps,
            src_w=video_info.width,
            src_h=video_info.height,
            analysis_width=analysis_width,
            start_sec=start_sec,
            duration_sec=seg_sec,
            sample_fps=sample_fps,
            quality_mode="sharpness",
            compute_feature_motion=False,
        )

        if len(change_scores) < 2:
            print(f"[sample] segment {idx}/{len(windows)} skipped: insufficient frames")
            continue

        selected, seg_min_gap_frames, _ = select_change(
            change_scores,
            seg_fps,
            threshold,
            min_gap_sec,
            max_gap_sec,
        )
        decoded_sec = len(change_scores) / seg_fps if seg_fps > 0 else seg_sec
        if decoded_sec <= 0:
            continue

        analysis_w = out_w
        analysis_h = out_h
        analysis_fps = seg_fps
        min_gap_frames = seg_min_gap_frames

        # Each sampled segment seeds one first frame; remove that bias before extrapolation.
        segment_selected = max(0, len(selected) - 1)
        selected_per_sec = segment_selected / decoded_sec

        weighted_rate_sum += selected_per_sec * decoded_sec
        weight_sum += decoded_sec
        used_segments += 1
        sampled_frames += len(change_scores)
        sampled_duration += decoded_sec

        print(
            f"[sample] segment {idx}/{len(windows)} start={start_sec:.2f}s "
            f"dur={decoded_sec:.2f}s selected={len(selected)}"
        )

    if used_segments == 0 or weight_sum <= 0:
        raise RuntimeError("Sampled estimate failed: no valid segment data")

    selected_rate = weighted_rate_sum / weight_sum
    estimated = 1 + int(round(selected_rate * max(video_info.duration, 0.0)))
    range_min, range_max = estimate_count_range(video_info.duration, min_gap_sec, max_gap_sec)
    estimated = max(range_min, min(range_max, estimated))

    return {
        "selected_count": estimated,
        "replaced_count": 0,
        "fallback_keep_count": 0,
        "analysis_w": analysis_w,
        "analysis_h": analysis_h,
        "analysis_fps": analysis_fps,
        "min_gap_frames": min_gap_frames,
        "sampled_segments_requested": len(windows),
        "sampled_segments_used": used_segments,
        "sampled_duration_sec": sampled_duration,
        "sampled_frames": sampled_frames,
        "range_min_count": range_min,
        "range_max_count": range_max,
    }


def _candidate_window(
    selected_indices: list[int],
    pos: int,
    total_frames: int,
) -> tuple[int, int]:
    original_idx = selected_indices[pos]
    if pos == 0:
        low = 0
    else:
        prev_idx = selected_indices[pos - 1]
        low = ((prev_idx + original_idx) // 2) + 1

    if pos + 1 >= len(selected_indices):
        high = total_frames - 1
    else:
        next_idx = selected_indices[pos + 1]
        high = (original_idx + next_idx) // 2

    low = max(0, min(low, original_idx))
    high = min(total_frames - 1, max(high, original_idx))
    return low, high


def representative_window_for_report(
    selected_indices: Sequence[int],
    total_frames: int,
) -> int:
    if not selected_indices or total_frames <= 0:
        return 0
    max_radius = 0
    selected = list(selected_indices)
    for pos, original_idx in enumerate(selected):
        low, high = _candidate_window(selected, pos, total_frames)
        max_radius = max(max_radius, original_idx - low, high - original_idx)
    return max_radius


def select_representative_frames(
    selected_indices: list[int],
    quality_scores: list[float],
    quality_min_score: float,
    quality_min_improvement: float,
    center_bias: float,
) -> list[dict]:
    """Choose one SfM-oriented representative frame for each extraction anchor.

    The initial fixed/change selection defines anchors. Each anchor owns a
    non-overlapping candidate window (midpoint to neighboring anchors). We
    select a better nearby frame only when the absolute quality improvement is
    clear; no global percentile is used.
    """
    if not selected_indices:
        return []

    n = len(quality_scores)
    if n <= 0:
        return []

    rows: list[dict] = []
    for pos, original_idx in enumerate(selected_indices):
        if original_idx < 0 or original_idx >= n:
            continue

        low, high = _candidate_window(selected_indices, pos, n)
        radius = max(abs(original_idx - low), abs(high - original_idx), 1)

        def objective(idx: int, original_idx: int = original_idx, radius: int = radius) -> float:
            center_score = 1.0 - (abs(idx - original_idx) / float(radius))
            return float(quality_scores[idx]) + center_bias * _bounded01(center_score)

        candidates = list(range(low, high + 1))
        best_idx = max(candidates, key=objective) if candidates else original_idx
        original_quality = float(quality_scores[original_idx])
        best_quality = float(quality_scores[best_idx])

        final_idx = original_idx
        status = "ok"
        if (
            best_idx != original_idx
            and best_quality - original_quality >= quality_min_improvement
            and objective(best_idx) > objective(original_idx)
        ):
            final_idx = best_idx
            status = "replaced"
        elif original_quality < quality_min_score:
            status = "fallback_keep"

        rows.append(
            {
                "original_index": original_idx,
                "final_index": final_idx,
                "status": status,
                "quality_min_score": quality_min_score,
                "quality_score_original": original_quality,
                "quality_score_final": float(quality_scores[final_idx]),
                "candidate_low": low,
                "candidate_high": high,
            }
        )

    return rows
def build_select_expr(frame_indices: list[int]) -> str:
    return "+".join(f"eq(n\\,{idx})" for idx in frame_indices)


def extract_selected_frames(
    video_path: Path,
    ffmpeg_bin: str,
    frame_indices: list[int],
    output_dir: Path,
    image_ext: str,
    jpg_quality: int,
    filename_prefix: str,
    frame_digits: int,
    allow_partial_tail: bool = False,
) -> list[int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_tmp_extract"
    if tmp_dir.exists():
        for p in tmp_dir.glob("*"):
            p.unlink(missing_ok=True)
    tmp_dir.mkdir(exist_ok=True)

    select_expr = build_select_expr(frame_indices)

    quality_args: list[str] = []
    if image_ext == "jpg":
        quality_args = ["-q:v", str(jpg_quality)]

    out_pattern = str(tmp_dir / f"%08d.{image_ext}")

    # Try filter script first to avoid command-length issues.
    with tempfile.NamedTemporaryFile("w", suffix=".ffscript", delete=False, encoding="utf-8") as tf:
        tf.write(f"select='{select_expr}'\n")
        filter_script_path = tf.name

    try:
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-progress",
            "pipe:2",
            "-y",
            "-i",
            str(video_path),
            "-filter_script:v",
            filter_script_path,
            "-vsync",
            "vfr",
            *quality_args,
            out_pattern,
        ]
        proc = run_cmd_with_ffmpeg_progress(cmd, phase="extract", total_items=len(frame_indices))

        if proc.returncode != 0:
            # Fallback when filter_script:v is unsupported by ffmpeg build.
            cmd = [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostats",
                "-progress",
                "pipe:2",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"select='{select_expr}'",
                "-vsync",
                "vfr",
                *quality_args,
                out_pattern,
            ]
            proc = run_cmd_with_ffmpeg_progress(cmd, phase="extract", total_items=len(frame_indices))
            stderr_text = (proc.stderr or "").lower()
            if proc.returncode != 0 and "unrecognized option" in stderr_text and "progress" in stderr_text:
                cmd = [
                    ffmpeg_bin,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(video_path),
                    "-vf",
                    f"select='{select_expr}'",
                    "-vsync",
                    "vfr",
                    *quality_args,
                    out_pattern,
                ]
                proc = run_cmd(cmd, capture=True)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg extraction failed: {proc.stderr.strip()}")
    finally:
        Path(filter_script_path).unlink(missing_ok=True)

    extracted_files = sorted(tmp_dir.glob(f"*.{image_ext}"))
    if len(extracted_files) != len(frame_indices):
        if allow_partial_tail and 0 < len(extracted_files) < len(frame_indices):
            missing = len(frame_indices) - len(extracted_files)
            print(
                "[warn] ffmpeg produced fewer frames than requested; "
                f"keeping {len(extracted_files)} extracted frame(s) and dropping {missing} trailing request(s)",
                flush=True,
            )
            frame_indices = frame_indices[:len(extracted_files)]
        else:
            for p in extracted_files:
                p.unlink(missing_ok=True)
            tmp_dir.rmdir()
            raise RuntimeError(
                f"Expected {len(frame_indices)} extracted files, got {len(extracted_files)}"
            )
    rename_total = len(frame_indices)
    rename_step = max(1, rename_total // 100)
    last_rename_report = 0
    print(f"[progress] finalize 0/{rename_total} files (0.0%)", flush=True)
    for seq, (src, frame_idx) in enumerate(zip(extracted_files, frame_indices, strict=True), start=1):
        dst_name = frame_filename(filename_prefix, frame_idx, image_ext, frame_digits)
        dst_path = output_dir / dst_name
        if dst_path.exists():
            dst_path.unlink()
        src.rename(dst_path)
        if seq - last_rename_report >= rename_step or seq == rename_total:
            pct = min(100.0, (seq / float(rename_total)) * 100.0)
            print(f"[progress] finalize {seq}/{rename_total} files ({pct:.1f}%)", flush=True)
            last_rename_report = seq

    tmp_dir.rmdir()
    return list(frame_indices)


SELECTED_CSV_FIELDNAMES = [
    "seq",
    "source_session",
    "source_video",
    "original_index",
    "final_index",
    "timestamp_sec",
    "change_score_original",
    "change_score_final",
    "blur_score_original",
    "blur_score_final",
    "sharpness_baseline",
    "sharpness_ratio",
    "quality_score_original",
    "quality_score_final",
    "status",
    "decision",
    "analysis_pipeline",
    "selection_reason",
    "review_required",
    "prev_kept_index",
    "gap_sec",
    "yaw_shift_px",
    "yaw_shift_deg",
    "residual_score",
    "raw_change_score",
    "track_count",
    "track_coverage",
    "match_confidence",
    "risk_flags",
    "analysis_width",
    "pair_gate_width",
    "pair_motion_profile",
    "pair_threshold_mode",
    "pair_drop_threshold",
    "pair_add_threshold",
    "output_file",
]


def _csv_score(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def build_quick_extract_rows(frame_indices: Sequence[int]) -> list[dict]:
    return [
        {
            "original_index": idx,
            "final_index": idx,
            "change_score_original": None,
            "change_score_final": None,
            "blur_score_original": None,
            "blur_score_final": None,
            "quality_score_original": None,
            "quality_score_final": None,
            "status": "ok",
            "decision": "keep",
            "analysis_pipeline": "quick",
            "selection_reason": "quick_extract",
            "review_required": "0",
        }
        for idx in frame_indices
    ]


def build_selected_csv_rows(
    rows: list[dict],
    fps: float,
    image_ext: str,
    filename_prefix: str,
    frame_digits: int,
    session_id: str = "",
    source_video: str = "",
) -> list[dict]:
    out_rows: list[dict] = []
    for i, row in enumerate(rows, start=1):
        final_idx = row["final_index"]
        timestamp_sec = f"{final_idx / fps:.6f}" if fps > 0 else ""
        out_rows.append(
            {
                "seq": i,
                "source_session": session_id,
                "source_video": source_video,
                "original_index": row["original_index"],
                "final_index": final_idx,
                "timestamp_sec": timestamp_sec,
                "change_score_original": _csv_score(row.get("change_score_original")),
                "change_score_final": _csv_score(row.get("change_score_final")),
                "blur_score_original": _csv_score(row.get("blur_score_original")),
                "blur_score_final": _csv_score(row.get("blur_score_final")),
                "sharpness_baseline": _csv_score(row.get("sharpness_baseline")),
                "sharpness_ratio": _csv_score(row.get("sharpness_ratio")),
                "quality_score_original": _csv_score(row.get("quality_score_original", 0.0)),
                "quality_score_final": _csv_score(row.get("quality_score_final", 0.0)),
                "status": row["status"],
                "decision": row.get("decision", "keep"),
                "analysis_pipeline": row.get("analysis_pipeline", "legacy"),
                "selection_reason": row.get("selection_reason", ""),
                "review_required": row.get("review_required", "1" if row.get("status", "ok") != "ok" else "0"),
                "prev_kept_index": row.get("prev_kept_index", ""),
                "gap_sec": _csv_score(row.get("gap_sec")),
                "yaw_shift_px": row.get("yaw_shift_px", ""),
                "yaw_shift_deg": _csv_score(row.get("yaw_shift_deg")),
                "residual_score": _csv_score(row.get("residual_score")),
                "raw_change_score": _csv_score(row.get("raw_change_score")),
                "track_count": row.get("track_count", ""),
                "track_coverage": _csv_score(row.get("track_coverage")),
                "match_confidence": _csv_score(row.get("match_confidence")),
                "risk_flags": row.get("risk_flags", ""),
                "analysis_width": row.get("analysis_width", ""),
                "pair_gate_width": row.get("pair_gate_width", ""),
                "pair_motion_profile": row.get("pair_motion_profile", ""),
                "pair_threshold_mode": row.get("pair_threshold_mode", ""),
                "pair_drop_threshold": _csv_score(row.get("pair_drop_threshold")),
                "pair_add_threshold": _csv_score(row.get("pair_add_threshold")),
                "output_file": f"images/{frame_filename(filename_prefix, final_idx, image_ext, frame_digits)}",
            }
        )
    return out_rows


def read_selected_csv(csv_path: Path) -> tuple[list[str], list[dict]]:
    if not csv_path.exists():
        return list(SELECTED_CSV_FIELDNAMES), []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or SELECTED_CSV_FIELDNAMES), list(reader)


def write_selected_csv_rows(csv_path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_fields = list(SELECTED_CSV_FIELDNAMES)
    for name in fieldnames:
        if name not in ordered_fields:
            ordered_fields.append(name)
    for row in rows:
        for name in row.keys():
            if name not in ordered_fields:
                ordered_fields.append(name)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fields)
        writer.writeheader()
        for seq, row in enumerate(rows, start=1):
            updated = dict(row)
            updated["seq"] = seq
            writer.writerow(updated)


def write_selected_csv(
    rows: list[dict],
    csv_path: Path,
    fps: float,
    image_ext: str,
    filename_prefix: str,
    frame_digits: int,
    existing_rows: Sequence[dict] | None = None,
    existing_fieldnames: Sequence[str] | None = None,
    session_id: str = "",
    source_video: str = "",
) -> None:
    if existing_rows is None:
        existing_rows = []
    if existing_fieldnames is None:
        existing_fieldnames = SELECTED_CSV_FIELDNAMES

    new_rows = build_selected_csv_rows(
        rows,
        fps=fps,
        image_ext=image_ext,
        filename_prefix=filename_prefix,
        frame_digits=frame_digits,
        session_id=session_id,
        source_video=source_video,
    )
    write_selected_csv_rows(csv_path, existing_fieldnames, [*existing_rows, *new_rows])


def write_report(
    report_path: Path,
    args: argparse.Namespace,
    video_info: VideoInfo,
    analysis_w: int,
    analysis_h: int,
    selected_rows: list[dict],
    window_frames: int,
    min_gap_frames: int,
    filename_prefix: str,
    frame_digits: int,
) -> None:
    pair_thresholds = resolve_pair_thresholds(
        args.interval_sec,
        getattr(args, "pair_motion_profile", "walk"),
        getattr(args, "pair_drop_threshold", -1.0),
        getattr(args, "pair_add_threshold", -1.0),
    )
    report = {
        "input_video": str(Path(args.input_video).resolve()),
        "mode": args.mode,
        "video": {
            "width": video_info.width,
            "height": video_info.height,
            "fps": video_info.fps,
            "duration_sec": video_info.duration,
            "total_frames": video_info.total_frames,
        },
        "analysis": {
            "width": analysis_w,
            "height": analysis_h,
            "pipeline": getattr(args, "analysis_pipeline", "legacy"),
            "quality_mode": "skipped" if getattr(args, "quick_extract", False) else QUALITY_MODE,
            "representative_window_frames": window_frames,
            "min_gap_frames": min_gap_frames,
        },
        "params": {
            "interval_sec": args.interval_sec,
            "change_threshold": args.change_threshold,
            "min_gap_sec": args.min_gap_sec,
            "max_gap_sec": args.max_gap_sec,
            "fixed_smart": args.fixed_smart,
            "quick_extract": getattr(args, "quick_extract", False),
            "analysis_pipeline": getattr(args, "analysis_pipeline", "legacy"),
            "fixed_smart_change_threshold": args.fixed_smart_change_threshold,
            "fixed_smart_feature_threshold": args.fixed_smart_feature_threshold,
            "fixed_smart_max_inserts_per_interval": args.fixed_smart_max_inserts_per_interval,
            "pair_motion_profile": getattr(args, "pair_motion_profile", "walk"),
            "pair_threshold_mode": pair_thresholds.mode,
            "pair_drop_threshold": getattr(args, "pair_drop_threshold", -1.0),
            "pair_add_threshold": getattr(args, "pair_add_threshold", -1.0),
            "pair_drop_threshold_resolved": pair_thresholds.drop,
            "pair_add_threshold_resolved": pair_thresholds.add,
            "pair_track_min_count": getattr(args, "pair_track_min_count", 0),
            "pair_track_min_confidence": getattr(args, "pair_track_min_confidence", 0.0),
            "quality_min_score": args.quality_min_score,
            "quality_min_improvement": args.quality_min_improvement,
            "center_bias": args.center_bias,
            "filename_prefix": filename_prefix,
            "frame_number_digits": frame_digits,
            "output_mode": getattr(args, "output_mode", "overwrite"),
        },
        "result": {
            "selected_count": len(selected_rows),
            "smart_added_count": sum(1 for r in selected_rows if "smart_added" in r.get("status", "")),
            "novelty_added_count": sum(1 for r in selected_rows if "novelty_added" in r.get("status", "")),
            "redundant_drop_count": sum(1 for r in selected_rows if "redundant_drop" in r.get("status", "")),
            "gap_forced_count": sum(1 for r in selected_rows if "gap_forced" in r.get("status", "")),
            "weak_match_count": sum(1 for r in selected_rows if "weak_match" in r.get("status", "")),
            "replaced_count": sum(1 for r in selected_rows if "replaced" in r.get("status", "")),
            "fallback_keep_count": sum(1 for r in selected_rows if "fallback_keep" in r.get("status", "")),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def build_summary_from_counts(
    args: argparse.Namespace,
    video_info: VideoInfo,
    analysis_w: int,
    analysis_h: int,
    min_gap_frames: int,
    window_frames: int,
    selected_count: int,
    replaced_count: int,
    fallback_keep_count: int,
    estimate_mode: str,
    filename_prefix: str,
    frame_digits: int,
    estimate_meta: dict | None = None,
) -> dict:
    pair_thresholds = resolve_pair_thresholds(
        args.interval_sec,
        getattr(args, "pair_motion_profile", "walk"),
        getattr(args, "pair_drop_threshold", -1.0),
        getattr(args, "pair_add_threshold", -1.0),
    )
    summary = {
        "input_video": str(Path(args.input_video).resolve()),
        "mode": args.mode,
        "estimate_mode": estimate_mode,
        "video": {
            "width": video_info.width,
            "height": video_info.height,
            "fps": video_info.fps,
            "duration_sec": video_info.duration,
            "total_frames": video_info.total_frames,
        },
        "analysis": {
            "width": analysis_w,
            "height": analysis_h,
            "pipeline": getattr(args, "analysis_pipeline", "legacy"),
            "min_gap_frames": min_gap_frames,
            "representative_window_frames": window_frames,
            "quality_mode": "skipped" if getattr(args, "quick_extract", False) else QUALITY_MODE,
        },
        "params": {
            "interval_sec": args.interval_sec,
            "change_threshold": args.change_threshold,
            "min_gap_sec": args.min_gap_sec,
            "max_gap_sec": args.max_gap_sec,
            "fixed_smart": args.fixed_smart,
            "quick_extract": getattr(args, "quick_extract", False),
            "analysis_pipeline": getattr(args, "analysis_pipeline", "legacy"),
            "fixed_smart_change_threshold": args.fixed_smart_change_threshold,
            "fixed_smart_feature_threshold": args.fixed_smart_feature_threshold,
            "fixed_smart_max_inserts_per_interval": args.fixed_smart_max_inserts_per_interval,
            "pair_motion_profile": getattr(args, "pair_motion_profile", "walk"),
            "pair_threshold_mode": pair_thresholds.mode,
            "pair_drop_threshold": getattr(args, "pair_drop_threshold", -1.0),
            "pair_add_threshold": getattr(args, "pair_add_threshold", -1.0),
            "pair_drop_threshold_resolved": pair_thresholds.drop,
            "pair_add_threshold_resolved": pair_thresholds.add,
            "pair_track_min_count": getattr(args, "pair_track_min_count", 0),
            "pair_track_min_confidence": getattr(args, "pair_track_min_confidence", 0.0),
            "analysis_width": args.analysis_width,
            "pair_gate_width": pair_gate_dimensions(analysis_w, analysis_h)[0] if analysis_w > 0 else 0,
            "quality_min_score": args.quality_min_score,
            "quality_min_improvement": args.quality_min_improvement,
            "center_bias": args.center_bias,
            "filename_prefix": filename_prefix,
            "frame_number_digits": frame_digits,
        },
        "result": {
            "selected_count": selected_count,
            "replaced_count": replaced_count,
            "fallback_keep_count": fallback_keep_count,
        },
    }
    if estimate_meta:
        summary["estimate"] = estimate_meta
    return summary


def build_summary(
    args: argparse.Namespace,
    video_info: VideoInfo,
    analysis_w: int,
    analysis_h: int,
    selected_rows: list[dict],
    min_gap_frames: int,
    window_frames: int,
    filename_prefix: str,
    frame_digits: int,
    estimate_mode: str = "full",
) -> dict:
    replaced_count = sum(1 for r in selected_rows if "replaced" in r.get("status", ""))
    smart_added_count = sum(1 for r in selected_rows if "smart_added" in r.get("status", ""))
    novelty_added_count = sum(1 for r in selected_rows if "novelty_added" in r.get("status", ""))
    redundant_drop_count = sum(1 for r in selected_rows if "redundant_drop" in r.get("status", ""))
    gap_forced_count = sum(1 for r in selected_rows if "gap_forced" in r.get("status", ""))
    motion_blur_count = sum(1 for r in selected_rows if "motion_blur" in r.get("status", ""))
    low_texture_count = sum(1 for r in selected_rows if "low_texture" in r.get("status", ""))
    weak_match_count = sum(1 for r in selected_rows if "weak_match" in r.get("status", ""))
    fallback_keep_count = sum(1 for r in selected_rows if "fallback_keep" in r.get("status", ""))
    thinned_count = sum(
        1 for r in selected_rows
        if r.get("decision") == "drop" and "thinned" in r.get("status", "")
    )
    kept_count = sum(1 for r in selected_rows if r.get("decision", "keep") != "drop")

    summary = build_summary_from_counts(
        args=args,
        video_info=video_info,
        analysis_w=analysis_w,
        analysis_h=analysis_h,
        min_gap_frames=min_gap_frames,
        window_frames=window_frames,
        selected_count=kept_count,
        replaced_count=replaced_count,
        fallback_keep_count=fallback_keep_count,
        estimate_mode=estimate_mode,
        filename_prefix=filename_prefix,
        frame_digits=frame_digits,
    )
    summary["result"]["smart_added_count"] = smart_added_count
    summary["result"]["novelty_added_count"] = novelty_added_count
    summary["result"]["redundant_drop_count"] = redundant_drop_count
    summary["result"]["gap_forced_count"] = gap_forced_count
    summary["result"]["motion_blur_count"] = motion_blur_count
    summary["result"]["low_texture_count"] = low_texture_count
    summary["result"]["weak_match_count"] = weak_match_count
    summary["result"]["thinned_count"] = thinned_count
    summary["result"]["selected_before_thin"] = len(selected_rows)
    return summary


def video_info_to_dict(video_info: VideoInfo) -> dict:
    return {
        "width": video_info.width,
        "height": video_info.height,
        "fps": video_info.fps,
        "duration_sec": video_info.duration,
        "total_frames": video_info.total_frames,
    }


def output_files_for_indices(
    final_indices: Sequence[int],
    filename_prefix: str,
    image_ext: str,
    frame_digits: int,
) -> list[str]:
    return [
        f"images/{frame_filename(filename_prefix, frame_idx, image_ext, frame_digits)}"
        for frame_idx in final_indices
    ]


def remove_session_outputs(scene_dir: Path, output_files: Sequence[str]) -> int:
    removed = 0
    images_dir = (scene_dir / "images").resolve()
    for rel in output_files:
        path = (scene_dir / rel).resolve()
        try:
            path.relative_to(images_dir)
        except ValueError:
            continue
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def filter_rows_for_replaced_sessions(
    rows: Sequence[dict],
    replaced_session_ids: set[str],
    replaced_output_files: set[str],
) -> list[dict]:
    kept: list[dict] = []
    for row in rows:
        session_id = row.get("source_session", "")
        output_file = row.get("output_file", "")
        if session_id and session_id in replaced_session_ids:
            continue
        if output_file in replaced_output_files:
            continue
        kept.append(dict(row))
    return kept


def total_frames_for_fixed_selection(video_info: VideoInfo) -> int:
    total_frames = video_info.total_frames
    if total_frames <= 0 and video_info.duration > 0:
        total_frames = max(1, int(round(video_info.duration * video_info.fps)))
    return max(total_frames, 1)


def analyze_with_cache(
    args: argparse.Namespace,
    input_video: Path,
    video_info: VideoInfo,
    scene_dir: Path,
) -> tuple[list[float], list[float], list[float], list[float], int, int]:
    ensure_python_deps()

    cache_path = cache_path_for(scene_dir)
    cached: tuple[list[float], list[float], list[float], list[float], int, int] | None = None
    if not args.no_cache:
        needs_feature_motion = args.mode == "fixed" and args.fixed_smart
        cached = load_analysis_cache(
            cache_path,
            input_video,
            video_info,
            args.analysis_width,
            QUALITY_MODE,
            require_feature_motion=needs_feature_motion,
        )
        if cached is not None:
            print(f"[cache] reusing analysis cache: {cache_path}")

    if cached is not None:
        return cached

    needs_feature_motion = args.mode == "fixed" and args.fixed_smart
    progress_step = max(10, video_info.total_frames // 100) if video_info.total_frames > 0 else max(
        10, int(round(video_info.fps * 2.0))
    )
    blur_scores, change_scores, quality_scores, feature_motion_scores, analysis_w, analysis_h = analyze_video(
        input_video,
        args.ffmpeg,
        video_info.fps,
        video_info.width,
        video_info.height,
        args.analysis_width,
        progress_phase="analyze",
        progress_total_frames=video_info.total_frames,
        progress_step_frames=progress_step,
        quality_mode=QUALITY_MODE,
        compute_feature_motion=needs_feature_motion,
    )
    if not args.no_cache:
        try:
            save_analysis_cache(
                cache_path, input_video, video_info,
                analysis_w,
                analysis_h,
                blur_scores,
                change_scores,
                quality_scores,
                feature_motion_scores,
                QUALITY_MODE,
                feature_motion_computed=needs_feature_motion,
            )
        except Exception as e:
            print(f"[cache] failed to save cache (non-fatal): {e}")
    return blur_scores, change_scores, quality_scores, feature_motion_scores, analysis_w, analysis_h


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract equirectangular frames via FFmpeg with SfM-oriented representative frame selection."
    )
    parser.add_argument("input_video", help="Input video file path")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=".",
        help="Output root directory (default='.')",
    )

    parser.add_argument("--mode", choices=["fixed", "change"], default="change")
    parser.add_argument("--interval-sec", type=float, default=0.5, help="Fixed mode interval in seconds")
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=0.04,
        help="Change mode threshold (0.0-1.0) based on normalized frame difference",
    )
    parser.add_argument("--min-gap-sec", type=float, default=0.25, help="Minimum gap in seconds")
    parser.add_argument("--max-gap-sec", type=float, default=2.0, help="Maximum safety gap in seconds")
    parser.add_argument(
        "--fixed-smart",
        action="store_true",
        help=(
            "Fixed mode helper: keep the base fixed interval, skip low-change candidates, "
            "and add extra anchors where image/feature motion is high."
        ),
    )
    parser.add_argument(
        "--quick-extract",
        action="store_true",
        help=(
            "Fixed mode shortcut for test runs: extract the requested cadence directly "
            "without change adjustment or quality-based representative selection."
        ),
    )
    parser.add_argument(
        "--fixed-smart-change-threshold",
        type=float,
        default=0.04,
        help="Normalized frame-difference threshold used for fixed-smart high-motion insertion.",
    )
    parser.add_argument(
        "--fixed-smart-feature-threshold",
        type=float,
        default=0.012,
        help="Sparse feature-motion threshold used for fixed-smart high-motion insertion.",
    )
    parser.add_argument(
        "--fixed-smart-max-inserts-per-interval",
        type=int,
        default=2,
        help="Maximum extra anchors inserted inside each fixed interval by --fixed-smart.",
    )
    parser.add_argument(
        "--analysis-pipeline",
        choices=["pair", "legacy"],
        default="pair",
        help=(
            "Frame analysis pipeline. pair uses last-kept-frame residual and candidate pair tracking; "
            "legacy uses the previous whole-video quality score and fixed-smart logic."
        ),
    )
    parser.add_argument(
        "--pair-track-min-count",
        type=int,
        default=36,
        help="Minimum tracked feature count before a kept pair is flagged as weak_match.",
    )
    parser.add_argument(
        "--pair-motion-profile",
        choices=sorted(PAIR_THRESHOLD_PROFILES.keys()),
        default="walk",
        help="Auto threshold profile for pair analysis. walk is the GUI default; drone is for aerial 360 capture.",
    )
    parser.add_argument(
        "--pair-drop-threshold",
        type=float,
        default=-1.0,
        help="Pair residual below this value drops fixed candidates. Negative uses profile-based auto.",
    )
    parser.add_argument(
        "--pair-add-threshold",
        type=float,
        default=-1.0,
        help="Pair residual at or above this value adds novelty candidates. Negative uses profile-based auto.",
    )
    parser.add_argument(
        "--pair-track-min-confidence",
        type=float,
        default=0.25,
        help="Minimum pair tracking confidence before a kept pair is flagged as weak_match.",
    )

    parser.add_argument(
        "--analysis-width",
        type=int,
        default=1920,
        help=(
            "Analysis decode width for change/quality scoring (default=1920). "
            "Higher values give more accurate feature/quality scoring at the cost of "
            "analysis time. Set to 0 or a value >= source width to use full resolution."
        ),
    )
    parser.add_argument(
        "--quality-min-score",
        type=float,
        default=0.35,
        help=(
            "0.0-1.0 SfM-oriented quality score floor. Frames below this score are marked "
            "for review if no better representative is found."
        ),
    )
    parser.add_argument(
        "--quality-min-improvement",
        type=float,
        default=0.08,
        help=(
            "0.0-1.0 quality-score gain required to replace an anchor with a nearby "
            "representative. Computed as candidate quality minus original quality."
        ),
    )
    parser.add_argument(
        "--center-bias",
        type=float,
        default=0.05,
        help="Small preference for frames close to the original extraction anchor.",
    )

    parser.add_argument("--image-ext", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpg-quality", type=int, default=2, help="JPEG quality for ffmpeg -q:v (2 is high quality)")
    parser.add_argument(
        "--filename-prefix",
        default="",
        help="Output filename prefix. Default is input video stem.",
    )
    parser.add_argument(
        "--output-mode",
        choices=["overwrite", "append", "replace-video"],
        default="overwrite",
        help=(
            "How selected_frames.csv and extract_sessions.json are updated. "
            "overwrite=current single-extraction behavior, append=add a new video session, "
            "replace-video=remove prior sessions for the same video then append."
        ),
    )
    parser.add_argument(
        "--allow-duplicate-video",
        action="store_true",
        help="Allow appending a video that already exists in extract_sessions.json.",
    )

    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg executable")
    parser.add_argument("--ffprobe", default="ffprobe", help="Path to ffprobe executable")
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Run probe/analysis/selection and print estimated selected count without image extraction",
    )
    parser.add_argument(
        "--estimate-mode",
        choices=["full", "sampled"],
        default="full",
        help="Estimate mode when --estimate-only is set (full=all frames, sampled=window sampling)",
    )
    parser.add_argument(
        "--sample-segments",
        type=int,
        default=5,
        help="Number of temporal windows for sampled estimate mode",
    )
    parser.add_argument(
        "--sample-segment-sec",
        type=float,
        default=12.0,
        help="Duration (seconds) for each sampled estimate window",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=8.0,
        help="Temporal fps used in sampled estimate windows",
    )
    parser.add_argument(
        "--print-summary-json",
        action="store_true",
        help="Print one-line JSON summary prefixed with SUMMARY_JSON:",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write the analysis cache (extract_cache.npz). Forces full re-analysis.",
    )
    parser.add_argument(
        "--thin-motion-threshold",
        type=float,
        default=0.6,
        help=(
            "Low-change thinning: drop selected frames whose cumulative change_score since the "
            "last kept frame is below this threshold. change_score is mean absolute luma "
            "difference / 255 per analyzed frame, summed between kept frames. Default=0.6. "
            "Set to 0 to disable. 0.3-1.0 is a typical range."
        ),
    )
    parser.add_argument(
        "--no-thin-keep-endpoints",
        dest="thin_keep_endpoints",
        action="store_false",
        default=True,
        help="When thinning, allow the last frame to be dropped too (default keeps endpoints to preserve time coverage).",
    )
    parser.add_argument(
        "--no-extract-thinned",
        dest="extract_thinned",
        action="store_false",
        default=True,
        help=(
            "Skip image extraction for thinned frames (saves disk; default is to extract them so they "
            "can be previewed and unthinned in the review GUI). Thinned rows always remain in CSV "
            "marked decision=drop regardless of this flag."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quality_min_score < 0.0 or args.quality_min_score > 1.0:
        print("Error: --quality-min-score must be between 0 and 1")
        sys.exit(1)
    if args.quality_min_improvement < 0.0 or args.quality_min_improvement > 1.0:
        print("Error: --quality-min-improvement must be between 0 and 1")
        sys.exit(1)
    if args.center_bias < 0.0:
        print("Error: --center-bias must be >= 0")
        sys.exit(1)
    if args.fixed_smart_change_threshold < 0.0 or args.fixed_smart_feature_threshold < 0.0:
        print("Error: fixed-smart thresholds must be >= 0")
        sys.exit(1)
    if args.fixed_smart_max_inserts_per_interval < 0:
        print("Error: --fixed-smart-max-inserts-per-interval must be >= 0")
        sys.exit(1)
    if args.pair_track_min_count < 0:
        print("Error: --pair-track-min-count must be >= 0")
        sys.exit(1)
    try:
        resolve_pair_thresholds(
            args.interval_sec,
            args.pair_motion_profile,
            args.pair_drop_threshold,
            args.pair_add_threshold,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    if args.pair_track_min_confidence < 0.0 or args.pair_track_min_confidence > 1.0:
        print("Error: --pair-track-min-confidence must be between 0 and 1")
        sys.exit(1)
    if args.quick_extract and args.mode != "fixed":
        print("Error: --quick-extract requires --mode fixed")
        sys.exit(1)
    if args.quick_extract and args.fixed_smart:
        print("Error: --quick-extract cannot be combined with --fixed-smart")
        sys.exit(1)

    input_video = Path(args.input_video)
    if not input_video.exists():
        print(f"Error: input video not found: {input_video}")
        sys.exit(1)

    output_root = Path(args.output_dir)
    scene_dir = output_root.resolve()
    images_dir = scene_dir / "images"
    csv_path = scene_dir / "selected_frames.csv"
    report_path = scene_dir / "extract_report.json"

    try:
        ensure_binary(args.ffmpeg, "ffmpeg")
        ensure_binary(args.ffprobe, "ffprobe")
        video_info = probe_video(input_video, args.ffprobe)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Input video: {input_video}")
    print(f"Video: {video_info.width}x{video_info.height} @ {video_info.fps:.3f} fps")
    resolved_prefix = sanitize_filename_prefix(args.filename_prefix)
    if not resolved_prefix:
        resolved_prefix = sanitize_filename_prefix(input_video.stem)
    if not resolved_prefix:
        resolved_prefix = "frame"
    print(f"Filename prefix: {resolved_prefix}")
    video_frame_digits = frame_index_digits(video_info.total_frames)
    current_video_identity = video_identity(input_video)
    manifest = load_manifest(scene_dir)
    manifest_sessions = [
        session for session in manifest.get("sessions", []) if isinstance(session, dict)
    ]
    matching_sessions = [
        session for session in manifest_sessions if session_matches_video(session, current_video_identity)
    ]
    if args.output_mode == "append" and matching_sessions and not args.allow_duplicate_video:
        print(
            "Error: this video already exists in extract_sessions.json. "
            "Use --output-mode replace-video to re-extract it, or --allow-duplicate-video "
            "with a unique --filename-prefix to add it as a separate session."
        )
        sys.exit(1)

    if args.estimate_only and args.mode == "fixed" and not args.fixed_smart:
        total_frames = total_frames_for_fixed_selection(video_info)

        try:
            selected, min_gap_frames = select_fixed(total_frames, video_info.fps, args.interval_sec)
        except Exception as e:
            print(f"Error while selecting frames: {e}")
            sys.exit(1)

        if args.quick_extract:
            analysis_w, analysis_h = 0, 0
            window_frames = 0
            estimate_mode = "quick_extract"
        else:
            analysis_w, analysis_h = scaled_dimensions(video_info.width, video_info.height, args.analysis_width)
            window_frames = representative_window_for_report(selected, total_frames)
            estimate_mode = "fixed_exact"
        summary = build_summary_from_counts(
            args=args,
            video_info=video_info,
            analysis_w=analysis_w,
            analysis_h=analysis_h,
            min_gap_frames=min_gap_frames,
            window_frames=window_frames,
            selected_count=len(selected),
            replaced_count=0,
            fallback_keep_count=0,
            estimate_mode=estimate_mode,
            filename_prefix=resolved_prefix,
            frame_digits=frame_index_digits(video_info.total_frames, selected),
        )
        print(f"Estimated selected frames: {summary['result']['selected_count']}")
        print("Estimated representative replacements: 0")
        print("Estimated low-quality review frames: 0")
        if args.print_summary_json:
            print("SUMMARY_JSON:" + json.dumps(summary, ensure_ascii=False))
        return

    if args.estimate_only and args.mode == "change" and args.estimate_mode == "sampled":
        try:
            ensure_python_deps()
            sampled = estimate_change_sampled(
                video_path=input_video,
                ffmpeg_bin=args.ffmpeg,
                video_info=video_info,
                analysis_width=args.analysis_width,
                threshold=args.change_threshold,
                min_gap_sec=args.min_gap_sec,
                max_gap_sec=args.max_gap_sec,
                sample_segments=args.sample_segments,
                sample_segment_sec=args.sample_segment_sec,
                sample_fps=args.sample_fps,
            )
        except Exception as e:
            print(f"Error during sampled estimate: {e}")
            sys.exit(1)

        summary = build_summary_from_counts(
            args=args,
            video_info=video_info,
            analysis_w=sampled["analysis_w"],
            analysis_h=sampled["analysis_h"],
            min_gap_frames=sampled["min_gap_frames"],
            window_frames=0,
            selected_count=sampled["selected_count"],
            replaced_count=sampled["replaced_count"],
            fallback_keep_count=sampled["fallback_keep_count"],
            estimate_mode="sampled",
            filename_prefix=resolved_prefix,
            frame_digits=video_frame_digits,
            estimate_meta={
                "sampled_segments_requested": sampled["sampled_segments_requested"],
                "sampled_segments_used": sampled["sampled_segments_used"],
                "sampled_duration_sec": sampled["sampled_duration_sec"],
                "sampled_frames": sampled["sampled_frames"],
                "sampled_fps": sampled["analysis_fps"],
                "range_min_count": sampled["range_min_count"],
                "range_max_count": sampled["range_max_count"],
            },
        )

        print(f"Estimated selected frames: {summary['result']['selected_count']}")
        print(f"Estimated representative replacements: {summary['result']['replaced_count']}")
        print(f"Estimated low-quality review frames: {summary['result']['fallback_keep_count']}")
        print(
            "[sample] used segments: "
            f"{sampled['sampled_segments_used']}/{sampled['sampled_segments_requested']} "
            f"(duration={sampled['sampled_duration_sec']:.2f}s, decoded={sampled['sampled_frames']} frames)"
        )
        if args.print_summary_json:
            print("SUMMARY_JSON:" + json.dumps(summary, ensure_ascii=False))
        return

    use_pair_pipeline = (
        args.analysis_pipeline == "pair"
        and not args.quick_extract
        and args.mode == "fixed"
    )
    enriched_rows: list[dict] = []
    window_frames_for_report = 0
    min_gap_frames = 1

    if use_pair_pipeline:
        try:
            ensure_python_deps()
            pair_thresholds = resolve_pair_thresholds(
                args.interval_sec,
                args.pair_motion_profile,
                args.pair_drop_threshold,
                args.pair_add_threshold,
            )
            print(
                "Pair thresholds: "
                f"profile={pair_thresholds.profile} mode={pair_thresholds.mode} "
                f"drop={pair_thresholds.drop:.5f} add={pair_thresholds.add:.5f}"
            )
            (
                enriched_rows,
                analysis_w,
                analysis_h,
                min_gap_frames,
                _pair_max_gap_frames,
                total_frames,
            ) = analyze_pair_selection(
                video_path=input_video,
                ffmpeg_bin=args.ffmpeg,
                video_info=video_info,
                analysis_width=args.analysis_width,
                interval_sec=args.interval_sec,
                fixed_smart=args.fixed_smart,
                min_gap_sec=args.min_gap_sec,
                max_gap_sec=args.max_gap_sec,
                drop_threshold=pair_thresholds.drop,
                add_threshold=pair_thresholds.add,
                threshold_profile=pair_thresholds.profile,
                threshold_mode=pair_thresholds.mode,
                max_inserts_per_interval=args.fixed_smart_max_inserts_per_interval,
                track_min_confidence=args.pair_track_min_confidence,
                track_min_count=args.pair_track_min_count,
                progress_phase="analyze",
            )
        except Exception as e:
            print(f"Error during pair analysis: {e}")
            sys.exit(1)
        print(f"Pair-analyzed frames: {total_frames} ({analysis_w}x{analysis_h})")
        selected = [int(r["final_index"]) for r in enriched_rows if r.get("decision", "keep") != "drop"]
        if not selected:
            print("Error: no frames selected")
            sys.exit(1)
    elif args.quick_extract:
        total_frames = total_frames_for_fixed_selection(video_info)
        blur_scores: list[float] = []
        change_scores: list[float] = []
        quality_scores: list[float] = []
        feature_motion_scores: list[float] = []
        analysis_w, analysis_h = 0, 0
        print("Quick extract: skipping analysis and quality scoring")
    else:
        try:
            blur_scores, change_scores, quality_scores, feature_motion_scores, analysis_w, analysis_h = analyze_with_cache(
                args,
                input_video,
                video_info,
                scene_dir,
            )
        except Exception as e:
            print(f"Error during analysis: {e}")
            sys.exit(1)

        total_frames = len(blur_scores)
        print(f"Analyzed frames: {total_frames} ({analysis_w}x{analysis_h})")

    if not use_pair_pipeline:
        try:
            smart_added_indices: set[int] = set()
            smart_thinned_indices: set[int] = set()
            if args.mode == "fixed":
                selected, min_gap_frames = select_fixed(total_frames, video_info.fps, args.interval_sec)
                if args.fixed_smart:
                    (
                        selected,
                        smart_added_indices,
                        smart_thinned_indices,
                        min_gap_frames,
                        _smart_max_gap_frames,
                    ) = select_fixed_smart(
                        selected,
                        change_scores,
                        feature_motion_scores,
                        video_info.fps,
                        args.min_gap_sec,
                        args.max_gap_sec,
                        change_threshold=args.fixed_smart_change_threshold,
                        feature_motion_threshold=args.fixed_smart_feature_threshold,
                        max_inserts_per_interval=args.fixed_smart_max_inserts_per_interval,
                    )
            else:
                selected, min_gap_frames, _ = select_change(
                    change_scores,
                    video_info.fps,
                    args.change_threshold,
                    args.min_gap_sec,
                    args.max_gap_sec,
                )
        except Exception as e:
            print(f"Error while selecting frames: {e}")
            sys.exit(1)

        if not selected:
            print("Error: no frames selected")
            sys.exit(1)

    if args.quick_extract:
        window_frames_for_report = 0
        enriched_rows = build_quick_extract_rows(selected)
    elif not use_pair_pipeline:
        window_frames_for_report = representative_window_for_report(selected, total_frames)

        rows = select_representative_frames(
            selected_indices=selected,
            quality_scores=quality_scores,
            quality_min_score=args.quality_min_score,
            quality_min_improvement=args.quality_min_improvement,
            center_bias=args.center_bias,
        )

        enriched_rows: list[dict] = []
        for row in rows:
            orig = row["original_index"]
            final = row["final_index"]
            status = row["status"]
            if orig in smart_added_indices:
                status = "smart_added" if status == "ok" else f"smart_added+{status}"
            if orig in smart_thinned_indices:
                status = "thinned" if status == "ok" else f"{status}+thinned"
            enriched_rows.append(
                {
                    **row,
                    "status": status,
                    "change_score_original": change_scores[orig],
                    "change_score_final": change_scores[final],
                    "blur_score_original": blur_scores[orig],
                    "blur_score_final": blur_scores[final],
                    "quality_score_original": row.get("quality_score_original", quality_scores[orig]),
                    "quality_score_final": row.get("quality_score_final", quality_scores[final]),
                    "decision": "drop" if orig in smart_thinned_indices else "keep",
                }
            )

        # 立ち止まり間引き: 累積モーションが閾値未満の連続区間を drop でマーク
        if args.thin_motion_threshold > 0.0 and not args.fixed_smart:
            enriched_rows = thin_stationary(
                enriched_rows,
                change_scores,
                motion_threshold=args.thin_motion_threshold,
                keep_endpoints=args.thin_keep_endpoints,
            )
            thinned_count = sum(
                1 for r in enriched_rows
                if r.get("decision") == "drop" and "thinned" in r.get("status", "")
            )
            kept_count = sum(1 for r in enriched_rows if r.get("decision") != "drop")
            print(
                f"Stationary thinning: dropped {thinned_count}, kept {kept_count} "
                f"(threshold={args.thin_motion_threshold:g})"
            )

    # 抽出対象の決定
    # - 既定: 間引き含めて全部抽出（review GUI で確認・unthin できるように）
    # - --no-extract-thinned: 間引きフレームを抽出しない（容量節約）
    # CSV 上は decision=drop のまま残るので、finalize_in_place が後で削除する
    if args.extract_thinned:
        final_indices = [r["final_index"] for r in enriched_rows]
    else:
        final_indices = [
            r["final_index"] for r in enriched_rows if r.get("decision", "keep") != "drop"
        ]

    summary = build_summary(
        args=args,
        video_info=video_info,
        analysis_w=analysis_w,
        analysis_h=analysis_h,
        selected_rows=enriched_rows,
        min_gap_frames=min_gap_frames,
        window_frames=window_frames_for_report,
        filename_prefix=resolved_prefix,
        frame_digits=frame_index_digits(video_info.total_frames, final_indices),
        estimate_mode="quick_extract" if args.quick_extract else "full",
    )

    if args.estimate_only:
        print(f"Estimated selected frames: {summary['result']['selected_count']}")
        if args.analysis_pipeline == "pair":
            print(f"Estimated pair novelty additions: {summary['result'].get('novelty_added_count', 0)}")
            print(f"Estimated pair redundant drops: {summary['result'].get('redundant_drop_count', 0)}")
            print(f"Estimated pair motion-blur review frames: {summary['result'].get('motion_blur_count', 0)}")
            print(f"Estimated pair low-texture review frames: {summary['result'].get('low_texture_count', 0)}")
            print(f"Estimated pair weak-match review frames: {summary['result'].get('weak_match_count', 0)}")
        else:
            print(f"Estimated representative replacements: {summary['result']['replaced_count']}")
            print(f"Estimated low-quality review frames: {summary['result']['fallback_keep_count']}")
        if args.print_summary_json:
            print("SUMMARY_JSON:" + json.dumps(summary, ensure_ascii=False))
        return

    if not final_indices:
        print("Error: no frames remain after thinning; skipping extraction")
        sys.exit(1)

    session_id = new_session_id()
    output_files = output_files_for_indices(
        final_indices,
        resolved_prefix,
        args.image_ext,
        summary["params"]["frame_number_digits"],
    )

    existing_fieldnames: list[str] = list(SELECTED_CSV_FIELDNAMES)
    existing_rows: list[dict] = []
    active_manifest_sessions = manifest_sessions
    if args.output_mode in {"append", "replace-video"}:
        existing_fieldnames, existing_rows = read_selected_csv(csv_path)

    replaced_session_ids: set[str] = set()
    replaced_output_files: set[str] = set()
    if args.output_mode == "replace-video" and matching_sessions:
        for session in matching_sessions:
            session_id_value = str(session.get("id") or "")
            if session_id_value:
                replaced_session_ids.add(session_id_value)
            for rel in session.get("output_files", []) or []:
                if isinstance(rel, str):
                    replaced_output_files.add(rel)
        removed = remove_session_outputs(scene_dir, sorted(replaced_output_files))
        existing_rows = filter_rows_for_replaced_sessions(
            existing_rows,
            replaced_session_ids,
            replaced_output_files,
        )
        active_manifest_sessions = [
            session
            for session in active_manifest_sessions
            if str(session.get("id") or "") not in replaced_session_ids
        ]
        print(f"Replaced prior sessions for this video: {len(matching_sessions)} session(s), {removed} file(s) removed")

    if args.output_mode in {"append", "replace-video"}:
        collisions = [
            rel
            for rel in output_files
            if (scene_dir / rel).exists() and rel not in replaced_output_files
        ]
        if collisions:
            preview = ", ".join(collisions[:3])
            print(
                f"Error: output files already exist ({len(collisions)}). "
                f"Use a unique --filename-prefix. Example: {preview}"
            )
            sys.exit(1)

    try:
        extracted_indices = extract_selected_frames(
            input_video,
            args.ffmpeg,
            final_indices,
            images_dir,
            args.image_ext,
            args.jpg_quality,
            resolved_prefix,
            summary["params"]["frame_number_digits"],
            allow_partial_tail=args.quick_extract,
        )
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)

    if extracted_indices != final_indices:
        extracted_set = set(extracted_indices)
        enriched_rows = [r for r in enriched_rows if r["final_index"] in extracted_set]
        final_indices = extracted_indices
        output_files = output_files_for_indices(
            final_indices,
            resolved_prefix,
            args.image_ext,
            summary["params"]["frame_number_digits"],
        )
        summary = build_summary(
            args=args,
            video_info=video_info,
            analysis_w=analysis_w,
            analysis_h=analysis_h,
            selected_rows=enriched_rows,
            min_gap_frames=min_gap_frames,
            window_frames=window_frames_for_report,
            filename_prefix=resolved_prefix,
            frame_digits=summary["params"]["frame_number_digits"],
            estimate_mode="quick_extract" if args.quick_extract else "full",
        )

    write_selected_csv(
        enriched_rows,
        csv_path,
        video_info.fps,
        args.image_ext,
        resolved_prefix,
        summary["params"]["frame_number_digits"],
        existing_rows=existing_rows if args.output_mode in {"append", "replace-video"} else [],
        existing_fieldnames=existing_fieldnames,
        session_id=session_id,
        source_video=str(input_video.resolve()),
    )
    session_record = build_session_record(
        session_id=session_id,
        input_video=input_video,
        video_info=video_info_to_dict(video_info),
        mode=args.mode,
        filename_prefix=resolved_prefix,
        image_ext=args.image_ext,
        output_files=output_files,
        selected_count=sum(1 for r in enriched_rows if r.get("decision", "keep") != "drop"),
        dropped_count=sum(1 for r in enriched_rows if r.get("decision", "keep") == "drop"),
    )
    if args.output_mode == "overwrite":
        manifest["sessions"] = [session_record]
    else:
        manifest["sessions"] = [*active_manifest_sessions, session_record]
    save_manifest(scene_dir, manifest)
    write_report(
        report_path,
        args,
        video_info,
        analysis_w,
        analysis_h,
        enriched_rows,
        window_frames_for_report,
        min_gap_frames,
        resolved_prefix,
        summary["params"]["frame_number_digits"],
    )

    replaced_count = sum(1 for r in enriched_rows if "replaced" in r.get("status", ""))
    smart_added_count = sum(1 for r in enriched_rows if "smart_added" in r.get("status", ""))
    novelty_added_count = sum(1 for r in enriched_rows if "novelty_added" in r.get("status", ""))
    redundant_drop_count = sum(1 for r in enriched_rows if "redundant_drop" in r.get("status", ""))
    gap_forced_count = sum(1 for r in enriched_rows if "gap_forced" in r.get("status", ""))
    motion_blur_count = sum(1 for r in enriched_rows if "motion_blur" in r.get("status", ""))
    low_texture_count = sum(1 for r in enriched_rows if "low_texture" in r.get("status", ""))
    weak_match_count = sum(1 for r in enriched_rows if "weak_match" in r.get("status", ""))
    fallback_count = sum(1 for r in enriched_rows if "fallback_keep" in r.get("status", ""))
    thinned_count = sum(
        1 for r in enriched_rows
        if r.get("decision") == "drop" and "thinned" in r.get("status", "")
    )
    kept_count = len(final_indices)

    print(f"Selected frames: {kept_count} (extracted)")
    if smart_added_count > 0:
        print(f"Smart fixed interval additions: {smart_added_count}")
    if novelty_added_count > 0:
        print(f"Pair novelty additions: {novelty_added_count}")
    if redundant_drop_count > 0:
        print(f"Pair redundant drops: {redundant_drop_count}")
    if gap_forced_count > 0:
        print(f"Pair gap-forced keeps: {gap_forced_count}")
    if motion_blur_count > 0:
        print(f"Pair motion-blur review frames: {motion_blur_count}")
    if low_texture_count > 0:
        print(f"Pair low-texture review frames: {low_texture_count}")
    if weak_match_count > 0:
        print(f"Pair weak-match review frames: {weak_match_count}")
    if thinned_count > 0:
        print(f"Thinned (stationary, recorded as drop in CSV): {thinned_count}")
    if args.analysis_pipeline == "legacy":
        print(f"Representative replacements: {replaced_count}")
        print(f"Low-quality review frames: {fallback_count}")
    print(f"Images: {images_dir}")
    print(f"Selection CSV: {csv_path}")
    print(f"Report: {report_path}")
    if args.print_summary_json:
        print("SUMMARY_JSON:" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
