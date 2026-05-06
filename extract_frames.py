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
from scene_layout import extract_report_path, selected_frames_path

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
    """Classify candidate-only SfM risks from pair metrics."""
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


def select_fixed(total_frames: int, fps: float, interval_sec: float) -> tuple[list[int], int]:
    if interval_sec <= 0:
        raise ValueError("--interval-sec must be > 0")

    step = max(1, int(round(interval_sec * fps)))
    indices = list(range(0, total_frames, step))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    return indices, step


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
                "status": row["status"],
                "decision": row.get("decision", "keep"),
                "analysis_pipeline": row.get("analysis_pipeline", "pair"),
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
    kept_count = sum(1 for r in selected_rows if r.get("decision", "keep") != "drop")
    dropped_count = sum(1 for r in selected_rows if r.get("decision", "keep") == "drop")
    pipeline = "quick" if getattr(args, "quick_extract", False) else "pair"
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
            "pipeline": pipeline,
            "min_gap_frames": min_gap_frames,
        },
        "params": {
            "interval_sec": args.interval_sec,
            "min_gap_sec": args.min_gap_sec,
            "max_gap_sec": args.max_gap_sec,
            "fixed_smart": args.fixed_smart,
            "quick_extract": getattr(args, "quick_extract", False),
            "fixed_smart_max_inserts_per_interval": args.fixed_smart_max_inserts_per_interval,
            "pair_motion_profile": getattr(args, "pair_motion_profile", "walk"),
            "pair_threshold_mode": pair_thresholds.mode,
            "pair_drop_threshold": getattr(args, "pair_drop_threshold", -1.0),
            "pair_add_threshold": getattr(args, "pair_add_threshold", -1.0),
            "pair_drop_threshold_resolved": pair_thresholds.drop,
            "pair_add_threshold_resolved": pair_thresholds.add,
            "pair_track_min_count": getattr(args, "pair_track_min_count", 0),
            "pair_track_min_confidence": getattr(args, "pair_track_min_confidence", 0.0),
            "filename_prefix": filename_prefix,
            "frame_number_digits": frame_digits,
            "output_mode": getattr(args, "output_mode", "overwrite"),
        },
        "result": {
            "selected_count": kept_count,
            "dropped_count": dropped_count,
            "review_row_count": len(selected_rows),
            "novelty_added_count": sum(1 for r in selected_rows if "novelty_added" in r.get("status", "")),
            "redundant_drop_count": sum(1 for r in selected_rows if "redundant_drop" in r.get("status", "")),
            "gap_forced_count": sum(1 for r in selected_rows if "gap_forced" in r.get("status", "")),
            "motion_blur_count": sum(1 for r in selected_rows if "motion_blur" in r.get("status", "")),
            "low_texture_count": sum(1 for r in selected_rows if "low_texture" in r.get("status", "")),
            "weak_match_count": sum(1 for r in selected_rows if "weak_match" in r.get("status", "")),
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def build_summary_from_counts(
    args: argparse.Namespace,
    video_info: VideoInfo,
    analysis_w: int,
    analysis_h: int,
    min_gap_frames: int,
    selected_count: int,
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
    pipeline = "quick" if getattr(args, "quick_extract", False) else "pair"
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
            "pipeline": pipeline,
            "min_gap_frames": min_gap_frames,
        },
        "params": {
            "interval_sec": args.interval_sec,
            "min_gap_sec": args.min_gap_sec,
            "max_gap_sec": args.max_gap_sec,
            "fixed_smart": args.fixed_smart,
            "quick_extract": getattr(args, "quick_extract", False),
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
            "filename_prefix": filename_prefix,
            "frame_number_digits": frame_digits,
        },
        "result": {
            "selected_count": selected_count,
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
    filename_prefix: str,
    frame_digits: int,
    estimate_mode: str = "full",
) -> dict:
    novelty_added_count = sum(1 for r in selected_rows if "novelty_added" in r.get("status", ""))
    redundant_drop_count = sum(1 for r in selected_rows if "redundant_drop" in r.get("status", ""))
    gap_forced_count = sum(1 for r in selected_rows if "gap_forced" in r.get("status", ""))
    motion_blur_count = sum(1 for r in selected_rows if "motion_blur" in r.get("status", ""))
    low_texture_count = sum(1 for r in selected_rows if "low_texture" in r.get("status", ""))
    weak_match_count = sum(1 for r in selected_rows if "weak_match" in r.get("status", ""))
    kept_count = sum(1 for r in selected_rows if r.get("decision", "keep") != "drop")
    dropped_count = sum(1 for r in selected_rows if r.get("decision", "keep") == "drop")

    summary = build_summary_from_counts(
        args=args,
        video_info=video_info,
        analysis_w=analysis_w,
        analysis_h=analysis_h,
        min_gap_frames=min_gap_frames,
        selected_count=kept_count,
        estimate_mode=estimate_mode,
        filename_prefix=filename_prefix,
        frame_digits=frame_digits,
    )
    summary["result"]["dropped_count"] = dropped_count
    summary["result"]["review_row_count"] = len(selected_rows)
    summary["result"]["novelty_added_count"] = novelty_added_count
    summary["result"]["redundant_drop_count"] = redundant_drop_count
    summary["result"]["gap_forced_count"] = gap_forced_count
    summary["result"]["motion_blur_count"] = motion_blur_count
    summary["result"]["low_texture_count"] = low_texture_count
    summary["result"]["weak_match_count"] = weak_match_count
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract equirectangular frames via FFmpeg with SfM-oriented pair analysis."
    )
    parser.add_argument("input_video", help="Input video file path")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=".",
        help="Output root directory (default='.')",
    )

    parser.set_defaults(mode="fixed")
    parser.add_argument("--interval-sec", type=float, default=0.5, help="Fixed interval in seconds")
    parser.add_argument("--min-gap-sec", type=float, default=0.25, help="Minimum gap in seconds")
    parser.add_argument("--max-gap-sec", type=float, default=2.0, help="Maximum safety gap in seconds")
    parser.add_argument(
        "--fixed-smart",
        action="store_true",
        help=(
            "Enable pair-analysis motion adjustment: drop redundant fixed candidates, "
            "add novelty candidates, and keep max-gap safety frames."
        ),
    )
    parser.add_argument(
        "--quick-extract",
        action="store_true",
        help=(
            "Extract the requested fixed cadence directly without pair analysis or motion adjustment."
        ),
    )
    parser.add_argument(
        "--fixed-smart-max-inserts-per-interval",
        type=int,
        default=2,
        help="Maximum novelty anchors inserted inside each fixed interval by --fixed-smart.",
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
            "Pair-analysis decode width for candidate tracking and sharpness checks. "
            "Yaw/residual monitoring is internally capped to a 1280px gate. "
            "Set to 0 or a value >= source width to use full resolution."
        ),
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
            "How _stechdrive/frames/selected_frames.csv and _stechdrive/frames/extract_sessions.json are updated. "
            "overwrite=current single-extraction behavior, append=add a new video session, "
            "replace-video=remove prior sessions for the same video then append."
        ),
    )
    parser.add_argument(
        "--allow-duplicate-video",
        action="store_true",
        help="Allow appending a video that already exists in _stechdrive/frames/extract_sessions.json.",
    )

    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg executable")
    parser.add_argument("--ffprobe", default="ffprobe", help="Path to ffprobe executable")
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Run probe/selection and print estimated selected count without image extraction",
    )
    parser.add_argument(
        "--print-summary-json",
        action="store_true",
        help="Print one-line JSON summary prefixed with SUMMARY_JSON:",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval_sec <= 0:
        print("Error: --interval-sec must be > 0")
        sys.exit(1)
    if args.min_gap_sec <= 0 or args.max_gap_sec <= 0:
        print("Error: --min-gap-sec and --max-gap-sec must be > 0")
        sys.exit(1)
    if args.max_gap_sec < args.min_gap_sec:
        print("Error: --max-gap-sec must be >= --min-gap-sec")
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
    csv_path = selected_frames_path(scene_dir)
    report_path = extract_report_path(scene_dir)

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
            "Error: this video already exists in _stechdrive/frames/extract_sessions.json. "
            "Use --output-mode replace-video to re-extract it, or --allow-duplicate-video "
            "with a unique --filename-prefix to add it as a separate session."
        )
        sys.exit(1)

    if args.quick_extract:
        total_frames = total_frames_for_fixed_selection(video_info)
        try:
            selected, min_gap_frames = select_fixed(total_frames, video_info.fps, args.interval_sec)
        except Exception as e:
            print(f"Error while selecting frames: {e}")
            sys.exit(1)
        enriched_rows = build_quick_extract_rows(selected)
        analysis_w, analysis_h = 0, 0
        print("Quick extract: skipping analysis and motion adjustment")
    else:
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

    kept_rows = [r for r in enriched_rows if r.get("decision", "keep") != "drop"]
    if not kept_rows:
        print("Error: no frames selected")
        sys.exit(1)

    final_indices = [int(r["final_index"]) for r in enriched_rows]
    frame_digits = frame_index_digits(video_info.total_frames, final_indices)
    summary = build_summary(
        args=args,
        video_info=video_info,
        analysis_w=analysis_w,
        analysis_h=analysis_h,
        selected_rows=enriched_rows,
        min_gap_frames=min_gap_frames,
        filename_prefix=resolved_prefix,
        frame_digits=frame_digits,
        estimate_mode="quick_extract" if args.quick_extract else "full",
    )

    if args.estimate_only:
        print(f"Estimated selected frames: {summary['result']['selected_count']}")
        if not args.quick_extract:
            print(f"Estimated pair novelty additions: {summary['result'].get('novelty_added_count', 0)}")
            print(f"Estimated pair redundant drops: {summary['result'].get('redundant_drop_count', 0)}")
            print(f"Estimated pair motion-blur review frames: {summary['result'].get('motion_blur_count', 0)}")
            print(f"Estimated pair low-texture review frames: {summary['result'].get('low_texture_count', 0)}")
            print(f"Estimated pair weak-match review frames: {summary['result'].get('weak_match_count', 0)}")
        if args.print_summary_json:
            print("SUMMARY_JSON:" + json.dumps(summary, ensure_ascii=False))
        return

    if not final_indices:
        print("Error: no frames selected; skipping extraction")
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
        enriched_rows = [r for r in enriched_rows if int(r["final_index"]) in extracted_set]
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
        min_gap_frames,
        resolved_prefix,
        summary["params"]["frame_number_digits"],
    )

    novelty_added_count = sum(1 for r in enriched_rows if "novelty_added" in r.get("status", ""))
    redundant_drop_count = sum(1 for r in enriched_rows if "redundant_drop" in r.get("status", ""))
    gap_forced_count = sum(1 for r in enriched_rows if "gap_forced" in r.get("status", ""))
    motion_blur_count = sum(1 for r in enriched_rows if "motion_blur" in r.get("status", ""))
    low_texture_count = sum(1 for r in enriched_rows if "low_texture" in r.get("status", ""))
    weak_match_count = sum(1 for r in enriched_rows if "weak_match" in r.get("status", ""))
    dropped_count = sum(1 for r in enriched_rows if r.get("decision", "keep") == "drop")

    print(f"Selected frames: {len(final_indices)} (extracted)")
    if dropped_count > 0:
        print(f"Dropped review rows: {dropped_count}")
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
    print(f"Images: {images_dir}")
    print(f"Selection CSV: {csv_path}")
    print(f"Report: {report_path}")
    if args.print_summary_json:
        print("SUMMARY_JSON:" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
