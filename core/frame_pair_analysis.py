from __future__ import annotations

import math
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from core.video_info import VideoInfo

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
    borderline_blur: bool
    low_texture: bool
    weak_match: bool


@dataclass
class PairCandidateFrame:
    idx: int
    frame: np.ndarray
    gate_frame: np.ndarray
    blur_score: float


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
PAIR_MOTION_BLUR_DROP_RATIO = 0.60
PAIR_MOTION_BLUR_REVIEW_RATIO = 0.80
PAIR_GATE_WIDTH_DEFAULT = 1280


_WALK_CLOSE_PROFILE = PairThresholdProfile(
    reference_interval_sec=1.0,
    min_interval_sec=0.35,
    max_interval_sec=2.5,
    base_drop_threshold=0.035,
    base_add_threshold=0.090,
)


PAIR_THRESHOLD_PROFILES: dict[str, PairThresholdProfile] = {
    # Walking footage with nearby subjects. This keeps the previous "walk"
    # threshold behavior while exposing it as an explicit close-range profile.
    "walk_close": _WALK_CLOSE_PROFILE,
    # Default walking profile: wider cadence than close-range capture, with
    # slightly conservative additions so ordinary facility footage does not
    # explode into unnecessary near-duplicates.
    "walk_standard": PairThresholdProfile(
        reference_interval_sec=1.5,
        min_interval_sec=0.8,
        max_interval_sec=4.0,
        base_drop_threshold=0.035,
        base_add_threshold=0.095,
    ),
    # Broad walking scenes such as parks and plazas usually have weaker
    # residual parallax for the same walking speed, so thresholds are lower.
    "walk_wide": PairThresholdProfile(
        reference_interval_sec=3.0,
        min_interval_sec=1.5,
        max_interval_sec=7.0,
        base_drop_threshold=0.030,
        base_add_threshold=0.075,
    ),
    # Aerial 360 capture tends to have the weakest residual parallax because
    # most features are far away, so the reference residual thresholds are
    # lower and the practical max interval is wider.
    "drone_distant": PairThresholdProfile(
        reference_interval_sec=3.0,
        min_interval_sec=1.5,
        max_interval_sec=8.0,
        base_drop_threshold=0.025,
        base_add_threshold=0.065,
    ),
    # Legacy CLI profiles kept for compatibility with existing commands.
    "walk": _WALK_CLOSE_PROFILE,
    "drone": PairThresholdProfile(
        reference_interval_sec=2.0,
        min_interval_sec=0.8,
        max_interval_sec=5.0,
        base_drop_threshold=0.025,
        base_add_threshold=0.065,
    ),
}

PAIR_THRESHOLD_PROFILE_CHOICES = sorted(PAIR_THRESHOLD_PROFILES)


def ensure_python_deps() -> None:
    missing = []
    if cv2 is None:
        missing.append(f"opencv-python (cv2 import failed: {_CV2_IMPORT_ERROR})")
    if np is None:
        missing.append(f"numpy (import failed: {_NP_IMPORT_ERROR})")
    if missing:
        raise RuntimeError("Missing required Python modules: " + "; ".join(missing))


def _drain_binary_pipe(pipe: BinaryIO, chunks: list[bytes]) -> None:
    try:
        while True:
            data = pipe.read(8192)
            if not data:
                break
            chunks.append(data)
    except OSError as e:
        chunks.append(f"\n[stderr read error: {e}]".encode("utf-8", errors="replace"))


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
        weak_for_blur = track.track_count < max(12, min_count // 2) or track.confidence < max(
            0.15, min_confidence * 0.75
        )
        low_texture_track = (
            track.track_count < max(12, min_count)
            or track.coverage < 0.12
            or track.confidence < max(0.12, min_confidence * 0.5)
        )

    has_sharpness_baseline = (
        sharpness_baseline is not None
        and sharpness_baseline >= PAIR_MOTION_BLUR_BASELINE_MIN
        and sharpness_ratio is not None
    )
    severe_sharpness_drop = has_sharpness_baseline and sharpness_ratio <= min(0.12, PAIR_MOTION_BLUR_RATIO)
    weak_track_blur = (
        has_sharpness_baseline
        and sharpness_ratio <= PAIR_MOTION_BLUR_RATIO
        and (weak_for_blur or severe_sharpness_drop)
    )
    motion_blur = bool(has_sharpness_baseline and sharpness_ratio <= PAIR_MOTION_BLUR_DROP_RATIO) or bool(
        weak_track_blur
    )
    borderline_blur = bool(
        has_sharpness_baseline
        and not motion_blur
        and sharpness_ratio <= PAIR_MOTION_BLUR_REVIEW_RATIO
    )
    low_texture = (
        not motion_blur
        and not borderline_blur
        and blur_score <= PAIR_LOW_TEXTURE_SHARPNESS
        and (sharpness_baseline is None or sharpness_baseline <= PAIR_MOTION_BLUR_BASELINE_MIN)
        and (track is None or low_texture_track)
    )

    return PairFrameRisk(
        blur_score=blur_score,
        sharpness_baseline=sharpness_baseline,
        sharpness_ratio=sharpness_ratio,
        motion_blur=motion_blur,
        borderline_blur=borderline_blur,
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
    risk_flags = [
        token for token in ("motion_blur", "borderline_blur", "low_texture", "weak_match") if token in status
    ]
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
    novelty_inserts_since_anchor = 0
    kept_sharpness: list[float] = []
    last_keep_frame: np.ndarray | None = None
    last_keep_gate_frame: np.ndarray | None = None
    last_keep_idx: int | None = None
    next_due_idx: int | None = None
    last_frame: np.ndarray | None = None
    last_frame_idx = -1
    last_progress_report = 0
    replacement_origin_row: dict | None = None
    replacement_candidates: list[PairCandidateFrame] = []

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

    def schedule_next_due(anchor_idx: int) -> None:
        nonlocal next_due_idx
        next_due_idx = int(anchor_idx) + interval_frames

    def remember_replacement_candidate(
        idx: int,
        frame: np.ndarray,
        gate_frame: np.ndarray,
        blur_score: float | None = None,
    ) -> None:
        replacement_candidates.append(
            PairCandidateFrame(
                idx=idx,
                frame=frame.copy(),
                gate_frame=gate_frame.copy(),
                blur_score=compute_pair_blur_score(frame) if blur_score is None else float(blur_score),
            )
        )

    def adopt_keep(row: dict, frame: np.ndarray, gate_frame: np.ndarray, *, reset_novelty: bool) -> None:
        nonlocal last_keep_frame, last_keep_gate_frame, last_keep_idx, novelty_inserts_since_anchor
        if row.get("decision") == "drop":
            return
        status = row.get("status", "")
        if "motion_blur" not in status and "borderline_blur" not in status:
            row_blur = row.get("blur_score_final")
            if row_blur not in (None, ""):
                kept_sharpness.append(float(row_blur))
                del kept_sharpness[:-PAIR_SHARPNESS_HISTORY]
        last_keep_frame = frame.copy()
        last_keep_gate_frame = gate_frame.copy()
        last_keep_idx = int(row["final_index"])
        schedule_next_due(last_keep_idx)
        if reset_novelty:
            novelty_inserts_since_anchor = 0

    def evaluate_row(
        *,
        idx: int,
        frame: np.ndarray,
        status_tokens: Sequence[str],
        decision: str,
        reason: str,
        metrics: PairMetrics,
        allow_motion_blur_drop: bool,
        blur_score: float | None = None,
    ) -> dict:
        track_shift_px = int(round(metrics.yaw_shift_px * (frame.shape[1] / float(max(1, gate_w)))))
        track = (
            compute_pair_track_metrics(last_keep_frame, frame, track_shift_px) if last_keep_frame is not None else None
        )
        tokens = list(status_tokens)
        risk = assess_pair_frame_risk(
            blur_score=compute_pair_blur_score(frame) if blur_score is None else float(blur_score),
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
        elif risk.borderline_blur:
            tokens.append("borderline_blur")
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

    def finalize_replacement_search() -> None:
        nonlocal replacement_origin_row, replacement_candidates
        if replacement_origin_row is None:
            return

        append_row(replacement_origin_row)
        accepted: tuple[dict, PairCandidateFrame] | None = None
        assert last_keep_idx is not None
        assert last_keep_gate_frame is not None

        for candidate in sorted(replacement_candidates, key=lambda item: item.blur_score, reverse=True):
            metrics = compute_pair_metrics(last_keep_gate_frame, candidate.gate_frame, weights)
            gap = candidate.idx - last_keep_idx
            if metrics.residual < thresholds.drop and gap < max_gap_frames:
                continue
            row = evaluate_row(
                idx=candidate.idx,
                frame=candidate.frame,
                status_tokens=["blur_replacement"],
                decision="keep",
                reason="blur_replacement",
                metrics=metrics,
                allow_motion_blur_drop=True,
                blur_score=candidate.blur_score,
            )
            status = row.get("status", "")
            if row.get("decision") != "drop" and "motion_blur" not in status and "borderline_blur" not in status:
                accepted = (row, candidate)
                break

        if accepted is not None:
            row, candidate = accepted
            append_row(row)
            adopt_keep(row, candidate.frame, candidate.gate_frame, reset_novelty=True)
        elif replacement_candidates:
            candidate = replacement_candidates[-1]
            metrics = compute_pair_metrics(last_keep_gate_frame, candidate.gate_frame, weights)
            row = evaluate_row(
                idx=candidate.idx,
                frame=candidate.frame,
                status_tokens=["gap_forced"],
                decision="keep",
                reason="gap_forced",
                metrics=metrics,
                allow_motion_blur_drop=False,
                blur_score=candidate.blur_score,
            )
            append_row(row)
            adopt_keep(row, candidate.frame, candidate.gate_frame, reset_novelty=True)

        replacement_origin_row = None
        replacement_candidates = []

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
                schedule_next_due(idx)
                emit_progress(idx + 1, force=True)
                continue

            assert last_keep_idx is not None
            gap = idx - last_keep_idx

            if replacement_origin_row is not None:
                gate_frame = pair_gate_frame(frame, gate_w, gate_h)
                remember_replacement_candidate(idx, frame, gate_frame)
                if gap >= max_gap_frames:
                    finalize_replacement_search()
                emit_progress(idx + 1)
                continue

            if gap < min_gap_frames:
                emit_progress(idx + 1)
                continue

            base_due = next_due_idx is not None and idx >= next_due_idx
            max_due = gap >= max_gap_frames
            assert last_keep_gate_frame is not None
            gate_frame = pair_gate_frame(frame, gate_w, gate_h)
            metrics = compute_pair_metrics(last_keep_gate_frame, gate_frame, weights)
            high_novelty = thresholds.add <= 0.0 or metrics.residual >= thresholds.add
            low_redundancy = metrics.residual < thresholds.drop

            row: dict | None = None
            keep_current = False
            status_tokens: list[str] = []
            reason = "ok"
            decision = "keep"

            if fixed_smart and not base_due and high_novelty and novelty_inserts_since_anchor < max_inserts:
                keep_current = True
                status_tokens.append("novelty_added")
                reason = "novelty_added"
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
                if (
                    fixed_smart
                    and row.get("decision") == "drop"
                    and "motion_blur" in row.get("status", "")
                    and not max_due
                ):
                    replacement_origin_row = row
                    replacement_candidates = []
                    schedule_next_due(idx)
                    emit_progress(idx + 1)
                    continue

                append_row(row)
                if decision == "drop" and row.get("selection_reason") == "redundant_drop":
                    schedule_next_due(idx)
                if keep_current and row.get("decision") != "drop":
                    if row.get("selection_reason") == "novelty_added":
                        novelty_inserts_since_anchor += 1
                        reset_novelty = False
                    else:
                        reset_novelty = True
                    adopt_keep(row, frame, gate_frame, reset_novelty=reset_novelty)

            emit_progress(idx + 1)
    finally:
        ret = proc.wait()
        stderr_thread.join()
        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")

    if ret != 0:
        raise RuntimeError(f"ffmpeg pair analysis failed: {stderr_text.strip()}")
    if last_frame_idx < 0 or last_frame is None:
        raise RuntimeError("No frames decoded during pair analysis")

    if replacement_origin_row is not None:
        finalize_replacement_search()

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
