from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from extract_frames import (
    PairTrackMetrics,
    VideoInfo,
    analyze_pair_selection,
    assess_pair_frame_risk,
    compute_pair_metrics,
    resolve_pair_thresholds,
)


def _fake_ffmpeg_wrapper(tmp_path: Path, script_name: str, lines: list[str]) -> Path:
    fake_script = tmp_path / script_name
    fake_script.write_text("\n".join(lines), encoding="utf-8")
    if os.name == "nt":
        wrapper = tmp_path / f"{fake_script.stem}.cmd"
        wrapper.write_text(f'@echo off\n"{sys.executable}" "{fake_script}" %*\n', encoding="utf-8")
    else:
        wrapper = tmp_path / f"{fake_script.stem}.sh"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake_script}" "$@"\n', encoding="utf-8")
        wrapper.chmod(0o755)
    return wrapper


def test_compute_pair_metrics_compensates_horizontal_yaw() -> None:
    base = np.tile(np.arange(16, dtype=np.uint8), (8, 1)) * 8
    shifted = np.roll(base, 3, axis=1)
    weights = np.ones((8, 1), dtype=np.float32)

    raw = np.mean(np.abs(base.astype(np.int16) - shifted.astype(np.int16))) / 255.0
    metrics = compute_pair_metrics(base, shifted, weights)

    assert metrics.raw_change > 0
    assert metrics.residual < raw
    assert abs(metrics.yaw_shift_px) == 3


def test_resolve_pair_thresholds_scales_and_clamps_walking_profile() -> None:
    base = resolve_pair_thresholds(1.0, "walk")
    shorter = resolve_pair_thresholds(0.8, "walk")
    tiny = resolve_pair_thresholds(0.05, "walk")
    huge = resolve_pair_thresholds(10.0, "walk")

    assert base.profile == "walk"
    assert base.mode == "auto"
    assert base.drop == pytest.approx(0.035)
    assert base.add == pytest.approx(0.090)
    assert shorter.drop == pytest.approx(0.035 * (0.8 ** 0.5))
    assert shorter.add == pytest.approx(0.090 * (0.8 ** 0.5))
    assert tiny.drop == pytest.approx(0.035 * (0.35 ** 0.5))
    assert huge.add == pytest.approx(0.090 * (2.5 ** 0.5))


def test_resolve_pair_thresholds_drone_profile_uses_lower_aerial_residuals() -> None:
    walking = resolve_pair_thresholds(2.0, "walk")
    drone = resolve_pair_thresholds(2.0, "drone")

    assert drone.profile == "drone"
    assert drone.drop == pytest.approx(0.025)
    assert drone.add == pytest.approx(0.065)
    assert drone.drop < walking.drop
    assert drone.add < walking.add


def test_resolve_pair_thresholds_rejects_add_not_greater_than_drop() -> None:
    with pytest.raises(ValueError):
        resolve_pair_thresholds(1.0, "walk", drop_threshold=0.08, add_threshold=0.08)


def test_assess_pair_frame_risk_drops_clear_ratio_blur_even_with_strong_tracking() -> None:
    risk = assess_pair_frame_risk(
        blur_score=60.0,
        sharpness_baseline=100.0,
        track=PairTrackMetrics(track_count=240, coverage=1.0, confidence=0.95, median_residual_motion=0.0),
        track_min_confidence=0.25,
        track_min_count=36,
    )

    assert risk.sharpness_ratio == pytest.approx(0.60)
    assert risk.motion_blur is True
    assert risk.borderline_blur is False
    assert risk.low_texture is False


def test_assess_pair_frame_risk_keeps_borderline_blur_for_review() -> None:
    risk = assess_pair_frame_risk(
        blur_score=75.0,
        sharpness_baseline=100.0,
        track=PairTrackMetrics(track_count=240, coverage=1.0, confidence=0.95, median_residual_motion=0.0),
        track_min_confidence=0.25,
        track_min_count=36,
    )

    assert risk.sharpness_ratio == pytest.approx(0.75)
    assert risk.motion_blur is False
    assert risk.borderline_blur is True
    assert risk.low_texture is False


def test_analyze_pair_selection_marks_redundant_and_gap_forced(tmp_path: Path) -> None:
    wrapper = _fake_ffmpeg_wrapper(
        tmp_path,
        "fake_ffmpeg_pair.py",
        [
            "import sys",
            "frames = [bytes([40]) * 8 for _ in range(5)]",
            "for frame in frames:",
            "    sys.stdout.buffer.write(frame)",
            "    sys.stdout.flush()",
        ],
    )

    rows, out_w, out_h, min_gap, max_gap, decoded = analyze_pair_selection(
        video_path=tmp_path / "dummy.mp4",
        ffmpeg_bin=str(wrapper),
        video_info=VideoInfo(width=4, height=2, fps=1.0, duration=5.0, total_frames=5),
        analysis_width=4,
        interval_sec=2.0,
        fixed_smart=True,
        min_gap_sec=1.0,
        max_gap_sec=3.0,
        drop_threshold=0.03,
        add_threshold=0.08,
        threshold_profile="walk",
        threshold_mode="manual",
        max_inserts_per_interval=2,
        track_min_confidence=0.25,
        track_min_count=36,
        progress_phase="",
    )

    statuses = [row["status"] for row in rows]
    assert (out_w, out_h, min_gap, max_gap, decoded) == (4, 2, 1, 3, 5)
    assert any("redundant_drop" in status for status in statuses)
    assert any("gap_forced" in status for status in statuses)
    assert any(row["decision"] == "drop" for row in rows)


def test_analyze_pair_selection_marks_motion_blur_candidate_for_review(tmp_path: Path) -> None:
    wrapper = _fake_ffmpeg_wrapper(
        tmp_path,
        "fake_ffmpeg_blur_pair.py",
        [
            "import sys",
            "w, h = 64, 32",
            "sharp = bytes((255 if ((x // 2 + y // 2) % 2) else 0) for y in range(h) for x in range(w))",
            "blur = bytes([128]) * (w * h)",
            "for frame in [sharp, blur, sharp, sharp]:",
            "    sys.stdout.buffer.write(frame)",
            "    sys.stdout.flush()",
        ],
    )

    rows, *_ = analyze_pair_selection(
        video_path=tmp_path / "dummy.mp4",
        ffmpeg_bin=str(wrapper),
        video_info=VideoInfo(width=64, height=32, fps=1.0, duration=4.0, total_frames=4),
        analysis_width=64,
        interval_sec=1.0,
        fixed_smart=True,
        min_gap_sec=1.0,
        max_gap_sec=3.0,
        drop_threshold=0.03,
        add_threshold=0.08,
        threshold_profile="walk",
        threshold_mode="manual",
        max_inserts_per_interval=2,
        track_min_confidence=0.25,
        track_min_count=36,
        progress_phase="",
    )

    blur_rows = [row for row in rows if "motion_blur" in row["status"]]
    assert blur_rows
    assert blur_rows[0]["decision"] == "drop"
    assert blur_rows[0]["review_required"] == "1"
    assert float(blur_rows[0]["sharpness_ratio"]) <= 0.35


def test_analyze_pair_selection_replaces_blur_with_sharp_candidate(tmp_path: Path) -> None:
    wrapper = _fake_ffmpeg_wrapper(
        tmp_path,
        "fake_ffmpeg_blur_replacement.py",
        [
            "import sys",
            "w, h = 64, 32",
            "def frame_a():",
            "    return bytes((255 if ((x // 2 + y // 2) % 2) else 0) for y in range(h) for x in range(w))",
            "def frame_b():",
            "    return bytes((255 if (((x * 3 + y * 5) // 2) % 2) else 0) for y in range(h) for x in range(w))",
            "def frame_c():",
            "    return bytes((255 if ((x // 8 + y // 8) % 2) else 0) for y in range(h) for x in range(w))",
            "blur = bytes([128]) * (w * h)",
            "frames = [frame_a(), frame_a(), blur, frame_b(), frame_c(), frame_a()]",
            "for frame in frames:",
            "    sys.stdout.buffer.write(frame)",
            "    sys.stdout.flush()",
        ],
    )

    rows, *_ = analyze_pair_selection(
        video_path=tmp_path / "dummy.mp4",
        ffmpeg_bin=str(wrapper),
        video_info=VideoInfo(width=64, height=32, fps=1.0, duration=6.0, total_frames=6),
        analysis_width=64,
        interval_sec=2.0,
        fixed_smart=True,
        min_gap_sec=1.0,
        max_gap_sec=4.0,
        drop_threshold=0.03,
        add_threshold=0.80,
        threshold_profile="walk",
        threshold_mode="manual",
        max_inserts_per_interval=0,
        track_min_confidence=0.0,
        track_min_count=0,
        progress_phase="",
    )

    blur_rows = [row for row in rows if "motion_blur" in row["status"]]
    replacement_rows = [row for row in rows if "blur_replacement" in row["status"]]

    assert blur_rows
    assert blur_rows[0]["final_index"] == 2
    assert blur_rows[0]["decision"] == "drop"
    assert replacement_rows
    assert replacement_rows[0]["final_index"] == 3
    assert replacement_rows[0]["decision"] == "keep"
    assert replacement_rows[0]["prev_kept_index"] == 0

    post_replacement_rows = [row for row in rows if row["final_index"] == 5]
    assert post_replacement_rows
    assert post_replacement_rows[0]["prev_kept_index"] == 3
