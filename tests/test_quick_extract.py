from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from extract_frames import (
    VideoInfo,
    build_quick_extract_rows,
    build_selected_csv_rows,
    build_summary_from_counts,
    extract_selected_frames,
)


def _args(input_video: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input_video=str(input_video),
        mode="fixed",
        interval_sec=0.8,
        change_threshold=0.04,
        min_gap_sec=0.25,
        max_gap_sec=2.0,
        fixed_smart=False,
        quick_extract=True,
        fixed_smart_change_threshold=0.04,
        fixed_smart_feature_threshold=0.012,
        fixed_smart_max_inserts_per_interval=2,
        analysis_width=1920,
        quality_min_score=0.35,
        quality_min_improvement=0.08,
        center_bias=0.05,
    )


def test_quick_extract_rows_keep_review_status_ok() -> None:
    rows = build_quick_extract_rows([0, 24, 48])

    assert [row["final_index"] for row in rows] == [0, 24, 48]
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["decision"] == "keep" for row in rows)


def test_quick_extract_csv_leaves_uncomputed_scores_blank() -> None:
    rows = build_selected_csv_rows(
        build_quick_extract_rows([0, 24]),
        fps=30.0,
        image_ext="jpg",
        filename_prefix="clip",
        frame_digits=3,
    )

    assert rows[0]["timestamp_sec"] == "0.000000"
    assert rows[1]["timestamp_sec"] == "0.800000"
    for key in (
        "change_score_original",
        "change_score_final",
        "blur_score_original",
        "blur_score_final",
        "sharpness_baseline",
        "sharpness_ratio",
        "quality_score_original",
        "quality_score_final",
    ):
        assert rows[0][key] == ""


def test_quick_extract_summary_marks_quality_mode_skipped(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    summary = build_summary_from_counts(
        args=_args(video),
        video_info=VideoInfo(width=3840, height=1920, fps=30.0, duration=10.0, total_frames=300),
        analysis_w=0,
        analysis_h=0,
        min_gap_frames=24,
        window_frames=0,
        selected_count=14,
        replaced_count=0,
        fallback_keep_count=0,
        estimate_mode="quick_extract",
        filename_prefix="input",
        frame_digits=3,
    )

    assert summary["estimate_mode"] == "quick_extract"
    assert summary["analysis"]["quality_mode"] == "skipped"
    assert summary["params"]["quick_extract"] is True


def test_quick_extract_allows_missing_trailing_frame_outputs(tmp_path: Path, monkeypatch) -> None:
    def fake_run_cmd_with_ffmpeg_progress(cmd: list[str], **_kwargs) -> subprocess.CompletedProcess:
        out_pattern = Path(cmd[-1])
        out_dir = out_pattern.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "00000001.jpg").write_bytes(b"a")
        (out_dir / "00000002.jpg").write_bytes(b"b")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(
        "extract_frames.run_cmd_with_ffmpeg_progress",
        fake_run_cmd_with_ffmpeg_progress,
    )

    extracted = extract_selected_frames(
        video_path=tmp_path / "input.mp4",
        ffmpeg_bin="ffmpeg",
        frame_indices=[0, 10, 20],
        output_dir=tmp_path / "images",
        image_ext="jpg",
        jpg_quality=2,
        filename_prefix="clip",
        frame_digits=2,
        allow_partial_tail=True,
    )

    assert extracted == [0, 10]
    assert (tmp_path / "images" / "clip_00.jpg").read_bytes() == b"a"
    assert (tmp_path / "images" / "clip_10.jpg").read_bytes() == b"b"
    assert not (tmp_path / "images" / "clip_20.jpg").exists()
