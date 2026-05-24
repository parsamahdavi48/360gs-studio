from __future__ import annotations

from pathlib import Path

import pytest

from core.extract_frames import ExtractFramesOptions
from core.frame_job_runner import _run_extract_video
from core.frame_job_spec import (
    JOB_KIND_APPLY_FRAME_DECISIONS,
    JOB_KIND_EXTRACT_VIDEO,
    JOB_KIND_IMPORT_IMAGE_SEQUENCE,
    apply_frame_decisions_job,
    extract_video_job,
    import_image_sequence_job,
    load_frame_job,
    write_frame_job,
)


def test_extract_video_frame_job_round_trips(tmp_path: Path) -> None:
    payload = extract_video_job(
        input_video=tmp_path / "clip.mp4",
        scene_dir=tmp_path / "scene",
        image_ext="jpg",
        jpg_quality=2,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        output_mode="append",
        filename_prefix="clip",
        interval_sec=1.5,
        quick_extract=False,
        pair_motion_profile="walk_standard",
        analysis_width=1920,
        fixed_smart=True,
        min_gap_sec=0.5,
        max_gap_sec=3.0,
    )

    path = write_frame_job(tmp_path / "extract.json", payload)
    loaded = load_frame_job(path, expected_kind=JOB_KIND_EXTRACT_VIDEO)

    assert loaded["kind"] == JOB_KIND_EXTRACT_VIDEO
    assert loaded["input_video"] == str(tmp_path / "clip.mp4")
    assert loaded["schema_version"] == 1
    assert loaded["analysis_width"] == 1920
    assert loaded["fixed_smart"] is True


def test_extract_video_frame_job_runner_passes_typed_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = extract_video_job(
        input_video=tmp_path / "clip.mp4",
        scene_dir=tmp_path / "scene",
        image_ext="png",
        jpg_quality=2,
        ffmpeg="custom-ffmpeg",
        ffprobe="custom-ffprobe",
        output_mode="append",
        filename_prefix="clip",
        interval_sec=1.5,
        quick_extract=False,
        pair_motion_profile="walk_standard",
        analysis_width=0,
        fixed_smart=True,
        min_gap_sec=0.5,
        max_gap_sec=3.0,
        allow_duplicate_video=True,
        estimate_only=True,
        print_summary_json=True,
    )
    captured: dict[str, ExtractFramesOptions] = {}

    def fake_run_extract_frames(options: ExtractFramesOptions) -> int:
        captured["options"] = options
        return 0

    monkeypatch.setattr("core.extract_frames.run_extract_frames", fake_run_extract_frames)

    _run_extract_video(payload)

    options = captured["options"]
    assert isinstance(options, ExtractFramesOptions)
    assert options.input_video == tmp_path / "clip.mp4"
    assert options.output_dir == tmp_path / "scene"
    assert options.image_ext == "png"
    assert options.ffmpeg == "custom-ffmpeg"
    assert options.ffprobe == "custom-ffprobe"
    assert options.output_mode == "append"
    assert options.filename_prefix == "clip"
    assert options.interval_sec == 1.5
    assert options.pair_motion_profile == "walk_standard"
    assert options.analysis_width == 0
    assert options.fixed_smart is True
    assert options.min_gap_sec == 0.5
    assert options.max_gap_sec == 3.0
    assert options.allow_duplicate_video is True
    assert options.estimate_only is True
    assert options.print_summary_json is True


def test_import_image_sequence_frame_job_round_trips(tmp_path: Path) -> None:
    payload = import_image_sequence_job(
        source_dir=tmp_path / "frames",
        scene_dir=tmp_path / "scene",
        prefix="take",
        recursive=False,
    )

    path = write_frame_job(tmp_path / "import.json", payload)
    loaded = load_frame_job(path, expected_kind=JOB_KIND_IMPORT_IMAGE_SEQUENCE)

    assert loaded["kind"] == JOB_KIND_IMPORT_IMAGE_SEQUENCE
    assert loaded["source_dir"] == str(tmp_path / "frames")
    assert loaded["recursive"] is False


def test_apply_frame_decisions_frame_job_round_trips(tmp_path: Path) -> None:
    payload = apply_frame_decisions_job(
        scene_dir=tmp_path / "scene",
        finalize_in_place=True,
        renumber_kept_images=True,
    )

    path = write_frame_job(tmp_path / "apply.json", payload)
    loaded = load_frame_job(path, expected_kind=JOB_KIND_APPLY_FRAME_DECISIONS)

    assert loaded["kind"] == JOB_KIND_APPLY_FRAME_DECISIONS
    assert loaded["csv"] == "selected_frames.csv"
    assert loaded["output"] == "metashape_images"
    assert loaded["renumber_kept_images"] is True


def test_extract_video_frame_job_rejects_invalid_ranges(tmp_path: Path) -> None:
    payload = extract_video_job(
        input_video=tmp_path / "clip.mp4",
        scene_dir=tmp_path / "scene",
        image_ext="jpg",
        jpg_quality=32,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        output_mode="append",
        filename_prefix="clip",
        interval_sec=1.5,
        quick_extract=False,
        pair_motion_profile="walk_standard",
        analysis_width=1920,
        fixed_smart=True,
        min_gap_sec=0.5,
        max_gap_sec=3.0,
    )

    with pytest.raises(ValueError, match="jpg_quality"):
        write_frame_job(tmp_path / "bad_quality.json", payload)

    payload["jpg_quality"] = 2
    payload["max_gap_sec"] = 0.25
    with pytest.raises(ValueError, match="max_gap_sec"):
        write_frame_job(tmp_path / "bad_gap.json", payload)


def test_extract_video_frame_job_rejects_incompatible_modes(tmp_path: Path) -> None:
    payload = extract_video_job(
        input_video=tmp_path / "clip.mp4",
        scene_dir=tmp_path / "scene",
        image_ext="jpg",
        jpg_quality=2,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        output_mode="append",
        filename_prefix="clip",
        interval_sec=1.5,
        quick_extract=True,
        pair_motion_profile="walk_standard",
        analysis_width=0,
        fixed_smart=True,
        min_gap_sec=0.5,
        max_gap_sec=3.0,
    )

    with pytest.raises(ValueError, match="quick_extract"):
        write_frame_job(tmp_path / "bad_mode.json", payload)


def test_apply_frame_decisions_rejects_renumber_without_finalize(tmp_path: Path) -> None:
    payload = apply_frame_decisions_job(
        scene_dir=tmp_path / "scene",
        finalize_in_place=False,
        renumber_kept_images=True,
    )

    with pytest.raises(ValueError, match="renumber_kept_images"):
        write_frame_job(tmp_path / "bad_apply.json", payload)
