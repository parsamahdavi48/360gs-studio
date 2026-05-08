from __future__ import annotations

from pathlib import Path

from core.extract_sessions import (
    build_session_record,
    matching_video_sessions,
    sanitize_filename_prefix,
    save_manifest,
)


def test_sanitize_filename_prefix_matches_extractor_rules() -> None:
    assert sanitize_filename_prefix("  GX 01/テスト  ") == "GX_01"
    assert sanitize_filename_prefix("...") == ""


def test_matching_video_sessions_uses_path_size_and_mtime(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    scene = tmp_path / "scene"
    session = build_session_record(
        session_id="session-1",
        input_video=video,
        video_info={"width": 10, "height": 10, "fps": 30, "duration_sec": 1, "total_frames": 30},
        mode="fixed",
        filename_prefix="clip",
        image_ext="jpg",
        output_files=["images/clip_001.jpg"],
        selected_count=1,
        dropped_count=0,
    )
    save_manifest(scene, {"version": 1, "sessions": [session]})

    assert [s["id"] for s in matching_video_sessions(scene, video)] == ["session-1"]

    video.write_bytes(b"changed")

    assert matching_video_sessions(scene, video) == []
