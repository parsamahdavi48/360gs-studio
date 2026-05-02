from __future__ import annotations

import csv
from pathlib import Path

from extract_frames import frame_filename, frame_index_digits, write_selected_csv


def test_frame_index_digits_uses_total_frame_count() -> None:
    assert frame_index_digits(10) == 1
    assert frame_index_digits(120) == 3
    assert frame_index_digits(7200) == 4
    assert frame_index_digits(108000) == 6


def test_frame_index_digits_keeps_selected_indices_sortable_when_total_is_unknown() -> None:
    assert frame_index_digits(0, [0, 120, 2400]) == 4
    assert frame_index_digits(0, []) == 6


def test_frame_filename_uses_requested_width() -> None:
    assert frame_filename("clip", 120, "jpg", 4) == "clip_0120.jpg"
    assert frame_filename("clip", 9, "png", 1) == "clip_9.png"


def test_write_selected_csv_uses_frame_digit_width(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_frames.csv"
    rows = [
        {
            "original_index": 120,
            "final_index": 120,
            "change_score_original": 0.0,
            "change_score_final": 0.0,
            "blur_score_original": 0.0,
            "blur_score_final": 0.0,
            "quality_score_original": 0.5,
            "quality_score_final": 0.5,
            "status": "ok",
            "decision": "keep",
        }
    ]

    write_selected_csv(rows, csv_path, fps=30.0, image_ext="jpg", filename_prefix="clip", frame_digits=4)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        written = list(csv.DictReader(f))
    assert written[0]["output_file"] == "images/clip_0120.jpg"
