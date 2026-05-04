from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scripts.benchmark_yolo_mask import compare_masks, select_configs


def test_select_configs_defaults_to_fast_representative_set() -> None:
    assert [config.name for config in select_configs(None)] == ["level1-standard", "level2-bottom-high"]


def test_compare_masks_reports_exact_and_pixel_differences(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()

    ref_a = np.full((4, 4), 255, dtype=np.uint8)
    cand_a = ref_a.copy()
    ref_b = np.full((4, 4), 255, dtype=np.uint8)
    cand_b = ref_b.copy()
    cand_b[1, 1] = 0

    cv2.imwrite(str(reference / "a.png"), ref_a)
    cv2.imwrite(str(candidate / "a.png"), cand_a)
    cv2.imwrite(str(reference / "b.png"), ref_b)
    cv2.imwrite(str(candidate / "b.png"), cand_b)

    summary = compare_masks(reference, candidate)

    assert summary["files_compared"] == 2
    assert summary["exact_match_count"] == 1
    assert summary["missing_candidate"] == []
    assert summary["missing_reference"] == []
    assert summary["max_diff_ratio"] == 1 / 16
    assert summary["mean_iou"] < 1.0
