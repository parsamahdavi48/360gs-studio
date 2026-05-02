from __future__ import annotations

from extract_frames import representative_window_for_report, select_representative_frames


def _scores(n: int, default: float = 0.6) -> list[float]:
    return [default] * n


def test_representative_selection_does_not_force_percentile_replacements():
    n = 40
    selected = [0, 10, 20, 30]
    quality = _scores(n, 0.7)

    rows = select_representative_frames(
        selected_indices=selected,
        quality_scores=quality,
        quality_min_score=0.35,
        quality_min_improvement=0.08,
        center_bias=0.05,
    )

    assert [r["final_index"] for r in rows] == selected
    assert [r["status"] for r in rows] == ["ok"] * len(selected)


def test_representative_selection_replaces_with_clearly_better_neighbor():
    n = 30
    quality = _scores(n, 0.45)
    quality[10] = 0.20
    quality[12] = 0.62

    rows = select_representative_frames(
        selected_indices=[10],
        quality_scores=quality,
        quality_min_score=0.35,
        quality_min_improvement=0.08,
        center_bias=0.05,
    )

    assert rows[0]["original_index"] == 10
    assert rows[0]["final_index"] == 12
    assert rows[0]["status"] == "replaced"


def test_representative_selection_marks_low_quality_without_better_candidate():
    n = 20
    quality = _scores(n, 0.2)

    rows = select_representative_frames(
        selected_indices=[8],
        quality_scores=quality,
        quality_min_score=0.35,
        quality_min_improvement=0.08,
        center_bias=0.05,
    )

    assert rows[0]["final_index"] == 8
    assert rows[0]["status"] == "fallback_keep"


def test_auto_window_uses_neighbor_midpoints_without_crossing_anchor_regions():
    n = 40
    quality = _scores(n, 0.4)
    quality[15] = 0.95  # belongs to the previous anchor region
    quality[24] = 0.90  # belongs to the middle anchor region

    rows = select_representative_frames(
        selected_indices=[10, 20, 30],
        quality_scores=quality,
        quality_min_score=0.35,
        quality_min_improvement=0.08,
        center_bias=0.05,
    )

    assert rows[1]["candidate_low"] == 16
    assert rows[1]["candidate_high"] == 25
    assert rows[1]["final_index"] == 24


def test_representative_window_for_report_returns_max_auto_radius():
    assert representative_window_for_report([10, 20, 30], 40) == 10
