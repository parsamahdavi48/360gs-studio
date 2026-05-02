from __future__ import annotations

import pytest

from extract_frames import compute_feature_motion_score, select_fixed_smart


def test_select_fixed_smart_inserts_feature_motion_peak() -> None:
    change_scores = [0.0] * 31
    feature_motion_scores = [0.0] * 31
    feature_motion_scores[15] = 0.02

    selected, added, thinned, min_gap_frames, max_gap_frames = select_fixed_smart(
        base_indices=[0, 30],
        change_scores=change_scores,
        feature_motion_scores=feature_motion_scores,
        fps=30.0,
        min_gap_sec=0.25,
        max_gap_sec=2.0,
        change_threshold=0.04,
        feature_motion_threshold=0.012,
        max_inserts_per_interval=2,
    )

    assert 15 in selected
    assert added == {15}
    assert 15 not in thinned
    assert min_gap_frames == 8
    assert max_gap_frames == 60


def test_select_fixed_smart_respects_min_gap() -> None:
    change_scores = [0.0] * 31
    feature_motion_scores = [0.0] * 31
    feature_motion_scores[3] = 0.05

    selected, added, thinned, _, _ = select_fixed_smart(
        base_indices=[0, 30],
        change_scores=change_scores,
        feature_motion_scores=feature_motion_scores,
        fps=30.0,
        min_gap_sec=0.25,
        max_gap_sec=2.0,
        feature_motion_threshold=0.012,
    )

    assert selected == [0, 30]
    assert added == set()
    assert thinned == set()


def test_select_fixed_smart_caps_insertions_per_interval() -> None:
    change_scores = [0.0] * 61
    feature_motion_scores = [0.0] * 61
    feature_motion_scores[15] = 0.03
    feature_motion_scores[30] = 0.04
    feature_motion_scores[45] = 0.02

    selected, added, thinned, _, _ = select_fixed_smart(
        base_indices=[0, 60],
        change_scores=change_scores,
        feature_motion_scores=feature_motion_scores,
        fps=30.0,
        min_gap_sec=0.25,
        max_gap_sec=2.0,
        feature_motion_threshold=0.012,
        max_inserts_per_interval=1,
    )

    assert len(added) == 1
    assert len(selected) == 3
    assert 30 in added
    assert added.isdisjoint(thinned)


def test_select_fixed_smart_marks_low_motion_base_candidates_only() -> None:
    change_scores = [0.0] * 91
    feature_motion_scores = [0.0] * 91
    feature_motion_scores[15] = 0.02

    selected, added, thinned, _, _ = select_fixed_smart(
        base_indices=[0, 30, 60, 90],
        change_scores=change_scores,
        feature_motion_scores=feature_motion_scores,
        fps=30.0,
        min_gap_sec=0.25,
        max_gap_sec=2.0,
        feature_motion_threshold=0.012,
        max_inserts_per_interval=1,
    )

    assert 15 in added
    assert 15 not in thinned
    assert 30 in thinned
    assert 60 in selected


def test_select_fixed_smart_can_skip_low_motion_when_insertions_disabled() -> None:
    change_scores = [0.0] * 91
    feature_motion_scores = [0.0] * 91

    selected, added, thinned, _, _ = select_fixed_smart(
        base_indices=[0, 30, 60, 90],
        change_scores=change_scores,
        feature_motion_scores=feature_motion_scores,
        fps=30.0,
        min_gap_sec=0.25,
        max_gap_sec=2.0,
        max_inserts_per_interval=0,
    )

    assert selected == [0, 30, 60, 90]
    assert added == set()
    assert 30 in thinned


def test_feature_motion_score_detects_sparse_translation() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    base = np.zeros((96, 96), dtype=np.uint8)
    for y in range(12, 84, 18):
        for x in range(12, 84, 18):
            base[y : y + 5, x : x + 5] = 255
    shifted = np.roll(base, shift=5, axis=1)

    assert compute_feature_motion_score(base, shifted) > 0.0
