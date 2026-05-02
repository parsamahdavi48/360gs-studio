"""extract_frames.py の立ち止まり間引き (thin_stationary) テスト。
"""
from __future__ import annotations

import numpy as np
import pytest

from extract_frames import thin_stationary


def _make_row(final_index: int, status: str = "ok") -> dict:
    return {
        "original_index": final_index,
        "final_index": final_index,
        "status": status,
        "quality_min_score": 0.35,
    }


# =============================================================================
# 無効化動作: threshold <= 0
# =============================================================================


def test_threshold_zero_keeps_all():
    rows = [_make_row(i) for i in [0, 5, 10, 15]]
    change_scores = [0.0] * 16
    out = thin_stationary(rows, change_scores, motion_threshold=0.0)
    assert all(r["decision"] == "keep" for r in out)


def test_threshold_negative_keeps_all():
    rows = [_make_row(i) for i in [0, 5, 10]]
    out = thin_stationary(rows, [0.0] * 11, motion_threshold=-1.0)
    assert all(r["decision"] == "keep" for r in out)


def test_single_row_keeps_all():
    rows = [_make_row(0)]
    out = thin_stationary(rows, [0.5], motion_threshold=0.5)
    assert out[0]["decision"] == "keep"


# =============================================================================
# 立ち止まり: すべて閾値未満 → 中間を全部 drop
# =============================================================================


def test_all_stationary_drops_middle(tmp_path):
    """すべての frame 間の change が閾値未満。中間は全部 drop、両端は keep。"""
    rows = [_make_row(i) for i in [0, 10, 20, 30, 40]]
    # change_scores 全 0.01 → 累積も小さい
    change_scores = [0.01] * 41

    out = thin_stationary(rows, change_scores, motion_threshold=1.0, keep_endpoints=True)

    # 先頭と末尾は keep、中間は drop (status=thinned)
    assert out[0]["decision"] == "keep"
    assert out[-1]["decision"] == "keep"
    for r in out[1:-1]:
        assert r["decision"] == "drop"
        assert "thinned" in r["status"]


# =============================================================================
# 全部歩行: change が閾値超え → 何も削らない
# =============================================================================


def test_all_walking_keeps_all():
    rows = [_make_row(i) for i in [0, 10, 20, 30]]
    # 各区間の累積 change が 1.0 を大きく超える
    change_scores = [0.2] * 31  # 10frame × 0.2 = 2.0/区間

    out = thin_stationary(rows, change_scores, motion_threshold=1.0)
    assert all(r["decision"] == "keep" for r in out)


# =============================================================================
# 混在: 立ち止まり → 歩行 → 立ち止まり
# =============================================================================


def test_mixed_stop_walk_stop():
    """0:立ち止まり開始 → 中盤歩行 → 末尾立ち止まり、というシナリオ。"""
    # final_index: 0, 10, 20, 30 (立ち止まり) | 40 (急に動く) | 50, 60, 70 (再び立ち止まり)
    rows = [_make_row(i) for i in [0, 10, 20, 30, 40, 50, 60, 70]]
    n = 71
    change_scores = [0.0] * n
    # 0~30 : 静止 (ほぼ 0)
    for i in range(31):
        change_scores[i] = 0.005
    # 30~40 : 歩行 (大きな変化)
    for i in range(31, 41):
        change_scores[i] = 0.3
    # 40~70 : 静止
    for i in range(41, 71):
        change_scores[i] = 0.005

    out = thin_stationary(rows, change_scores, motion_threshold=1.0, keep_endpoints=True)

    # 期待: 先頭/末尾は keep
    assert out[0]["decision"] == "keep"  # final_index=0
    assert out[-1]["decision"] == "keep"  # final_index=70

    # 立ち止まり中は累積 0.05 < 1.0 → drop されるはず
    # 歩行直後 (final_index=40) は 30→40 累積 ≈ 3.0 ≫ 1.0 → keep
    by_idx = {r["final_index"]: r for r in out}
    assert by_idx[40]["decision"] == "keep"
    # 1~3 (final_index=10,20,30) は静止のみで drop されるはず
    assert by_idx[10]["decision"] == "drop"
    assert by_idx[20]["decision"] == "drop"
    assert by_idx[30]["decision"] == "drop"


# =============================================================================
# 末端保持
# =============================================================================


def test_keep_endpoints_true_preserves_last():
    """keep_endpoints=True なら最後の row は静止判定でも keep。"""
    rows = [_make_row(i) for i in [0, 10, 20]]
    change_scores = [0.005] * 21

    out = thin_stationary(rows, change_scores, motion_threshold=1.0, keep_endpoints=True)
    assert out[0]["decision"] == "keep"
    assert out[-1]["decision"] == "keep"


def test_keep_endpoints_false_can_drop_last():
    """keep_endpoints=False なら末尾も判定対象。"""
    rows = [_make_row(i) for i in [0, 10, 20]]
    change_scores = [0.005] * 21  # 全部静止

    out = thin_stationary(rows, change_scores, motion_threshold=1.0, keep_endpoints=False)
    # 先頭は無条件 keep
    assert out[0]["decision"] == "keep"
    # 末尾は drop されうる
    assert out[-1]["decision"] == "drop"
    assert "thinned" in out[-1]["status"]


# =============================================================================
# status の合成
# =============================================================================


def test_status_thinned_appended_to_replaced():
    """既に status='replaced' のフレームを間引く場合、status='replaced+thinned' になる。"""
    rows = [
        _make_row(0, status="ok"),
        _make_row(10, status="replaced"),
        _make_row(20, status="ok"),
    ]
    change_scores = [0.005] * 21

    out = thin_stationary(rows, change_scores, motion_threshold=1.0, keep_endpoints=True)
    # final_index=10 は drop、status は replaced+thinned
    assert out[1]["decision"] == "drop"
    assert out[1]["status"] == "replaced+thinned"


# =============================================================================
# 境界条件
# =============================================================================


def test_invalid_indices_safe_keep():
    """index が範囲外でもクラッシュせず keep として扱う。"""
    rows = [_make_row(0), _make_row(100)]  # change_scores は 5 要素しかない
    change_scores = [0.005] * 5

    out = thin_stationary(rows, change_scores, motion_threshold=1.0)
    # 範囲外検出時は安全側で keep
    assert out[0]["decision"] == "keep"
    assert out[1]["decision"] == "keep"


def test_thinning_preserves_row_count():
    """間引きは row を削除せず、decision/status を変えるだけ（CSV メタとして残す）。"""
    rows = [_make_row(i) for i in [0, 10, 20, 30, 40]]
    change_scores = [0.005] * 41

    out = thin_stationary(rows, change_scores, motion_threshold=1.0)
    assert len(out) == len(rows)


def test_thinning_does_not_mutate_input_rows():
    """入力 rows を直接変更しない（コピーを返す）。"""
    rows = [_make_row(i) for i in [0, 10, 20]]
    change_scores = [0.005] * 21

    thin_stationary(rows, change_scores, motion_threshold=1.0)
    # 元の rows は decision を持っていない (default を持たないのを確認)
    for r in rows:
        assert "decision" not in r
