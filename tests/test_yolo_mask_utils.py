from core.yolo_mask_utils import EXPAND_DEFAULT, EXPAND_MAX, EXPAND_MIN, clamp_expand_px


def test_yolo_expand_default_is_neutral() -> None:
    assert EXPAND_DEFAULT == 0


def test_yolo_expand_clamp_keeps_manual_values_in_safe_range() -> None:
    assert clamp_expand_px(2) == 2
    assert clamp_expand_px(-4) == -4
    assert clamp_expand_px(999) == EXPAND_MAX
    assert clamp_expand_px(-999) == EXPAND_MIN


def test_yolo_expand_safe_range_is_bounded() -> None:
    assert EXPAND_MIN == -16
    assert EXPAND_MAX == 32
