"""Shared helpers for YOLO mask generation."""
from __future__ import annotations

EXPAND_MIN = -16
EXPAND_MAX = 32
EXPAND_DEFAULT = 0


def clamp_expand_px(value: int) -> int:
    """Clamp manual YOLO/SAM mask expansion to a safe post-processing range."""
    return max(EXPAND_MIN, min(EXPAND_MAX, int(value)))
