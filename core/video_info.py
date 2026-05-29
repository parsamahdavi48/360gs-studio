"""Shared video metadata contract for frame extraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    total_frames: int
    avg_frame_rate: float = 0.0
    r_frame_rate: float = 0.0
    variable_frame_rate: bool = False
    frame_rate_warning: str = ""


def frame_rates_indicate_vfr(avg_frame_rate: float, r_frame_rate: float) -> bool:
    if avg_frame_rate <= 0 or r_frame_rate <= 0:
        return False
    delta = abs(avg_frame_rate - r_frame_rate)
    return (delta / max(avg_frame_rate, r_frame_rate)) > 0.001
