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
