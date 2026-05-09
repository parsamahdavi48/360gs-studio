"""Lightweight metadata helpers for saved mask files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

PIXEL_STATS_SKIPPED = "skipped"


def mask_file_summary(path: Path) -> dict[str, Any]:
    """Return mask file metadata without decoding full pixel data."""

    try:
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
    except Exception:
        return {"readable": False}
    return {
        "readable": True,
        "width": int(width),
        "height": int(height),
        "mode": mode,
        "pixel_stats": PIXEL_STATS_SKIPPED,
    }


def summary_size(summary: dict[str, Any]) -> tuple[int, int] | None:
    if not summary.get("readable"):
        return None
    try:
        width = int(summary.get("width") or 0)
        height = int(summary.get("height") or 0)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height
