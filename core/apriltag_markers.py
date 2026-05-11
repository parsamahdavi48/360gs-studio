"""Shared AprilTag dictionary helpers."""

from __future__ import annotations

import base64
import html

import cv2
import numpy as np

DEFAULT_APRILTAG_FAMILY = "tag36h11"
DEFAULT_APRILTAG_ID = 7
DEFAULT_APRILTAG_SIZE_M = 0.160
MAX_APRILTAG_IDS_PER_RUN = 16

_FAMILY_TO_DICT = {
    "tag36h11": "DICT_APRILTAG_36H11",
    "tag36h10": "DICT_APRILTAG_36H10",
    "tag16h5": "DICT_APRILTAG_16H5",
    "tag25h9": "DICT_APRILTAG_25H9",
}


def available_families() -> tuple[str, ...]:
    return tuple(_FAMILY_TO_DICT)


def dictionary_for_family(family: str) -> cv2.aruco.Dictionary:
    key = family.strip().lower()
    attr = _FAMILY_TO_DICT.get(key)
    if attr is None:
        raise ValueError(f"Unsupported AprilTag family: {family}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, attr))


def tag_count_for_family(family: str) -> int:
    return int(len(dictionary_for_family(family).bytesList))


def tag_id_range(family: str) -> tuple[int, int]:
    return 0, tag_count_for_family(family) - 1


def clamp_tag_id(family: str, tag_id: int) -> int:
    low, high = tag_id_range(family)
    return min(max(int(tag_id), low), high)


def marker_image_array(family: str, tag_id: int, size_px: int) -> np.ndarray:
    if size_px <= 0:
        raise ValueError("size_px must be positive")
    marker_id = clamp_tag_id(family, tag_id)
    return cv2.aruco.generateImageMarker(dictionary_for_family(family), marker_id, int(size_px))


def parse_tag_ids(
    text: str,
    *,
    family: str,
    max_ids: int = MAX_APRILTAG_IDS_PER_RUN,
) -> tuple[int, ...]:
    """Parse comma/space separated tag IDs, preserving order and clamping count."""
    ids: list[int] = []
    seen: set[int] = set()
    low, high = tag_id_range(family)
    for raw in text.replace(",", " ").split():
        try:
            tag_id = int(raw)
        except ValueError as exc:
            raise ValueError(f"tag ID must be an integer: {raw}") from exc
        if not low <= tag_id <= high:
            raise ValueError(f"tag ID must be between {low} and {high}: {tag_id}")
        if tag_id not in seen:
            ids.append(tag_id)
            seen.add(tag_id)
        if len(ids) >= max(1, int(max_ids)):
            break
    return tuple(ids)


def marker_tooltip_html(family: str, tag_id: int, *, cell_px: int = 9) -> str:
    """Return a square inline-image preview for a marker without creating a file."""
    dictionary = dictionary_for_family(family)
    modules = int(dictionary.markerSize) + 2
    preview_px = modules * cell_px
    marker = cv2.aruco.generateImageMarker(dictionary, clamp_tag_id(family, tag_id), preview_px)
    ok, encoded = cv2.imencode(".png", marker)
    if not ok:
        raise ValueError("Failed to encode AprilTag marker preview")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    label = html.escape(f"{family} / ID {clamp_tag_id(family, tag_id)}")
    return (
        f"<b>{label}</b><br>"
        f'<img src="data:image/png;base64,{data}" width="{preview_px}" height="{preview_px}">'
    )
