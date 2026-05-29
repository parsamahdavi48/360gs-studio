from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import cv2
import numpy as np

Pathish = str | PathLike[str]


_IMREAD_IGNORE_ORIENTATION = int(getattr(cv2, "IMREAD_IGNORE_ORIENTATION", 128))


def _ignore_exif_orientation_flags(flags: int) -> int:
    """Keep image reads in raw-pixel orientation unless the caller asks for unchanged data."""
    value = int(flags)
    if value == int(cv2.IMREAD_UNCHANGED):
        return value
    return value | _IMREAD_IGNORE_ORIENTATION


def imread_unicode(path: Pathish, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read an image without using OpenCV's Windows filename APIs."""
    try:
        data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, _ignore_exif_orientation_flags(flags))


def image_size_unicode(path: Pathish) -> tuple[int, int] | None:
    """Return ``(width, height)`` without forcing a full pixel decode when possible."""
    image_path = Path(path)
    try:
        from PIL import Image

        with image_path.open("rb") as f:
            with Image.open(f) as img:
                width, height = img.size
                img.verify()
    except Exception:
        image = imread_unicode(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        height, width = image.shape[:2]
        return int(width), int(height)
    if not _can_decode_reduced_unicode(image_path):
        return None
    return int(width), int(height)


def _can_decode_reduced_unicode(path: Pathish) -> bool:
    try:
        data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    except OSError:
        return False
    if data.size == 0:
        return False
    try:
        return cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_8) is not None
    except cv2.error:
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED) is not None


def imwrite_unicode(
    path: Pathish,
    image: np.ndarray,
    params: Sequence[int] | None = None,
) -> bool:
    """Write an image without letting OpenCV encode the filesystem path."""
    output_path = Path(path)
    ext = output_path.suffix
    if not ext:
        return False
    try:
        ok, encoded = cv2.imencode(ext, image, list(params or []))
    except cv2.error:
        return False
    if not ok:
        return False
    try:
        output_path.write_bytes(encoded.tobytes())
    except OSError:
        return False
    return True
