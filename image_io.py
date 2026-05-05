from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import cv2
import numpy as np

Pathish = str | PathLike[str]


def imread_unicode(path: Pathish, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read an image without using OpenCV's Windows filename APIs."""
    try:
        data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def image_size_unicode(path: Pathish) -> tuple[int, int] | None:
    """Return ``(width, height)`` without forcing a full pixel decode when possible."""
    try:
        from PIL import Image

        with Path(path).open("rb") as f:
            with Image.open(f) as img:
                width, height = img.size
                img.verify()
                return int(width), int(height)
    except Exception:
        image = imread_unicode(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        height, width = image.shape[:2]
        return int(width), int(height)


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
