from __future__ import annotations

import math
from typing import Any

PROJECTION_EQUIRECTANGULAR = "equirectangular"
PROJECTION_NORMAL = "normal"
PROJECTION_UNKNOWN = "unknown"
PROJECTION_MIXED = "mixed"

MASK_PROJECTION_EQUIRECT = "equirect"
MASK_PROJECTION_NORMAL = "normal"

_EQUIRECT_ALIASES = {"equirect", "equirectangular", "erp", "360", "360deg", "360°"}
_NORMAL_ALIASES = {"normal", "perspective", "flat", "frame", "pinhole"}


def normalize_projection(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in _EQUIRECT_ALIASES:
        return PROJECTION_EQUIRECTANGULAR
    if text in _NORMAL_ALIASES:
        return PROJECTION_NORMAL
    if text == PROJECTION_MIXED:
        return PROJECTION_MIXED
    return PROJECTION_UNKNOWN


def projection_for_record(record: dict[str, Any]) -> str:
    override = normalize_projection(record.get("projection_override"))
    if override != PROJECTION_UNKNOWN:
        return override
    explicit = normalize_projection(record.get("projection"))
    if explicit != PROJECTION_UNKNOWN:
        return explicit
    return normalize_projection(record.get("detected_projection"))


def mask_projection_for_image_projection(projection: object) -> str:
    return MASK_PROJECTION_NORMAL if normalize_projection(projection) == PROJECTION_NORMAL else MASK_PROJECTION_EQUIRECT


def infer_projection_from_size(width: int, height: int, *, media_label: str) -> dict[str, Any]:
    if width > 0 and height > 0:
        ratio = width / height
        if math.isfinite(ratio) and abs(ratio - 2.0) <= 0.04:
            return {
                "projection": PROJECTION_EQUIRECTANGULAR,
                "confidence": "medium",
                "reason": f"2:1 {media_label} aspect ratio",
            }
        return {
            "projection": PROJECTION_NORMAL,
            "confidence": "medium",
            "reason": f"{media_label} aspect ratio is not 2:1",
        }
    return {
        "projection": PROJECTION_UNKNOWN,
        "confidence": "low",
        "reason": f"{media_label} dimensions unavailable",
    }
