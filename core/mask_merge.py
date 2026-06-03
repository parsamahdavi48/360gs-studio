"""Shared mask merge helpers.

Mask convention: white means usable pixels, black means excluded pixels.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.image_io import imread_unicode

MASK_MERGE_REPLACE = "replace"
MASK_MERGE_ADD = "add"
MASK_MERGE_SUBTRACT = "subtract"
SUPPORTED_MERGE_MODES = (MASK_MERGE_REPLACE, MASK_MERGE_ADD, MASK_MERGE_SUBTRACT)


def normalize_mask_merge_mode(value: object) -> str:
    mode = str(value or MASK_MERGE_ADD).strip().lower().replace("_", "-")
    if mode in {"and", "merge"}:
        return MASK_MERGE_ADD
    if mode not in SUPPORTED_MERGE_MODES:
        raise ValueError(f"merge mode must be one of: {', '.join(SUPPORTED_MERGE_MODES)}")
    return mode


def merge_mask_arrays(
    existing: np.ndarray | None,
    candidate: np.ndarray,
    *,
    merge_mode: str = MASK_MERGE_ADD,
    resize_existing: bool = False,
) -> np.ndarray:
    mode = normalize_mask_merge_mode(merge_mode)
    candidate_u8 = _as_u8_mask(candidate)
    if mode == MASK_MERGE_REPLACE:
        return candidate_u8
    if existing is None:
        if mode == MASK_MERGE_SUBTRACT:
            return np.full_like(candidate_u8, 255, dtype=np.uint8)
        return candidate_u8

    existing_u8 = _as_u8_mask(existing)
    if existing_u8.shape != candidate_u8.shape:
        if not resize_existing:
            raise ValueError(
                "existing mask size does not match candidate mask size: "
                f"existing={existing_u8.shape[1]}x{existing_u8.shape[0]} "
                f"candidate={candidate_u8.shape[1]}x{candidate_u8.shape[0]}"
            )
        existing_u8 = cv2.resize(
            existing_u8,
            (candidate_u8.shape[1], candidate_u8.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    if mode == MASK_MERGE_SUBTRACT:
        merged = existing_u8.copy()
        merged[candidate_u8 == 0] = 255
        return merged
    return cv2.bitwise_and(existing_u8, candidate_u8)


def merge_mask_file(
    mask_path: str | Path,
    candidate: np.ndarray,
    *,
    merge_mode: str = MASK_MERGE_ADD,
    resize_existing: bool = False,
) -> np.ndarray:
    path = Path(mask_path)
    existing = imread_unicode(path, cv2.IMREAD_GRAYSCALE) if path.is_file() else None
    return merge_mask_arrays(
        existing,
        candidate,
        merge_mode=merge_mode,
        resize_existing=resize_existing,
    )


def _as_u8_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype == np.uint8:
        return array
    return np.where(array > 0, 255, 0).astype(np.uint8)
