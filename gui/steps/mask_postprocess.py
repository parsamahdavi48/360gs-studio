"""Pure helpers for Step 3 saved-mask post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.custom_mask import load_custom_mask
from core.image_io import imread_unicode, imwrite_unicode
from core.mask_metadata import PIXEL_STATS_SKIPPED
from core.overexposure_mask import detect_overexposure, read_image_preserve_depth
from core.stitch_mask import boundary_width_to_limit_angle, create_angular_stitched_mask


@dataclass(frozen=True)
class MaskPostprocessOptions:
    projection: str
    equirect_projection: str
    apply_stitch: bool
    stitch_boundary_width: float
    apply_overexposure: bool
    overexposure_threshold: int
    overexposure_dilate: int
    apply_custom: bool
    custom_mask_path: str
    replace: bool = False
    preview_load_fail_message: str = "Failed to load preview image"
    custom_required_message: str = "Custom mask is required"
    custom_not_found_message: str = "Custom mask not found: {path}"
    write_failed_message: str = "Failed to write mask"


def mask_stats(mask_path: Path) -> dict:
    return {
        "readable": mask_path.is_file(),
        "pixel_stats": PIXEL_STATS_SKIPPED,
    }


def apply_mask_postprocess(image_path: Path, mask_path: Path, options: MaskPostprocessOptions) -> None:
    source_img: np.ndarray | None = None
    mask: np.ndarray | None = None

    def load_source() -> np.ndarray:
        nonlocal source_img
        if source_img is None:
            source_img = read_image_preserve_depth(str(image_path))
            if source_img is None:
                raise RuntimeError(options.preview_load_fail_message)
        return source_img

    def current_mask(target_shape: tuple[int, int]) -> np.ndarray:
        nonlocal mask
        if mask is None:
            existing = None if options.replace else imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
            if existing is None:
                mask = np.full(target_shape, 255, dtype=np.uint8)
            else:
                mask = existing
        if mask.shape != target_shape:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    if options.projection == options.equirect_projection and options.apply_stitch:
        if mask is None and not options.replace and mask_path.is_file():
            existing = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
            target_shape = existing.shape[:2] if existing is not None else load_source().shape[:2]
        else:
            target_shape = load_source().shape[:2]
        base = current_mask(target_shape)
        h, w = base.shape[:2]
        stitch = create_angular_stitched_mask(
            w,
            h,
            boundary_width_to_limit_angle(options.stitch_boundary_width),
        )
        mask = cv2.bitwise_and(base, stitch)

    if options.apply_overexposure:
        source = load_source()
        overexp = detect_overexposure(
            source,
            threshold=options.overexposure_threshold,
            dilate_px=options.overexposure_dilate,
        )
        mask = cv2.bitwise_and(current_mask(overexp.shape), overexp)

    if options.apply_custom:
        if not options.custom_mask_path:
            raise RuntimeError(options.custom_required_message)
        loaded_custom, load_error = load_custom_mask(options.custom_mask_path)
        if loaded_custom is None:
            raise RuntimeError(load_error or options.custom_not_found_message.format(path=options.custom_mask_path))
        source_shape = load_source().shape[:2]
        if loaded_custom.mask.shape != source_shape:
            raise RuntimeError(
                f"Skipped (size mismatch): {image_path.name} "
                f"image={source_shape[1]}x{source_shape[0]} "
                f"custom={loaded_custom.mask.shape[1]}x{loaded_custom.mask.shape[0]}"
            )
        mask = cv2.bitwise_and(current_mask(loaded_custom.mask.shape), loaded_custom.mask)

    if mask is not None:
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not imwrite_unicode(mask_path, mask):
            raise RuntimeError(options.write_failed_message)
