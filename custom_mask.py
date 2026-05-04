"""Merge a user-provided static PNG mask into generated masks.

Mask convention: white means keep, black means exclude. The custom mask is
normalized to binary 0/255, then AND-merged with each existing mask. If no
existing mask is present, or --replace is used, the custom mask itself becomes
the output mask.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from image_io import imread_unicode, imwrite_unicode

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class CustomMaskMergeResult:
    applied: bool = False
    skipped: bool = False
    failed: bool = False
    message: str | None = None


@dataclass(frozen=True)
class LoadedCustomMask:
    mask: np.ndarray
    description: str


@dataclass
class CustomMaskRunResult:
    total: int = 0
    applied: int = 0
    skipped: int = 0
    failed: int = 0
    messages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0 and (self.total == 0 or self.applied > 0)


def iter_image_files(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    return sorted(
        (path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
        key=lambda path: str(path).lower(),
    )


def mask_output_path_for_image(image_path: Path, images_dir: Path, masks_dir: Path) -> Path:
    rel_parent = image_path.resolve().relative_to(images_dir.resolve()).parent
    return masks_dir / rel_parent / f"{image_path.stem}.png"


def load_custom_mask(custom_mask_path: str | Path) -> tuple[LoadedCustomMask | None, str | None]:
    """Load a PNG custom mask and normalize it to a binary uint8 mask."""
    custom_path = Path(custom_mask_path)
    if custom_path.suffix.lower() != ".png":
        return None, f"Custom mask must be a PNG file: {custom_path}"

    raw = imread_unicode(custom_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None, f"Custom mask read error: {custom_path}"

    if raw.dtype == np.uint8:
        threshold = 128
        bit_depth = 8
    elif raw.dtype == np.uint16:
        threshold = 32768
        bit_depth = 16
    else:
        return None, f"Custom mask must be 8-bit or 16-bit PNG: {custom_path} ({raw.dtype})"

    conversion = "grayscale"
    if raw.ndim == 2:
        gray = raw
    elif raw.ndim == 3 and raw.shape[2] == 3:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        conversion = "color converted to grayscale"
    elif raw.ndim == 3 and raw.shape[2] == 4:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        conversion = "color converted to grayscale; alpha ignored"
    else:
        shape_text = "x".join(str(v) for v in raw.shape)
        return None, f"Custom mask must be grayscale, RGB, or RGBA PNG: {custom_path} (shape={shape_text})"

    binary = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    return LoadedCustomMask(
        mask=binary,
        description=f"Custom mask loaded: {bit_depth}-bit PNG, {conversion}, threshold={threshold}",
    ), None


def merge_custom_mask_for_image(
    image_path: Path,
    images_dir: Path,
    masks_dir: Path,
    custom_mask: np.ndarray,
    *,
    replace: bool = False,
) -> CustomMaskMergeResult:
    source = imread_unicode(image_path, cv2.IMREAD_UNCHANGED)
    if source is None:
        return CustomMaskMergeResult(skipped=True, message=f"Skipped (image read error): {image_path.name}")
    source_shape = source.shape[:2]
    if custom_mask.shape != source_shape:
        return CustomMaskMergeResult(
            skipped=True,
            message=(
                f"Skipped (size mismatch): {image_path.name} "
                f"image={source_shape[1]}x{source_shape[0]} custom={custom_mask.shape[1]}x{custom_mask.shape[0]}"
            ),
        )

    mask_out = mask_output_path_for_image(image_path, images_dir, masks_dir)
    existing = None if replace else imread_unicode(mask_out, cv2.IMREAD_GRAYSCALE) if mask_out.is_file() else None
    if existing is not None:
        if existing.shape != custom_mask.shape:
            return CustomMaskMergeResult(
                skipped=True,
                message=(
                    f"Skipped (existing mask size mismatch): {mask_out.name} "
                    f"existing={existing.shape[1]}x{existing.shape[0]} custom={custom_mask.shape[1]}x{custom_mask.shape[0]}"
                ),
            )
        output = cv2.bitwise_and(existing, custom_mask)
    else:
        output = custom_mask

    mask_out.parent.mkdir(parents=True, exist_ok=True)
    if not imwrite_unicode(mask_out, output):
        return CustomMaskMergeResult(failed=True, message=f"Failed (write error): {mask_out.name}")
    return CustomMaskMergeResult(applied=True)


def run(
    images_dir: str | Path,
    masks_dir: str | Path,
    custom_mask_path: str | Path,
    *,
    replace: bool = False,
) -> CustomMaskRunResult:
    images_path = Path(images_dir)
    masks_path = Path(masks_dir)
    custom_path = Path(custom_mask_path)
    loaded_custom, load_error = load_custom_mask(custom_path)
    if loaded_custom is None:
        return CustomMaskRunResult(failed=1, messages=[load_error or f"Custom mask read error: {custom_path}"])
    custom = loaded_custom.mask

    image_files = iter_image_files(images_path)
    if not image_files:
        print(f"No images found in {images_path}")
        return CustomMaskRunResult()

    print(f"Applying custom mask to {len(image_files)} images (replace={replace})")
    print(loaded_custom.description)
    print(f"[progress] 0/{len(image_files)}", flush=True)
    result = CustomMaskRunResult(total=len(image_files))
    for done, image_path in enumerate(image_files, start=1):
        merge_result = merge_custom_mask_for_image(image_path, images_path, masks_path, custom, replace=replace)
        if merge_result.applied:
            result.applied += 1
        elif merge_result.skipped:
            result.skipped += 1
        elif merge_result.failed:
            result.failed += 1
        if merge_result.message is not None:
            result.messages.append(merge_result.message)
        print(f"[progress] {done}/{len(image_files)}", flush=True)

    for message in result.messages:
        print(message)
    print(f"Done: {result.applied} applied, {result.skipped} skipped, {result.failed} failed")
    if result.total > 0 and result.applied == 0 and result.failed == 0:
        print("No images matched the custom mask size; nothing was written.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AND-merge a user-provided static mask into mask images.",
    )
    parser.add_argument("images_dir", help="Source images directory")
    parser.add_argument("masks_dir", help="Mask output directory")
    parser.add_argument("custom_mask", help="PNG custom mask to apply to every source image")
    parser.add_argument("--replace", action="store_true", help="Ignore existing masks and write custom-only masks")
    args = parser.parse_args()

    result = run(args.images_dir, args.masks_dir, args.custom_mask, replace=bool(args.replace))
    if not result.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
