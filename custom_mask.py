"""Merge a user-provided static mask into generated masks.

Mask convention: white means keep, black means exclude. The custom mask is
AND-merged with each existing mask. If no existing mask is present for an image,
the custom mask itself becomes the output mask.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from image_io import imread_unicode, imwrite_unicode

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


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


def merge_custom_mask_for_image(
    image_path: Path,
    images_dir: Path,
    masks_dir: Path,
    custom_mask: np.ndarray,
) -> str | None:
    source = imread_unicode(image_path, cv2.IMREAD_UNCHANGED)
    if source is None:
        return f"Skipped (image read error): {image_path.name}"
    source_shape = source.shape[:2]
    if custom_mask.shape != source_shape:
        return (
            f"Skipped (size mismatch): {image_path.name} "
            f"image={source_shape[1]}x{source_shape[0]} custom={custom_mask.shape[1]}x{custom_mask.shape[0]}"
        )

    mask_out = mask_output_path_for_image(image_path, images_dir, masks_dir)
    existing = imread_unicode(mask_out, cv2.IMREAD_GRAYSCALE) if mask_out.is_file() else None
    if existing is not None:
        if existing.shape != custom_mask.shape:
            return (
                f"Skipped (existing mask size mismatch): {mask_out.name} "
                f"existing={existing.shape[1]}x{existing.shape[0]} custom={custom_mask.shape[1]}x{custom_mask.shape[0]}"
            )
        output = cv2.bitwise_and(existing, custom_mask)
    else:
        output = custom_mask

    mask_out.parent.mkdir(parents=True, exist_ok=True)
    if not imwrite_unicode(mask_out, output):
        return f"Skipped (write error): {mask_out.name}"
    return None


def run(images_dir: str | Path, masks_dir: str | Path, custom_mask_path: str | Path) -> list[str]:
    images_path = Path(images_dir)
    masks_path = Path(masks_dir)
    custom_path = Path(custom_mask_path)
    custom = imread_unicode(custom_path, cv2.IMREAD_GRAYSCALE)
    if custom is None:
        return [f"Custom mask read error: {custom_path}"]

    image_files = iter_image_files(images_path)
    if not image_files:
        print(f"No images found in {images_path}")
        return []

    print(f"Applying custom mask to {len(image_files)} images")
    print(f"[progress] 0/{len(image_files)}", flush=True)
    errors: list[str] = []
    for done, image_path in enumerate(image_files, start=1):
        error = merge_custom_mask_for_image(image_path, images_path, masks_path, custom)
        if error is not None:
            errors.append(error)
        print(f"[progress] {done}/{len(image_files)}", flush=True)

    for error in errors:
        print(error)
    print(f"Done: {len(image_files) - len(errors)} succeeded, {len(errors)} skipped")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AND-merge a user-provided static mask into mask images.",
    )
    parser.add_argument("images_dir", help="Source images directory")
    parser.add_argument("masks_dir", help="Mask output directory")
    parser.add_argument("custom_mask", help="Custom mask image to apply to every source image")
    args = parser.parse_args()

    errors = run(args.images_dir, args.masks_dir, args.custom_mask)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
