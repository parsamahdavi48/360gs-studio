"""Initialize white mask files for source images.

Mask convention: white means keep, black means exclude. This helper is used
when the selected mask steps start with a merge-only operation such as stitch
seam masking.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from image_io import image_size_unicode, imwrite_unicode

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def iter_image_files(images: Path) -> tuple[Path, list[Path]]:
    if images.is_file() and images.suffix.lower() in IMAGE_EXTS:
        return images.parent, [images]
    if not images.is_dir():
        return images, []
    files = sorted(
        (path for path in images.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
        key=lambda path: str(path).lower(),
    )
    return images, files


def mask_output_path_for_image(image_path: Path, images_root: Path, masks_dir: Path, *, add_ext: bool = False) -> Path:
    try:
        rel_parent = image_path.resolve().relative_to(images_root.resolve()).parent
    except Exception:
        rel_parent = Path()
    name = f"{image_path.name}.png" if add_ext else f"{image_path.stem}.png"
    return masks_dir / rel_parent / name


def run(images: str | Path, masks_dir: str | Path, *, add_ext: bool = False) -> int:
    images_path = Path(images)
    masks_path = Path(masks_dir)
    images_root, image_files = iter_image_files(images_path)
    if not image_files:
        print(f"No images found in {images_path}")
        return 0

    print(f"Initializing {len(image_files)} masks")
    print(f"[progress] 0/{len(image_files)}", flush=True)
    failed = 0
    white_masks: dict[tuple[int, int], np.ndarray] = {}
    for done, image_path in enumerate(image_files, start=1):
        size = image_size_unicode(image_path)
        if size is None:
            failed += 1
            print(f"Skipped (read error): {image_path.name}")
            print(f"[progress] {done}/{len(image_files)}", flush=True)
            continue
        w, h = size
        mask = white_masks.get(size)
        if mask is None:
            mask = np.full((h, w), 255, dtype=np.uint8)
            white_masks[size] = mask
        mask_out = mask_output_path_for_image(image_path, images_root, masks_path, add_ext=add_ext)
        mask_out.parent.mkdir(parents=True, exist_ok=True)
        if not imwrite_unicode(mask_out, mask):
            failed += 1
            print(f"Skipped (write error): {mask_out.name}")
        print(f"[progress] {done}/{len(image_files)}", flush=True)

    print(f"Done: {len(image_files) - failed} initialized, {failed} failed")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize white PNG masks for source images.")
    parser.add_argument("images", help="Source image file or directory")
    parser.add_argument("masks_dir", help="Mask output directory")
    parser.add_argument("--add-ext", action="store_true", help="Use source filename plus extension as the mask stem")
    args = parser.parse_args()
    sys.exit(run(args.images, args.masks_dir, add_ext=bool(args.add_ext)))


if __name__ == "__main__":
    main()
