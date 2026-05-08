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
from mask_targets import collect_image_targets


def run(
    images: str | Path,
    masks_dir: str | Path,
    *,
    add_ext: bool = False,
    image_list: str | Path | None = None,
) -> int:
    images_path = Path(images)
    masks_path = Path(masks_dir)
    _images_root, targets = collect_image_targets(
        images_path,
        masks_path,
        add_ext=add_ext,
        image_list=image_list,
    )
    if not targets:
        print(f"No images found in {images_path}")
        return 0

    print(f"Initializing {len(targets)} masks")
    print(f"[progress] 0/{len(targets)}", flush=True)
    failed = 0
    white_masks: dict[tuple[int, int], np.ndarray] = {}
    for done, target in enumerate(targets, start=1):
        image_path = target.image_path
        size = image_size_unicode(image_path)
        if size is None:
            failed += 1
            print(f"Skipped (read error): {image_path.name}")
            print(f"[progress] {done}/{len(targets)}", flush=True)
            continue
        w, h = size
        mask = white_masks.get(size)
        if mask is None:
            mask = np.full((h, w), 255, dtype=np.uint8)
            white_masks[size] = mask
        mask_out = target.mask_path
        mask_out.parent.mkdir(parents=True, exist_ok=True)
        if not imwrite_unicode(mask_out, mask):
            failed += 1
            print(f"Skipped (write error): {mask_out.name}")
        print(f"[progress] {done}/{len(targets)}", flush=True)

    print(f"Done: {len(targets) - failed} initialized, {failed} failed")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize white PNG masks for source images.")
    parser.add_argument("images", help="Source image file or directory")
    parser.add_argument("masks_dir", help="Mask output directory")
    parser.add_argument("--add-ext", action="store_true", help="Use source filename plus extension as the mask stem")
    parser.add_argument("--image-list", default=None, help="JSON or JSONL list of images to process")
    args = parser.parse_args()
    sys.exit(run(args.images, args.masks_dir, add_ext=bool(args.add_ext), image_list=args.image_list))


if __name__ == "__main__":
    main()
