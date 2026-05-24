"""Prepare a scene folder for SphereSfM execution."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from core.path_safety import safe_clear_path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


def iter_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def validate_spheresfm_colmap(colmap: str) -> None:
    candidates = [
        [colmap, "help"],
        [colmap, "-h"],
    ]
    last_error = ""
    for cmd in candidates:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_error = str(exc)
            continue
        output = f"{result.stdout}\n{result.stderr}"
        if result.returncode == 0 and "sphere_cubic_reprojecer" in output:
            print("SphereSfM executable verified.", flush=True)
            return
        last_error = output.strip()[-1200:]
    raise RuntimeError(
        "The selected executable does not look like SphereSfM's COLMAP build. "
        "Expected command list to contain sphere_cubic_reprojecer."
        + (f"\nLast output:\n{last_error}" if last_error else "")
    )


def source_mask_candidates(source_masks_dir: Path, rel_image: Path) -> list[Path]:
    rel_parent = rel_image.parent
    return [
        source_masks_dir / rel_parent / f"{rel_image.name}.png",
        source_masks_dir / rel_parent / f"{rel_image.stem}.png",
    ]


def prepare_masks(images_dir: Path, source_masks_dir: Path, output_masks_dir: Path) -> tuple[int, int]:
    if output_masks_dir.exists():
        safe_clear_path(output_masks_dir, allowed_roots=[output_masks_dir.parent])
    output_masks_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    images = iter_images(images_dir)
    total = len(images)
    for index, image in enumerate(images, start=1):
        rel_image = image.relative_to(images_dir)
        source = next((p for p in source_mask_candidates(source_masks_dir, rel_image) if p.is_file()), None)
        if source is None:
            missing += 1
            continue
        target = output_masks_dir / rel_image.parent / f"{rel_image.name}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
        if copied == 1 or copied % 50 == 0 or index == total:
            print(f"[progress] {index} / {total}", flush=True)
    return copied, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare masks and validate a SphereSfM COLMAP executable.")
    parser.add_argument("--colmap", required=True, help="SphereSfM colmap executable")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--source-masks-dir", type=Path)
    parser.add_argument("--output-masks-dir", type=Path)
    parser.add_argument("--use-masks", action="store_true")
    args = parser.parse_args(argv)

    images_dir = args.images_dir
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    images = iter_images(images_dir)
    if not images:
        raise FileNotFoundError(f"No supported images found: {images_dir}")

    validate_spheresfm_colmap(args.colmap)

    if args.use_masks:
        if args.source_masks_dir is None or not args.source_masks_dir.is_dir():
            raise FileNotFoundError(f"Masks folder not found: {args.source_masks_dir}")
        if args.output_masks_dir is None:
            raise ValueError("--output-masks-dir is required with --use-masks")
        copied, missing = prepare_masks(images_dir, args.source_masks_dir, args.output_masks_dir)
        print(f"Prepared SphereSfM masks: copied={copied}, missing={missing}", flush=True)
    else:
        print("SphereSfM masks disabled.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
