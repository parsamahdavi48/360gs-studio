"""Prepare a scene folder for COLMAP spherical SfM execution."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

from core.path_safety import safe_clear_path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


def iter_images(images_dir: Path) -> list[Path]:
    # Spherical SfM project preparation intentionally scans the prepared image root:
    # GUI preflight already constrains this route to same-resolution ERP input.
    return sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _parse_colmap_version(output: str) -> tuple[int, int, int] | None:
    match = re.search(r"\bCOLMAP\s+(\d+)\.(\d+)(?:\.(\d+))?", output)
    if match is None:
        return None
    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3) or 0)
    return major, minor, patch


def validate_spheresfm_colmap(colmap: str) -> None:
    candidates = [
        [colmap, "version"],
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
        version = _parse_colmap_version(output)
        if result.returncode == 0 and version is not None and version >= (4, 1, 0):
            print("COLMAP 4.1+ spherical SfM executable verified.", flush=True)
            return
        last_error = output.strip()[-1200:]
    raise RuntimeError(
        "The selected executable must be COLMAP 4.1.0 or newer with native EQUIRECTANGULAR camera support. "
        "Run `colmap version` and select a COLMAP 4.1+ executable."
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
    if total > 0:
        print(f"[progress] 0/{total}", flush=True)
    for index, image in enumerate(images, start=1):
        rel_image = image.relative_to(images_dir)
        source = next((p for p in source_mask_candidates(source_masks_dir, rel_image) if p.is_file()), None)
        if source is None:
            missing += 1
            if index == 1 or index % 50 == 0 or index == total:
                print(f"[progress] {index}/{total}", flush=True)
            continue
        target = output_masks_dir / rel_image.parent / f"{rel_image.name}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
        if index == 1 or index % 50 == 0 or index == total:
            print(f"[progress] {index}/{total}", flush=True)
    return copied, missing


def main(argv: list[str] | None = None) -> int:
    from core.spheresfm_project_cli import main as _main

    return _main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
