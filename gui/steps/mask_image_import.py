"""Image import helpers for Step 3 mask generation."""
from __future__ import annotations

import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def import_external_images(source_dir: Path, images_dir: Path) -> tuple[int, int]:
    """Copy supported image files into a scene images directory.

    Returns ``(added, skipped)``. Existing destination names are skipped; overwrite
    behavior remains a user responsibility outside this helper.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    added = 0
    skipped = 0
    for src in sorted(source_dir.iterdir(), key=lambda p: p.name.lower()):
        if not src.is_file() or src.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        dst = images_dir / src.name
        try:
            if src.resolve() == dst.resolve():
                skipped += 1
                continue
        except OSError:
            pass
        if dst.exists():
            skipped += 1
            continue
        shutil.copy2(src, dst)
        added += 1
    return added, skipped
