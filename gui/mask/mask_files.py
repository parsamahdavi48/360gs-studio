"""File discovery helpers shared by mask preview widgets."""
from __future__ import annotations

from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def iter_image_files(images_dir: str | Path) -> list[Path]:
    """Return image files below images_dir without falling back to cwd."""

    if not images_dir:
        return []
    root = Path(images_dir)
    if not root.is_dir():
        return []

    result: list[Path] = []
    seen: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        key = path_key(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    result.sort(key=lambda x: str(x).lower())
    return result


def mask_candidates_for_image(
    image_path: Path,
    images_dir: str | Path,
    masks_dir: str | Path,
) -> list[Path]:
    """Return candidate generated mask paths for an image, preserving subfolders."""

    if not masks_dir:
        return []
    masks_root = Path(masks_dir)
    if not masks_root.is_dir():
        return []

    rel_parent = Path()
    try:
        images_root = Path(images_dir).resolve()
        rel_parent = image_path.resolve().relative_to(images_root).parent
    except Exception:
        pass

    return [
        masks_root / rel_parent / f"{image_path.stem}.png",
        masks_root / rel_parent / f"{image_path.name}.png",
        masks_root / f"{image_path.name}.png",
        masks_root / f"{image_path.stem}.png",
    ]


def path_key(path: Path) -> str:
    try:
        return str(path.resolve()).lower()
    except Exception:
        return str(path).lower()
