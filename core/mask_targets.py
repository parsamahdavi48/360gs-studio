"""Shared image-list handling for mask generation CLIs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.path_safety import is_path_inside

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class MaskTarget:
    image_path: Path
    mask_path: Path
    projection: str = ""
    rel_path: str = ""


def normalize_projection(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text in {"equirect", "equirectangular", "360", "360deg", "360°"}:
        return "equirect"
    if text in {"normal", "perspective", "flat"}:
        return "normal"
    return ""


def iter_image_files(images: Path) -> tuple[Path, list[Path]]:
    if images.is_file() and images.suffix.lower() in IMAGE_EXTS:
        return images.parent, [images]
    if not images.is_dir():
        return images, []
    # Worker fallback only. Source-scoped GUI runs pass an image_list manifest
    # so projection/source ownership is resolved before the worker starts.
    files = sorted(
        (path for path in images.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
        key=lambda path: str(path).lower(),
    )
    return images, files


def mask_output_path_for_image(
    image_path: Path,
    images_root: Path,
    masks_dir: Path,
    *,
    add_ext: bool = False,
) -> Path:
    try:
        root = images_root.parent if images_root.is_file() else images_root
        rel_parent = image_path.resolve().relative_to(root.resolve()).parent
    except Exception:
        rel_parent = Path()
    name = f"{image_path.name}.png" if add_ext else f"{image_path.stem}.png"
    return masks_dir / rel_parent / name


def collect_image_targets(
    images: str | Path,
    masks_dir: str | Path,
    *,
    add_ext: bool = False,
    image_list: str | Path | None = None,
    projection_filter: str | None = None,
    allow_cwd_fallback: bool = False,
) -> tuple[Path, list[MaskTarget]]:
    images_root = Path(images)
    masks_root = Path(masks_dir)
    if image_list:
        return images_root, load_image_targets(
            image_list,
            images_root=images_root,
            masks_root=masks_root,
            add_ext=add_ext,
            projection_filter=projection_filter,
            allow_cwd_fallback=allow_cwd_fallback,
        )

    root, image_files = iter_image_files(images_root)
    targets = [
        MaskTarget(
            image_path=path,
            mask_path=mask_output_path_for_image(path, root, masks_root, add_ext=add_ext),
            rel_path=_relative_key(path, root),
        )
        for path in image_files
    ]
    return root, targets


def load_image_targets(
    image_list: str | Path,
    *,
    images_root: Path,
    masks_root: Path,
    add_ext: bool = False,
    projection_filter: str | None = None,
    allow_cwd_fallback: bool = False,
) -> list[MaskTarget]:
    filter_projection = normalize_projection(projection_filter)
    entries = _load_entries(Path(image_list))
    targets: list[MaskTarget] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        parsed = _parse_entry(entry)
        if parsed is None:
            continue
        image_value, mask_value, projection = parsed
        projection = normalize_projection(projection)
        if filter_projection and projection and projection != filter_projection:
            continue
        image_path = _resolve_image_path(
            image_value,
            images_root,
            manifest_base=Path(image_list).parent,
            allow_cwd_fallback=allow_cwd_fallback,
        )
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        mask_path = (
            _resolve_mask_path(mask_value, masks_root)
            if mask_value
            else mask_output_path_for_image(image_path, _target_root(images_root), masks_root, add_ext=add_ext)
        )
        key = (_path_key(image_path), _path_key(mask_path))
        if key in seen:
            continue
        seen.add(key)
        targets.append(
            MaskTarget(
                image_path=image_path,
                mask_path=mask_path,
                projection=projection,
                rel_path=_relative_key(image_path, _target_root(images_root)),
            )
        )
    return targets


def load_mask_paths_from_image_list(
    image_list: str | Path,
    *,
    masks_root: str | Path,
    images_root: str | Path | None = None,
    add_ext: bool = False,
    projection_filter: str | None = None,
    allow_cwd_fallback: bool = False,
) -> list[Path]:
    masks_dir = Path(masks_root)
    images_dir = Path(images_root) if images_root is not None else masks_dir.parent / "images"
    targets = load_image_targets(
        image_list,
        images_root=images_dir,
        masks_root=masks_dir,
        add_ext=add_ext,
        projection_filter=projection_filter,
        allow_cwd_fallback=allow_cwd_fallback,
    )
    return [target.mask_path for target in targets]


def _target_root(path: Path) -> Path:
    return path.parent if path.is_file() else path


def _load_entries(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else []
    if stripped.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return _load_json_lines(text)
        if isinstance(data, dict):
            for key in ("images", "targets", "files", "items"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        return _load_json_lines(text)
    return _load_json_lines(text)


def _load_json_lines(text: str) -> list[Any]:
    entries: list[Any] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            entries.append(line)
    return entries


def _parse_entry(entry: Any) -> tuple[str, str, str] | None:
    if isinstance(entry, str):
        return entry, "", ""
    if not isinstance(entry, dict):
        return None
    image_value = entry.get("image") or entry.get("image_path") or entry.get("path") or entry.get("source") or ""
    if not image_value:
        return None
    mask_value = entry.get("mask") or entry.get("mask_path") or ""
    projection = entry.get("projection") or entry.get("image_type") or ""
    return str(image_value), str(mask_value), str(projection)


def _resolve_image_path(
    value: str,
    images_root: Path,
    *,
    manifest_base: Path,
    allow_cwd_fallback: bool,
) -> Path:
    return _resolve_relative_path(
        value,
        images_root,
        default_base=_target_root(images_root),
        manifest_base=manifest_base,
        allow_cwd_fallback=allow_cwd_fallback,
    )


def _resolve_mask_path(value: str, masks_root: Path) -> Path:
    return _resolve_relative_path(value, masks_root, default_base=masks_root, manifest_base=None)


def _resolve_relative_path(
    value: str,
    root: Path,
    *,
    default_base: Path,
    manifest_base: Path | None,
    allow_cwd_fallback: bool = False,
) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return _require_path_inside(raw, _target_root(root), original=value)

    root_base = _target_root(root)
    parts = raw.parts
    if parts and parts[0].casefold() == root_base.name.casefold():
        candidate = root_base.parent / raw
        if candidate.exists():
            return _require_path_inside(candidate, root_base, original=value)

    candidate = root_base / raw
    if candidate.exists():
        return _require_path_inside(candidate, root_base, original=value)

    if manifest_base is not None:
        manifest_candidate = manifest_base / raw
        if manifest_candidate.exists():
            return _require_path_inside(manifest_candidate, root_base, original=value)

    if allow_cwd_fallback:
        cwd_candidate = Path.cwd() / raw
        if cwd_candidate.exists():
            return _require_path_inside(cwd_candidate, root_base, original=value)

    if parts and parts[0].casefold() == root_base.name.casefold():
        return _require_path_inside(root_base.parent / raw, root_base, original=value)
    return _require_path_inside(default_base / raw, root_base, original=value)


def _require_path_inside(path: Path, root: Path, *, original: str) -> Path:
    if not is_path_inside(path, root, allow_equal=False):
        raise ValueError(f"Image-list path escapes its allowed root: {original}")
    return path


def _relative_key(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.name


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve()).casefold()
    except Exception:
        return str(path).casefold()
