from __future__ import annotations

from pathlib import Path

REALITYSCAN_PRIMARY_IMAGE_DIR = "images"
REALITYSCAN_EXTRA_IMAGE_DIR = "extra_images"
REALITYSCAN_IMAGE_DIR_NAMES = (REALITYSCAN_PRIMARY_IMAGE_DIR, REALITYSCAN_EXTRA_IMAGE_DIR)
REALITYSCAN_LEGACY_MASK_DIR_NAMES = ("masks", "extra_masks")
REALITYSCAN_MASK_DIR_NAMES = REALITYSCAN_LEGACY_MASK_DIR_NAMES

REALITYSCAN_GEOMETRY_LAYER_DIR = "_geometry"
REALITYSCAN_MASK_LAYER_DIR = "_mask"
REALITYSCAN_GEOMETRY_LAYER_NAMES = (".geometry", "_geometry", "@geometry", "#geometry", "!geometry")
REALITYSCAN_MASK_LAYER_NAMES = (".mask", "_mask", "@mask", "#mask", "!mask")
REALITYSCAN_LAYER_DIR_NAMES = REALITYSCAN_GEOMETRY_LAYER_NAMES + REALITYSCAN_MASK_LAYER_NAMES

MASK_SEARCH_EXTENSIONS = (".png", ".jpg", ".jpeg", ".mask.png")


def primary_geometry_dir(output_dir: Path) -> Path:
    return Path(output_dir) / REALITYSCAN_PRIMARY_IMAGE_DIR / REALITYSCAN_GEOMETRY_LAYER_DIR


def primary_mask_dir(output_dir: Path) -> Path:
    return Path(output_dir) / REALITYSCAN_PRIMARY_IMAGE_DIR / REALITYSCAN_MASK_LAYER_DIR


def extra_geometry_dir(output_dir: Path) -> Path:
    return Path(output_dir) / REALITYSCAN_EXTRA_IMAGE_DIR / REALITYSCAN_GEOMETRY_LAYER_DIR


def extra_mask_dir(output_dir: Path) -> Path:
    return Path(output_dir) / REALITYSCAN_EXTRA_IMAGE_DIR / REALITYSCAN_MASK_LAYER_DIR


def geometry_file_path(output_dir: Path, name: str | Path, *, extra: bool = False) -> Path:
    root = extra_geometry_dir(output_dir) if extra else primary_geometry_dir(output_dir)
    return root / Path(name).name


def mask_file_path_for_geometry(image_path: Path) -> Path:
    image_path = Path(image_path)
    if _is_geometry_layer_dir(image_path.parent):
        return image_path.parent.parent / REALITYSCAN_MASK_LAYER_DIR / f"{image_path.stem}.png"
    if _is_image_asset_container(image_path.parent):
        return image_path.parent / REALITYSCAN_MASK_LAYER_DIR / f"{image_path.stem}.png"
    return image_path.parent / REALITYSCAN_MASK_LAYER_DIR / f"{image_path.stem}.png"


def related_realityscan_asset_roots(asset_dir: Path, dir_names: tuple[str, ...]) -> tuple[Path, ...]:
    if _same_names(dir_names, REALITYSCAN_IMAGE_DIR_NAMES):
        return related_realityscan_image_roots(asset_dir)
    if _same_names(dir_names, REALITYSCAN_MASK_DIR_NAMES):
        return related_realityscan_mask_roots(asset_dir)

    asset_dir = Path(asset_dir)
    names = {name.casefold() for name in dir_names}
    roots: list[Path] = [asset_dir]
    if asset_dir.name.casefold() in names:
        roots.extend(asset_dir.parent / name for name in dir_names)
    else:
        roots.extend(asset_dir / name for name in dir_names)
    return _dedupe_paths(roots)


def related_realityscan_image_roots(images_dir: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    base = Path(images_dir)
    for container in _related_image_containers(base):
        roots.append(container / REALITYSCAN_GEOMETRY_LAYER_DIR)
        roots.append(container)
    if _is_geometry_layer_dir(base):
        roots.insert(0, base)
    return _dedupe_paths(roots)


def related_realityscan_mask_roots(masks_dir: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    base = Path(masks_dir)
    for container in _related_image_containers(base):
        roots.append(container / REALITYSCAN_MASK_LAYER_DIR)
    if _is_mask_layer_dir(base):
        roots.insert(0, base)

    root = _realityscan_root_from_any_asset(base)
    if root is not None:
        roots.extend(root / name for name in REALITYSCAN_LEGACY_MASK_DIR_NAMES)
    else:
        roots.append(base)
        if base.name.casefold() in {name.casefold() for name in REALITYSCAN_LEGACY_MASK_DIR_NAMES}:
            roots.extend(base.parent / name for name in REALITYSCAN_LEGACY_MASK_DIR_NAMES)
    return _dedupe_paths(roots)


def strip_leading_realityscan_asset_dir(path: Path, dir_names: tuple[str, ...]) -> Path:
    parts = Path(path).parts
    if not parts:
        return Path(path)
    names = {name.casefold() for name in dir_names}
    start = 0
    if parts[start].casefold() in names and len(parts) > start + 1:
        start += 1
    if start < len(parts) and parts[start].casefold() in _layer_name_set() and len(parts) > start + 1:
        start += 1
    return Path(*parts[start:]) if start else Path(path)


def resolve_realityscan_image_path(images_dir: Path, name: str) -> Path:
    raw = Path(name)
    if raw.is_absolute():
        return raw
    stripped = strip_leading_realityscan_asset_dir(raw, REALITYSCAN_IMAGE_DIR_NAMES)
    seen: set[str] = set()
    for root in related_realityscan_image_roots(images_dir):
        for rel in (raw, stripped):
            candidate = root / rel
            key = candidate.resolve(strict=False).as_posix().casefold()
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists():
                return candidate
    return Path(images_dir) / raw


def realityscan_image_asset_relative_path(path: Path, images_dir: Path) -> Path | None:
    path = Path(path)
    for root in related_realityscan_image_roots(images_dir):
        try:
            rel = path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        return _asset_relative_path_for_root(root, rel, images_dir)
    return None


def mask_lookup_candidates(image_name: str) -> list[Path]:
    raw = Path(str(image_name).replace("\\", "/"))
    bases = [
        raw,
        strip_leading_realityscan_asset_dir(raw, REALITYSCAN_IMAGE_DIR_NAMES),
        strip_leading_realityscan_asset_dir(raw, REALITYSCAN_MASK_DIR_NAMES),
        _replace_layer_dir(raw, REALITYSCAN_GEOMETRY_LAYER_NAMES, REALITYSCAN_MASK_LAYER_DIR),
    ]
    candidates: list[Path] = []
    for image_path in bases:
        if image_path is None:
            continue
        if image_path.name:
            candidates.extend(_mask_candidates_for_base(image_path))
        stripped = strip_leading_realityscan_asset_dir(image_path, REALITYSCAN_IMAGE_DIR_NAMES)
        if stripped != image_path and stripped.name:
            candidates.extend(_mask_candidates_for_base(stripped))
    return _dedupe_relative_paths(candidates)


def _mask_candidates_for_base(image_path: Path) -> list[Path]:
    stem_path = image_path.parent / image_path.stem
    layer_stem_path = _replace_layer_dir(stem_path, REALITYSCAN_GEOMETRY_LAYER_NAMES, REALITYSCAN_MASK_LAYER_DIR)
    candidates = [image_path]
    for base in (stem_path, layer_stem_path):
        if base is None:
            continue
        for ext in MASK_SEARCH_EXTENSIONS:
            candidates.append(base.with_suffix(ext))
    for ext in MASK_SEARCH_EXTENSIONS:
        candidates.append(Path(f"{image_path.as_posix()}{ext}"))
    return candidates


def _related_image_containers(base: Path) -> list[Path]:
    root = _realityscan_root_from_any_asset(base)
    if root is None:
        return [base.parent if _is_layer_dir(base) else base]
    return [root / name for name in REALITYSCAN_IMAGE_DIR_NAMES]


def _realityscan_root_from_any_asset(path: Path) -> Path | None:
    path = Path(path)
    current = path.parent if _is_layer_dir(path) else path
    if _is_image_asset_container(current) or _is_legacy_mask_container(current):
        return current.parent
    if (current / REALITYSCAN_PRIMARY_IMAGE_DIR).exists() or (current / REALITYSCAN_EXTRA_IMAGE_DIR).exists():
        return current
    return None


def _asset_relative_path_for_root(root: Path, rel: Path, images_dir: Path) -> Path:
    root = Path(root)
    container = root.parent if _is_geometry_layer_dir(root) else root
    try:
        container_rel = container.resolve().relative_to(Path(images_dir).resolve())
    except ValueError:
        container_rel = Path(container.name) if _is_image_asset_container(container) else Path()
    if str(container_rel) in {"", "."} and _is_image_asset_container(container):
        container_rel = Path(container.name)
    if _is_geometry_layer_dir(root):
        return container_rel / root.name / rel
    return container_rel / rel


def _replace_layer_dir(path: Path, source_names: tuple[str, ...], target_name: str) -> Path | None:
    parts = list(Path(path).parts)
    sources = {name.casefold() for name in source_names}
    for index, part in enumerate(parts):
        if part.casefold() in sources:
            parts[index] = target_name
            return Path(*parts)
    return None


def _is_image_asset_container(path: Path) -> bool:
    return Path(path).name.casefold() in {name.casefold() for name in REALITYSCAN_IMAGE_DIR_NAMES}


def _is_legacy_mask_container(path: Path) -> bool:
    return Path(path).name.casefold() in {name.casefold() for name in REALITYSCAN_LEGACY_MASK_DIR_NAMES}


def _is_geometry_layer_dir(path: Path) -> bool:
    return Path(path).name.casefold() in {name.casefold() for name in REALITYSCAN_GEOMETRY_LAYER_NAMES}


def _is_mask_layer_dir(path: Path) -> bool:
    return Path(path).name.casefold() in {name.casefold() for name in REALITYSCAN_MASK_LAYER_NAMES}


def _is_layer_dir(path: Path) -> bool:
    return Path(path).name.casefold() in _layer_name_set()


def _layer_name_set() -> set[str]:
    return {name.casefold() for name in REALITYSCAN_LAYER_DIR_NAMES}


def _same_names(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return {name.casefold() for name in left} == {name.casefold() for name in right}


def _dedupe_paths(paths: list[Path]) -> tuple[Path, ...]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = path.resolve(strict=False).as_posix().casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return tuple(deduped)


def _dedupe_relative_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = path.as_posix().casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped
