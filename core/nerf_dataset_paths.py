from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_PROFILE_RE = re.compile(r"[^a-z0-9]+")
_PROFILE_TRANSFORMS_RE = re.compile(r"^transforms_([a-z0-9_]+)\.json$", re.IGNORECASE)
_DEFAULT_TRANSFORMS_NAMES = ("transforms.json", "transforms_train.json")


def nerf_profile_suffix(value: object, *, fallback: str = "custom") -> str:
    raw = str(value or "").strip().lower()
    suffix = _PROFILE_RE.sub("_", raw).strip("_")
    return suffix or fallback


def transforms_name_for_profile(profile: object) -> str:
    return f"transforms_{nerf_profile_suffix(profile)}.json"


def pointcloud_name_for_profile(profile: object) -> str:
    return f"pointcloud_{nerf_profile_suffix(profile)}.ply"


def profile_from_transforms_name(path: str | Path) -> str:
    name = Path(path).name
    match = _PROFILE_TRANSFORMS_RE.match(name)
    return nerf_profile_suffix(match.group(1)) if match else ""


def iter_nerf_transforms_paths(
    root: str | Path,
    *,
    preferred_profiles: Iterable[object] = (),
) -> tuple[Path, ...]:
    dataset_root = Path(root)
    candidates: list[Path] = []
    for profile in preferred_profiles:
        suffix = nerf_profile_suffix(profile, fallback="")
        if suffix:
            candidates.append(dataset_root / transforms_name_for_profile(suffix))
    candidates.extend(dataset_root / name for name in _DEFAULT_TRANSFORMS_NAMES)
    candidates.extend(sorted(dataset_root.glob("transforms_*.json")))
    return tuple(path for path in _dedupe_paths(candidates) if path.is_file())


def find_nerf_transforms_path(
    root: str | Path,
    *,
    preferred_profiles: Iterable[object] = (),
) -> Path | None:
    return next(iter(iter_nerf_transforms_paths(root, preferred_profiles=preferred_profiles)), None)


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def declared_pointcloud_path(root: str | Path, transforms_json: str | Path) -> Path | None:
    dataset_root = Path(root)
    transforms = Path(transforms_json)
    data = load_json_object(transforms)
    raw = str(data.get("ply_file_path") or "").strip()
    if raw:
        candidate = _resolve_dataset_path(raw, transforms.parent)
        return candidate if candidate.is_file() else None
    source = data.get("source")
    if isinstance(source, dict):
        raw = str(source.get("raw_metashape_pointcloud_path") or source.get("external_pointcloud_path") or "").strip()
        if raw:
            candidate = _resolve_dataset_path(raw, transforms.parent)
            return candidate if candidate.is_file() else None
    profile = profile_from_transforms_name(transforms)
    if profile:
        candidate = dataset_root / pointcloud_name_for_profile(profile)
        if candidate.is_file():
            return candidate
    default = dataset_root / "pointcloud.ply"
    return default if default.is_file() else None


def find_nerf_pointcloud_path(
    root: str | Path,
    *,
    transforms_json: str | Path | None = None,
    preferred_profiles: Iterable[object] = (),
) -> Path | None:
    dataset_root = Path(root)
    transforms = Path(transforms_json) if transforms_json is not None else find_nerf_transforms_path(
        dataset_root,
        preferred_profiles=preferred_profiles,
    )
    if transforms is not None:
        declared = declared_pointcloud_path(dataset_root, transforms)
        if declared is not None:
            return declared
    for profile in preferred_profiles:
        suffix = nerf_profile_suffix(profile, fallback="")
        if not suffix:
            continue
        candidate = dataset_root / pointcloud_name_for_profile(suffix)
        if candidate.is_file():
            return candidate
    default = dataset_root / "pointcloud.ply"
    return default if default.is_file() else None


def _resolve_dataset_path(raw: str, base: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else base / path


def _dedupe_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = str(path.resolve()).casefold()
        except OSError:
            key = str(path.absolute()).casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return tuple(result)
