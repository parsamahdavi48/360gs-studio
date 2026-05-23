from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.scene_project import file_identity, load_json, mask_item_path, scene_relative

MASK_SCOPE_MISSING = "missing"
MASK_SCOPE_STALE = "stale"
MASK_SCOPE_ALL = "all"
MASK_SCOPES = {MASK_SCOPE_MISSING, MASK_SCOPE_STALE, MASK_SCOPE_ALL}


@dataclass(frozen=True, slots=True)
class MaskRefreshPlan:
    targets: tuple[Path, ...]
    current: tuple[Path, ...]
    protected: tuple[Path, ...]


def normalize_mask_scope(value: str | None) -> str:
    scope = str(value or "").strip().lower()
    return scope if scope in MASK_SCOPES else MASK_SCOPE_MISSING


def build_mask_refresh_plan(
    *,
    scene_dir: str | Path,
    image_paths: list[Path] | tuple[Path, ...],
    mask_path_for_image: Callable[[Path], Path],
    settings: dict[str, Any],
    scope: str,
) -> MaskRefreshPlan:
    normalized_scope = normalize_mask_scope(scope)
    scene = Path(scene_dir)
    targets: list[Path] = []
    current: list[Path] = []
    protected: list[Path] = []

    for image_path in image_paths:
        image = Path(image_path)
        mask = mask_path_for_image(image)
        if normalized_scope == MASK_SCOPE_ALL:
            targets.append(image)
            continue
        if not mask.is_file():
            targets.append(image)
            continue
        if normalized_scope == MASK_SCOPE_MISSING:
            current.append(image)
            continue

        item = _load_mask_item(scene, image)
        if not item:
            protected.append(image)
            continue
        if not _file_identity_matches(item.get("mask_file"), mask):
            protected.append(image)
            continue
        if not _file_identity_matches(item.get("image_file"), image):
            targets.append(image)
            continue
        if not _settings_equal(item.get("settings"), settings):
            targets.append(image)
            continue
        current.append(image)

    return MaskRefreshPlan(tuple(targets), tuple(current), tuple(protected))


def _load_mask_item(scene_dir: Path, image_path: Path) -> dict[str, Any]:
    rel = scene_relative(scene_dir, image_path)
    return load_json(mask_item_path(scene_dir, rel), {})


def _file_identity_matches(recorded: object, path: Path) -> bool:
    if not isinstance(recorded, dict):
        return False
    current = file_identity(path)
    keys = ("path", "exists", "is_file", "is_dir", "size", "mtime_ns")
    return all(recorded.get(key) == current.get(key) for key in keys)


def _settings_equal(recorded: object, current: dict[str, Any]) -> bool:
    if not isinstance(recorded, dict):
        return False
    return _stable_json(recorded) == _stable_json(current)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
