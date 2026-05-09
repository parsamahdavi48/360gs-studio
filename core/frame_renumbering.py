from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from core.scene_layout import (
    extract_sessions_path,
    mask_items_dir,
    mask_runs_path,
    scene_images_dir,
    scene_masks_dir,
    scene_output_dir,
    source_image_sets_path,
    step4_meta_dir,
)
from core.scene_project import file_identity, load_json, scene_relative, utc_now_iso, write_json


@dataclass(frozen=True)
class FrameRename:
    old_rel: str
    new_rel: str
    old_path: Path
    new_path: Path


def _path_key(path: Path) -> str:
    return str(path.resolve(strict=False)).replace("\\", "/").casefold()


def _rel_key(value: str) -> str:
    return value.replace("\\", "/").strip("/").casefold()


def _has_files(root: Path) -> bool:
    if not root.exists():
        return False
    if root.is_file():
        return True
    return any(path.is_file() for path in root.rglob("*"))


def find_renumber_blockers(scene_dir: Path) -> list[str]:
    blockers: list[str] = []
    if _has_files(scene_masks_dir(scene_dir)):
        blockers.append("masks/ contains files")
    if mask_runs_path(scene_dir).is_file() or _has_files(mask_items_dir(scene_dir)):
        blockers.append("_stechdrive/masks contains metadata")
    if _has_files(scene_output_dir(scene_dir)):
        blockers.append("output/ contains files")
    if _has_files(step4_meta_dir(scene_dir)):
        blockers.append("_stechdrive/step4 contains metadata")
    return blockers


def ensure_can_renumber(scene_dir: Path) -> None:
    blockers = find_renumber_blockers(scene_dir)
    if blockers:
        joined = "; ".join(blockers)
        raise RuntimeError(f"Cannot renumber kept images after downstream outputs exist: {joined}")


def build_renumber_plan(
    scene_dir: Path,
    source_paths: Sequence[Path],
    *,
    allowed_existing_targets: Iterable[Path] = (),
) -> list[FrameRename]:
    total = len(source_paths)
    if total <= 0:
        return []
    images_dir = scene_images_dir(scene_dir)
    digits = max(6, len(str(total)))
    allowed_keys = {_path_key(path) for path in source_paths}
    allowed_keys.update(_path_key(path) for path in allowed_existing_targets)
    seen_targets: set[str] = set()
    plan: list[FrameRename] = []

    for seq, source in enumerate(source_paths, start=1):
        old_rel = scene_relative(scene_dir, source)
        new_path = images_dir / f"frame_{seq:0{digits}d}{source.suffix}"
        new_rel = scene_relative(scene_dir, new_path)
        target_key = _path_key(new_path)
        if target_key in seen_targets:
            raise RuntimeError(f"Duplicate renumber target: {new_rel}")
        seen_targets.add(target_key)
        if new_path.exists() and target_key not in allowed_keys:
            raise RuntimeError(f"Renumber target already exists: {new_rel}")
        plan.append(FrameRename(old_rel=old_rel, new_rel=new_rel, old_path=source, new_path=new_path))
    return plan


def plan_path_mapping(plan: Sequence[FrameRename]) -> dict[str, str]:
    return {
        _rel_key(item.old_rel): item.new_rel
        for item in plan
        if _rel_key(item.old_rel) != _rel_key(item.new_rel)
    }


def plan_source_mapping(plan: Sequence[FrameRename]) -> dict[str, str]:
    return {
        _path_key(item.old_path): item.new_rel
        for item in plan
        if _rel_key(item.old_rel) != _rel_key(item.new_rel)
    }


def rename_records(plan: Sequence[FrameRename]) -> list[dict[str, str]]:
    return [
        {"from": item.old_rel, "to": item.new_rel}
        for item in plan
        if _rel_key(item.old_rel) != _rel_key(item.new_rel)
    ]


def apply_renumber_plan(plan: Sequence[FrameRename]) -> None:
    moves = [item for item in plan if _path_key(item.old_path) != _path_key(item.new_path)]
    if not moves:
        return

    staged: list[tuple[Path, FrameRename]] = []
    completed: list[FrameRename] = []
    try:
        for item in moves:
            temp = item.old_path.with_name(f".{item.old_path.name}.renumber-{uuid.uuid4().hex}.tmp")
            item.old_path.replace(temp)
            staged.append((temp, item))

        for temp, item in staged:
            item.new_path.parent.mkdir(parents=True, exist_ok=True)
            temp.replace(item.new_path)
            completed.append(item)
    except Exception:
        for item in reversed(completed):
            if item.new_path.exists() and not item.old_path.exists():
                item.new_path.replace(item.old_path)
        for temp, item in reversed(staged):
            if temp.exists() and not item.old_path.exists():
                temp.replace(item.old_path)
        raise


def update_frame_path_metadata(scene_dir: Path, plan: Sequence[FrameRename]) -> None:
    mapping = plan_path_mapping(plan)
    if not mapping:
        return
    source_mapping = {
        _path_key(item.old_path): item.new_path
        for item in plan
        if _rel_key(item.old_rel) != _rel_key(item.new_rel)
    }
    _update_extract_sessions(scene_dir, mapping)
    _update_source_image_sets(scene_dir, mapping, source_mapping)


def _update_extract_sessions(scene_dir: Path, mapping: dict[str, str]) -> None:
    path = extract_sessions_path(scene_dir)
    data = load_json(path, {"version": 1, "sessions": []})
    sessions = data.get("sessions")
    if not isinstance(sessions, list):
        return

    changed = False
    for session in sessions:
        if not isinstance(session, dict):
            continue
        files = session.get("output_files")
        if not isinstance(files, list):
            continue
        updated: list[object] = []
        for value in files:
            if isinstance(value, str):
                replacement = mapping.get(_rel_key(value))
                if replacement is not None:
                    updated.append(replacement)
                    changed = True
                    continue
            updated.append(value)
        session["output_files"] = updated

    if changed:
        data["version"] = int(data.get("version") or 1)
        write_json(path, data)


def _source_path_key(scene_dir: Path, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = scene_dir / path
    return _path_key(path)


def _update_source_image_sets(scene_dir: Path, mapping: dict[str, str], source_mapping: dict[str, Path]) -> None:
    path = source_image_sets_path(scene_dir)
    data = load_json(path, {"version": 1, "image_sets": []})
    image_sets = data.get("image_sets")
    if not isinstance(image_sets, list):
        return

    changed = False
    for image_set in image_sets:
        if not isinstance(image_set, dict):
            continue
        files = image_set.get("files")
        if not isinstance(files, list):
            continue
        set_changed = False
        for item in files:
            if not isinstance(item, dict):
                continue
            old_rel = str(item.get("scene_path") or "")
            replacement = mapping.get(_rel_key(old_rel))
            source_path = str(item.get("source_path") or "")
            source_replacement = source_mapping.get(_source_path_key(scene_dir, source_path)) if source_path else None
            if replacement is None and source_replacement is None:
                continue
            if replacement is not None:
                item["scene_path"] = replacement
                item["file"] = file_identity(scene_dir / replacement)
            if source_replacement is not None:
                item["source_path"] = str(source_replacement)
                item["source_file"] = file_identity(source_replacement)
            changed = True
            set_changed = True
        if set_changed:
            image_set["updated_at"] = utc_now_iso()

    if changed:
        data["version"] = int(data.get("version") or 1)
        write_json(path, data)
