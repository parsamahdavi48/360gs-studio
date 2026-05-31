from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.nerf_dataset_paths import find_nerf_transforms_path


@dataclass(frozen=True, slots=True)
class DatasetMaskPathResult:
    transforms_json: Path
    frame_count: int
    mask_path_count: int
    missing_mask_count: int


def attach_nerf_mask_paths(
    *,
    dataset_root: str | Path,
    transforms_json: str | Path | None = None,
    masks_dir: str | Path | None = None,
    clear_missing: bool = True,
) -> DatasetMaskPathResult:
    root = Path(dataset_root)
    transforms = Path(transforms_json) if transforms_json else find_nerf_transforms_path(root)
    if transforms is None or not transforms.is_file():
        raise FileNotFoundError(f"transforms.json not found under dataset: {root}")
    masks = Path(masks_dir) if masks_dir else root / "masks"

    data = json.loads(transforms.read_text(encoding="utf-8"))
    frames = data.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"transforms.json has no frames list: {transforms}")

    attached = 0
    missing = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        mask = _mask_path_for_frame(root, masks, frame)
        if mask is not None and mask.is_file():
            frame["mask_path"] = mask.relative_to(root).as_posix()
            attached += 1
        else:
            missing += 1
            if clear_missing:
                frame.pop("mask_path", None)

    transforms.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return DatasetMaskPathResult(
        transforms_json=transforms,
        frame_count=len(frames),
        mask_path_count=attached,
        missing_mask_count=missing,
    )


def clear_nerf_mask_paths(
    *,
    dataset_root: str | Path,
    transforms_json: str | Path | None = None,
) -> DatasetMaskPathResult:
    root = Path(dataset_root)
    transforms = Path(transforms_json) if transforms_json else find_nerf_transforms_path(root)
    if transforms is None or not transforms.is_file():
        raise FileNotFoundError(f"transforms.json not found under dataset: {root}")
    data = json.loads(transforms.read_text(encoding="utf-8"))
    frames = data.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"transforms.json has no frames list: {transforms}")
    removed = 0
    for frame in frames:
        if isinstance(frame, dict) and "mask_path" in frame:
            frame.pop("mask_path", None)
            removed += 1
    transforms.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return DatasetMaskPathResult(
        transforms_json=transforms,
        frame_count=len(frames),
        mask_path_count=0,
        missing_mask_count=removed,
    )


def _mask_path_for_frame(root: Path, masks: Path, frame: dict[str, Any]) -> Path | None:
    raw = str(frame.get("file_path") or "").strip()
    if not raw:
        return None
    image_rel = Path(raw)
    if image_rel.is_absolute():
        try:
            image_rel = image_rel.relative_to(root / "images")
        except ValueError:
            image_rel = Path(image_rel.name)
    parts = image_rel.parts
    if parts and parts[0].casefold() == "images":
        image_rel = Path(*parts[1:]) if len(parts) > 1 else Path(image_rel.name)
    return masks / image_rel.with_suffix(".png")
