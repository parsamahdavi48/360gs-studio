from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.scene_inventory import SceneImage
from core.scene_layout import selected_frames_path, source_image_sets_path
from core.scene_project import load_json

MASK_SOURCE_ALL = "all"
_SOURCE_KEY_SEPARATOR = "\x1f"


@dataclass(frozen=True, slots=True)
class MaskSourceOption:
    key: str
    label: str
    source_kind: str
    source_id: str
    image_count: int


def source_scope_key(source_kind: object, source_id: object) -> str:
    kind = str(source_kind or "unknown").strip() or "unknown"
    source = str(source_id or "").strip()
    return f"{kind}{_SOURCE_KEY_SEPARATOR}{source}"


def source_scope_key_for_image(image: SceneImage) -> str:
    return source_scope_key(image.source_kind, image.source_id)


def build_mask_source_options(scene_dir: str | Path, images: tuple[SceneImage, ...]) -> list[MaskSourceOption]:
    scene = Path(scene_dir)
    labels = _source_label_map(scene)
    counts: dict[str, int] = {}
    first_source: dict[str, tuple[str, str]] = {}
    for image in images:
        key = source_scope_key_for_image(image)
        counts[key] = counts.get(key, 0) + 1
        first_source.setdefault(key, (image.source_kind or "unknown", image.source_id or ""))

    options: list[MaskSourceOption] = []
    for key, count in counts.items():
        source_kind, source_id = first_source[key]
        label = labels.get(key) or _fallback_source_label(source_kind, source_id)
        options.append(
            MaskSourceOption(
                key=key,
                label=label,
                source_kind=source_kind,
                source_id=source_id,
                image_count=count,
            )
        )
    return sorted(options, key=lambda option: (option.label.casefold(), option.source_kind, option.source_id))


def filter_images_by_source(images: tuple[SceneImage, ...], source_key: str) -> list[SceneImage]:
    if not source_key or source_key == MASK_SOURCE_ALL:
        return list(images)
    return [image for image in images if source_scope_key_for_image(image) == source_key]


def _source_label_map(scene: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    labels.update(_image_set_label_map(scene))
    labels.update(_selected_frame_label_map(scene))
    return labels


def _selected_frame_label_map(scene: Path) -> dict[str, str]:
    path = selected_frames_path(scene)
    if not path.is_file():
        return {}
    labels: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kind = str(row.get("source_type") or "video_extract").strip() or "video_extract"
                source_id = str(row.get("source_session") or row.get("import_id") or "").strip()
                key = source_scope_key(kind, source_id)
                label = _label_from_selected_row(row)
                if label and key not in labels:
                    labels[key] = label
    except OSError:
        return {}
    return labels


def _label_from_selected_row(row: dict[str, Any]) -> str:
    source_label = str(row.get("source_label") or "").strip()
    if source_label:
        return source_label
    source_video = str(row.get("source_video") or "").strip()
    if source_video:
        return Path(source_video).name
    source_id = str(row.get("source_session") or row.get("import_id") or "").strip()
    return source_id


def _image_set_label_map(scene: Path) -> dict[str, str]:
    data = load_json(source_image_sets_path(scene), {"image_sets": []})
    image_sets = data.get("image_sets")
    if not isinstance(image_sets, list):
        return {}
    labels: dict[str, str] = {}
    for image_set in image_sets:
        if not isinstance(image_set, dict):
            continue
        kind = str(image_set.get("source_type") or "external_images").strip() or "external_images"
        source_id = str(image_set.get("id") or "").strip()
        label = _label_from_image_set(image_set)
        if label:
            labels[source_scope_key(kind, source_id)] = label
    return labels


def _label_from_image_set(image_set: dict[str, Any]) -> str:
    source_dir = str(image_set.get("source_dir") or "").strip()
    if source_dir:
        return Path(source_dir).name or source_dir
    source_id = str(image_set.get("id") or "").strip()
    return source_id


def _fallback_source_label(source_kind: str, source_id: str) -> str:
    if source_id:
        return source_id
    return source_kind or "unknown"
