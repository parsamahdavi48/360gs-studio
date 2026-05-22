from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from core.mask_metadata import mask_file_summary, summary_size
from core.projection_contract import (
    PROJECTION_EQUIRECTANGULAR,
    PROJECTION_NORMAL,
    PROJECTION_UNKNOWN,
    normalize_projection,
)
from core.scene_import_contracts import IMAGE_EXTS
from core.scene_layout import scene_images_dir, scene_masks_dir, selected_frames_path, source_image_sets_path
from core.scene_project import image_header_info, load_json, scene_image_projection_map

MASK_POLARITY_WHITE_KEEP = "white_keep"


@dataclass(frozen=True, slots=True)
class MaskArtifact:
    path: Path
    rel_path: str
    exists: bool
    readable: bool
    width: int
    height: int
    matches_image_size: bool
    polarity: str = MASK_POLARITY_WHITE_KEEP

    @property
    def size(self) -> tuple[int, int] | None:
        if self.width <= 0 or self.height <= 0:
            return None
        return self.width, self.height


@dataclass(frozen=True, slots=True)
class SceneImage:
    path: Path
    rel_path: str
    width: int
    height: int
    suffix: str
    projection: str
    projection_source: str
    source_kind: str
    source_id: str
    sequence_index: int | None
    mask: MaskArtifact | None

    @property
    def size(self) -> tuple[int, int] | None:
        if self.width <= 0 or self.height <= 0:
            return None
        return self.width, self.height


@dataclass(frozen=True, slots=True)
class SceneInventory:
    scene_dir: Path
    images_dir: Path
    masks_dir: Path
    images: tuple[SceneImage, ...]

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def projections(self) -> set[str]:
        return {image.projection for image in self.images}

    @property
    def image_sizes(self) -> set[tuple[int, int]]:
        return {image.size for image in self.images if image.size is not None}

    @property
    def projection_counts(self) -> Counter[str]:
        return Counter(image.projection for image in self.images)

    @property
    def size_counts(self) -> Counter[tuple[int, int]]:
        return Counter(image.size for image in self.images if image.size is not None)

    @property
    def missing_masks(self) -> tuple[SceneImage, ...]:
        return tuple(image for image in self.images if image.mask is None or not image.mask.exists)

    @property
    def mismatched_masks(self) -> tuple[SceneImage, ...]:
        return tuple(
            image
            for image in self.images
            if image.mask is not None and image.mask.exists and image.mask.readable and not image.mask.matches_image_size
        )

    def equirectangular_images(self) -> tuple[SceneImage, ...]:
        return tuple(image for image in self.images if image.projection == PROJECTION_EQUIRECTANGULAR)

    def normal_images(self) -> tuple[SceneImage, ...]:
        return tuple(image for image in self.images if image.projection == PROJECTION_NORMAL)


def build_scene_inventory(
    scene_dir: str | Path,
    *,
    images_dir: str | Path | None = None,
    masks_dir: str | Path | None = None,
) -> SceneInventory:
    scene = Path(scene_dir)
    images_root = Path(images_dir) if images_dir is not None else scene_images_dir(scene)
    masks_root = Path(masks_dir) if masks_dir is not None else scene_masks_dir(scene)
    image_paths = _iter_image_files(images_root)
    projection_map = scene_image_projection_map(scene, image_paths) if image_paths else {}
    selected_map = _selected_frame_metadata(scene)
    image_set_map = _image_set_metadata(scene)

    images: list[SceneImage] = []
    for path in image_paths:
        rel_path = _inventory_rel_path(scene, images_root, path)
        width, height = _image_size(path)
        metadata = _metadata_for(rel_path, selected_map, image_set_map)
        projection = normalize_projection(projection_map.get(rel_path) or metadata.get("projection") or "")
        if projection == PROJECTION_UNKNOWN:
            projection = normalize_projection(image_header_info(path).get("detected_projection"))
        projection_source = str(metadata.get("projection_source") or "project")
        source_kind = str(metadata.get("source_kind") or "unknown")
        source_id = str(metadata.get("source_id") or "")
        sequence_index = _optional_int(metadata.get("sequence_index"))
        mask = _mask_artifact(scene, images_root, masks_root, path, image_size=(width, height))
        images.append(
            SceneImage(
                path=path,
                rel_path=rel_path,
                width=width,
                height=height,
                suffix=path.suffix.lower(),
                projection=projection,
                projection_source=projection_source,
                source_kind=source_kind,
                source_id=source_id,
                sequence_index=sequence_index,
                mask=mask,
            )
        )

    return SceneInventory(scene_dir=scene, images_dir=images_root, masks_dir=masks_root, images=tuple(images))


def _iter_image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
        key=lambda path: str(path).lower(),
    )


def _image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return 0, 0


def _inventory_rel_path(scene: Path, images_root: Path, image_path: Path) -> str:
    try:
        resolved = image_path.resolve()
        scene_root = scene.resolve()
        return resolved.relative_to(scene_root).as_posix()
    except Exception:
        pass
    try:
        return image_path.resolve().relative_to(images_root.resolve()).as_posix()
    except Exception:
        return image_path.name


def _mask_artifact(
    scene: Path,
    images_root: Path,
    masks_root: Path,
    image_path: Path,
    *,
    image_size: tuple[int, int],
) -> MaskArtifact | None:
    if not masks_root.is_dir():
        return None
    for candidate in _mask_candidates(images_root, masks_root, image_path):
        if not candidate.is_file():
            continue
        summary = mask_file_summary(candidate)
        size = summary_size(summary)
        width, height = size if size is not None else (0, 0)
        return MaskArtifact(
            path=candidate,
            rel_path=_mask_rel_path(scene, masks_root, candidate),
            exists=True,
            readable=bool(summary.get("readable")),
            width=width,
            height=height,
            matches_image_size=size == image_size,
        )
    return None


def _mask_candidates(images_root: Path, masks_root: Path, image_path: Path) -> list[Path]:
    try:
        rel = image_path.resolve().relative_to(images_root.resolve())
    except Exception:
        rel = Path(image_path.name)
    parent = rel.parent
    return [
        masks_root / parent / f"{rel.stem}.png",
        masks_root / parent / f"{rel.name}.png",
        masks_root / rel,
    ]


def _mask_rel_path(scene: Path, masks_root: Path, mask_path: Path) -> str:
    try:
        return mask_path.resolve().relative_to(scene.resolve()).as_posix()
    except Exception:
        pass
    try:
        return mask_path.resolve().relative_to(masks_root.resolve()).as_posix()
    except Exception:
        return mask_path.name


def _selected_frame_metadata(scene: Path) -> dict[str, dict[str, Any]]:
    csv_path = selected_frames_path(scene)
    if not csv_path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel = str(row.get("output_file") or "").replace("\\", "/").strip("/")
                if not rel:
                    continue
                result[rel] = {
                    "source_kind": row.get("source_type") or "video_extract",
                    "source_id": row.get("source_session") or row.get("import_id") or "",
                    "sequence_index": row.get("final_index") or row.get("seq") or "",
                }
    except OSError:
        return {}
    return result


def _image_set_metadata(scene: Path) -> dict[str, dict[str, Any]]:
    data = load_json(source_image_sets_path(scene), {"image_sets": []})
    image_sets = data.get("image_sets")
    if not isinstance(image_sets, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for image_set in image_sets:
        if not isinstance(image_set, dict):
            continue
        set_kind = str(image_set.get("source_type") or "external_images")
        set_id = str(image_set.get("id") or "")
        files = image_set.get("files")
        if not isinstance(files, list):
            continue
        for item in files:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("scene_path") or "").replace("\\", "/").strip("/")
            if not rel:
                continue
            result[rel] = {
                "source_kind": set_kind,
                "source_id": set_id,
                "sequence_index": item.get("sequence_index"),
                "projection": item.get("projection") or item.get("detected_projection") or image_set.get("projection"),
                "projection_source": item.get("projection_source") or image_set.get("projection_source") or "image_header",
            }
    return result


def _metadata_for(
    rel_path: str,
    selected_map: dict[str, dict[str, Any]],
    image_set_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    data.update(image_set_map.get(rel_path, {}))
    data.update(selected_map.get(rel_path, {}))
    return data


def _optional_int(value: object) -> int | None:
    try:
        parsed = int(str(value or "").strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
