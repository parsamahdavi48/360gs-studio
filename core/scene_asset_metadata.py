from __future__ import annotations

from pathlib import Path
from typing import Any

from core.cancellation import CancellationToken, raise_if_cancelled
from core.mask_metadata import mask_file_summary, summary_size
from core.projection_contract import normalize_projection
from core.scene_layout import scene_asset_metadata_path, scene_images_dir, scene_masks_dir
from core.scene_project import (
    file_identity,
    image_header_info,
    load_json,
    scene_relative,
    utc_now_iso,
    write_json,
)

SCENE_ASSET_METADATA_SCHEMA_VERSION = 1


def load_scene_asset_metadata(scene_dir: str | Path) -> dict[str, Any]:
    return load_json(
        scene_asset_metadata_path(Path(scene_dir)),
        {
            "version": SCENE_ASSET_METADATA_SCHEMA_VERSION,
            "images": {},
            "masks": {},
        },
    )


def rebuild_scene_asset_metadata(
    scene_dir: str | Path,
    *,
    cancel_event: CancellationToken | None = None,
) -> dict[str, Any]:
    """Reprobe scene images/masks and write the app-internal asset ledger."""

    raise_if_cancelled(cancel_event)
    from core import scene_inventory

    scene = Path(scene_dir)
    scene_inventory.clear_scene_inventory_cache()
    inventory = scene_inventory.build_scene_inventory(scene)
    raise_if_cancelled(cancel_event)
    return save_scene_asset_metadata_from_inventory(inventory)


def save_scene_asset_metadata_from_inventory(inventory: Any) -> dict[str, Any]:
    scene = Path(inventory.scene_dir)
    images_dir = Path(inventory.images_dir)
    masks_dir = Path(inventory.masks_dir)
    if not _is_standard_scene_root(scene, images_dir, scene_images_dir(scene)):
        return load_scene_asset_metadata(scene)
    if not _is_standard_scene_root(scene, masks_dir, scene_masks_dir(scene)):
        return load_scene_asset_metadata(scene)

    images: dict[str, dict[str, Any]] = {}
    masks: dict[str, dict[str, Any]] = {}
    for image in inventory.images:
        image_rel = _normalize_rel(image.rel_path)
        images[image_rel] = {
            "path": image_rel,
            "file": relative_file_identity(scene, image.path),
            "readable": bool(image.width > 0 and image.height > 0),
            "width": int(image.width),
            "height": int(image.height),
            "suffix": str(image.suffix or Path(image.path).suffix).lower(),
            "projection": normalize_projection(image.projection),
            "projection_source": str(image.projection_source or ""),
        }
        if image.mask is None or not image.mask.exists:
            continue
        mask_rel = _normalize_rel(image.mask.rel_path)
        masks[image_rel] = {
            "image": image_rel,
            "mask": mask_rel,
            "file": relative_file_identity(scene, image.mask.path),
            "readable": bool(image.mask.readable),
            "width": int(image.mask.width),
            "height": int(image.mask.height),
            "polarity": str(image.mask.polarity or "white_keep"),
        }

    payload = {
        "version": SCENE_ASSET_METADATA_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "images_root": "images",
        "masks_root": "masks",
        "images": images,
        "masks": masks,
    }
    write_json(scene_asset_metadata_path(scene), payload)
    return payload


def update_scene_asset_mask_metadata(
    scene_dir: str | Path,
    *,
    image_path: str | Path,
    mask_path: str | Path,
) -> None:
    scene = Path(scene_dir)
    image = Path(image_path)
    mask = Path(mask_path)
    if not _is_under_standard_root(scene, image, scene_images_dir(scene)):
        return
    if not _is_under_standard_root(scene, mask, scene_masks_dir(scene)):
        return

    payload = load_scene_asset_metadata(scene)
    images = payload.get("images")
    if not isinstance(images, dict):
        images = {}
    masks = payload.get("masks")
    if not isinstance(masks, dict):
        masks = {}

    image_rel = _normalize_rel(scene_relative(scene, image))
    mask_rel = _normalize_rel(scene_relative(scene, mask))
    summary = mask_file_summary(mask)
    size = summary_size(summary)
    width, height = size if size is not None else (0, 0)
    masks[image_rel] = {
        "image": image_rel,
        "mask": mask_rel,
        "file": relative_file_identity(scene, mask),
        "readable": bool(summary.get("readable")),
        "width": int(width),
        "height": int(height),
        "polarity": "white_keep",
    }
    payload["version"] = SCENE_ASSET_METADATA_SCHEMA_VERSION
    payload["updated_at"] = utc_now_iso()
    payload["images_root"] = "images"
    payload["masks_root"] = "masks"
    payload["images"] = images
    payload["masks"] = masks
    write_json(scene_asset_metadata_path(scene), payload)


def current_image_metadata(payload: dict[str, Any], scene_dir: str | Path, image_path: str | Path) -> dict[str, Any] | None:
    images = payload.get("images")
    if not isinstance(images, dict):
        return None
    if not images:
        return None
    scene = Path(scene_dir)
    path = Path(image_path)
    rel = _normalize_scene_rel(scene, path)
    record = images.get(rel)
    if not isinstance(record, dict):
        return None
    if not relative_file_identity_matches(record.get("file"), scene, path):
        return None
    return record


def current_mask_metadata_for_image(
    payload: dict[str, Any],
    scene_dir: str | Path,
    image_rel: str,
    mask_path: str | Path,
) -> dict[str, Any] | None:
    masks = payload.get("masks")
    if not isinstance(masks, dict):
        return None
    if not masks:
        return None
    scene = Path(scene_dir)
    mask = Path(mask_path)
    record = masks.get(_normalize_rel(image_rel))
    if not isinstance(record, dict):
        return None
    if _normalize_rel(str(record.get("mask") or "")) != _normalize_scene_rel(scene, mask):
        return None
    if not relative_file_identity_matches(record.get("file"), scene, mask):
        return None
    return record


def probe_image_metadata(scene_dir: str | Path, image_path: str | Path) -> dict[str, Any]:
    scene = Path(scene_dir)
    path = Path(image_path)
    header = image_header_info(path)
    return {
        "path": _normalize_rel(scene_relative(scene, path)),
        "file": relative_file_identity(scene, path),
        "readable": bool(int(header.get("width") or 0) > 0 and int(header.get("height") or 0) > 0),
        "width": int(header.get("width") or 0),
        "height": int(header.get("height") or 0),
        "mode": str(header.get("mode") or ""),
        "suffix": path.suffix.lower(),
        "projection": normalize_projection(header.get("detected_projection")),
        "projection_source": "image_header",
    }


def relative_file_identity(scene_dir: str | Path, path: str | Path) -> dict[str, Any]:
    scene = Path(scene_dir)
    identity = file_identity(path)
    identity["path"] = _normalize_scene_rel(scene, Path(path))
    return identity


def relative_file_identity_matches(recorded: object, scene_dir: str | Path, path: str | Path) -> bool:
    if not isinstance(recorded, dict):
        return False
    current = relative_file_identity(scene_dir, path)
    keys = ("path", "exists", "is_file", "is_dir", "size", "mtime_ns")
    return all(recorded.get(key) == current.get(key) for key in keys)


def _is_standard_scene_root(scene: Path, path: Path, expected: Path) -> bool:
    try:
        return path.resolve(strict=False) == expected.resolve(strict=False)
    except OSError:
        return False


def _is_under_standard_root(scene: Path, path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False


def _normalize_rel(value: str) -> str:
    return str(value or "").replace("\\", "/").strip("/")


def _normalize_scene_rel(scene: Path, path: Path) -> str:
    try:
        return path.relative_to(scene).as_posix()
    except ValueError:
        pass
    return _normalize_rel(scene_relative(scene, path))
