from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.colmap_rig_export import (
    DEFAULT_RIG_NAME,
    colmap_rig_root,
    prepare_views_for_colmap,
    write_rig_config_json,
)
from core.cubemap_transforms_json import (
    convert_images_colmap_rig,
    infer_image_only_sizes,
    load_custom_views,
    make_colmap_rig_jobs,
    write_colmap_rig_metadata,
)
from core.dataset_writer_colmap import replace_file_with_link_or_copy
from core.scene_inventory import SceneImage, SceneInventory, build_scene_inventory
from core.sfm_input_plan import (
    SFM_ACTION_EXPAND_ERP_TO_RIG_VIEWS,
    SFM_ACTION_LINK_OR_COPY_NORMAL_IMAGE,
    build_colmap_mixed_sfm_input_plan,
)

COLMAP_MIXED_MANIFEST = "stechdrive_colmap_mixed_project.json"
COLMAP_RIG_IMAGE_LIST = "rig_image_list.txt"
COLMAP_NORMAL_IMAGE_LIST = "normal_image_list.txt"
COLMAP_NORMAL_CAMERA_MODEL = "SIMPLE_RADIAL"


@dataclass(frozen=True, slots=True)
class ColmapMixedProjectResult:
    project_dir: Path
    rig_image_list: Path
    normal_image_list: Path
    manifest_path: Path
    erp_source_count: int
    normal_source_count: int
    rig_image_count: int
    normal_image_count: int
    warnings: tuple[str, ...] = ()


def prepare_colmap_mixed_project(
    scene_dir: str | Path,
    output_dir: str | Path,
    *,
    views_json: str | Path,
    output_scale: float,
    output_format: str = "auto",
    output_bit_depth: str = "8",
    jpg_quality: int = 95,
    write_images: bool = True,
    write_masks: bool = True,
    invert_masks: bool = False,
    workers: str | int = "auto",
    remap_cache_limit: str | int = "auto",
    rig_name: str = DEFAULT_RIG_NAME,
) -> ColmapMixedProjectResult:
    scene = Path(scene_dir)
    output = Path(output_dir)
    inventory = build_scene_inventory(scene)
    plan = build_colmap_mixed_sfm_input_plan(inventory)
    if plan.issues:
        details = "; ".join(issue.message for issue in plan.issues)
        raise ValueError(f"COLMAP mixed project input is invalid: {details}")

    by_rel = {image.rel_path: image for image in inventory.images}
    erp_images = _images_for_action(by_rel, plan.items_for_action(SFM_ACTION_EXPAND_ERP_TO_RIG_VIEWS))
    normal_images = _images_for_action(by_rel, plan.items_for_action(SFM_ACTION_LINK_OR_COPY_NORMAL_IMAGE))

    project_dir = colmap_rig_root(output)
    project_images_dir = project_dir / "images"
    project_masks_dir = project_dir / "masks"
    project_dir.mkdir(parents=True, exist_ok=True)

    views = load_custom_views(str(views_json))
    rig_image_names: list[str] = []
    warnings: list[str] = []

    if erp_images:
        erp_files = [_image_rel_to_images_root(inventory, image) for image in erp_images]
        input_size, output_size = infer_image_only_sizes(str(inventory.images_dir), erp_files, output_scale)
        prepared_views = prepare_views_for_colmap([{**view, "fov": 90.0} for view in views])
        rig_path = write_rig_config_json(output, prepared_views, (output_size, output_size), rig_name=rig_name)
        write_colmap_rig_metadata(
            output_dir=str(output),
            image_dir=str(inventory.images_dir),
            mask_dir=str(inventory.masks_dir),
            image_files=erp_files,
            prepared_views=prepared_views,
            fov=90.0,
            output_scale=float(output_scale),
            input_size=input_size,
            output_size=output_size,
            rig_name=rig_name,
            export_images=write_images,
            export_masks=write_masks,
        )
        if write_images or write_masks:
            convert_images_colmap_rig(
                image_files=erp_files,
                input_size=input_size,
                output_size=output_size,
                views=prepared_views,
                fov=90.0,
                image_dir=str(inventory.images_dir),
                mask_dir=str(inventory.masks_dir),
                output_dir=str(output),
                rig_name=rig_name,
                mask_from_alpha=False,
                invert_masks=invert_masks,
                output_format=output_format,
                output_bit_depth=output_bit_depth,
                jpg_quality=jpg_quality,
                export_images=write_images,
                export_masks=write_masks,
                workers=workers,
                remap_cache_limit=remap_cache_limit,
            )
        rig_image_names = _rig_image_names(erp_files, prepared_views, output_format, rig_name)
        print(f"COLMAP rig images prepared: {len(rig_image_names)} ({rig_path})", flush=True)

    normal_image_names = _link_normal_images(
        normal_images,
        project_images_dir,
        project_masks_dir,
        write_images=write_images,
        write_masks=write_masks,
        warnings=warnings,
    )

    rig_list = project_dir / COLMAP_RIG_IMAGE_LIST
    normal_list = project_dir / COLMAP_NORMAL_IMAGE_LIST
    _write_image_list(rig_list, rig_image_names)
    _write_image_list(normal_list, normal_image_names)

    manifest_path = project_dir / COLMAP_MIXED_MANIFEST
    manifest = _manifest(
        project_dir=project_dir,
        inventory=inventory,
        erp_images=erp_images,
        normal_images=normal_images,
        rig_image_names=rig_image_names,
        normal_image_names=normal_image_names,
        warnings=warnings,
        rig_name=rig_name,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        "COLMAP mixed project prepared: "
        f"{len(erp_images)} ERP source(s), {len(normal_images)} normal source(s)",
        flush=True,
    )
    return ColmapMixedProjectResult(
        project_dir=project_dir,
        rig_image_list=rig_list,
        normal_image_list=normal_list,
        manifest_path=manifest_path,
        erp_source_count=len(erp_images),
        normal_source_count=len(normal_images),
        rig_image_count=len(rig_image_names),
        normal_image_count=len(normal_image_names),
        warnings=tuple(warnings),
    )


def _images_for_action(
    by_rel: dict[str, SceneImage],
    items: tuple[Any, ...],
) -> list[SceneImage]:
    images: list[SceneImage] = []
    for item in items:
        image = by_rel.get(item.image_rel_path)
        if image is not None:
            images.append(image)
    return images


def _image_rel_to_images_root(inventory: SceneInventory, image: SceneImage) -> str:
    try:
        return image.path.resolve().relative_to(inventory.images_dir.resolve()).as_posix()
    except Exception:
        parts = Path(image.rel_path).parts
        if parts and parts[0].lower() == "images" and len(parts) > 1:
            return Path(*parts[1:]).as_posix()
        return Path(image.rel_path).name


def _rig_image_names(
    erp_files: list[str],
    prepared_views: list[dict[str, Any]],
    output_format: str,
    rig_name: str,
) -> list[str]:
    names: list[str] = []
    jobs = make_colmap_rig_jobs(erp_files, output_format)
    for _source, output_filename in jobs:
        for view in prepared_views:
            names.append(f"{rig_name}/{view['camera_name']}/{output_filename}")
    return names


def _link_normal_images(
    images: list[SceneImage],
    project_images_dir: Path,
    project_masks_dir: Path,
    *,
    write_images: bool,
    write_masks: bool,
    warnings: list[str],
) -> list[str]:
    names: list[str] = []
    total = len(images)
    for index, image in enumerate(images, start=1):
        rel = Path("normal") / _normal_group_name(image) / _normal_filename(image, index, total)
        image_dest = project_images_dir / rel
        if write_images:
            replace_file_with_link_or_copy(image.path, image_dest)
        elif not image_dest.is_file():
            warnings.append(f"Normal image is not present in project output: {rel.as_posix()}")
            continue
        names.append(rel.as_posix())

        if not write_masks:
            continue
        if image.mask is None or not image.mask.exists:
            continue
        if not image.mask.readable:
            warnings.append(f"Skipped unreadable normal mask: {image.mask.rel_path}")
            continue
        if not image.mask.matches_image_size:
            warnings.append(f"Skipped size-mismatched normal mask: {image.mask.rel_path}")
            continue
        mask_dest = project_masks_dir / Path(f"{rel.as_posix()}.png")
        replace_file_with_link_or_copy(image.mask.path, mask_dest)
    return names


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(value: str, *, fallback: str) -> str:
    text = _SAFE_NAME_RE.sub("_", value.strip()).strip("._-")
    return text or fallback


def _normal_group_name(image: SceneImage) -> str:
    source = image.source_id or image.source_kind or "normal"
    source = _safe_name(source, fallback="source")
    width = image.width if image.width > 0 else 0
    height = image.height if image.height > 0 else 0
    return f"{source}_{width}x{height}"


def _normal_filename(image: SceneImage, index: int, total: int) -> str:
    parts = Path(image.rel_path).parts
    if parts and parts[0].lower() == "images" and len(parts) > 1:
        parts = parts[1:]
    raw = "__".join(parts) if parts else image.path.name
    name = _safe_name(raw, fallback=f"image{image.path.suffix.lower() or '.jpg'}")
    digits = max(5, len(str(max(1, total))))
    return f"normal_{index:0{digits}d}_{name}"


def _write_image_list(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(names)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _manifest(
    *,
    project_dir: Path,
    inventory: SceneInventory,
    erp_images: list[SceneImage],
    normal_images: list[SceneImage],
    rig_image_names: list[str],
    normal_image_names: list[str],
    warnings: list[str],
    rig_name: str,
) -> dict[str, Any]:
    return {
        "app": "stechdrive-3dgs-utils",
        "export_type": "colmap_mixed_project",
        "project_dir": str(project_dir),
        "images_dir": "images",
        "masks_dir": "masks",
        "database": "database.db",
        "sparse_dir": "sparse",
        "rig_config": "rig_config.json" if erp_images else "",
        "rig_name": rig_name if erp_images else "",
        "rig_image_list": COLMAP_RIG_IMAGE_LIST,
        "normal_image_list": COLMAP_NORMAL_IMAGE_LIST,
        "normal_camera_model": COLMAP_NORMAL_CAMERA_MODEL,
        "source_images_dir": str(inventory.images_dir),
        "source_masks_dir": str(inventory.masks_dir),
        "erp_source_count": len(erp_images),
        "normal_source_count": len(normal_images),
        "rig_image_count": len(rig_image_names),
        "normal_image_count": len(normal_image_names),
        "warnings": warnings,
    }
