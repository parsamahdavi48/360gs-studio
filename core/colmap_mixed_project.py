from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.colmap_normal_camera_contract import (
    COLMAP_NORMAL_CAMERA_MODEL,
    normal_camera_group_for_image,
    normal_camera_groups_for_images,
)
from core.colmap_rig_export import (
    DEFAULT_RIG_NAME,
    build_rig_config,
    colmap_rig_root,
    pinhole_camera_params,
    prepare_views_for_colmap,
    write_rig_config_payload_json,
)
from core.cubemap_image_conversion import (
    convert_images_colmap_rig,
    make_colmap_rig_jobs,
)
from core.cubemap_view_spec import load_views_json, views_to_dicts
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


@dataclass(frozen=True, slots=True)
class ColmapErpRigGroup:
    rig_name: str
    image_list_name: str
    input_size: tuple[int, int]
    output_size: int
    image_files: tuple[str, ...]
    image_names: tuple[str, ...]
    prepared_views: tuple[dict[str, Any], ...]
    camera_params: tuple[float, ...]


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
    views_json: str | Path | None = None,
    views: list[dict[str, Any]] | None = None,
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

    views = _load_or_normalize_views(views_json=views_json, views=views)
    rig_image_names: list[str] = []
    rig_groups: list[ColmapErpRigGroup] = []
    warnings: list[str] = []

    if erp_images:
        rig_groups = colmap_erp_rig_groups_for_images(
            inventory,
            erp_images,
            views=views,
            output_scale=output_scale,
            output_format=output_format,
            rig_name=rig_name,
        )
        rig_payload: list[dict] = []
        for group in rig_groups:
            prepared_views = [dict(view) for view in group.prepared_views]
            rig_payload.extend(
                build_rig_config(
                    prepared_views,
                    (group.output_size, group.output_size),
                    rig_name=group.rig_name,
                )
            )
        rig_path = write_rig_config_payload_json(output, rig_payload)
        _write_colmap_multi_rig_metadata(
            output_dir=output,
            image_dir=inventory.images_dir,
            mask_dir=inventory.masks_dir,
            groups=rig_groups,
            fov=90.0,
            output_scale=float(output_scale),
            export_images=write_images,
            export_masks=write_masks,
        )
        for group in rig_groups:
            rig_image_names.extend(group.image_names)
            if not (write_images or write_masks):
                continue
            convert_images_colmap_rig(
                image_files=list(group.image_files),
                input_size=group.input_size,
                output_size=group.output_size,
                views=[dict(view) for view in group.prepared_views],
                fov=90.0,
                image_dir=str(inventory.images_dir),
                mask_dir=str(inventory.masks_dir),
                output_dir=str(output),
                rig_name=group.rig_name,
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
        print(f"COLMAP rig images prepared: {len(rig_image_names)} ({rig_path})", flush=True)

    normal_image_names, normal_group_image_names = _link_normal_images(
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
    for group in rig_groups:
        if group.image_list_name == COLMAP_RIG_IMAGE_LIST:
            continue
        _write_image_list(project_dir / group.image_list_name, list(group.image_names))
    for group_id, image_names in sorted(normal_group_image_names.items()):
        _write_image_list(project_dir / _normal_group_image_list_name(group_id), image_names)

    manifest_path = project_dir / COLMAP_MIXED_MANIFEST
    manifest = _manifest(
        project_dir=project_dir,
        inventory=inventory,
        erp_images=erp_images,
        normal_images=normal_images,
        rig_image_names=rig_image_names,
        rig_groups=rig_groups,
        normal_image_names=normal_image_names,
        normal_group_image_names=normal_group_image_names,
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


def _load_or_normalize_views(
    *,
    views_json: str | Path | None,
    views: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if views is None:
        if views_json is None:
            raise ValueError("views_json or views is required")
        return views_to_dicts(load_views_json(views_json))

    normalized: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for index, item in enumerate(views):
        if not isinstance(item, dict):
            raise ValueError(f"views[{index}] must be an object")
        if not bool(item.get("enabled", True)):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            raise ValueError(f"views[{index}].name is required")
        if name in used_names:
            raise ValueError(f"views has duplicated name: {name}")
        try:
            yaw = float(item["yaw"])
            pitch = float(item["pitch"])
        except KeyError as exc:
            raise ValueError(f"views[{index}] missing field: {exc}") from exc
        normalized.append({"name": name, "yaw": yaw, "pitch": pitch})
        used_names.add(name)
    if not normalized:
        raise ValueError("views has no enabled views")
    return normalized


def colmap_erp_rig_groups_for_images(
    inventory: SceneInventory,
    erp_images: list[SceneImage],
    *,
    views: list[dict[str, Any]],
    output_scale: float,
    output_format: str | None,
    rig_name: str = DEFAULT_RIG_NAME,
) -> list[ColmapErpRigGroup]:
    if output_scale <= 0.0:
        raise ValueError("output_scale must be positive")
    prepared_views = prepare_views_for_colmap([{**view, "fov": 90.0} for view in views])
    grouped: OrderedDict[tuple[int, int], list[SceneImage]] = OrderedDict()
    for image in erp_images:
        if image.width <= 0 or image.height <= 0:
            raise ValueError(f"ERP image size is unknown: {image.rel_path}")
        grouped.setdefault((int(image.width), int(image.height)), []).append(image)

    total_groups = len(grouped)
    groups: list[ColmapErpRigGroup] = []
    for index, (input_size, images) in enumerate(grouped.items(), start=1):
        group_rig_name = _rig_name_for_group(rig_name, index, total_groups)
        image_files = [_image_rel_to_images_root(inventory, image) for image in images]
        output_size = max(1, int(round(input_size[1] * float(output_scale))))
        image_names = _rig_image_names(image_files, prepared_views, output_format, group_rig_name)
        groups.append(
            ColmapErpRigGroup(
                rig_name=group_rig_name,
                image_list_name=_rig_group_image_list_name(group_rig_name, total_groups),
                input_size=input_size,
                output_size=output_size,
                image_files=tuple(image_files),
                image_names=tuple(image_names),
                prepared_views=tuple(dict(view) for view in prepared_views),
                camera_params=tuple(pinhole_camera_params(output_size, output_size, 90.0)),
            )
        )
    return groups


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
) -> tuple[list[str], dict[str, list[str]]]:
    names: list[str] = []
    group_names: dict[str, list[str]] = {}
    total = len(images)
    for index, image in enumerate(images, start=1):
        group = normal_camera_group_for_image(image)
        rel = Path(group.image_dir) / _normal_filename(image, index, total)
        image_dest = project_images_dir / rel
        if write_images:
            replace_file_with_link_or_copy(image.path, image_dest)
        elif not image_dest.is_file():
            warnings.append(f"Normal image is not present in project output: {rel.as_posix()}")
            continue
        rel_text = rel.as_posix()
        names.append(rel_text)
        group_names.setdefault(group.group_id, []).append(rel_text)

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
    return names, group_names


def _normal_filename(image: SceneImage, index: int, total: int) -> str:
    parts = Path(image.rel_path).parts
    if parts and parts[0].lower() == "images" and len(parts) > 1:
        parts = parts[1:]
    raw = "__".join(parts) if parts else image.path.name
    name = _safe_path_name(raw, fallback=f"image{image.path.suffix.lower() or '.jpg'}")
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
    rig_groups: list[ColmapErpRigGroup],
    normal_image_names: list[str],
    normal_group_image_names: dict[str, list[str]],
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
        "rig_camera_groups": [
            {
                "id": group.rig_name,
                "rig_name": group.rig_name,
                "image_list": group.image_list_name,
                "camera_model": "PINHOLE",
                "camera_params": list(group.camera_params),
                "width": group.output_size,
                "height": group.output_size,
                "input_width": group.input_size[0],
                "input_height": group.input_size[1],
                "source_count": len(group.image_files),
                "image_count": len(group.image_names),
            }
            for group in rig_groups
        ],
        "normal_image_list": COLMAP_NORMAL_IMAGE_LIST,
        "normal_camera_model": COLMAP_NORMAL_CAMERA_MODEL,
        "normal_camera_groups": [
            {
                "id": group.group_id,
                "image_dir": group.image_dir,
                "image_list": _normal_group_image_list_name(group.group_id),
                "camera_model": group.camera_model,
                "camera_params": list(group.camera_params),
                "camera_source": group.camera_source,
                "width": group.width,
                "height": group.height,
                "source_kind": group.source_kind,
                "source_id": group.source_id,
                "image_count": len(normal_group_image_names.get(group.group_id, ())),
            }
            for group in normal_camera_groups_for_images(normal_images)
        ],
        "source_images_dir": str(inventory.images_dir),
        "source_masks_dir": str(inventory.masks_dir),
        "erp_source_count": len(erp_images),
        "normal_source_count": len(normal_images),
        "rig_image_count": len(rig_image_names),
        "normal_image_count": len(normal_image_names),
        "warnings": warnings,
    }


def _safe_path_name(value: str, *, fallback: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip()).strip("._-")
    return safe or fallback


def _normal_group_image_list_name(group_id: str) -> str:
    return f"normal_image_list_{_safe_path_name(group_id, fallback='group')}.txt"


def _rig_group_image_list_name(rig_name: str, total_groups: int) -> str:
    if total_groups <= 1:
        return COLMAP_RIG_IMAGE_LIST
    return f"rig_image_list_{_safe_path_name(rig_name, fallback='rig')}.txt"


def _rig_name_for_group(base_name: str, index: int, total_groups: int) -> str:
    if total_groups <= 1:
        return base_name
    stripped = base_name.rstrip("0123456789")
    prefix = stripped or base_name
    return f"{prefix}{index}"


def _write_colmap_multi_rig_metadata(
    *,
    output_dir: Path,
    image_dir: Path,
    mask_dir: Path,
    groups: list[ColmapErpRigGroup],
    fov: float,
    output_scale: float,
    export_images: bool,
    export_masks: bool,
) -> None:
    root = colmap_rig_root(output_dir)
    payload = {
        "export_type": "colmap_rig",
        "camera_model": "PINHOLE",
        "fov": float(fov),
        "output_scale": float(output_scale),
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "export_images": bool(export_images),
        "export_masks": bool(export_masks),
        "yaw_offset_per_frame": 0.0,
        "rig_config": "rig_config.json",
        "images_dir": "images",
        "masks_dir": "masks",
        "rig_groups": [
            {
                "rig_name": group.rig_name,
                "image_list": group.image_list_name,
                "input_size": {"w": group.input_size[0], "h": group.input_size[1]},
                "output_size": {"w": group.output_size, "h": group.output_size},
                "camera_params": list(group.camera_params),
                "source_images": list(group.image_files),
                "views": [
                    {
                        "name": view["name"],
                        "camera_name": view["camera_name"],
                        "yaw": float(view["yaw"]),
                        "pitch": float(view["pitch"]),
                    }
                    for view in group.prepared_views
                ],
            }
            for group in groups
        ],
    }
    if len(groups) == 1:
        group = groups[0]
        payload.update(
            {
                "rig_name": group.rig_name,
                "camera_params": list(group.camera_params),
                "input_size": {"w": group.input_size[0], "h": group.input_size[1]},
                "output_size": {"w": group.output_size, "h": group.output_size},
                "views": payload["rig_groups"][0]["views"],
                "source_images": list(group.image_files),
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "view_export_settings.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
