"""Temporary projection helpers for AprilTag detection."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from core.cubemap_image_conversion import convert_images
from core.cubemap_transform_export import transform_json
from core.cubemap_view_spec import make_default_cube6_views, views_to_dicts


@dataclass(frozen=True)
class EquirectProjectionConfig:
    transforms_json: Path
    output_dir: Path
    image_root: Path | None = None
    output_scale: float = 0.5
    fov: float = 90.0
    yaw: float = 45.0
    workers: str | int | None = "auto"
    remap_cache_limit: str | int | None = "auto"


def camera_model(transforms_json: Path) -> str:
    data = json.loads(transforms_json.read_text(encoding="utf-8"))
    return str(data.get("camera_model") or "")


def prepare_equirect_detection_dataset(config: EquirectProjectionConfig) -> Path:
    """Project an EQUIRECTANGULAR transforms dataset to temporary pinhole views."""
    if camera_model(config.transforms_json) != "EQUIRECTANGULAR":
        raise ValueError("Temporary projection expects an EQUIRECTANGULAR transforms.json")
    if config.output_scale <= 0.0 or config.output_scale > 1.0:
        raise ValueError("output_scale must be in (0, 1.0]")
    if config.fov <= 0.0 or config.fov >= 180.0:
        raise ValueError("fov must be in (0, 180)")

    input_dir = config.transforms_json.parent
    input_json = config.transforms_json.name
    image_root = config.image_root or input_dir
    output_dir = config.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    views = views_to_dicts(make_default_cube6_views(config.yaw, 0.0, no_top=False, no_bottom=False))
    image_files, frame_yaw_offsets, input_size, output_size = transform_json(
        input_dir=str(input_dir),
        input_json=input_json,
        image_dir=str(image_root),
        output_dir=str(output_dir),
        views=views,
        fov=config.fov,
        output_scale=config.output_scale,
        no_transform=True,
        allow_duplicate=False,
        brush_mode=False,
        yaw_offset_per_frame=0.0,
        output_format="png",
    )
    if not image_files:
        raise ValueError("No equirectangular frames were available for temporary projection")

    convert_images(
        image_files=image_files,
        input_size=input_size,
        output_size=output_size,
        views=views,
        fov=config.fov,
        image_dir=str(image_root),
        mask_dir="",
        output_image_dir=str(output_dir / "images"),
        output_mask_dir=str(output_dir / "masks"),
        mask_from_alpha=False,
        invert_masks=False,
        output_format="png",
        output_bit_depth="8",
        jpg_quality=95,
        frame_yaw_offsets=frame_yaw_offsets,
        export_images=True,
        export_masks=False,
        workers=config.workers,
        remap_cache_limit=config.remap_cache_limit,
    )
    return output_dir / "transforms.json"
