from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image

from core.colmap_rig_export import colmap_rig_root, pinhole_camera_params
from core.cubemap_image_io import RAW_IMAGE_EXTS


def collect_image_files(image_dir: str) -> list[str]:
    """Collect input equirectangular images relative to ``image_dir`` for image-only export."""
    root = Path(image_dir)
    if not root.is_dir():
        return []
    files = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in RAW_IMAGE_EXTS]
    return sorted(files, key=lambda x: x.lower())


def infer_image_only_sizes(
    image_dir: str,
    image_files: list[str],
    output_scale: float,
) -> tuple[tuple[int, int], int]:
    """Infer source equirectangular size and output face size from the first readable image."""
    for rel in image_files:
        path = Path(image_dir) / rel
        if not path.is_file():
            continue
        with Image.open(path) as img:
            input_size = img.size
        return input_size, max(1, int(round(input_size[1] * output_scale)))
    return (7840, 3920), max(1, int(round(3920 * output_scale)))


def write_image_only_metadata(
    output_dir: str,
    image_dir: str,
    mask_dir: str,
    image_files: list[str],
    views: list[dict],
    fov: float,
    output_scale: float,
    input_size: tuple[int, int],
    output_size: int,
    yaw_offset_per_frame: float,
    export_images: bool = True,
    export_masks: bool = True,
) -> None:
    """Write a small manifest for SfM-oriented image-only exports."""
    payload = {
        "export_type": "image_only",
        "camera_model": "PINHOLE",
        "fov": float(fov),
        "input_size": {"w": int(input_size[0]), "h": int(input_size[1])},
        "output_size": {"w": int(output_size), "h": int(output_size)},
        "output_scale": float(output_scale),
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "export_images": bool(export_images),
        "export_masks": bool(export_masks),
        "yaw_offset_per_frame": float(yaw_offset_per_frame),
        "views": [{"name": v["name"], "yaw": float(v["yaw"]), "pitch": float(v["pitch"])} for v in views],
        "source_images": image_files,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "view_export_settings.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_colmap_rig_metadata(
    output_dir: str,
    image_dir: str,
    mask_dir: str,
    image_files: list[str],
    prepared_views: list[dict],
    fov: float,
    output_scale: float,
    input_size: tuple[int, int],
    output_size: int,
    rig_name: str,
    export_images: bool = True,
    export_masks: bool = True,
) -> None:
    root = colmap_rig_root(output_dir)
    payload = {
        "export_type": "colmap_rig",
        "camera_model": "PINHOLE",
        "camera_params": pinhole_camera_params(output_size, output_size, fov),
        "fov": float(fov),
        "rig_name": rig_name,
        "input_size": {"w": int(input_size[0]), "h": int(input_size[1])},
        "output_size": {"w": int(output_size), "h": int(output_size)},
        "output_scale": float(output_scale),
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "export_images": bool(export_images),
        "export_masks": bool(export_masks),
        "yaw_offset_per_frame": 0.0,
        "rig_config": "rig_config.json",
        "images_dir": "images",
        "masks_dir": "masks",
        "views": [
            {
                "name": v["name"],
                "camera_name": v["camera_name"],
                "yaw": float(v["yaw"]),
                "pitch": float(v["pitch"]),
            }
            for v in prepared_views
        ],
        "source_images": image_files,
    }
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "view_export_settings.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
