from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.colmap_rig_export import pinhole_camera_params
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


def infer_image_only_frame_output_sizes(
    image_dir: str,
    image_files: list[str],
    output_scale: float,
) -> list[int]:
    """Infer one output face size per source image for mixed-resolution image-only export."""
    fallback_input_size, fallback_output_size = infer_image_only_sizes(image_dir, image_files, output_scale)
    fallback_height = int(fallback_input_size[1])
    sizes: list[int] = []
    for rel in image_files:
        path = Path(image_dir) / rel
        height = fallback_height
        if path.is_file():
            try:
                with Image.open(path) as img:
                    height = int(img.height)
            except Exception:
                height = fallback_height
        sizes.append(max(1, int(round(height * output_scale))) if height > 0 else fallback_output_size)
    return sizes


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
    frame_output_sizes: list[int] | None = None,
) -> dict:
    """Return image-only export metadata for app-side persistence."""
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
    if frame_output_sizes is not None and len(frame_output_sizes) == len(image_files):
        payload["source_output_sizes"] = [
            {"file": file_path, "w": int(size), "h": int(size)}
            for file_path, size in zip(image_files, frame_output_sizes, strict=True)
        ]
        payload["mixed_camera_intrinsics"] = len({int(size) for size in frame_output_sizes}) > 1
    return payload


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
) -> dict:
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
    return payload
