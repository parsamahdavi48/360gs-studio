"""COLMAP rig export helpers.

The GUI exports perspective views from one equirectangular frame as a fixed
multi-camera rig. COLMAP's ``rig_configurator`` reads the JSON produced here
and assigns one sensor per view folder.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

COLMAP_RIG_DIRNAME = "colmap_rig"
COLMAP_IMAGES_DIRNAME = "images"
COLMAP_MASKS_DIRNAME = "masks"
DEFAULT_RIG_NAME = "rig1"
DEFAULT_CAMERA_PREFIX = "cam"
DEFAULT_FRAME_PREFIX = "frame"
DEFAULT_FRAME_DIGITS = 5


def _view_sort_key(view: dict) -> tuple[float, float, str]:
    return (
        float(view.get("pitch", 0.0)),
        float(view.get("yaw", 0.0)),
        str(view.get("name", "")),
    )


def sort_views_for_colmap(views: list[dict]) -> list[dict]:
    return sorted(views, key=_view_sort_key)


def camera_name_for_index(index: int, total_cameras: int, prefix: str = DEFAULT_CAMERA_PREFIX) -> str:
    digits = max(2, len(str(max(1, total_cameras))))
    return f"{prefix}{index:0{digits}d}"


def prepare_views_for_colmap(views: list[dict], camera_prefix: str = DEFAULT_CAMERA_PREFIX) -> list[dict]:
    sorted_views = sort_views_for_colmap(views)
    total = len(sorted_views)
    prepared: list[dict] = []
    for idx, view in enumerate(sorted_views, start=1):
        item = dict(view)
        item["camera_index"] = idx
        item["camera_name"] = camera_name_for_index(idx, total, camera_prefix)
        prepared.append(item)
    return prepared


def colmap_rig_root(output_dir: str | Path) -> Path:
    return Path(output_dir) / COLMAP_RIG_DIRNAME


def colmap_images_root(output_dir: str | Path) -> Path:
    return colmap_rig_root(output_dir) / COLMAP_IMAGES_DIRNAME


def colmap_masks_root(output_dir: str | Path) -> Path:
    return colmap_rig_root(output_dir) / COLMAP_MASKS_DIRNAME


def colmap_camera_image_dir(output_dir: str | Path, rig_name: str, camera_name: str) -> Path:
    return colmap_images_root(output_dir) / rig_name / camera_name


def colmap_camera_mask_dir(output_dir: str | Path, rig_name: str, camera_name: str) -> Path:
    return colmap_masks_root(output_dir) / rig_name / camera_name


def colmap_image_prefix(rig_name: str, camera_name: str) -> str:
    return f"{rig_name}/{camera_name}/"


def rig_config_path(output_dir: str | Path) -> Path:
    return colmap_rig_root(output_dir) / "rig_config.json"


def frame_filename(index: int, total_frames: int, ext: str) -> str:
    digits = max(DEFAULT_FRAME_DIGITS, len(str(max(1, total_frames))))
    suffix = ext if ext.startswith(".") else f".{ext}"
    return f"{DEFAULT_FRAME_PREFIX}_{index:0{digits}d}{suffix.lower()}"


def _quat_multiply(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = q
    return (w, -x, -y, -z)


def cam_from_rig_rotation_quaternion(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> list[float]:
    yaw_rad = math.radians(float(yaw_deg))
    pitch_rad = math.radians(float(pitch_deg))
    roll_rad = math.radians(float(roll_deg))

    q_yaw = (math.cos(yaw_rad / 2.0), 0.0, math.sin(yaw_rad / 2.0), 0.0)
    q_pitch = (math.cos(pitch_rad / 2.0), math.sin(pitch_rad / 2.0), 0.0, 0.0)
    q_roll = (math.cos(roll_rad / 2.0), 0.0, 0.0, math.sin(roll_rad / 2.0))

    q_rig_from_cam = _quat_multiply(_quat_multiply(q_yaw, q_pitch), q_roll)
    q_cam_from_rig = _quat_conjugate(q_rig_from_cam)
    return [q_cam_from_rig[0], q_cam_from_rig[1], q_cam_from_rig[2], q_cam_from_rig[3]]


def pinhole_camera_params(width: int, height: int, fov_deg: float) -> list[float]:
    safe_fov = max(1e-6, min(float(fov_deg), 179.999))
    half_fov_rad = math.radians(safe_fov) / 2.0
    focal = 0.5 / math.tan(half_fov_rad)
    return [
        focal * float(width),
        focal * float(height),
        (float(width) - 1.0) / 2.0,
        (float(height) - 1.0) / 2.0,
    ]


def build_rig_config(
    prepared_views: list[dict],
    output_size: tuple[int, int],
    *,
    rig_name: str = DEFAULT_RIG_NAME,
) -> list[dict]:
    cameras: list[dict] = []
    for idx, view in enumerate(prepared_views, start=1):
        camera_name = str(
            view.get("camera_name") or camera_name_for_index(idx, len(prepared_views))
        )
        camera = {
            "image_prefix": colmap_image_prefix(rig_name, camera_name),
            "camera_model_name": "PINHOLE",
            "camera_params": pinhole_camera_params(
                output_size[0],
                output_size[1],
                float(view.get("fov", 90.0)),
            ),
        }
        if idx == 1:
            camera["ref_sensor"] = True
        else:
            camera["cam_from_rig_rotation"] = cam_from_rig_rotation_quaternion(
                float(view.get("yaw", 0.0)),
                float(view.get("pitch", 0.0)),
            )
            camera["cam_from_rig_translation"] = [0.0, 0.0, 0.0]
        cameras.append(camera)
    return [{"cameras": cameras}]


def write_rig_config_json(
    output_dir: str | Path,
    prepared_views: list[dict],
    output_size: tuple[int, int],
    *,
    rig_name: str = DEFAULT_RIG_NAME,
) -> Path:
    root = colmap_rig_root(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = rig_config_path(output_dir)
    payload = build_rig_config(prepared_views, output_size, rig_name=rig_name)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
