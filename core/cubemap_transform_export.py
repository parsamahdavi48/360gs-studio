from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from core.cubemap_image_io import resolve_output_ext
from core.cubemap_remap import rot4, rotation_matrix
from core.orientation_correction import (
    FINAL_ORIENTATION_NONE,
    FINAL_ORIENTATION_STAGE_CUBEMAP_CLI,
    final_orientation_is_applied,
    final_orientation_matrix,
    final_orientation_writes_pointcloud,
    mark_final_orientation,
    normalize_final_orientation,
    resolve_pointcloud_path,
    write_final_orientation_pointcloud,
)


def rotation_angle_diff(r1: np.ndarray, r2: np.ndarray) -> float:
    r = r1.T @ r2
    cos_theta = (np.trace(r) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def make_output_file_path(file_path: str, view_name: str, output_format: str | None = None) -> str:
    root, ext = os.path.splitext(file_path)
    if ext:
        out_ext = resolve_output_ext(ext, output_format)
        return f"{root}_{view_name}{out_ext}"
    return f"{file_path}_{view_name}"


def frame_yaw_offset(frame_index: int, step_deg: float) -> float:
    if step_deg == 0.0:
        return 0.0
    return (float(frame_index) * float(step_deg)) % 360.0


def transform_json(
    input_dir: str,
    input_json: str,
    image_dir: str,
    output_dir: str,
    views: list[dict],
    fov: float,
    output_scale: float,
    no_transform: bool,
    allow_duplicate: bool,
    brush_mode: bool = False,
    yaw_offset_per_frame: float = 0.0,
    final_orientation: str = FINAL_ORIENTATION_NONE,
    output_format: str | None = None,
) -> tuple[list[str], list[float], tuple[int, int], int]:
    json_path = os.path.join(input_dir, input_json)
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        return [], [], (0, 0), 0

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if data.get("camera_model") != "EQUIRECTANGULAR":
        print("Error: camera_model is not EQUIRECTANGULAR")
        return [], [], (0, 0), 0

    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        print("Error: frames in transforms.json is empty")
        return [], [], (0, 0), 0

    input_size = (7840, 3920)
    output_size = max(1, int(round(input_size[1] * output_scale)))

    for frame in frames:
        file_path = frame.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            continue
        probe = os.path.join(image_dir, file_path)
        if os.path.exists(probe):
            with Image.open(probe) as first_img:
                input_size = first_img.size
            output_size = max(1, int(round(input_size[1] * output_scale)))
            break

    if no_transform:
        axis_transform = np.eye(4)
    else:
        axis_transform = rot4(np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]))  # for Postshot/Brush
        if brush_mode:
            brush_rot = rot4(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))  # for Brush
            axis_transform = brush_rot @ axis_transform

    final_orientation = normalize_final_orientation(final_orientation)
    final_orientation_already_applied = final_orientation_is_applied(data, final_orientation)
    if final_orientation != FINAL_ORIENTATION_NONE and not final_orientation_already_applied:
        axis_transform = final_orientation_matrix(final_orientation) @ axis_transform

    new_frames: list[dict] = []
    image_files: list[str] = []
    frame_yaw_offsets: list[float] = []
    image_map: dict[str, np.ndarray] = {}

    for frame in frames:
        file_path = frame.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            print("Skipped frame without file_path")
            continue

        try:
            t = np.array(frame["transform_matrix"], dtype=float)
        except Exception:
            print(f"Skipped frame with invalid transform_matrix: {file_path}")
            continue

        if t.shape != (4, 4):
            print(f"Skipped frame with non 4x4 transform_matrix: {file_path}")
            continue

        if not allow_duplicate and file_path in image_map:
            r_diff = rotation_angle_diff(image_map[file_path][:3, :3], t[:3, :3])
            t_diff = image_map[file_path][:3, 3] - t[:3, 3]
            print(
                "Skipped duplicated image: "
                f"{file_path} (diff={np.rad2deg(r_diff):.3f} deg, {np.linalg.norm(t_diff):.4f} dist.)"
            )
            continue

        t_world = axis_transform @ t

        frame_index = len(image_files)
        yaw_offset = frame_yaw_offset(frame_index, yaw_offset_per_frame)

        image_map[file_path] = t
        image_files.append(file_path)
        frame_yaw_offsets.append(yaw_offset)

        for view_index, view in enumerate(views):
            view_name = view["name"]
            yaw = float(view["yaw"]) + yaw_offset
            pitch = view["pitch"]

            new_frame: dict = {
                "file_path": make_output_file_path(file_path, view_name, output_format),
                "source_file_path": file_path,
                "source_image_index": frame_index,
                "view_name": view_name,
                "view_index": view_index,
                "yaw_offset_deg": yaw_offset,
            }

            r = rotation_matrix(yaw, pitch, True)
            t_face = t_world @ rot4(r.T)
            new_frame["transform_matrix"] = t_face.tolist()

            new_frames.append(new_frame)

    focal = output_size / 2.0 / np.tan(np.deg2rad(fov) / 2.0)
    principal = (output_size - 1) / 2.0
    out = {
        "camera_model": "PINHOLE",
        "w": output_size,
        "h": output_size,
        "fl_x": focal,
        "fl_y": focal,
        "cx": principal,
        "cy": principal,
        "frames": new_frames,
    }
    if final_orientation != FINAL_ORIENTATION_NONE:
        mark_final_orientation(out, final_orientation, FINAL_ORIENTATION_STAGE_CUBEMAP_CLI)
        if final_orientation_writes_pointcloud(final_orientation):
            ply_source = resolve_pointcloud_path(Path(input_dir), data.get("ply_file_path"))
            if ply_source is None:
                print("Warning: final orientation requested, but input ply_file_path was not found")
            else:
                ply_dest = Path(output_dir) / "pointcloud.ply"
                write_final_orientation_pointcloud(
                    ply_source,
                    ply_dest,
                    final_orientation,
                    already_applied=final_orientation_already_applied,
                )
                out["ply_file_path"] = ply_dest.name
    elif data.get("ply_file_path"):
        out["ply_file_path"] = data["ply_file_path"]

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "transforms.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved transforms.json in {output_dir}")

    return image_files, frame_yaw_offsets, input_size, output_size
