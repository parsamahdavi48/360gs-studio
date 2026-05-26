"""Apply an estimated metric scale to generated dataset geometry."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from core.apriltag_colmap_dataset import (
    scale_colmap_images_txt,
    scale_colmap_points3d_txt,
    validate_colmap_apriltag_dataset,
)
from core.nerf_dataset_paths import find_nerf_pointcloud_path, find_nerf_transforms_path
from core.scene_layout import scene_output_dir, step4_export_settings_path


@dataclass(frozen=True)
class ScaleOutputDataset:
    transforms_json: Path
    pointcloud_ply: Path | None
    frame_count: int
    checked_image_count: int
    kind: str = "transforms"
    root: Path | None = None
    sparse_dir: Path | None = None
    images_dir: Path | None = None
    geometry_label: str = "transforms.json"
    pointcloud_label: str = "pointcloud"
    can_apply_scale: bool = True

    @property
    def estimation_input(self) -> Path:
        if self.kind == "colmap":
            return self.root or self.transforms_json.parent
        return self.transforms_json


@dataclass(frozen=True)
class ScaleApplyResult:
    transforms_json: Path
    transforms_backup: Path
    pointcloud_ply: Path | None
    pointcloud_backup: Path | None
    scale: float
    frames_scaled: int
    points_scaled: int
    kind: str = "transforms"
    geometry_label: str = "transforms.json"
    pointcloud_label: str = "pointcloud"


def _load_transforms(transforms_json: Path) -> dict:
    try:
        data = json.loads(transforms_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid transforms.json: {transforms_json}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"transforms.json must contain a JSON object: {transforms_json}")
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames in transforms.json is empty")
    return data


def _resolve_relative(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def _pointcloud_path(transforms_json: Path, data: dict) -> Path | None:
    raw = data.get("ply_file_path")
    if isinstance(raw, str) and raw.strip():
        path = _resolve_relative(raw.strip(), transforms_json.parent)
        if not path.is_file():
            raise ValueError(f"Point cloud referenced by transforms.json was not found: {path}")
        return path
    return find_nerf_pointcloud_path(transforms_json.parent, transforms_json=transforms_json)


def _iter_frame_image_paths(transforms_json: Path, data: dict) -> tuple[Path, ...]:
    paths: list[Path] = []
    for frame in data.get("frames", []):
        if not isinstance(frame, dict):
            continue
        file_path = frame.get("file_path")
        if isinstance(file_path, str) and file_path.strip():
            paths.append(_resolve_relative(file_path.strip(), transforms_json.parent))
    return tuple(paths)


def _configured_output_dir(scene: Path) -> Path:
    settings_path = step4_export_settings_path(scene)
    if not settings_path.is_file():
        return scene_output_dir(scene)
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return scene_output_dir(scene)
    if not isinstance(settings, dict):
        return scene_output_dir(scene)
    output_dir = str(settings.get("output_dir") or "").strip()
    if output_dir:
        path = Path(output_dir)
        return path if path.is_absolute() else scene / path
    portable = settings.get("portable_output")
    if isinstance(portable, dict):
        portable_root = str(portable.get("root") or "").strip()
        if portable_root:
            return scene / portable_root
    return scene_output_dir(scene)


def validate_scale_output_dataset(
    scene_dir: Path,
    *,
    output_dir: Path | None = None,
    max_image_checks: int = 12,
) -> ScaleOutputDataset:
    scene = Path(scene_dir)
    if not scene.is_dir():
        raise ValueError(f"Scene folder was not found: {scene}")
    output = Path(output_dir) if output_dir is not None else _configured_output_dir(scene)
    if not output.is_dir():
        raise ValueError(f"Output folder was not found: {output}")
    transforms_json = find_nerf_transforms_path(output)
    if transforms_json is None:
        try:
            colmap = validate_colmap_apriltag_dataset(output, max_image_checks=max_image_checks)
        except Exception as exc:
            raise ValueError(
                f"Output folder must contain a projected NeRF transforms.json or COLMAP images/sparse dataset: {output}"
            ) from exc
        return ScaleOutputDataset(
            transforms_json=colmap.sparse_dir / "images.txt",
            pointcloud_ply=colmap.sparse_dir / "points3D.txt",
            frame_count=colmap.frame_count,
            checked_image_count=colmap.checked_image_count,
            kind="colmap",
            root=colmap.root,
            sparse_dir=colmap.sparse_dir,
            images_dir=colmap.images_dir,
            geometry_label="COLMAP images.txt",
            pointcloud_label="COLMAP points3D.txt",
            can_apply_scale=colmap.text_model,
        )
    data = _load_transforms(transforms_json)
    camera_model = str(data.get("camera_model") or "")
    if camera_model not in {"PINHOLE", "SIMPLE_PINHOLE"}:
        raise ValueError(
            "AprilTag scale estimation requires projected Cubemap output with PINHOLE/SIMPLE_PINHOLE "
            "transforms.json. Run Step 4 with Cubemap image output first."
        )
    pointcloud = _pointcloud_path(transforms_json, data)

    image_paths = _iter_frame_image_paths(transforms_json, data)
    if not image_paths:
        raise ValueError("No frame image paths were found in transforms.json")
    missing = [path for path in image_paths[: max(1, int(max_image_checks))] if not path.is_file()]
    if missing:
        raise ValueError(f"Frame image referenced by transforms.json was not found: {missing[0]}")
    return ScaleOutputDataset(
        transforms_json=transforms_json,
        pointcloud_ply=pointcloud,
        frame_count=len(data.get("frames", [])),
        checked_image_count=min(len(image_paths), max(1, int(max_image_checks))),
        root=output,
        geometry_label=transforms_json.name,
        pointcloud_label=pointcloud.name if pointcloud is not None else "pointcloud",
    )


def _validate_scale(scale: float) -> float:
    value = float(scale)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("scale must be a positive finite number")
    return value


def _default_backup_dir(transforms_json: Path) -> Path:
    output_dir = transforms_json.parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"apriltag_scale_backup_{stamp}"
    if not base.exists():
        return base
    index = 2
    while True:
        candidate = output_dir / f"{base.name}_{index}"
        if not candidate.exists():
            return candidate
        index += 1


def _copy_backup(path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / path.name
    if backup.exists():
        stem = backup.stem
        suffix = backup.suffix
        index = 2
        while backup.exists():
            backup = backup_dir / f"{stem}_{index}{suffix}"
            index += 1
    shutil.copy2(path, backup)
    return backup


def _scaled_transforms_data(data: dict, scale: float) -> tuple[dict, int]:
    copied = json.loads(json.dumps(data))
    frames_scaled = 0
    for frame in copied.get("frames", []):
        if not isinstance(frame, dict):
            continue
        try:
            matrix = np.asarray(frame.get("transform_matrix"), dtype=float)
        except Exception as exc:
            raise ValueError(f"Invalid transform_matrix for {frame.get('file_path', '-')}") from exc
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"Invalid transform_matrix for {frame.get('file_path', '-')}")
        matrix[:3, 3] *= scale
        frame["transform_matrix"] = matrix.tolist()
        frames_scaled += 1
    if frames_scaled <= 0:
        raise ValueError("No valid frames were available to scale")
    return copied, frames_scaled


def _scale_pointcloud_with_open3d(ply_path: Path, scale: float) -> int | None:
    try:
        import open3d as o3d  # type: ignore
    except Exception:
        return None
    pc = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pc.points)
    if points.size == 0:
        return 0
    pc.points = o3d.utility.Vector3dVector(points * scale)
    if not o3d.io.write_point_cloud(str(ply_path), pc):
        raise ValueError(f"Failed to write scaled point cloud: {ply_path}")
    return int(points.shape[0])


def _scale_pointcloud_ascii_fallback(ply_path: Path, scale: float) -> int:
    raw = ply_path.read_text(encoding="ascii")
    lines = raw.splitlines()
    header_end = None
    vertex_count = 0
    vertex_properties: list[str] = []
    in_vertex = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "end_header":
            header_end = index
            break
        parts = stripped.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            vertex_count = int(parts[2])
            in_vertex = True
            vertex_properties = []
            continue
        if parts[:2] == ["element", "face"]:
            in_vertex = False
        if in_vertex and len(parts) >= 3 and parts[0] == "property":
            vertex_properties.append(parts[-1])
    if header_end is None:
        raise ValueError(f"PLY header missing end_header: {ply_path}")
    if not any(line.strip() == "format ascii 1.0" for line in lines[: header_end + 1]):
        raise ValueError("Binary PLY scaling requires open3d")
    try:
        x_index = vertex_properties.index("x")
        y_index = vertex_properties.index("y")
        z_index = vertex_properties.index("z")
    except ValueError as exc:
        raise ValueError(f"PLY missing x/y/z vertex properties: {ply_path}") from exc

    first = header_end + 1
    last = first + vertex_count
    if len(lines) < last:
        raise ValueError(f"Truncated vertex data in PLY: {ply_path}")
    updated = list(lines)
    for row in range(first, last):
        parts = updated[row].split()
        if len(parts) <= max(x_index, y_index, z_index):
            raise ValueError(f"Invalid vertex row in PLY: {ply_path}")
        for axis_index in (x_index, y_index, z_index):
            parts[axis_index] = f"{float(parts[axis_index]) * scale:.9g}"
        updated[row] = " ".join(parts)
    trailing_newline = "\n" if raw.endswith("\n") else ""
    ply_path.write_text("\n".join(updated) + trailing_newline, encoding="ascii")
    return vertex_count


def _scale_pointcloud(ply_path: Path, scale: float) -> int:
    points = _scale_pointcloud_with_open3d(ply_path, scale)
    if points is not None:
        return points
    return _scale_pointcloud_ascii_fallback(ply_path, scale)


def apply_scale_to_transforms_and_pointcloud(
    transforms_json: Path,
    scale: float,
    *,
    backup_dir: Path | None = None,
) -> ScaleApplyResult:
    value = _validate_scale(scale)
    transforms_json = Path(transforms_json)
    if not transforms_json.is_file():
        raise ValueError(f"transforms.json was not found: {transforms_json}")
    data = _load_transforms(transforms_json)
    pointcloud = _pointcloud_path(transforms_json, data)
    target_backup_dir = backup_dir or _default_backup_dir(transforms_json)

    transforms_backup = _copy_backup(transforms_json, target_backup_dir)
    pointcloud_backup = _copy_backup(pointcloud, target_backup_dir) if pointcloud is not None else None

    scaled, frames_scaled = _scaled_transforms_data(data, value)
    transforms_json.write_text(json.dumps(scaled, indent=2), encoding="utf-8")

    points_scaled = 0
    if pointcloud is not None:
        try:
            points_scaled = _scale_pointcloud(pointcloud, value)
        except Exception:
            shutil.copy2(transforms_backup, transforms_json)
            if pointcloud_backup is not None:
                shutil.copy2(pointcloud_backup, pointcloud)
            raise

    return ScaleApplyResult(
        transforms_json=transforms_json,
        transforms_backup=transforms_backup,
        pointcloud_ply=pointcloud,
        pointcloud_backup=pointcloud_backup,
        scale=value,
        frames_scaled=frames_scaled,
        points_scaled=points_scaled,
        kind="transforms",
        geometry_label=transforms_json.name,
        pointcloud_label=pointcloud.name if pointcloud is not None else "pointcloud",
    )


def apply_scale_to_colmap_dataset(dataset_root: Path, scale: float) -> ScaleApplyResult:
    value = _validate_scale(scale)
    dataset = validate_colmap_apriltag_dataset(Path(dataset_root))
    sparse_dir = dataset.sparse_dir
    images_txt = sparse_dir / "images.txt"
    points_txt = sparse_dir / "points3D.txt"
    if not images_txt.is_file() or not points_txt.is_file():
        raise ValueError(
            "AprilTag scale application for COLMAP currently requires text sparse files "
            "(cameras.txt, images.txt, points3D.txt). Export or convert the sparse model to COLMAP text first."
        )
    backup_dir = _default_backup_dir(images_txt)
    images_backup = _copy_backup(images_txt, backup_dir)
    points_backup = _copy_backup(points_txt, backup_dir)

    try:
        frames_scaled = scale_colmap_images_txt(images_txt, value)
        points_scaled = scale_colmap_points3d_txt(points_txt, value)
    except Exception:
        shutil.copy2(images_backup, images_txt)
        shutil.copy2(points_backup, points_txt)
        raise

    return ScaleApplyResult(
        transforms_json=images_txt,
        transforms_backup=images_backup,
        pointcloud_ply=points_txt,
        pointcloud_backup=points_backup,
        scale=value,
        frames_scaled=frames_scaled,
        points_scaled=points_scaled,
        kind="colmap",
        geometry_label="COLMAP images.txt",
        pointcloud_label="COLMAP points3D.txt",
    )


def apply_scene_output_scale(
    scene_dir: Path,
    scale: float,
    *,
    output_dir: Path | None = None,
) -> ScaleApplyResult:
    dataset = validate_scale_output_dataset(scene_dir, output_dir=output_dir)
    if dataset.kind == "colmap":
        return apply_scale_to_colmap_dataset(dataset.root or dataset.transforms_json.parent, scale)
    return apply_scale_to_transforms_and_pointcloud(dataset.transforms_json, scale)
