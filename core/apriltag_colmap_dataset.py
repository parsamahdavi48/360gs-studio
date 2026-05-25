"""COLMAP dataset adapter for AprilTag scale estimation and application."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.apriltag_geometry import PinholeFrame


@dataclass(frozen=True, slots=True)
class ColmapAprilTagDataset:
    root: Path
    sparse_dir: Path
    images_dir: Path
    pointcloud_ply: Path | None
    frame_count: int
    checked_image_count: int
    text_model: bool


def _read_model(path: Path):
    from core.spheresfm_to_transforms import read_model

    return read_model(path)


def _colmap_pose_to_c2w(image: object, *, opengl_camera: bool) -> np.ndarray:
    from core.spheresfm_to_transforms import colmap_pose_to_c2w

    return colmap_pose_to_c2w(image, opengl_camera=opengl_camera)


def _resolve_colmap_image_path(images_dir: Path, image_name: str) -> Path:
    from core.spheresfm_to_transforms import resolve_image_path

    return resolve_image_path(images_dir, image_name)


def _model_files_kind(path: Path) -> str:
    if all((path / name).is_file() for name in ("cameras.txt", "images.txt", "points3D.txt")):
        return "text"
    if all((path / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin")):
        return "binary"
    return ""


def _dataset_root_from_sparse(sparse_dir: Path) -> Path:
    if sparse_dir.name == "0" and sparse_dir.parent.name.lower() == "sparse":
        return sparse_dir.parent.parent
    if sparse_dir.name.lower() == "sparse":
        return sparse_dir.parent
    return sparse_dir


def resolve_colmap_sparse_dir(path: Path) -> Path:
    path = Path(path)
    candidates = (
        path,
        path / "sparse" / "0",
        path / "sparse",
    )
    errors: list[str] = []
    for candidate in candidates:
        try:
            _cameras, _images, _points, resolved = _read_model(candidate)
        except Exception as exc:
            errors.append(str(exc))
            continue
        return Path(resolved)
    detail = errors[-1] if errors else str(path)
    raise FileNotFoundError(f"No COLMAP sparse model found for AprilTag scale estimation: {path}\n{detail}")


def resolve_colmap_images_dir(dataset_root: Path, sparse_dir: Path, images_dir: Path | None = None) -> Path:
    if images_dir is not None:
        return Path(images_dir)
    for candidate in (
        dataset_root / "images",
        sparse_dir.parent.parent / "images" if sparse_dir.name == "0" else sparse_dir.parent / "images",
        dataset_root,
    ):
        if candidate.is_dir():
            return candidate
    return dataset_root / "images"


def _camera_intrinsics_and_distortion(camera: Any) -> tuple[float, float, float, float, np.ndarray | None]:
    model = str(camera.model).upper()
    params = tuple(float(value) for value in camera.params)
    if model == "SIMPLE_PINHOLE":
        if len(params) < 3:
            raise ValueError(f"Camera {getattr(camera, 'camera_id', '-')} SIMPLE_PINHOLE needs 3 params")
        f, cx, cy = params[:3]
        return f, f, cx, cy, None
    if model == "PINHOLE":
        if len(params) < 4:
            raise ValueError(f"Camera {getattr(camera, 'camera_id', '-')} PINHOLE needs 4 params")
        fx, fy, cx, cy = params[:4]
        return fx, fy, cx, cy, None
    if model == "SIMPLE_RADIAL":
        if len(params) < 4:
            raise ValueError(f"Camera {getattr(camera, 'camera_id', '-')} SIMPLE_RADIAL needs 4 params")
        f, cx, cy, k1 = params[:4]
        return f, f, cx, cy, np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64)
    if model == "RADIAL":
        if len(params) < 5:
            raise ValueError(f"Camera {getattr(camera, 'camera_id', '-')} RADIAL needs 5 params")
        f, cx, cy, k1, k2 = params[:5]
        return f, f, cx, cy, np.array([k1, k2, 0.0, 0.0], dtype=np.float64)
    if model == "OPENCV":
        if len(params) < 8:
            raise ValueError(f"Camera {getattr(camera, 'camera_id', '-')} OPENCV needs 8 params")
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        return fx, fy, cx, cy, np.array([k1, k2, p1, p2], dtype=np.float64)
    if model == "FULL_OPENCV":
        if len(params) < 12:
            raise ValueError(f"Camera {getattr(camera, 'camera_id', '-')} FULL_OPENCV needs 12 params")
        fx, fy, cx, cy = params[:4]
        return fx, fy, cx, cy, np.array(params[4:12], dtype=np.float64)
    if model in {"SPHERE", "EQUIRECTANGULAR"}:
        raise ValueError("AprilTag scale estimation does not support raw ERP/SPHERE cameras. Use projected PINHOLE output.")
    if "FISHEYE" in model:
        raise ValueError(f"AprilTag scale estimation does not support fisheye COLMAP camera model: {model}")
    raise ValueError(f"Unsupported COLMAP camera model for AprilTag scale estimation: {model or '-'}")


def load_colmap_pinhole_frames(
    dataset_root: Path,
    *,
    images_dir: Path | None = None,
) -> tuple[PinholeFrame, ...]:
    sparse_dir = resolve_colmap_sparse_dir(Path(dataset_root))
    root = _dataset_root_from_sparse(sparse_dir)
    image_root = resolve_colmap_images_dir(root, sparse_dir, images_dir)
    cameras_by_id, images_by_id, _points_by_id, resolved_model = _read_model(sparse_dir)
    frames: list[PinholeFrame] = []
    for image_id in sorted(images_by_id):
        image = images_by_id[image_id]
        camera = cameras_by_id.get(image.camera_id)
        if camera is None:
            continue
        fl_x, fl_y, cx, cy, distortion = _camera_intrinsics_and_distortion(camera)
        c2w = _colmap_pose_to_c2w(image, opengl_camera=True)
        frames.append(
            PinholeFrame(
                frame_id=str(image.image_id),
                file_path=str(image.name),
                image_path=_resolve_colmap_image_path(image_root, str(image.name)),
                width=int(camera.width),
                height=int(camera.height),
                fl_x=float(fl_x),
                fl_y=float(fl_y),
                cx=float(cx),
                cy=float(cy),
                transform_matrix=c2w,
                distortion_coeffs=distortion,
            )
        )
    if not frames:
        raise ValueError(f"No registered COLMAP images found in sparse model: {resolved_model}")
    return tuple(frames)


def validate_colmap_apriltag_dataset(
    dataset_root: Path,
    *,
    images_dir: Path | None = None,
    max_image_checks: int = 12,
) -> ColmapAprilTagDataset:
    sparse_dir = resolve_colmap_sparse_dir(Path(dataset_root))
    root = _dataset_root_from_sparse(sparse_dir)
    image_root = resolve_colmap_images_dir(root, sparse_dir, images_dir)
    frames = load_colmap_pinhole_frames(root, images_dir=image_root)
    limit = max(1, int(max_image_checks))
    missing = [frame.image_path for frame in frames[:limit] if not frame.image_path.is_file()]
    if missing:
        raise ValueError(f"COLMAP image referenced by images.txt was not found: {missing[0]}")
    pointcloud_ply = sparse_dir / "points3D.ply"
    return ColmapAprilTagDataset(
        root=root,
        sparse_dir=sparse_dir,
        images_dir=image_root,
        pointcloud_ply=pointcloud_ply if pointcloud_ply.is_file() else None,
        frame_count=len(frames),
        checked_image_count=min(len(frames), limit),
        text_model=_model_files_kind(sparse_dir) == "text",
    )


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    from core.spheresfm_to_transforms import qvec_to_rotmat

    return qvec_to_rotmat(qvec)


def _scaled_colmap_image_line(line: str, scale: float) -> str:
    parts = line.split()
    if len(parts) < 10:
        return line
    qvec = np.array([float(value) for value in parts[1:5]], dtype=np.float64)
    tvec = np.array([float(value) for value in parts[5:8]], dtype=np.float64)
    r_cw = _qvec_to_rotmat(qvec)
    center = -r_cw.T @ tvec
    scaled_tvec = -r_cw @ (center * scale)
    name = " ".join(parts[9:])
    return (
        f"{int(parts[0])} "
        f"{qvec[0]:.12g} {qvec[1]:.12g} {qvec[2]:.12g} {qvec[3]:.12g} "
        f"{scaled_tvec[0]:.12g} {scaled_tvec[1]:.12g} {scaled_tvec[2]:.12g} "
        f"{int(parts[8])} {name}"
    )


def scale_colmap_images_txt(path: Path, scale: float) -> int:
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    updated: list[str] = []
    scaled = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            updated.append(line)
            index += 1
            continue
        parts = stripped.split()
        if len(parts) >= 10:
            updated.append(_scaled_colmap_image_line(stripped, scale))
            scaled += 1
            index += 1
            if index < len(lines):
                updated.append(lines[index])
                index += 1
            continue
        updated.append(line)
        index += 1
    trailing = "\n" if raw.endswith("\n") else ""
    path.write_text("\n".join(updated) + trailing, encoding="utf-8", newline="\n")
    return scaled


def scale_colmap_points3d_txt(path: Path, scale: float) -> int:
    raw = path.read_text(encoding="utf-8", errors="replace")
    updated: list[str] = []
    scaled = 0
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            updated.append(line)
            continue
        parts = stripped.split()
        if len(parts) >= 8:
            for axis in (1, 2, 3):
                parts[axis] = f"{float(parts[axis]) * scale:.12g}"
            updated.append(" ".join(parts))
            scaled += 1
        else:
            updated.append(line)
    trailing = "\n" if raw.endswith("\n") else ""
    path.write_text("\n".join(updated) + trailing, encoding="utf-8", newline="\n")
    return scaled


def copy_backup(path: Path, backup_dir: Path) -> Path:
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
