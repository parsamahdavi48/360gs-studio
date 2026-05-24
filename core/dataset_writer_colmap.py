from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SPARSE_RELATIVE_DIR = Path("sparse") / "0"


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


@dataclass(frozen=True, slots=True)
class ColmapDatasetWriteResult:
    root: Path
    sparse_dir: Path
    image_count: int
    camera_count: int
    pointcloud: Path | None = None


def camera_signature(model: str, width: int, height: int, params: tuple[float, ...]) -> tuple[Any, ...]:
    return (model, int(width), int(height), *(round(float(value), 10) for value in params))


def quaternion_from_matrix(r: np.ndarray) -> np.ndarray:
    trace = float(np.trace(r))
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (r[2, 1] - r[1, 2]) * s
        qy = (r[0, 2] - r[2, 0]) * s
        qz = (r[1, 0] - r[0, 1]) * s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s

    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(qvec))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid near-zero quaternion")
    qvec /= norm
    if qvec[0] < 0.0:
        qvec = -qvec
    return qvec


def write_colmap_text_dataset(
    dataset_root: Path,
    cameras: list[ColmapCamera],
    images: list[ColmapImage],
) -> ColmapDatasetWriteResult:
    sparse_dir = dataset_root / SPARSE_RELATIVE_DIR
    sparse_dir.mkdir(parents=True, exist_ok=True)
    write_cameras_txt(sparse_dir / "cameras.txt", cameras)
    write_images_txt(sparse_dir / "images.txt", images)
    write_empty_points3d_txt(sparse_dir / "points3D.txt")
    return ColmapDatasetWriteResult(
        root=dataset_root,
        sparse_dir=sparse_dir,
        image_count=len(images),
        camera_count=len(cameras),
    )


def write_cameras_txt(path: Path, cameras: list[ColmapCamera]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for camera in cameras:
            params = " ".join(f"{value:.12g}" for value in camera.params)
            f.write(f"{camera.camera_id} {camera.model} {camera.width} {camera.height} {params}\n")


def write_images_txt(path: Path, images: list[ColmapImage]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}\n")
        for image in images:
            qw, qx, qy, qz = image.qvec
            tx, ty, tz = image.tvec
            f.write(
                f"{image.image_id} {qw:.12g} {qx:.12g} {qy:.12g} {qz:.12g} "
                f"{tx:.12g} {ty:.12g} {tz:.12g} {image.camera_id} {image.name}\n\n"
            )


def write_empty_points3d_txt(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")


def paths_equivalent(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def replace_file_with_link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if paths_equivalent(source, destination):
        return ""
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def create_directory_link(link_path: Path, target_dir: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    target_dir = target_dir.resolve()

    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_path), str(target_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            return
        message = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "Failed to create a Windows directory junction. "
            f"Run manually: mklink /J {link_path} {target_dir}"
            + (f"\n{message}" if message else "")
        )

    try:
        os.symlink(target_dir, link_path, target_is_directory=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to create directory symlink: {link_path} -> {target_dir}") from exc


def ensure_dataset_asset_link(dataset_root: Path, name: str, source_dir: Path) -> str:
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        return ""

    link_path = dataset_root / name
    if paths_equivalent(link_path, source_dir):
        return ""
    if link_path.exists() or link_path.is_symlink():
        raise FileExistsError(f"Cannot link {name}: {link_path} already exists and does not point to {source_dir}")

    create_directory_link(link_path, source_dir)
    return str(link_path)
