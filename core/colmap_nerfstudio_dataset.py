"""Export a COLMAP sparse result as a Nerfstudio JSON/PLY dataset."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.colmap_sparse_model import Camera, ImagePose, Point3D, colmap_pose_to_c2w, read_model, resolve_model_dir
from core.dataset_writer_colmap import replace_file_with_link_or_copy

ProgressCallback = Callable[[int, int], None]

DEFAULT_NERFSTUDIO_POINTCLOUD_NAME = "pointcloud.ply"
NERFSTUDIO_COLMAP_WORLD_TRANSFORM = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
NERFSTUDIO_COLMAP_APPLIED_TRANSFORM = NERFSTUDIO_COLMAP_WORLD_TRANSFORM[:3, :]


@dataclass(frozen=True, slots=True)
class ColmapNerfstudioExportResult:
    output_dir: Path
    transforms_json: Path
    pointcloud: Path
    sparse_dir: Path
    image_count: int
    camera_count: int
    point_count: int
    mask_count: int
    action_counts: dict[str, int]
    warnings: tuple[str, ...] = ()


def export_colmap_nerfstudio_dataset(
    *,
    colmap_root: str | Path,
    output_dir: str | Path,
    images_dir: str | Path | None = None,
    masks_dir: str | Path | None = None,
    sparse_dir: str | Path | None = None,
    require_complete_masks: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> ColmapNerfstudioExportResult:
    """Create a Nerfstudio dataset from a finished COLMAP sparse model.

    The COLMAP world is converted with the same default coordinate contract as
    Nerfstudio's COLMAP processor: OpenCV camera axes are converted to OpenGL,
    then world axes are remapped by ``applied_transform``. The PLY points are
    written in that same transformed world so cameras and points stay aligned.
    """

    root = Path(colmap_root)
    out = Path(output_dir)
    source_images_root = Path(images_dir) if images_dir is not None else root / "images"
    source_masks_root = _resolve_masks_root(root, masks_dir)
    model_source = Path(sparse_dir) if sparse_dir is not None else _resolve_sparse_source(root)
    cameras, images, points, resolved_sparse = read_model(model_source)
    if not images:
        raise ValueError(f"No registered images found in COLMAP sparse model: {resolved_sparse}")
    if not source_images_root.is_dir():
        raise FileNotFoundError(f"COLMAP images directory not found: {source_images_root}")

    ordered_images = sorted(images.values(), key=lambda image: (image.name.lower(), image.image_id))
    _validate_registered_cameras(cameras, ordered_images)
    camera_model = _nerfstudio_camera_model_for_registered_images(cameras, ordered_images)
    mask_sources = _matching_masks(
        source_masks_root,
        source_images_root,
        ordered_images,
        require_complete=require_complete_masks,
    )

    total_steps = len(ordered_images) + len(mask_sources) + 2
    done = 0
    _notify(progress_callback, done, total_steps)

    out.mkdir(parents=True, exist_ok=True)
    action_counts: dict[str, int] = {}
    frames: list[dict[str, Any]] = []
    warnings: list[str] = []
    for image in ordered_images:
        camera = cameras[image.camera_id]
        source_image = _resolve_registered_image(source_images_root, image.name)
        if not source_image.is_file():
            raise FileNotFoundError(f"Registered COLMAP image was not found: {source_image}")
        rel_to_images = _relative_image_path(source_image, source_images_root)
        output_image_rel = Path("images") / rel_to_images
        action = replace_file_with_link_or_copy(source_image, out / output_image_rel)
        _increment_action(action_counts, action)

        frame: dict[str, Any] = {
            "file_path": output_image_rel.as_posix(),
            "transform_matrix": colmap_image_to_nerfstudio_transform(image).tolist(),
            "colmap_im_id": int(image.image_id),
        }
        frame.update(_nerfstudio_camera_payload(camera))
        mask_source = mask_sources.get(image.image_id)
        if mask_source is not None and source_masks_root is not None:
            output_mask_rel = Path("masks") / _relative_mask_path(mask_source, source_masks_root)
            frame["mask_path"] = output_mask_rel.as_posix()
        frames.append(frame)

        done += 1
        _notify(progress_callback, done, total_steps)

    for image in ordered_images:
        mask_source = mask_sources.get(image.image_id)
        if mask_source is None or source_masks_root is None:
            continue
        output_mask_rel = Path("masks") / _relative_mask_path(mask_source, source_masks_root)
        action = replace_file_with_link_or_copy(mask_source, out / output_mask_rel)
        _increment_action(action_counts, f"mask_{action}" if action else "")
        done += 1
        _notify(progress_callback, done, total_steps)

    ply_path = out / DEFAULT_NERFSTUDIO_POINTCLOUD_NAME
    write_nerfstudio_pointcloud_ply(ply_path, points)
    done += 1
    _notify(progress_callback, done, total_steps)

    data: dict[str, Any] = {
        "camera_model": camera_model,
        "frames": frames,
        "applied_transform": NERFSTUDIO_COLMAP_APPLIED_TRANSFORM.tolist(),
        "ply_file_path": DEFAULT_NERFSTUDIO_POINTCLOUD_NAME,
        "source": {
            "type": "colmap_nerfstudio_dataset",
            "colmap_root": str(root),
            "sparse_dir": str(resolved_sparse),
            "images_dir": str(source_images_root),
            "masks_dir": str(source_masks_root) if source_masks_root is not None and mask_sources else "",
            "coordinate_contract": "nerfstudio_colmap_to_json_default",
            "rig_contract": "per_image_colmap_poses_no_nerfstudio_rig",
            "pointcloud_space": "applied_transform_colmap_world",
        },
    }
    json_path = out / "transforms.json"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    done += 1
    _notify(progress_callback, done, total_steps)

    if source_masks_root is not None and not mask_sources:
        warnings.append(f"No matching masks found under: {source_masks_root}")

    return ColmapNerfstudioExportResult(
        output_dir=out,
        transforms_json=json_path,
        pointcloud=ply_path,
        sparse_dir=resolved_sparse,
        image_count=len(frames),
        camera_count=len({image.camera_id for image in ordered_images}),
        point_count=len(points),
        mask_count=len(mask_sources),
        action_counts=action_counts,
        warnings=tuple(warnings),
    )


def colmap_image_to_nerfstudio_transform(image: ImagePose) -> np.ndarray:
    c2w_opengl = colmap_pose_to_c2w(image, opengl_camera=True)
    return NERFSTUDIO_COLMAP_WORLD_TRANSFORM @ c2w_opengl


def transform_colmap_point_to_nerfstudio(point: Point3D) -> tuple[float, float, float]:
    xyz1 = np.array([point.xyz[0], point.xyz[1], point.xyz[2], 1.0], dtype=np.float64)
    transformed = NERFSTUDIO_COLMAP_WORLD_TRANSFORM @ xyz1
    return (float(transformed[0]), float(transformed[1]), float(transformed[2]))


def write_nerfstudio_pointcloud_ply(path: Path, points: dict[int, Point3D]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [points[key] for key in sorted(points)]
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(ordered)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point in ordered:
            x, y, z = transform_colmap_point_to_nerfstudio(point)
            r, g, b = point.rgb
            f.write(f"{x:.10g} {y:.10g} {z:.10g} {r:d} {g:d} {b:d}\n")


def _resolve_sparse_source(root: Path) -> Path:
    for candidate in (root / "sparse" / "0", root / "sparse", root):
        try:
            return resolve_model_dir(candidate)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No COLMAP sparse model found under: {root}")


def _resolve_masks_root(root: Path, masks_dir: str | Path | None) -> Path | None:
    if masks_dir is not None and str(masks_dir).strip():
        masks = Path(masks_dir)
        if not masks.is_dir():
            raise FileNotFoundError(f"COLMAP masks directory not found: {masks}")
        return masks
    masks = root / "masks"
    return masks if masks.is_dir() else None


def _validate_registered_cameras(cameras: dict[int, Camera], images: list[ImagePose]) -> None:
    missing = sorted({image.camera_id for image in images if image.camera_id not in cameras})
    if missing:
        raise ValueError(f"COLMAP images reference missing camera ids: {missing}")


def _nerfstudio_camera_model_for_registered_images(cameras: dict[int, Camera], images: list[ImagePose]) -> str:
    models = {_camera_model_family(cameras[image.camera_id]) for image in images}
    if len(models) == 1:
        return next(iter(models))
    raise ValueError(
        "Nerfstudio transforms.json supports one top-level camera_model; "
        f"registered COLMAP cameras mix incompatible models: {', '.join(sorted(models))}"
    )


def _camera_model_family(camera: Camera) -> str:
    if camera.model in {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"}:
        return "OPENCV"
    if camera.model in {"SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"}:
        return "OPENCV_FISHEYE"
    raise ValueError(f"Unsupported COLMAP camera model for Nerfstudio export: {camera.model}")


def _nerfstudio_camera_payload(camera: Camera) -> dict[str, float | int]:
    model = camera.model
    params = camera.params
    payload: dict[str, float | int] = {
        "w": int(camera.width),
        "h": int(camera.height),
    }
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        payload.update(_intrinsics(float(f), float(f), float(cx), float(cy)))
        payload.update(_distortion_zeros())
        return payload
    if model == "PINHOLE":
        fx, fy, cx, cy = params
        payload.update(_intrinsics(float(fx), float(fy), float(cx), float(cy)))
        payload.update(_distortion_zeros())
        return payload
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params
        payload.update(_intrinsics(float(f), float(f), float(cx), float(cy)))
        payload.update(_distortion_zeros(k1=float(k1)))
        return payload
    if model == "RADIAL":
        f, cx, cy, k1, k2 = params
        payload.update(_intrinsics(float(f), float(f), float(cx), float(cy)))
        payload.update(_distortion_zeros(k1=float(k1), k2=float(k2)))
        return payload
    if model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        payload.update(_intrinsics(float(fx), float(fy), float(cx), float(cy)))
        payload.update(_distortion_zeros(k1=float(k1), k2=float(k2), p1=float(p1), p2=float(p2)))
        return payload
    if model == "SIMPLE_RADIAL_FISHEYE":
        f, cx, cy, k1 = params
        payload.update(_intrinsics(float(f), float(f), float(cx), float(cy)))
        payload.update(_fisheye_distortion_zeros(k1=float(k1)))
        return payload
    if model == "RADIAL_FISHEYE":
        f, cx, cy, k1, k2 = params
        payload.update(_intrinsics(float(f), float(f), float(cx), float(cy)))
        payload.update(_fisheye_distortion_zeros(k1=float(k1), k2=float(k2)))
        return payload
    if model == "OPENCV_FISHEYE":
        fx, fy, cx, cy, k1, k2, k3, k4 = params
        payload.update(_intrinsics(float(fx), float(fy), float(cx), float(cy)))
        payload.update(_fisheye_distortion_zeros(k1=float(k1), k2=float(k2), k3=float(k3), k4=float(k4)))
        return payload
    raise ValueError(f"Unsupported COLMAP camera model for Nerfstudio export: {model}")


def _intrinsics(fx: float, fy: float, cx: float, cy: float) -> dict[str, float]:
    return {
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
    }


def _distortion_zeros(
    *,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    k4: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> dict[str, float]:
    return {
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "k4": k4,
        "p1": p1,
        "p2": p2,
    }


def _fisheye_distortion_zeros(
    *,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    k4: float = 0.0,
) -> dict[str, float]:
    return {
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "k4": k4,
    }


def _resolve_registered_image(images_dir: Path, image_name: str) -> Path:
    raw = Path(image_name)
    if raw.is_absolute():
        return raw
    candidate = images_dir / raw
    if candidate.exists():
        return candidate
    parts = raw.parts
    if parts and parts[0].lower() == "images" and len(parts) > 1 and images_dir.name.lower() == "images":
        return images_dir / Path(*parts[1:])
    return candidate


def _relative_image_path(image_path: Path, images_dir: Path) -> Path:
    try:
        return image_path.resolve().relative_to(images_dir.resolve())
    except ValueError:
        return Path(image_path.name)


def _matching_masks(
    masks_root: Path | None,
    images_root: Path,
    images: list[ImagePose],
    *,
    require_complete: bool,
) -> dict[int, Path]:
    if masks_root is None:
        return {}
    matches: dict[int, Path] = {}
    missing: list[str] = []
    for image in images:
        source_image = _resolve_registered_image(images_root, image.name)
        rel_to_images = _relative_image_path(source_image, images_root)
        mask = _first_existing_mask(masks_root, rel_to_images, image.name)
        if mask is None:
            missing.append(rel_to_images.as_posix())
            continue
        matches[image.image_id] = mask
    if matches and missing and require_complete:
        sample = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"COLMAP masks are incomplete; missing {len(missing)} registered masks: {sample}{suffix}")
    if missing:
        return {}
    return matches


def _first_existing_mask(masks_root: Path, rel_to_images: Path, image_name: str) -> Path | None:
    for candidate in _mask_candidates(masks_root, rel_to_images, image_name):
        if candidate.is_file():
            return candidate
    return None


def _mask_candidates(masks_root: Path, rel_to_images: Path, image_name: str) -> list[Path]:
    variants: list[Path] = [rel_to_images]
    raw = Path(image_name)
    if not raw.is_absolute():
        variants.append(raw)
        if raw.parts and raw.parts[0].lower() == "images" and len(raw.parts) > 1:
            variants.append(Path(*raw.parts[1:]))

    candidates: list[Path] = []
    for rel in variants:
        if rel.is_absolute():
            continue
        for candidate_rel in (
            rel,
            Path(f"{rel.as_posix()}.png"),
            rel.with_suffix(".png"),
            rel.parent / f"{rel.name}.png",
            rel.parent / f"{rel.stem}.png",
        ):
            safe = _candidate_under_root(masks_root, masks_root / candidate_rel)
            if safe is not None:
                candidates.append(safe)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.resolve(strict=False).as_posix().casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _candidate_under_root(root: Path, candidate: Path) -> Path | None:
    try:
        resolved_root = root.resolve(strict=False)
        resolved_candidate = candidate.resolve(strict=False)
        resolved_candidate.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    return candidate


def _relative_mask_path(mask_path: Path, masks_root: Path) -> Path:
    try:
        return mask_path.resolve().relative_to(masks_root.resolve())
    except ValueError:
        return Path(mask_path.name)


def _increment_action(counts: dict[str, int], action: str) -> None:
    if not action:
        return
    counts[action] = counts.get(action, 0) + 1


def _notify(callback: ProgressCallback | None, done: int, total: int) -> None:
    if callback is not None:
        callback(done, total)
