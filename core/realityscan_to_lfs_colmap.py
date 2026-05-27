"""Build a LichtFeld-compatible COLMAP text dataset from RealityScan CSV/PLY."""

from __future__ import annotations

import math
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from core.dataset_writer_colmap import (
    SPARSE_RELATIVE_DIR,
    ColmapCamera,
    ColmapImage,
    camera_signature,
    quaternion_from_matrix,
    write_colmap_text_dataset,
)
from core.dataset_writer_colmap import (
    paths_equivalent as _paths_equivalent,
)
from core.dataset_writer_colmap import (
    replace_file_with_link_or_copy as _safe_replace_file_link_or_copy,
)
from core.image_io import imread_unicode, imwrite_unicode
from core.realityscan_dataset_plan import build_realityscan_lfs_dataset_plan
from core.realityscan_layout import mask_lookup_candidates
from core.realityscan_to_transforms import (
    REALITYSCAN_IMAGE_DIR_NAMES,
    REALITYSCAN_MASK_DIR_NAMES,
    RealityScanCameraRow,
    camera_from_csv_row,
    image_size,
    read_realityscan_csv,
    realityscan_image_asset_relative_path,
    related_realityscan_asset_roots,
    resolve_image_path,
    row_has_distortion,
    row_to_transform,
    strip_leading_realityscan_asset_dir,
    transform_points,
)
from core.transforms_to_colmap import read_ply_points, write_points3d_txt

DEFAULT_DATASET_DIR_NAME = "lfs_colmap"
DEFAULT_UNDISTORTED_DATASET_DIR_NAME = "lfs_colmap_undistorted"
DEFAULT_UNDISTORT_ALPHA = 1.0
LICHTFELD_TRANSFORMS_MARKERS = ("transforms.json", "transforms_train.json")
IMAGE_WRITE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CameraPayload = tuple[str, int, int, tuple[float, ...]]
ProgressCallback = Callable[[int, int], None]

try:
    import cv2
except Exception:  # pragma: no cover - reported only when undistortion is requested
    cv2 = None  # type: ignore[assignment]


def x_axis_rotation_matrix(rotation_x_deg: float) -> np.ndarray:
    radians = math.radians(float(rotation_x_deg))
    cos_v = math.cos(radians)
    sin_v = math.sin(radians)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_v, -sin_v],
            [0.0, sin_v, cos_v],
        ],
        dtype=np.float64,
    )
    return matrix


def lichtfeld_colmap_pointcloud_matrix(rotation_x_deg: float = 90.0) -> np.ndarray:
    return x_axis_rotation_matrix(rotation_x_deg)

def realityscan_row_to_colmap_w2c(
    row: RealityScanCameraRow,
    *,
    camera_rotation_x_deg: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    c2w = row_to_transform(row)
    c2w = x_axis_rotation_matrix(camera_rotation_x_deg) @ c2w
    # RealityScan CSV rows match this toolkit's OpenGL-style c2w convention.
    # COLMAP images.txt expects OpenCV camera axes, so flip camera Y/Z before
    # inverting to world-to-camera.
    c2w[:3, 1:3] *= -1.0
    r_wc = c2w[:3, :3]
    t_wc = c2w[:3, 3]
    r_cw = r_wc.T
    t_cw = -r_cw @ t_wc
    return r_cw, t_cw


def colmap_camera_payload(row: RealityScanCameraRow, image_path: Path) -> tuple[str, int, int, tuple[float, ...]]:
    width, height = image_size(image_path)
    camera = camera_from_csv_row(row, width, height)
    fl_x = float(camera["fl_x"])
    fl_y = float(camera["fl_y"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    if row_has_distortion(row):
        if abs(row.k3) > 1e-12 or abs(row.k4) > 1e-12:
            return (
                "FULL_OPENCV",
                width,
                height,
                (fl_x, fl_y, cx, cy, row.k1, row.k2, row.t1, row.t2, row.k3, row.k4, 0.0, 0.0),
            )
        return ("OPENCV", width, height, (fl_x, fl_y, cx, cy, row.k1, row.k2, row.t1, row.t2))
    if abs(fl_x - fl_y) < 1e-9:
        return ("SIMPLE_PINHOLE", width, height, (fl_x, cx, cy))
    return ("PINHOLE", width, height, (fl_x, fl_y, cx, cy))


def pinhole_camera_payload_from_matrix(width: int, height: int, camera_matrix: np.ndarray) -> tuple[str, int, int, tuple[float, ...]]:
    return (
        "PINHOLE",
        int(width),
        int(height),
        (
            float(camera_matrix[0, 0]),
            float(camera_matrix[1, 1]),
            float(camera_matrix[0, 2]),
            float(camera_matrix[1, 2]),
        ),
    )


def opencv_camera_matrix_and_distortion_for_size(
    row: RealityScanCameraRow,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    camera = camera_from_csv_row(row, width, height)
    matrix = np.array(
        [
            [float(camera["fl_x"]), 0.0, float(camera["cx"])],
            [0.0, float(camera["fl_y"]), float(camera["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.array(
        [row.k1, row.k2, row.t1, row.t2, row.k3, row.k4, 0.0, 0.0],
        dtype=np.float64,
    )
    return matrix, distortion


def opencv_camera_matrix_and_distortion(row: RealityScanCameraRow, image_path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    width, height = image_size(image_path)
    matrix, distortion = opencv_camera_matrix_and_distortion_for_size(row, width, height)
    return matrix, distortion, width, height


def image_name_for_colmap(image_path: Path, images_dir: Path) -> str:
    try:
        rel = image_path.resolve().relative_to(images_dir.resolve())
    except ValueError:
        rel = Path(image_path.name)
    return rel.as_posix()


def image_name_for_realityscan_asset(image_path: Path, images_dir: Path) -> str:
    asset_rel = realityscan_image_asset_relative_path(image_path, images_dir)
    if asset_rel is not None:
        return asset_rel.as_posix()
    return image_name_for_colmap(image_path, images_dir)


def find_matching_mask(masks_dir: Path, image_name: str) -> Path | None:
    roots = tuple(root for root in related_realityscan_asset_roots(masks_dir, REALITYSCAN_MASK_DIR_NAMES) if root.is_dir())
    if not roots:
        return None
    for rel in mask_lookup_candidates(image_name):
        for root in roots:
            candidate = root / rel
            if candidate.is_file():
                return candidate
    return None


def mask_output_name(image_name: str, source_mask: Path | None) -> Path:
    image_path = Path(image_name)
    if source_mask is not None:
        suffix = source_mask.suffix.lower() or ".png"
        if suffix in {".jpg", ".jpeg"}:
            suffix = ".png"
        return image_path.with_suffix(suffix)
    return image_path.with_suffix(".png")


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix not in IMAGE_WRITE_EXTENSIONS:
        path = path.with_suffix(".png")
    params: list[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"} and cv2 is not None:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    if path.suffix.lower() == ".png" and cv2 is not None:
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    if not imwrite_unicode(path, image, params):
        raise RuntimeError(f"Failed to write image: {path}")


def _write_white_mask_for_image(image_path: Path, output_mask_path: Path) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required to generate masks")
    image = imread_unicode(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image for mask generation: {image_path}")
    height, width = image.shape[:2]
    _write_image(output_mask_path, np.full((height, width), 255, dtype=np.uint8))


def undistort_image_and_optional_mask(
    row: RealityScanCameraRow,
    image_path: Path,
    output_image_path: Path,
    *,
    source_mask_path: Path | None,
    output_mask_path: Path | None,
    alpha: float,
) -> tuple[int, int, np.ndarray]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for --pre-undistort-distorted-images")

    image = imread_unicode(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    height, width = image.shape[:2]
    camera_matrix, distortion = opencv_camera_matrix_and_distortion_for_size(row, width, height)
    new_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion,
        (width, height),
        float(alpha),
        (width, height),
        centerPrincipalPoint=False,
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        new_camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    undistorted = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    _write_image(output_image_path, undistorted)

    if output_mask_path is not None:
        if source_mask_path is not None:
            mask = imread_unicode(source_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to read mask: {source_mask_path}")
            if mask.shape[:2] != (height, width):
                raise ValueError(
                    "Mask size must match its image before undistortion: "
                    f"{source_mask_path} is {mask.shape[1]}x{mask.shape[0]}, "
                    f"{image_path} is {width}x{height}"
                )
        else:
            mask = np.full((height, width), 255, dtype=np.uint8)
        undistorted_mask = cv2.remap(
            mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        _threshold, binary = cv2.threshold(undistorted_mask, 127, 255, cv2.THRESH_BINARY)
        _write_image(output_mask_path, binary)

    return width, height, new_camera_matrix


def _remove_directory_link(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
        return
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        path.rmdir()


def _clear_directory_contents(path: Path) -> None:
    if not path.is_dir():
        return
    for child in path.iterdir():
        is_junction = getattr(child, "is_junction", None)
        if callable(is_junction) and is_junction():
            child.rmdir()
        elif child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _remove_empty_directory(path: Path) -> None:
    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()


def _prepare_output_asset_dir(path: Path, source_dir: Path, *, create: bool) -> bool:
    _remove_directory_link(path)
    protect_existing = _paths_equivalent(path, source_dir)
    if not protect_existing:
        if path.exists() and not path.is_dir():
            path.unlink()
        _clear_directory_contents(path)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return protect_existing


def _dataset_image_base_name(row_name: str, source_image: Path) -> str:
    raw = Path(row_name)
    stripped = strip_leading_realityscan_asset_dir(raw, REALITYSCAN_IMAGE_DIR_NAMES)
    name = stripped.name or source_image.name
    suffix = Path(name).suffix or source_image.suffix or ".jpg"
    stem = Path(name).stem or source_image.stem or "image"
    return f"{stem}{suffix.lower()}"


def _unique_dataset_image_name(
    row_name: str,
    source_image: Path,
    used_names: set[str],
    *,
    output_images_dir: Path,
    protect_existing: bool,
) -> str:
    candidate = _dataset_image_base_name(row_name, source_image)
    suffix = Path(candidate).suffix
    base = candidate[: -len(suffix)] if suffix else candidate
    index = 2
    while _dataset_name_is_used(
        candidate,
        used_names,
        output_images_dir=output_images_dir,
        source_image=source_image,
        protect_existing=protect_existing,
    ):
        candidate = f"{base}_{index}{suffix}"
        index += 1
    used_names.add(candidate.casefold())
    return candidate


def _dataset_name_is_used(
    candidate: str,
    used_names: set[str],
    *,
    output_images_dir: Path,
    source_image: Path,
    protect_existing: bool,
) -> bool:
    if candidate.casefold() in used_names:
        return True
    destination = output_images_dir / candidate
    return protect_existing and destination.exists() and not _paths_equivalent(destination, source_image)


def _copy_or_link_source_mask(
    source_mask: Path | None,
    output_mask: Path,
    stats: dict[str, int],
) -> None:
    if source_mask is None:
        return
    mask_link_kind = _safe_replace_file_link_or_copy(source_mask, output_mask)
    if mask_link_kind == "hardlink":
        stats["linked_masks"] += 1
    elif mask_link_kind == "copy":
        stats["copied_masks"] += 1


def _empty_asset_stats() -> dict[str, int]:
    return {
        "undistorted_images": 0,
        "linked_images": 0,
        "copied_images": 0,
        "undistorted_masks": 0,
        "linked_masks": 0,
        "copied_masks": 0,
        "generated_valid_masks": 0,
        "missing_images": 0,
    }


def _notify_progress(callback: ProgressCallback | None, done: int, total: int) -> None:
    if callback is not None:
        callback(max(0, int(done)), max(0, int(total)))


def prepare_undistorted_asset_dataset(
    rows: list[RealityScanCameraRow],
    source_images_dir: Path,
    source_masks_dir: Path,
    output_dir: Path,
    *,
    skip_missing_images: bool,
    alpha: float,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, Path, list[CameraPayload | None], list[str | None], dict[str, int]]:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Undistort alpha must be between 0 and 1: {alpha}")

    output_images_dir = output_dir / "images"
    output_masks_dir = output_dir / "masks"
    generate_valid_masks = alpha > 0.0

    if _paths_equivalent(output_images_dir, source_images_dir):
        raise ValueError(f"Refusing to pre-undistort into source images directory: {output_images_dir}")
    if source_masks_dir.is_dir() and _paths_equivalent(output_masks_dir, source_masks_dir):
        raise ValueError(f"Refusing to pre-undistort into source masks directory: {output_masks_dir}")

    protect_existing_images = _prepare_output_asset_dir(output_images_dir, source_images_dir, create=True)
    has_source_masks = any(
        root.is_dir() for root in related_realityscan_asset_roots(source_masks_dir, REALITYSCAN_MASK_DIR_NAMES)
    )
    protect_existing_masks = _prepare_output_asset_dir(
        output_masks_dir,
        source_masks_dir,
        create=has_source_masks or generate_valid_masks,
    )

    camera_payloads: list[CameraPayload | None] = []
    output_names: list[str | None] = []
    stats = _empty_asset_stats()
    used_names: set[str] = set()

    row_total = len(rows)
    for row_index, row in enumerate(rows, start=1):
        source_image = resolve_image_path(source_images_dir, row.name)
        if not source_image.is_file():
            if skip_missing_images:
                stats["missing_images"] += 1
                camera_payloads.append(None)
                output_names.append(None)
                _notify_progress(progress_callback, row_index, row_total)
                continue
            raise FileNotFoundError(f"Image referenced by RealityScan CSV was not found: {source_image}")

        output_name = _unique_dataset_image_name(
            row.name,
            source_image,
            used_names,
            output_images_dir=output_images_dir,
            protect_existing=protect_existing_images,
        )
        output_names.append(output_name)
        output_image = output_images_dir / output_name
        source_mask = find_matching_mask(source_masks_dir, image_name_for_realityscan_asset(source_image, source_images_dir))
        output_mask = output_masks_dir / mask_output_name(output_name, source_mask)

        if row_has_distortion(row):
            width, height, new_camera_matrix = undistort_image_and_optional_mask(
                row,
                source_image,
                output_image,
                source_mask_path=source_mask,
                output_mask_path=output_mask if source_mask is not None or generate_valid_masks else None,
                alpha=alpha,
            )
            camera_payloads.append(pinhole_camera_payload_from_matrix(width, height, new_camera_matrix))
            stats["undistorted_images"] += 1
            if source_mask is not None:
                stats["undistorted_masks"] += 1
            elif generate_valid_masks:
                stats["generated_valid_masks"] += 1
        else:
            camera_payloads.append(None)
            link_kind = _safe_replace_file_link_or_copy(source_image, output_image)
            if link_kind == "hardlink":
                stats["linked_images"] += 1
            elif link_kind == "copy":
                stats["copied_images"] += 1

            if source_mask is not None:
                _copy_or_link_source_mask(source_mask, output_mask, stats)
            elif generate_valid_masks:
                _write_white_mask_for_image(source_image, output_mask)
                stats["generated_valid_masks"] += 1

        _notify_progress(progress_callback, row_index, row_total)

    if not protect_existing_masks:
        _remove_empty_directory(output_masks_dir)
    return output_images_dir, output_masks_dir, camera_payloads, output_names, stats


def prepare_linked_asset_dataset(
    rows: list[RealityScanCameraRow],
    source_images_dir: Path,
    source_masks_dir: Path,
    output_dir: Path,
    *,
    skip_missing_images: bool,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, Path, list[str | None], dict[str, int]]:
    output_images_dir = output_dir / "images"
    output_masks_dir = output_dir / "masks"
    protect_existing_images = _prepare_output_asset_dir(output_images_dir, source_images_dir, create=True)
    protect_existing_masks = _prepare_output_asset_dir(output_masks_dir, source_masks_dir, create=False)

    stats = _empty_asset_stats()
    output_names: list[str | None] = []
    used_names: set[str] = set()

    row_total = len(rows)
    for row_index, row in enumerate(rows, start=1):
        source_image = resolve_image_path(source_images_dir, row.name)
        if not source_image.is_file():
            if skip_missing_images:
                stats["missing_images"] += 1
                output_names.append(None)
                _notify_progress(progress_callback, row_index, row_total)
                continue
            raise FileNotFoundError(f"Image referenced by RealityScan CSV was not found: {source_image}")

        output_name = _unique_dataset_image_name(
            row.name,
            source_image,
            used_names,
            output_images_dir=output_images_dir,
            protect_existing=protect_existing_images,
        )
        output_names.append(output_name)
        output_image = output_images_dir / output_name
        link_kind = _safe_replace_file_link_or_copy(source_image, output_image)
        if link_kind == "hardlink":
            stats["linked_images"] += 1
        elif link_kind == "copy":
            stats["copied_images"] += 1

        source_image_name = image_name_for_realityscan_asset(source_image, source_images_dir)
        source_mask = find_matching_mask(source_masks_dir, source_image_name)
        if source_mask is None:
            _notify_progress(progress_callback, row_index, row_total)
            continue
        output_masks_dir.mkdir(parents=True, exist_ok=True)
        output_mask = output_masks_dir / mask_output_name(output_name, source_mask)
        _copy_or_link_source_mask(source_mask, output_mask, stats)
        _notify_progress(progress_callback, row_index, row_total)

    if not protect_existing_masks:
        _remove_empty_directory(output_masks_dir)
    return output_images_dir, output_masks_dir, output_names, stats


def find_lfs_loader_conflict_markers(dataset_root: Path) -> list[Path]:
    return [dataset_root / name for name in LICHTFELD_TRANSFORMS_MARKERS if (dataset_root / name).is_file()]


def build_colmap_records(
    rows: list[RealityScanCameraRow],
    images_dir: Path,
    *,
    skip_missing_images: bool = False,
    camera_rotation_x_deg: float = 90.0,
    camera_payloads_by_name: dict[str, CameraPayload] | None = None,
    camera_payloads_by_row_index: Sequence[CameraPayload | None] | None = None,
    image_names_by_row_index: Sequence[str | None] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[ColmapCamera], list[ColmapImage], int]:
    cameras: list[ColmapCamera] = []
    images: list[ColmapImage] = []
    camera_ids: dict[tuple[Any, ...], int] = {}
    missing = 0

    row_total = len(rows)
    for row_index, row in enumerate(rows):
        image_name = image_names_by_row_index[row_index] if image_names_by_row_index is not None else None
        image_path = images_dir / image_name if image_name is not None else resolve_image_path(images_dir, row.name)
        if not image_path.is_file():
            if skip_missing_images:
                missing += 1
                _notify_progress(progress_callback, row_index + 1, row_total)
                continue
            raise FileNotFoundError(f"Image referenced by RealityScan CSV was not found: {image_path}")

        indexed_payload = (
            camera_payloads_by_row_index[row_index]
            if camera_payloads_by_row_index is not None and row_index < len(camera_payloads_by_row_index)
            else None
        )
        if indexed_payload is not None:
            model, width, height, params = indexed_payload
        elif camera_payloads_by_name is not None and row.name in camera_payloads_by_name:
            model, width, height, params = camera_payloads_by_name[row.name]
        else:
            model, width, height, params = colmap_camera_payload(row, image_path)
        signature = camera_signature(model, width, height, params)
        camera_id = camera_ids.get(signature)
        if camera_id is None:
            camera_id = len(cameras) + 1
            camera_ids[signature] = camera_id
            cameras.append(ColmapCamera(camera_id, model, width, height, params))

        r_cw, t_cw = realityscan_row_to_colmap_w2c(row, camera_rotation_x_deg=camera_rotation_x_deg)
        images.append(
            ColmapImage(
                image_id=len(images) + 1,
                qvec=quaternion_from_matrix(r_cw),
                tvec=t_cw,
                camera_id=camera_id,
                name=image_name if image_name is not None else image_name_for_colmap(image_path, images_dir),
            )
        )
        _notify_progress(progress_callback, row_index + 1, row_total)

    if not images:
        raise ValueError("No images were converted")
    return cameras, images, missing


def convert(
    csv_path: Path,
    output_dir: Path,
    *,
    images_dir: Path | None = None,
    masks_dir: Path | None = None,
    ply_path: Path | None = None,
    camera_rotation_x_deg: float = 90.0,
    pointcloud_rotation_x_deg: float = 90.0,
    skip_missing_images: bool = False,
    allow_mixed_loader_root: bool = False,
    pre_undistort_distorted_images: bool = False,
    undistort_alpha: float = DEFAULT_UNDISTORT_ALPHA,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    images_dir = Path(images_dir) if images_dir is not None else csv_path.parent / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    masks_dir = Path(masks_dir) if masks_dir is not None else csv_path.parent / "masks"

    conflict_markers = find_lfs_loader_conflict_markers(output_dir)
    if conflict_markers and not allow_mixed_loader_root:
        markers = ", ".join(str(path) for path in conflict_markers)
        raise ValueError(
            "This output directory also contains a LichtFeld transforms dataset marker. "
            "Use a dedicated COLMAP dataset root such as '<csv folder>\\lfs_colmap', "
            f"or pass --allow-mixed-loader-root if you intentionally want this layout: {markers}"
        )

    rows = read_realityscan_csv(csv_path)
    row_total = len(rows)
    progress_total = max(1, row_total * 2 + 1 + (1 if ply_path is not None else 0))

    def emit_progress(done: int) -> None:
        _notify_progress(progress_callback, min(done, progress_total), progress_total)

    emit_progress(0)
    plan = build_realityscan_lfs_dataset_plan(
        rows,
        images_dir,
        masks_dir,
        pre_undistort_distorted_images=pre_undistort_distorted_images,
        skip_missing_images=skip_missing_images,
    )
    if plan.issues:
        raise FileNotFoundError("\n".join(plan.issues))
    asset_stats: dict[str, int] = {}
    camera_payloads_by_row_index: list[CameraPayload | None] | None = None
    image_names_by_row_index: list[str | None] | None = None
    effective_images_dir = images_dir
    effective_masks_dir = masks_dir
    if pre_undistort_distorted_images:
        (
            effective_images_dir,
            effective_masks_dir,
            camera_payloads_by_row_index,
            image_names_by_row_index,
            asset_stats,
        ) = prepare_undistorted_asset_dataset(
            rows,
            images_dir,
            masks_dir,
            output_dir,
            skip_missing_images=skip_missing_images,
            alpha=undistort_alpha,
            progress_callback=lambda done, _total: emit_progress(done),
        )
    else:
        effective_images_dir, effective_masks_dir, image_names_by_row_index, asset_stats = prepare_linked_asset_dataset(
            rows,
            images_dir,
            masks_dir,
            output_dir,
            skip_missing_images=skip_missing_images,
            progress_callback=lambda done, _total: emit_progress(done),
        )

    cameras, images, missing = build_colmap_records(
        rows,
        effective_images_dir,
        skip_missing_images=skip_missing_images,
        camera_rotation_x_deg=camera_rotation_x_deg,
        camera_payloads_by_row_index=camera_payloads_by_row_index,
        image_names_by_row_index=image_names_by_row_index,
        progress_callback=lambda done, _total: emit_progress(row_total + done),
    )

    linked_assets: list[str] = []

    sparse_dir = write_colmap_text_dataset(output_dir, cameras, images).sparse_dir
    emit_progress(row_total * 2 + 1)

    pointcloud_output = ""
    if ply_path is not None:
        ply_path = Path(ply_path)
        if not ply_path.is_file():
            raise FileNotFoundError(f"PLY not found: {ply_path}")
        pointcloud_dest = sparse_dir / "points3D.txt"
        points, colors = read_ply_points(ply_path)
        transformed = transform_points(points, lichtfeld_colmap_pointcloud_matrix(pointcloud_rotation_x_deg))
        write_points3d_txt(pointcloud_dest, transformed, colors)
        pointcloud_output = str(pointcloud_dest)
        emit_progress(progress_total)

    metadata = _dataset_metadata(
        effective_masks_dir=effective_masks_dir,
        plan_action_counts=plan.action_counts,
        asset_stats=asset_stats,
        pre_undistort_distorted_images=pre_undistort_distorted_images,
    )

    return {
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
        "images_dir": str(effective_images_dir),
        "masks_dir": str(effective_masks_dir) if effective_masks_dir.is_dir() else "",
        "linked_assets": linked_assets,
        "asset_stats": asset_stats,
        "plan_action_counts": plan.action_counts,
        "sparse_dir": str(sparse_dir),
        "pointcloud": pointcloud_output,
        "num_csv_rows": len(rows),
        "num_images": len(images),
        "num_cameras": len(cameras),
        "num_missing_images": missing,
        "camera_rotation_x_deg": float(camera_rotation_x_deg),
        "pointcloud_rotation_x_deg": float(pointcloud_rotation_x_deg),
        "pre_undistort_distorted_images": bool(pre_undistort_distorted_images),
        "undistort_alpha": float(undistort_alpha),
        "metadata": metadata,
    }


def _dataset_metadata(
    *,
    effective_masks_dir: Path,
    plan_action_counts: dict[str, int],
    asset_stats: dict[str, int],
    pre_undistort_distorted_images: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "lichtfeld_colmap",
        "source_kind": "realityscan_csv_ply",
        "images_dir": "images",
        "masks_dir": "masks" if effective_masks_dir.is_dir() else "",
        "sparse_dir": SPARSE_RELATIVE_DIR.as_posix(),
        "plan_action_counts": plan_action_counts,
        "asset_stats": asset_stats,
        "pre_undistort_distorted_images": bool(pre_undistort_distorted_images),
    }


def parse_args(argv: list[str] | None = None):
    from core.realityscan_to_lfs_colmap_cli import parse_args as _parse_args

    return _parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from core.realityscan_to_lfs_colmap_cli import main as _main

    return _main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
