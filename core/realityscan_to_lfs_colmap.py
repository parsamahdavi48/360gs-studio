"""Build a LichtFeld-compatible COLMAP text dataset from RealityScan CSV/PLY."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from core.dataset_job_spec import JOB_KIND_REALITYSCAN_LFS_COLMAP, load_dataset_job
from core.dataset_writer_colmap import (
    SPARSE_RELATIVE_DIR,
    ColmapCamera,
    ColmapImage,
    camera_signature,
    ensure_dataset_asset_link,
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
from core.realityscan_to_transforms import (
    RealityScanCameraRow,
    camera_from_csv_row,
    image_size,
    read_realityscan_csv,
    resolve_image_path,
    row_has_distortion,
    row_to_transform,
    write_transformed_ply,
)

DEFAULT_DATASET_DIR_NAME = "lfs_colmap"
DEFAULT_UNDISTORTED_DATASET_DIR_NAME = "lfs_colmap_undistorted"
DEFAULT_UNDISTORT_ALPHA = 1.0
LICHTFELD_TRANSFORMS_MARKERS = ("transforms.json", "transforms_train.json")
MASK_SEARCH_EXTENSIONS = (".png", ".jpg", ".jpeg", ".mask.png")
IMAGE_WRITE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

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

def _mask_lookup_candidates(image_name: str) -> list[Path]:
    image_path = Path(image_name)
    stem_path = image_path.parent / image_path.stem
    candidates: list[Path] = [image_path]
    for ext in MASK_SEARCH_EXTENSIONS:
        candidates.append(stem_path.with_suffix(ext))
    for ext in MASK_SEARCH_EXTENSIONS:
        candidates.append(Path(f"{image_name}{ext}"))

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.as_posix().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def find_matching_mask(masks_dir: Path, image_name: str) -> Path | None:
    if not masks_dir.is_dir():
        return None
    for rel in _mask_lookup_candidates(image_name):
        candidate = masks_dir / rel
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


def prepare_undistorted_asset_dataset(
    rows: list[RealityScanCameraRow],
    source_images_dir: Path,
    source_masks_dir: Path,
    output_dir: Path,
    *,
    skip_missing_images: bool,
    alpha: float,
) -> tuple[Path, Path, dict[str, tuple[str, int, int, tuple[float, ...]]], dict[str, int]]:
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

    output_images_dir.mkdir(parents=True, exist_ok=True)
    if source_masks_dir.is_dir() or generate_valid_masks:
        output_masks_dir.mkdir(parents=True, exist_ok=True)

    camera_payloads: dict[str, tuple[str, int, int, tuple[float, ...]]] = {}
    stats = {
        "undistorted_images": 0,
        "linked_images": 0,
        "copied_images": 0,
        "undistorted_masks": 0,
        "linked_masks": 0,
        "copied_masks": 0,
        "generated_valid_masks": 0,
        "missing_images": 0,
    }

    for row in rows:
        source_image = resolve_image_path(source_images_dir, row.name)
        if not source_image.is_file():
            if skip_missing_images:
                stats["missing_images"] += 1
                continue
            raise FileNotFoundError(f"Image referenced by RealityScan CSV was not found: {source_image}")

        output_image = output_images_dir / image_name_for_colmap(source_image, source_images_dir)
        source_mask = find_matching_mask(source_masks_dir, image_name_for_colmap(source_image, source_images_dir))
        output_mask = output_masks_dir / mask_output_name(image_name_for_colmap(source_image, source_images_dir), source_mask)

        if row_has_distortion(row):
            width, height, new_camera_matrix = undistort_image_and_optional_mask(
                row,
                source_image,
                output_image,
                source_mask_path=source_mask,
                output_mask_path=output_mask if source_mask is not None or generate_valid_masks else None,
                alpha=alpha,
            )
            camera_payloads[row.name] = pinhole_camera_payload_from_matrix(width, height, new_camera_matrix)
            stats["undistorted_images"] += 1
            if source_mask is not None:
                stats["undistorted_masks"] += 1
            elif generate_valid_masks:
                stats["generated_valid_masks"] += 1
        else:
            link_kind = _safe_replace_file_link_or_copy(source_image, output_image)
            if link_kind == "hardlink":
                stats["linked_images"] += 1
            elif link_kind == "copy":
                stats["copied_images"] += 1

            if source_mask is not None:
                mask_link_kind = _safe_replace_file_link_or_copy(source_mask, output_mask)
                if mask_link_kind == "hardlink":
                    stats["linked_masks"] += 1
                elif mask_link_kind == "copy":
                    stats["copied_masks"] += 1
            elif generate_valid_masks:
                _write_white_mask_for_image(source_image, output_mask)
                stats["generated_valid_masks"] += 1

    return output_images_dir, output_masks_dir, camera_payloads, stats


def find_lfs_loader_conflict_markers(dataset_root: Path) -> list[Path]:
    return [dataset_root / name for name in LICHTFELD_TRANSFORMS_MARKERS if (dataset_root / name).is_file()]


def build_colmap_records(
    rows: list[RealityScanCameraRow],
    images_dir: Path,
    *,
    skip_missing_images: bool = False,
    camera_rotation_x_deg: float = 90.0,
    camera_payloads_by_name: dict[str, tuple[str, int, int, tuple[float, ...]]] | None = None,
) -> tuple[list[ColmapCamera], list[ColmapImage], int]:
    cameras: list[ColmapCamera] = []
    images: list[ColmapImage] = []
    camera_ids: dict[tuple[Any, ...], int] = {}
    missing = 0

    for row in rows:
        image_path = resolve_image_path(images_dir, row.name)
        if not image_path.is_file():
            if skip_missing_images:
                missing += 1
                continue
            raise FileNotFoundError(f"Image referenced by RealityScan CSV was not found: {image_path}")

        model, width, height, params = (
            camera_payloads_by_name[row.name]
            if camera_payloads_by_name is not None and row.name in camera_payloads_by_name
            else colmap_camera_payload(row, image_path)
        )
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
                name=image_name_for_colmap(image_path, images_dir),
            )
        )

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
    camera_payloads_by_name: dict[str, tuple[str, int, int, tuple[float, ...]]] | None = None
    effective_images_dir = images_dir
    effective_masks_dir = masks_dir
    if pre_undistort_distorted_images:
        effective_images_dir, effective_masks_dir, camera_payloads_by_name, asset_stats = prepare_undistorted_asset_dataset(
            rows,
            images_dir,
            masks_dir,
            output_dir,
            skip_missing_images=skip_missing_images,
            alpha=undistort_alpha,
        )

    cameras, images, missing = build_colmap_records(
        rows,
        effective_images_dir,
        skip_missing_images=skip_missing_images,
        camera_rotation_x_deg=camera_rotation_x_deg,
        camera_payloads_by_name=camera_payloads_by_name,
    )

    linked_assets: list[str] = []
    if not pre_undistort_distorted_images:
        image_link = ensure_dataset_asset_link(output_dir, "images", images_dir)
        if image_link:
            linked_assets.append(image_link)
        mask_link = ensure_dataset_asset_link(output_dir, "masks", masks_dir)
        if mask_link:
            linked_assets.append(mask_link)

    sparse_dir = write_colmap_text_dataset(output_dir, cameras, images).sparse_dir

    pointcloud_output = ""
    if ply_path is not None:
        ply_path = Path(ply_path)
        if not ply_path.is_file():
            raise FileNotFoundError(f"PLY not found: {ply_path}")
        pointcloud_dest = sparse_dir / "points3D.ply"
        write_transformed_ply(ply_path, pointcloud_dest, lichtfeld_colmap_pointcloud_matrix(pointcloud_rotation_x_deg))
        pointcloud_output = str(pointcloud_dest)

    _write_manifest(
        output_dir,
        {
            "schema_version": 1,
            "kind": "lichtfeld_colmap",
            "source_kind": "realityscan_csv_ply",
            "images_dir": "images" if (output_dir / "images").exists() else str(effective_images_dir),
            "masks_dir": "masks" if (output_dir / "masks").exists() else str(effective_masks_dir) if effective_masks_dir.is_dir() else "",
            "sparse_dir": SPARSE_RELATIVE_DIR.as_posix(),
            "plan_action_counts": plan.action_counts,
            "asset_stats": asset_stats,
            "pre_undistort_distorted_images": bool(pre_undistort_distorted_images),
        },
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
    }


def _write_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    (output_dir / "stechdrive_dataset_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a LichtFeld-compatible COLMAP text dataset from RealityScan CSV + PLY exports.",
    )
    parser.add_argument("csv_path", nargs="?", help="RealityScan registration CSV exported for Postshot")
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Dataset root. Defaults to <csv folder>/lfs_colmap",
    )
    parser.add_argument("--job", default="", help="Versioned dataset job JSON")
    parser.add_argument("--images-dir", help="Existing images directory. Defaults to <csv folder>/images")
    parser.add_argument("--masks-dir", help="Existing masks directory. Defaults to <csv folder>/masks when present")
    parser.add_argument("--ply", help="RealityScan PLY to rotate for LichtFeld COLMAP loading")
    parser.add_argument(
        "--camera-rotation-x-deg",
        type=float,
        default=90.0,
        help="X-axis world rotation applied to COLMAP camera poses (default: 90)",
    )
    parser.add_argument(
        "--pointcloud-rotation-x-deg",
        type=float,
        default=90.0,
        help="X-axis rotation applied only to points3D.ply (default: 90)",
    )
    parser.add_argument("--skip-missing-images", action="store_true", help="Skip CSV rows whose images are missing")
    parser.add_argument(
        "--allow-mixed-loader-root",
        action="store_true",
        help="Allow writing into a root that also contains transforms.json/transforms_train.json",
    )
    parser.add_argument(
        "--pre-undistort-distorted-images",
        action="store_true",
        help="Pre-undistort distorted RealityScan rows and write them as PINHOLE cameras",
    )
    parser.add_argument(
        "--undistort-alpha",
        type=float,
        default=DEFAULT_UNDISTORT_ALPHA,
        help="OpenCV undistort alpha for pre-undistorted images: 0 crops black borders, 1 keeps full FOV (default: 1)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.job:
            job = load_dataset_job(args.job, expected_kind=JOB_KIND_REALITYSCAN_LFS_COLMAP)
            result = convert(
                Path(str(job["csv_path"])),
                Path(str(job["output_dir"])),
                images_dir=Path(str(job["images_dir"])) if str(job.get("images_dir") or "") else None,
                masks_dir=Path(str(job["masks_dir"])) if str(job.get("masks_dir") or "") else None,
                ply_path=Path(str(job["ply_path"])) if str(job.get("ply_path") or "") else None,
                camera_rotation_x_deg=float(job.get("camera_rotation_x_deg", 90.0)),
                pointcloud_rotation_x_deg=float(job.get("pointcloud_rotation_x_deg", 90.0)),
                skip_missing_images=bool(job.get("skip_missing_images")),
                pre_undistort_distorted_images=bool(job.get("pre_undistort_distorted_images")),
                undistort_alpha=float(job.get("undistort_alpha", DEFAULT_UNDISTORT_ALPHA)),
            )
        else:
            if not args.csv_path:
                raise ValueError("csv_path is required unless --job is used")
            csv_path = Path(args.csv_path)
            default_name = DEFAULT_UNDISTORTED_DATASET_DIR_NAME if args.pre_undistort_distorted_images else DEFAULT_DATASET_DIR_NAME
            output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent / default_name
            result = convert(
                csv_path,
                output_dir,
                images_dir=Path(args.images_dir) if args.images_dir else None,
                masks_dir=Path(args.masks_dir) if args.masks_dir else None,
                ply_path=Path(args.ply) if args.ply else None,
                camera_rotation_x_deg=args.camera_rotation_x_deg,
                pointcloud_rotation_x_deg=args.pointcloud_rotation_x_deg,
                skip_missing_images=args.skip_missing_images,
                allow_mixed_loader_root=args.allow_mixed_loader_root,
                pre_undistort_distorted_images=args.pre_undistort_distorted_images,
                undistort_alpha=args.undistort_alpha,
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved COLMAP sparse text: {result['sparse_dir']}")
    for linked_asset in result["linked_assets"]:
        print(f"Linked dataset asset folder: {linked_asset}")
    if result["pre_undistort_distorted_images"]:
        stats = result["asset_stats"]
        print(
            "Pre-undistorted assets: "
            f"{stats.get('undistorted_images', 0)} images, {stats.get('undistorted_masks', 0)} masks; "
            f"linked {stats.get('linked_images', 0)} images, {stats.get('linked_masks', 0)} masks; "
            f"generated {stats.get('generated_valid_masks', 0)} valid masks"
        )
    if result["pointcloud"]:
        print(f"Saved LichtFeld points3D.ply: {result['pointcloud']}")
    print(f"Images: {result['num_images']} / CSV rows: {result['num_csv_rows']}")
    print(f"Cameras: {result['num_cameras']}")
    print(f"Camera X rotation: {result['camera_rotation_x_deg']} deg")
    print(f"Point cloud X rotation: {result['pointcloud_rotation_x_deg']} deg")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
