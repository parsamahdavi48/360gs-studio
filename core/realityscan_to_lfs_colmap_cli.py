"""CLI adapter for RealityScan CSV/PLY to LichtFeld COLMAP conversion."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from core.dataset_job_spec import JOB_KIND_REALITYSCAN_LFS_COLMAP, load_dataset_job
from core.realityscan_to_lfs_colmap import (
    DEFAULT_DATASET_DIR_NAME,
    DEFAULT_UNDISTORT_ALPHA,
    DEFAULT_UNDISTORTED_DATASET_DIR_NAME,
    convert,
)


def build_arg_parser() -> argparse.ArgumentParser:
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
        help="X-axis rotation applied only to the COLMAP point cloud (default: 90)",
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
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
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
            default_name = (
                DEFAULT_UNDISTORTED_DATASET_DIR_NAME
                if args.pre_undistort_distorted_images
                else DEFAULT_DATASET_DIR_NAME
            )
            result = convert(
                csv_path,
                Path(args.output_dir) if args.output_dir else csv_path.parent / default_name,
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
        print(f"Saved COLMAP points3D.txt: {result['pointcloud']}")
    print(f"Images: {result['num_images']} / CSV rows: {result['num_csv_rows']}")
    print(f"Cameras: {result['num_cameras']}")
    print(f"Camera X rotation: {result['camera_rotation_x_deg']} deg")
    print(f"Point cloud X rotation: {result['pointcloud_rotation_x_deg']} deg")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
