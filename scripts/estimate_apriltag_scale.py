#!/usr/bin/env python3
"""Estimate scene scale from AprilTags detected in pinhole/cubemap images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.apriltag_detection import available_families
from core.apriltag_pipeline import AprilTagScaleRun, run_apriltag_scale_estimation
from core.apriltag_projection import EquirectProjectionConfig, camera_model, prepare_equirect_detection_dataset


def _parse_tag_ids(values: list[str]) -> set[int] | None:
    ids: set[int] = set()
    for value in values:
        for part in value.replace(",", " ").split():
            ids.add(int(part))
    return ids or None


def _report(run: AprilTagScaleRun, args: argparse.Namespace) -> dict:
    estimate = run.estimate
    observations = [
        {
            "tag_id": obs.tag_id,
            "frame_id": obs.frame_id,
            "score": obs.score,
            "camera_position_sfm": obs.camera_position_sfm.tolist(),
            "camera_to_tag_m": obs.camera_to_tag_m.tolist(),
        }
        for obs in run.observations
    ]
    frame_detections = [
        {
            "frame_id": item.frame.frame_id,
            "file_path": item.frame.file_path,
            "detection_count": len(item.detections),
            "detections": [
                {
                    "tag_id": detection.tag_id,
                    "score": detection.score,
                    "area_px": detection.area_px,
                    "center_px": list(detection.center_px),
                    "reprojection_error_px": detection.reprojection_error_px,
                    "camera_to_tag_m": detection.camera_to_tag_m.tolist(),
                }
                for detection in item.detections
            ],
        }
        for item in run.frame_detections
    ]
    return {
        "schema_version": 1,
        "transforms_json": str(args.transforms_json),
        "image_root": str(args.image_root) if args.image_root else None,
        "equirect_temp_dir": str(args.equirect_temp_dir) if args.equirect_temp_dir else None,
        "family": args.family,
        "tag_size_m": args.tag_size_m,
        "tag_ids": sorted(args.tag_ids) if args.tag_ids else None,
        "min_score": args.min_score,
        "estimate": {
            "scale": estimate.scale,
            "observation_count": estimate.observation_count,
            "pair_count": estimate.pair_count,
            "inlier_count": estimate.inlier_count,
            "rms_residual_m": estimate.rms_residual_m,
            "median_pair_scale": estimate.median_pair_scale,
            "mad_pair_scale": estimate.mad_pair_scale,
        },
        "observations": observations,
        "frames": frame_detections,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate meters-per-SfM-unit scale from AprilTags.")
    parser.add_argument("transforms_json", type=Path, help="PINHOLE/SIMPLE_PINHOLE or EQUIRECTANGULAR transforms.json")
    parser.add_argument("--image-root", type=Path, default=None, help="Image root for relative file_path entries")
    parser.add_argument("--tag-size-m", type=float, required=True, help="Physical AprilTag side length in meters")
    parser.add_argument("--family", default="tag36h11", choices=available_families(), help="AprilTag family")
    parser.add_argument("--tag-id", action="append", default=[], help="Allowed tag ID. May be repeated or comma-separated")
    parser.add_argument("--min-score", type=float, default=0.0, help="Drop observations below this score")
    parser.add_argument("--min-baseline-sfm", type=float, default=1e-6, help="Minimum camera baseline in SfM units")
    parser.add_argument("--report-json", type=Path, default=None, help="Write detailed JSON report")
    parser.add_argument(
        "--equirect-temp-dir",
        type=Path,
        default=None,
        help="Temporary output directory used when input transforms.json is EQUIRECTANGULAR",
    )
    parser.add_argument(
        "--equirect-output-scale",
        type=float,
        default=0.5,
        help="Temporary cubemap face size ratio for EQUIRECTANGULAR input",
    )
    parser.add_argument("--workers", default="auto", help="Temporary projection worker count: auto or positive integer")
    parser.add_argument(
        "--remap-cache-limit",
        default="auto",
        help="Temporary projection per-worker remap cache limit: auto or positive integer",
    )
    args = parser.parse_args()
    if not args.transforms_json.is_file():
        parser.error(f"transforms_json not found: {args.transforms_json}")
    if args.tag_size_m <= 0.0:
        parser.error("--tag-size-m must be positive")
    if args.min_score < 0.0:
        parser.error("--min-score must be non-negative")
    if args.min_baseline_sfm < 0.0:
        parser.error("--min-baseline-sfm must be non-negative")
    try:
        args.tag_ids = _parse_tag_ids(args.tag_id)
    except ValueError as e:
        parser.error(f"--tag-id must contain integers: {e}")
    return args


def _resolve_estimation_input(args: argparse.Namespace) -> Path:
    model = camera_model(args.transforms_json)
    if model in {"PINHOLE", "SIMPLE_PINHOLE"}:
        return args.transforms_json
    if model != "EQUIRECTANGULAR":
        raise ValueError(f"Unsupported camera_model for AprilTag scale estimation: {model or '-'}")
    if args.equirect_temp_dir is None:
        raise ValueError("EQUIRECTANGULAR input requires --equirect-temp-dir for temporary pinhole projection")
    return prepare_equirect_detection_dataset(
        EquirectProjectionConfig(
            transforms_json=args.transforms_json,
            output_dir=args.equirect_temp_dir,
            image_root=args.image_root,
            output_scale=args.equirect_output_scale,
            workers=args.workers,
            remap_cache_limit=args.remap_cache_limit,
        )
    )


def main() -> int:
    args = parse_args()
    try:
        estimation_transforms = _resolve_estimation_input(args)
        run = run_apriltag_scale_estimation(
            estimation_transforms,
            image_root=None,
            tag_size_m=args.tag_size_m,
            family=args.family,
            tag_ids=args.tag_ids,
            min_score=args.min_score,
            min_baseline_sfm=args.min_baseline_sfm,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    report = _report(run, args)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    estimate = report["estimate"]
    print(
        "AprilTag scale estimate: "
        f"scale={estimate['scale']:.9g}, "
        f"observations={estimate['observation_count']}, "
        f"pairs={estimate['pair_count']}, "
        f"inliers={estimate['inlier_count']}, "
        f"rms={estimate['rms_residual_m']:.6g} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
