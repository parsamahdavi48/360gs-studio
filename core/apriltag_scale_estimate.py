"""Estimate scene scale from AprilTags detected in pinhole/Cubemap images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.apriltag_colmap_dataset import validate_colmap_apriltag_dataset
from core.apriltag_cubemap import CUBEMAP_POSE_PRESETS, cubemap_view_metadata_for_pose_preset
from core.apriltag_detection import available_families
from core.apriltag_pipeline import AprilTagScaleRun, run_apriltag_scale_estimation
from core.apriltag_projection import camera_model


def parse_tag_id_values(values: list[str]) -> set[int] | None:
    ids: set[int] = set()
    for value in values:
        for part in value.replace(",", " ").split():
            ids.add(int(part))
    return ids or None


def build_report(run: AprilTagScaleRun, args: argparse.Namespace) -> dict:
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
        "input_dataset": str(args.dataset),
        "transforms_json": str(args.dataset) if args.dataset.is_file() else None,
        "image_root": str(args.image_root) if args.image_root else None,
        "family": args.family,
        "tag_size_m": args.tag_size_m,
        "tag_ids": sorted(args.tag_ids) if args.tag_ids else None,
        "min_score": args.min_score,
        "workers": args.workers,
        "cubemap_pose_preset": args.cubemap_pose_preset,
        "timings_sec": dict(run.timings_sec),
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate meters-per-SfM-unit scale from AprilTags.")
    parser.add_argument(
        "dataset",
        type=Path,
        help="PINHOLE/SIMPLE_PINHOLE Cubemap transforms.json or COLMAP dataset root",
    )
    parser.add_argument("--image-root", type=Path, default=None, help="Image root for relative file_path entries")
    parser.add_argument("--tag-size-m", type=float, required=True, help="Physical AprilTag side length in meters")
    parser.add_argument("--family", default="tag36h11", choices=available_families(), help="AprilTag family")
    parser.add_argument("--tag-id", action="append", default=[], help="Allowed tag ID. May be repeated or comma-separated")
    parser.add_argument("--min-score", type=float, default=0.0, help="Drop observations below this score")
    parser.add_argument("--min-baseline-sfm", type=float, default=1e-6, help="Minimum camera baseline in SfM units")
    parser.add_argument("--report-json", type=Path, default=None, help="Write detailed JSON report")
    parser.add_argument("--workers", default="auto", help="AprilTag detection worker count: auto or positive integer")
    parser.add_argument(
        "--cubemap-pose-preset",
        choices=CUBEMAP_POSE_PRESETS,
        default="auto",
        help="Conversion preset used to create the Cube6 output. Usually keep auto.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.dataset.exists():
        parser.error(f"dataset not found: {args.dataset}")
    if args.tag_size_m <= 0.0:
        parser.error("--tag-size-m must be positive")
    if args.min_score < 0.0:
        parser.error("--min-score must be non-negative")
    if args.min_baseline_sfm < 0.0:
        parser.error("--min-baseline-sfm must be non-negative")
    try:
        args.tag_ids = parse_tag_id_values(args.tag_id)
    except ValueError as e:
        parser.error(f"--tag-id must contain integers: {e}")
    return args


def resolve_estimation_input(args: argparse.Namespace) -> Path:
    if args.dataset.is_dir():
        validate_colmap_apriltag_dataset(args.dataset, images_dir=args.image_root)
        return args.dataset
    model = camera_model(args.dataset)
    if model in {"PINHOLE", "SIMPLE_PINHOLE"}:
        return args.dataset
    if model == "EQUIRECTANGULAR":
        raise ValueError(
            "AprilTag scale estimation requires projected Cubemap output images. "
            "Run Step 4 with Cubemap image output first, then use output/metashape_cubemap/transforms.json."
        )
    raise ValueError(f"Unsupported camera_model for AprilTag scale estimation: {model or '-'}")


def _log(message: str) -> None:
    print(f"[apriltag] {message}", flush=True)


def _progress(done: int, total: int) -> None:
    print(f"[progress] {done}/{total}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _log("validating input dataset")
        estimation_input = resolve_estimation_input(args)
        cubemap_view_params = cubemap_view_metadata_for_pose_preset(args.cubemap_pose_preset)
        run = run_apriltag_scale_estimation(
            estimation_input,
            image_root=args.image_root,
            tag_size_m=args.tag_size_m,
            family=args.family,
            tag_ids=args.tag_ids,
            min_score=args.min_score,
            min_baseline_sfm=args.min_baseline_sfm,
            cubemap_view_params=cubemap_view_params,
            workers=args.workers,
            progress_callback=_progress,
            log_callback=_log,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    report = build_report(run, args)
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
