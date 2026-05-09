#!/usr/bin/env python3
"""Inject a supplied AprilTag image into a cubemap transforms.json dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.apriltag_synthetic import SyntheticAprilTagConfig, inject_synthetic_apriltag


def _parse_vec3(text: str, name: str) -> np.ndarray:
    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{name} must contain three numbers")
    try:
        vec = np.array([float(p) for p in parts], dtype=float)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{name} must contain three numbers") from e
    if not np.all(np.isfinite(vec)):
        raise argparse.ArgumentTypeError(f"{name} must contain finite numbers")
    return vec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject a supplied AprilTag image into cubemap frames for scale tests.")
    parser.add_argument("input_transforms", type=Path, help="Input cubemap transforms.json")
    parser.add_argument("output_dir", type=Path, help="Output dataset directory")
    parser.add_argument("--tag-image", type=Path, required=True, help="AprilTag PNG/JPG to warp into visible frames")
    parser.add_argument("--tag-size-m", type=float, required=True, help="Physical tag side length in meters")
    parser.add_argument("--true-scale", type=float, required=True, help="Synthetic ground-truth meters per SfM unit")
    parser.add_argument("--tag-center-sfm", type=lambda s: _parse_vec3(s, "--tag-center-sfm"), required=True)
    parser.add_argument(
        "--tag-normal-sfm",
        type=lambda s: _parse_vec3(s, "--tag-normal-sfm"),
        default=np.array([0.0, 0.0, -1.0], dtype=float),
    )
    parser.add_argument(
        "--tag-up-sfm",
        type=lambda s: _parse_vec3(s, "--tag-up-sfm"),
        default=np.array([0.0, 1.0, 0.0], dtype=float),
    )
    args = parser.parse_args()
    if args.tag_size_m <= 0.0:
        parser.error("--tag-size-m must be positive")
    if args.true_scale <= 0.0:
        parser.error("--true-scale must be positive")
    if not args.input_transforms.is_file():
        parser.error(f"Input transforms.json not found: {args.input_transforms}")
    if not args.tag_image.is_file():
        parser.error(f"Tag image not found: {args.tag_image}")
    return args


def main() -> int:
    args = parse_args()
    config = SyntheticAprilTagConfig(
        input_transforms=args.input_transforms,
        output_dir=args.output_dir,
        tag_image=args.tag_image,
        tag_size_m=args.tag_size_m,
        true_scale=args.true_scale,
        tag_center_sfm=args.tag_center_sfm,
        tag_normal_sfm=args.tag_normal_sfm,
        tag_up_sfm=args.tag_up_sfm,
    )
    try:
        report = inject_synthetic_apriltag(config)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    print(
        "Synthetic AprilTag injection complete: "
        f"{report['frames_written']} tagged, {report['frames_skipped']} copied/skipped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
