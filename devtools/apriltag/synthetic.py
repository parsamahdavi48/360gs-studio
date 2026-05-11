"""Synthetic AprilTag injection for development validation."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from core.apriltag_cubemap import CubemapViewMetadata
from core.apriltag_geometry import load_pinhole_frames, points_intersect_image, project_sfm_points, tag_corners_sfm
from core.image_io import imread_unicode, imwrite_unicode


@dataclass(frozen=True)
class SyntheticAprilTagConfig:
    input_transforms: Path
    output_dir: Path
    tag_image: Path
    tag_size_m: float
    true_scale: float
    tag_center_sfm: np.ndarray
    tag_normal_sfm: np.ndarray
    tag_up_sfm: np.ndarray
    frame_file_paths: frozenset[str] | None = None
    copy_unselected_frames: bool = True
    output_tagged_only: bool = False
    cubemap_view_params: CubemapViewMetadata | Mapping[str, tuple[float, float]] | None = None
    frame_transform_overrides: Mapping[str, np.ndarray] | None = None
    write_normalized_transforms: bool = False


def _load_tag_rgba(path: Path) -> np.ndarray:
    tag = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if tag is None:
        raise ValueError(f"Failed to read tag image: {path}")
    if tag.ndim == 2:
        bgr = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
        alpha = np.full(tag.shape, 255, dtype=np.uint8)
    elif tag.shape[2] == 4:
        bgr = tag[:, :, :3]
        alpha = tag[:, :, 3]
    else:
        bgr = tag[:, :, :3]
        alpha = np.full(tag.shape[:2], 255, dtype=np.uint8)
    return np.dstack([bgr, alpha])


def _warp_tag(base: np.ndarray, tag_rgba: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    height, width = base.shape[:2]
    tag_h, tag_w = tag_rgba.shape[:2]
    src_points = np.array([[0, 0], [tag_w - 1, 0], [tag_w - 1, tag_h - 1], [0, tag_h - 1]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src_points, dst_points.astype(np.float32))
    warped = cv2.warpPerspective(tag_rgba, homography, (width, height), flags=cv2.INTER_LINEAR)

    base_bgr = base[:, :, :3] if base.ndim == 3 else cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    blended = base_bgr.astype(np.float32) * (1.0 - alpha) + warped[:, :, :3].astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def inject_synthetic_apriltag(config: SyntheticAprilTagConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_pinhole_frames(
        config.input_transforms,
        cubemap_view_params=config.cubemap_view_params,
    )
    if config.frame_transform_overrides:
        frames = tuple(
            replace(
                frame,
                transform_matrix=np.asarray(config.frame_transform_overrides[frame.file_path], dtype=float),
            )
            if frame.file_path in config.frame_transform_overrides
            else frame
            for frame in frames
        )
    frames_by_path = {frame.file_path: frame for frame in frames}
    allowed_paths = None if config.frame_file_paths is None else set(config.frame_file_paths)
    tag_rgba = _load_tag_rgba(config.tag_image)
    corners = tag_corners_sfm(
        config.tag_center_sfm,
        config.tag_normal_sfm,
        config.tag_up_sfm,
        config.tag_size_m,
        config.true_scale,
    )

    written = 0
    copied = 0
    skipped = 0
    written_paths: set[str] = set()
    for frame in frames:
        dst = config.output_dir / frame.file_path
        if allowed_paths is not None and frame.file_path not in allowed_paths:
            if config.copy_unselected_frames:
                if frame.image_path.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(frame.image_path, dst)
                    copied += 1
                else:
                    skipped += 1
            else:
                skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        image = imread_unicode(frame.image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            skipped += 1
            continue
        projected = project_sfm_points(frame, corners)
        if projected is None or not points_intersect_image(projected, image.shape[1], image.shape[0]):
            if config.copy_unselected_frames:
                shutil.copy2(frame.image_path, dst)
                copied += 1
            else:
                skipped += 1
            continue
        if not imwrite_unicode(dst, _warp_tag(image, tag_rgba, projected)):
            skipped += 1
            continue
        written += 1
        written_paths.add(frame.file_path)

    data = json.loads(config.input_transforms.read_text(encoding="utf-8"))
    frames_data = data.get("frames", [])
    if config.write_normalized_transforms and isinstance(frames_data, list):
        for frame_data in frames_data:
            if not isinstance(frame_data, dict):
                continue
            file_path = frame_data.get("file_path")
            frame = frames_by_path.get(file_path) if isinstance(file_path, str) else None
            if frame is not None:
                frame_data["transform_matrix"] = frame.transform_matrix.tolist()
    if config.output_tagged_only:
        if isinstance(frames_data, list):
            data["frames"] = [
                frame
                for frame in frames_data
                if isinstance(frame, dict) and frame.get("file_path") in written_paths
            ]
    (config.output_dir / "transforms.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    report = {
        "schema_version": 1,
        "input_transforms": str(config.input_transforms),
        "output_dir": str(config.output_dir),
        "tag_size_m": config.tag_size_m,
        "true_scale": config.true_scale,
        "tag_center_sfm": config.tag_center_sfm.tolist(),
        "tag_normal_sfm": config.tag_normal_sfm.tolist(),
        "tag_up_sfm": config.tag_up_sfm.tolist(),
        "frame_file_paths": None if config.frame_file_paths is None else sorted(config.frame_file_paths),
        "copy_unselected_frames": config.copy_unselected_frames,
        "output_tagged_only": config.output_tagged_only,
        "frame_transform_override_count": 0
        if config.frame_transform_overrides is None
        else len(config.frame_transform_overrides),
        "write_normalized_transforms": config.write_normalized_transforms,
        "frames_written": written,
        "frames_copied": copied,
        "frames_skipped": skipped,
        "tagged_frame_file_paths": sorted(written_paths),
        "transforms_frame_count": len(data.get("frames", [])) if isinstance(data.get("frames"), list) else 0,
    }
    (config.output_dir / "synthetic_apriltag_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
