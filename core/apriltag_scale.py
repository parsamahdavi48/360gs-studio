"""AprilTag scale-estimation primitives.

Detection and GUI code should stay outside this module. Convert each accepted
tag detection into :class:`TagObservation`, then call :func:`estimate_scene_scale`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import exp, isfinite
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TagObservation:
    """One PnP observation of one physical AprilTag."""

    tag_id: int
    frame_id: str
    camera_position_sfm: np.ndarray
    camera_to_world_rotation: np.ndarray
    camera_to_tag_m: np.ndarray
    score: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_position_sfm", _vec3(self.camera_position_sfm, "camera_position_sfm"))
        object.__setattr__(
            self,
            "camera_to_world_rotation",
            _mat3(self.camera_to_world_rotation, "camera_to_world_rotation"),
        )
        object.__setattr__(self, "camera_to_tag_m", _vec3(self.camera_to_tag_m, "camera_to_tag_m"))
        score = float(self.score)
        object.__setattr__(self, "score", score if isfinite(score) and score > 0.0 else 0.0)


@dataclass(frozen=True)
class ScalePair:
    """Scale sample derived from two observations of the same tag."""

    tag_id: int
    frame_a: str
    frame_b: str
    scale: float
    weight: float
    baseline_sfm: float
    residual_m: float = 0.0


@dataclass(frozen=True)
class ScaleEstimate:
    """Robust scene scale estimate in meters per SfM unit."""

    scale: float
    observation_count: int
    pair_count: int
    inlier_count: int
    rms_residual_m: float
    median_pair_scale: float
    mad_pair_scale: float
    pairs: tuple[ScalePair, ...]


def _vec3(value: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def _mat3(value: np.ndarray | Iterable[Iterable[float]], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 matrix")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def score_tag_observation(
    *,
    tag_area_px: float,
    image_size: tuple[int, int],
    center_px: tuple[float, float],
    reprojection_error_px: float,
    reference_area_ratio: float = 0.02,
    reference_error_px: float = 2.0,
) -> float:
    """Score a detected tag using size, reprojection error, and centrality."""
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")

    area_ratio = max(0.0, float(tag_area_px)) / float(width * height)
    size_score = min(1.0, area_ratio / max(reference_area_ratio, 1e-12))

    error = max(0.0, float(reprojection_error_px))
    reprojection_score = exp(-error / max(reference_error_px, 1e-12))

    cx, cy = center_px
    nx = (float(cx) - (width - 1) / 2.0) / max(width / 2.0, 1e-12)
    ny = (float(cy) - (height - 1) / 2.0) / max(height / 2.0, 1e-12)
    center_distance = min(1.0, float(np.hypot(nx, ny)))
    center_score = max(0.0, 1.0 - center_distance)

    return size_score * reprojection_score * center_score


def build_scale_pairs(
    observations: Iterable[TagObservation],
    *,
    min_baseline_sfm: float = 1e-6,
) -> tuple[ScalePair, ...]:
    """Build pairwise scale samples from repeated observations of each tag."""
    grouped: dict[int, list[TagObservation]] = {}
    for obs in observations:
        grouped.setdefault(obs.tag_id, []).append(obs)

    pairs: list[ScalePair] = []
    min_baseline = max(0.0, float(min_baseline_sfm))
    for tag_id, tag_observations in grouped.items():
        for a, b in combinations(tag_observations, 2):
            camera_delta = a.camera_position_sfm - b.camera_position_sfm
            baseline = float(np.linalg.norm(camera_delta))
            if baseline <= min_baseline:
                continue

            metric_delta = b.camera_to_world_rotation @ b.camera_to_tag_m - a.camera_to_world_rotation @ a.camera_to_tag_m
            denom = float(camera_delta @ camera_delta)
            if denom <= 0.0:
                continue
            scale = float((camera_delta @ metric_delta) / denom)
            if not isfinite(scale) or scale <= 0.0:
                continue

            residual_vec = scale * camera_delta - metric_delta
            weight = max(a.score, 0.0) * max(b.score, 0.0) * baseline * baseline
            if weight <= 0.0:
                continue
            pairs.append(
                ScalePair(
                    tag_id=tag_id,
                    frame_a=a.frame_id,
                    frame_b=b.frame_id,
                    scale=scale,
                    weight=weight,
                    baseline_sfm=baseline,
                    residual_m=float(np.linalg.norm(residual_vec)),
                )
            )

    return tuple(pairs)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    total = float(np.sum(sorted_weights))
    if total <= 0.0:
        return float(np.median(values))
    cutoff = total * 0.5
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    index = min(index, len(sorted_values) - 1)
    return float(sorted_values[index])


def estimate_scene_scale(
    observations: Iterable[TagObservation],
    *,
    min_baseline_sfm: float = 1e-6,
    mad_sigma: float = 3.5,
) -> ScaleEstimate:
    """Estimate meters-per-SfM-unit scale from AprilTag observations."""
    obs = tuple(observations)
    pairs = build_scale_pairs(obs, min_baseline_sfm=min_baseline_sfm)
    if not pairs:
        raise ValueError("At least two valid observations of the same tag with camera baseline are required")

    scales = np.array([pair.scale for pair in pairs], dtype=float)
    weights = np.array([pair.weight for pair in pairs], dtype=float)
    median = _weighted_median(scales, weights)
    mad = _weighted_median(np.abs(scales - median), weights)

    if mad <= 1e-12:
        inlier_mask = np.abs(scales - median) <= 1e-9
        if not np.any(inlier_mask):
            inlier_mask = np.ones_like(scales, dtype=bool)
    else:
        robust_sigma = 1.4826 * mad
        inlier_mask = np.abs(scales - median) <= max(mad_sigma, 0.0) * robust_sigma
        if not np.any(inlier_mask):
            inlier_mask = np.ones_like(scales, dtype=bool)

    scale = float(np.average(scales[inlier_mask], weights=weights[inlier_mask]))
    residuals = [pair.residual_m for pair, inlier in zip(pairs, inlier_mask, strict=True) if inlier]
    rms = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else 0.0

    return ScaleEstimate(
        scale=scale,
        observation_count=len(obs),
        pair_count=len(pairs),
        inlier_count=int(np.count_nonzero(inlier_mask)),
        rms_residual_m=rms,
        median_pair_scale=median,
        mad_pair_scale=mad,
        pairs=pairs,
    )
