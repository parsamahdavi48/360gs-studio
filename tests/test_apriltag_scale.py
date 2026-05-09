from __future__ import annotations

import math

import numpy as np

from core.apriltag_scale import TagObservation, estimate_scene_scale, score_tag_observation


def _obs(frame_id: str, camera_x: float, camera_to_tag_x_m: float, *, score: float = 1.0) -> TagObservation:
    return TagObservation(
        tag_id=7,
        frame_id=frame_id,
        camera_position_sfm=np.array([camera_x, 0.0, 0.0], dtype=float),
        camera_to_world_rotation=np.eye(3),
        camera_to_tag_m=np.array([camera_to_tag_x_m, 0.0, 2.0], dtype=float),
        score=score,
    )


def test_estimate_scene_scale_from_repeated_tag_observations() -> None:
    estimate = estimate_scene_scale(
        [
            _obs("a", 0.0, 2.0),
            _obs("b", 4.0, 1.0),
            _obs("c", 8.0, 0.0),
        ]
    )

    assert estimate.observation_count == 3
    assert estimate.pair_count == 3
    assert estimate.inlier_count == 3
    assert estimate.scale == 0.25
    assert estimate.rms_residual_m == 0.0


def test_estimate_scene_scale_rejects_large_pair_outlier() -> None:
    estimate = estimate_scene_scale(
        [
            _obs("a", 0.0, 2.0),
            _obs("b", 4.0, 1.0),
            _obs("c", 8.0, 0.0),
            _obs("bad", 12.0, -2.0, score=0.001),
        ]
    )

    assert math.isclose(estimate.scale, 0.25, rel_tol=1e-9)
    assert estimate.inlier_count < estimate.pair_count


def test_score_tag_observation_prefers_large_centered_low_error_tags() -> None:
    good = score_tag_observation(
        tag_area_px=20000,
        image_size=(1000, 1000),
        center_px=(500, 500),
        reprojection_error_px=0.2,
    )
    poor = score_tag_observation(
        tag_area_px=2000,
        image_size=(1000, 1000),
        center_px=(800, 800),
        reprojection_error_px=5.0,
    )

    assert 0.0 < poor < good <= 1.0
