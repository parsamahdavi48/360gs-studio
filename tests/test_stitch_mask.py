import numpy as np

from stitch_mask import (
    boundary_width_to_fov,
    boundary_width_to_limit_angle,
    create_angular_stitched_mask,
    resolve_limit_angle,
)


def test_boundary_width_five_matches_legacy_fov_175():
    assert boundary_width_to_fov(5.0) == 175.0
    assert boundary_width_to_limit_angle(5.0) == 87.5
    assert resolve_limit_angle(None, 5.0) == 87.5
    assert resolve_limit_angle(170.0, 5.0) == 85.0


def test_larger_boundary_width_excludes_more_pixels():
    narrow = create_angular_stitched_mask(240, 120, boundary_width_to_limit_angle(5.0))
    wide = create_angular_stitched_mask(240, 120, boundary_width_to_limit_angle(10.0))

    assert np.count_nonzero(wide == 0) > np.count_nonzero(narrow == 0)
