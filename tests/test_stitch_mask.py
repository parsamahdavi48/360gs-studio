import cv2
import numpy as np

from stitch_mask import (
    boundary_width_to_fov,
    boundary_width_to_limit_angle,
    create_angular_stitched_mask,
    init_worker,
    process_single_image,
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


def test_resized_input_mask_stays_binary(tmp_path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    source = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    cv2.imwrite(str(input_path), source)
    init_worker(np.full((3, 3), 255, dtype=np.uint8))

    assert process_single_image((str(input_path), str(output_path))) is None

    written = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    assert written is not None
    assert set(np.unique(written).tolist()) <= {0, 255}
