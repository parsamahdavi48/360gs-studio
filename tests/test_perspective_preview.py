import cv2
import numpy as np

from gui.common.perspective_preview import PerspectiveParams, equirect_to_perspective, params_from_drag


def test_equirect_to_perspective_samples_front_center() -> None:
    img = np.zeros((80, 160, 3), dtype=np.uint8)
    img[:, 76:84] = (0, 0, 255)

    out = equirect_to_perspective(
        img,
        PerspectiveParams(yaw_deg=0.0, pitch_deg=0.0),
        output_size=40,
        interpolation=cv2.INTER_NEAREST,
    )

    assert out.shape == (40, 40, 3)
    assert int(out[20, 20, 2]) == 255


def test_perspective_params_from_drag_follows_grab_direction() -> None:
    params = params_from_drag(PerspectiveParams(yaw_deg=0.0, pitch_deg=0.0), 20, 20)

    assert params.yaw_deg < 0.0
    assert params.pitch_deg < 0.0


def test_perspective_params_from_drag_wraps_yaw_and_clamps_pitch() -> None:
    params = params_from_drag(PerspectiveParams(yaw_deg=-179.0, pitch_deg=88.0), -20, -20)

    assert -180.0 <= params.yaw_deg <= 180.0
    assert params.pitch_deg == 89.0
