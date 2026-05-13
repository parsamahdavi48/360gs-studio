import cv2
import numpy as np

from gui.common.perspective_image_view import PerspectiveLabelOverlay, mirror_perspective_overlay_x
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


def test_equirect_to_perspective_can_flip_screen_x_contract() -> None:
    img = np.zeros((80, 160, 3), dtype=np.uint8)
    img[:, :, 0] = np.arange(160, dtype=np.uint8).reshape(1, 160)

    normal = equirect_to_perspective(
        img,
        PerspectiveParams(yaw_deg=0.0, pitch_deg=0.0),
        output_size=40,
        interpolation=cv2.INTER_NEAREST,
    )
    flipped = equirect_to_perspective(
        img,
        PerspectiveParams(yaw_deg=0.0, pitch_deg=0.0),
        output_size=40,
        interpolation=cv2.INTER_NEAREST,
        screen_x_sign=-1.0,
    )

    assert np.array_equal(flipped, normal[:, ::-1])


def test_mirror_perspective_overlay_x_keeps_text_readable_coordinates() -> None:
    overlay = PerspectiveLabelOverlay(
        label="tag",
        box=(10, 20, 30, 40),
        origin=(12, 18),
        color_bgr=(0, 255, 0),
        polygon=((10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)),
        polyline=((10.0, 20.0), (30.0, 40.0)),
        points=((15.0, 25.0),),
    )

    mirrored = mirror_perspective_overlay_x(overlay, 100)

    assert mirrored.box == (70, 20, 90, 40)
    assert mirrored.origin == (72, 18)
    assert np.allclose(mirrored.polygon, ((90.0, 20.0), (70.0, 20.0), (70.0, 40.0), (90.0, 40.0)))
    assert np.allclose(mirrored.polyline, ((90.0, 20.0), (70.0, 40.0)))
    assert np.allclose(mirrored.points, ((85.0, 25.0),))


def test_perspective_params_from_drag_follows_grab_direction() -> None:
    params = params_from_drag(PerspectiveParams(yaw_deg=0.0, pitch_deg=0.0), 20, 20)

    assert params.yaw_deg < 0.0
    assert params.pitch_deg < 0.0


def test_perspective_params_from_drag_wraps_yaw_and_clamps_pitch() -> None:
    params = params_from_drag(PerspectiveParams(yaw_deg=-179.0, pitch_deg=88.0), -20, -20)

    assert -180.0 <= params.yaw_deg <= 180.0
    assert params.pitch_deg == 89.0
