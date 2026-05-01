import numpy as np

from overexposure_mask import detect_overexposure


def test_overexposure_default_only_masks_pure_white_without_dilate() -> None:
    image = np.array(
        [
            [[255, 255, 255], [254, 255, 255], [254, 254, 254]],
        ],
        dtype=np.uint8,
    )

    mask = detect_overexposure(image, dilate_px=0)

    assert mask[0, 0] == 0
    assert mask[0, 1] == 255
    assert mask[0, 2] == 255


def test_overexposure_threshold_255_masks_nothing() -> None:
    image = np.full((1, 1, 3), 255, dtype=np.uint8)

    mask = detect_overexposure(image, threshold=255, dilate_px=0)

    assert mask[0, 0] == 255


def test_overexposure_scales_threshold_for_16bit_images() -> None:
    image = np.array(
        [
            [[65535, 65535, 65535], [65278, 65535, 65535], [32768, 32768, 32768]],
        ],
        dtype=np.uint16,
    )

    mask = detect_overexposure(image, threshold=254, dilate_px=0)

    assert mask[0, 0] == 0
    assert mask[0, 1] == 255
    assert mask[0, 2] == 255


def test_overexposure_threshold_255_masks_nothing_for_16bit_images() -> None:
    image = np.full((1, 1, 3), 65535, dtype=np.uint16)

    mask = detect_overexposure(image, threshold=255, dilate_px=0)

    assert mask[0, 0] == 255
