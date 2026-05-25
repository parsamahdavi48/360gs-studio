import cv2
import numpy as np

from core.overexposure_mask import _init_worker, _process_one, detect_overexposure, run


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


def test_resized_existing_mask_stays_binary(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    existing_mask = tmp_path / "existing.png"
    output_mask = tmp_path / "output.png"
    cv2.imwrite(str(image_path), np.full((3, 3, 3), 128, dtype=np.uint8))
    cv2.imwrite(str(existing_mask), np.array([[0, 255], [255, 0]], dtype=np.uint8))
    _init_worker(254, 0)

    assert _process_one((str(image_path), str(output_mask), str(existing_mask))) is None

    written = cv2.imread(str(output_mask), cv2.IMREAD_GRAYSCALE)
    assert written is not None
    assert set(np.unique(written).tolist()) <= {0, 255}


def test_overexposure_replace_ignores_existing_mask(tmp_path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    image = np.full((4, 6, 3), 120, dtype=np.uint8)
    image[1, 2] = 255
    cv2.imwrite(str(images / "frame.png"), image)
    cv2.imwrite(str(masks / "frame.png"), np.zeros((4, 6), dtype=np.uint8))

    run(str(images), str(masks), threshold=254, dilate_px=0, workers=1, replace=True)

    written = cv2.imread(str(masks / "frame.png"), cv2.IMREAD_GRAYSCALE)
    assert written is not None
    assert written[1, 2] == 0
    assert written[3, 5] == 255
