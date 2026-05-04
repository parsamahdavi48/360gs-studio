from pathlib import Path

import cv2
import numpy as np

from custom_mask import run


def test_custom_mask_merges_with_existing_masks_and_preserves_subfolders(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    nested_images = images / "nested"
    nested_masks = masks / "nested"
    nested_images.mkdir(parents=True)
    nested_masks.mkdir(parents=True)
    image_path = nested_images / "frame_0001.png"
    cv2.imwrite(str(image_path), np.full((4, 6, 3), 128, dtype=np.uint8))

    existing = np.full((4, 6), 255, dtype=np.uint8)
    existing[:, :2] = 0
    cv2.imwrite(str(nested_masks / "frame_0001.png"), existing)

    custom = np.full((4, 6), 255, dtype=np.uint8)
    custom[1:3, :] = 0
    custom_path = tmp_path / "custom.png"
    cv2.imwrite(str(custom_path), custom)

    errors = run(images, masks, custom_path)

    assert errors == []
    output = cv2.imread(str(nested_masks / "frame_0001.png"), cv2.IMREAD_GRAYSCALE)
    assert output is not None
    assert output[0, 0] == 0
    assert output[2, 5] == 0
    assert output[3, 5] == 255


def test_custom_mask_creates_mask_when_no_existing_mask(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    image_path = images / "frame_0001.png"
    cv2.imwrite(str(image_path), np.full((4, 6, 3), 128, dtype=np.uint8))
    custom = np.full((4, 6), 255, dtype=np.uint8)
    custom[:, 0] = 0
    custom_path = tmp_path / "custom.png"
    cv2.imwrite(str(custom_path), custom)

    errors = run(images, masks, custom_path)

    assert errors == []
    output = cv2.imread(str(masks / "frame_0001.png"), cv2.IMREAD_GRAYSCALE)
    assert output is not None
    assert np.array_equal(output, custom)


def test_custom_mask_reports_size_mismatch_without_writing(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    image_path = images / "frame_0001.png"
    cv2.imwrite(str(image_path), np.full((4, 6, 3), 128, dtype=np.uint8))
    custom_path = tmp_path / "custom.png"
    cv2.imwrite(str(custom_path), np.full((5, 6), 255, dtype=np.uint8))

    errors = run(images, masks, custom_path)

    assert len(errors) == 1
    assert "size mismatch" in errors[0]
    assert not (masks / "frame_0001.png").exists()
