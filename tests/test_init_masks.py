from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from init_masks import run


def test_init_masks_writes_white_masks_and_preserves_subfolders(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    nested = images / "nested"
    nested.mkdir(parents=True)
    cv2.imwrite(str(images / "frame_0001.png"), np.full((12, 16, 3), 120, dtype=np.uint8))
    cv2.imwrite(str(nested / "frame_0002.jpg"), np.full((10, 14, 3), 80, dtype=np.uint8))
    old = np.zeros((12, 16), dtype=np.uint8)
    masks.mkdir()
    cv2.imwrite(str(masks / "frame_0001.png"), old)

    assert run(images, masks) == 0

    first = cv2.imread(str(masks / "frame_0001.png"), cv2.IMREAD_GRAYSCALE)
    second = cv2.imread(str(masks / "nested" / "frame_0002.png"), cv2.IMREAD_GRAYSCALE)
    assert first is not None
    assert second is not None
    assert first.shape == (12, 16)
    assert second.shape == (10, 14)
    assert np.all(first == 255)
    assert np.all(second == 255)


def test_init_masks_skips_unreadable_images_without_writing(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    (images / "broken.jpg").write_bytes(b"not an image")

    assert run(images, masks) == 1

    assert not (masks / "broken.png").exists()
