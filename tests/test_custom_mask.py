import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from core.custom_mask import run


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

    result = run(images, masks, custom_path)

    assert result.ok
    assert result.applied == 1
    assert result.skipped == 0
    output = cv2.imread(str(nested_masks / "frame_0001.png"), cv2.IMREAD_GRAYSCALE)
    assert output is not None
    assert output[0, 0] == 0
    assert output[2, 5] == 0
    assert output[3, 5] == 255


def test_custom_mask_replace_ignores_existing_masks(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    cv2.imwrite(str(images / "frame_0001.png"), np.full((4, 6, 3), 128, dtype=np.uint8))
    cv2.imwrite(str(masks / "frame_0001.png"), np.zeros((4, 6), dtype=np.uint8))
    custom = np.full((4, 6), 255, dtype=np.uint8)
    custom[:, 0] = 0
    custom_path = tmp_path / "custom.png"
    cv2.imwrite(str(custom_path), custom)

    result = run(images, masks, custom_path, replace=True)

    assert result.ok
    output = cv2.imread(str(masks / "frame_0001.png"), cv2.IMREAD_GRAYSCALE)
    assert output is not None
    assert np.array_equal(output, custom)


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

    result = run(images, masks, custom_path)

    assert result.ok
    assert result.applied == 1
    assert result.skipped == 0
    output = cv2.imread(str(masks / "frame_0001.png"), cv2.IMREAD_GRAYSCALE)
    assert output is not None
    assert np.array_equal(output, custom)


def test_custom_mask_binarizes_8bit_grayscale_png(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    cv2.imwrite(str(images / "frame_0001.png"), np.full((1, 4, 3), 128, dtype=np.uint8))
    custom_path = tmp_path / "custom.png"
    cv2.imwrite(str(custom_path), np.array([[0, 127, 128, 255]], dtype=np.uint8))

    result = run(images, masks, custom_path)

    assert result.ok
    output = cv2.imread(str(masks / "frame_0001.png"), cv2.IMREAD_GRAYSCALE)
    assert output is not None
    assert output.dtype == np.uint8
    assert np.array_equal(output, np.array([[0, 0, 255, 255]], dtype=np.uint8))


def test_custom_mask_binarizes_16bit_grayscale_png(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    cv2.imwrite(str(images / "frame_0001.png"), np.full((1, 4, 3), 128, dtype=np.uint8))
    custom_path = tmp_path / "custom.png"
    cv2.imwrite(str(custom_path), np.array([[0, 32767, 32768, 65535]], dtype=np.uint16))

    result = run(images, masks, custom_path)

    assert result.ok
    output = cv2.imread(str(masks / "frame_0001.png"), cv2.IMREAD_GRAYSCALE)
    assert output is not None
    assert output.dtype == np.uint8
    assert np.array_equal(output, np.array([[0, 0, 255, 255]], dtype=np.uint8))


def test_custom_mask_converts_rgba_png_and_ignores_alpha(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    cv2.imwrite(str(images / "frame_0001.png"), np.full((2, 2, 3), 128, dtype=np.uint8))
    custom = np.array(
        [
            [[255, 255, 255, 0], [0, 0, 0, 255]],
            [[255, 255, 255, 255], [0, 0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    custom_path = tmp_path / "custom.png"
    cv2.imwrite(str(custom_path), custom)

    result = run(images, masks, custom_path)

    assert result.ok
    output = cv2.imread(str(masks / "frame_0001.png"), cv2.IMREAD_GRAYSCALE)
    assert output is not None
    assert np.array_equal(output, np.array([[255, 0], [255, 0]], dtype=np.uint8))


def test_custom_mask_rejects_non_png_input(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    cv2.imwrite(str(images / "frame_0001.png"), np.full((4, 6, 3), 128, dtype=np.uint8))
    custom_path = tmp_path / "custom.jpg"
    cv2.imwrite(str(custom_path), np.full((4, 6), 255, dtype=np.uint8))

    result = run(images, masks, custom_path)

    assert not result.ok
    assert result.failed == 1
    assert any("PNG" in message for message in result.messages)
    assert not (masks / "frame_0001.png").exists()


def test_custom_mask_skips_size_mismatch_without_writing(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    image_path = images / "frame_0001.png"
    cv2.imwrite(str(image_path), np.full((4, 6, 3), 128, dtype=np.uint8))
    custom_path = tmp_path / "custom.png"
    cv2.imwrite(str(custom_path), np.full((5, 6), 255, dtype=np.uint8))

    result = run(images, masks, custom_path)

    assert not result.ok
    assert result.applied == 0
    assert result.skipped == 1
    assert result.failed == 0
    assert "size mismatch" in result.messages[0]
    assert not (masks / "frame_0001.png").exists()


def test_custom_mask_skips_unreadable_source_image(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    (images / "broken.jpg").write_bytes(b"not an image")
    custom_path = tmp_path / "custom.png"
    cv2.imwrite(str(custom_path), np.full((4, 6), 255, dtype=np.uint8))

    result = run(images, masks, custom_path)

    assert not result.ok
    assert result.applied == 0
    assert result.skipped == 1
    assert result.failed == 0
    assert "image read error" in result.messages[0]
    assert not (masks / "broken.png").exists()


def test_custom_mask_partial_size_mismatch_succeeds_and_skips_only_mismatches(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    cv2.imwrite(str(images / "small.png"), np.full((4, 6, 3), 128, dtype=np.uint8))
    cv2.imwrite(str(images / "large.png"), np.full((8, 12, 3), 128, dtype=np.uint8))
    custom = np.full((4, 6), 255, dtype=np.uint8)
    custom[:, 0] = 0
    custom_path = tmp_path / "custom.png"
    cv2.imwrite(str(custom_path), custom)

    result = run(images, masks, custom_path)

    assert result.ok
    assert result.applied == 1
    assert result.skipped == 1
    assert result.failed == 0
    assert (masks / "small.png").is_file()
    assert not (masks / "large.png").exists()


def test_custom_mask_cli_succeeds_for_partial_size_mismatch(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    cv2.imwrite(str(images / "small.png"), np.full((4, 6, 3), 128, dtype=np.uint8))
    cv2.imwrite(str(images / "large.png"), np.full((8, 12, 3), 128, dtype=np.uint8))
    custom_path = tmp_path / "custom.png"
    cv2.imwrite(str(custom_path), np.full((4, 6), 255, dtype=np.uint8))

    result = subprocess.run(
        [sys.executable, "-u", "-m", "core.custom_mask", str(images), str(masks), str(custom_path)],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 applied, 1 skipped, 0 failed" in result.stdout


def test_custom_mask_cli_fails_when_all_images_are_size_mismatched(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    cv2.imwrite(str(images / "large.png"), np.full((8, 12, 3), 128, dtype=np.uint8))
    custom_path = tmp_path / "custom.png"
    cv2.imwrite(str(custom_path), np.full((4, 6), 255, dtype=np.uint8))

    result = subprocess.run(
        [sys.executable, "-u", "-m", "core.custom_mask", str(images), str(masks), str(custom_path)],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "0 applied, 1 skipped, 0 failed" in result.stdout
    assert "nothing was written" in result.stdout
