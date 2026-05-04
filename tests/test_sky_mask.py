from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sky_mask import (
    SkyMaskOptions,
    auto_view_size,
    detect_sky_mask,
    mask_output_path_for_image,
    postprocess_sky_components,
    process_image,
    resolve_model_source,
)


class FakeSkySegmenter:
    def detect_sky(self, bgr: np.ndarray, options: SkyMaskOptions) -> np.ndarray:
        del options
        return bgr[:, :, 0] > 200


def test_sky_mask_blackens_detected_sky_and_keeps_other_pixels() -> None:
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    image[:8, :, 0] = 255
    mask = detect_sky_mask(
        image,
        FakeSkySegmenter(),
        SkyMaskOptions(projection="normal", mode="direct", min_area_ratio=0.0, top_connected=True),
    )

    assert mask[2, 4] == 0
    assert mask[12, 4] == 255


def test_sky_mask_top_connected_removes_detached_components() -> None:
    sky = np.zeros((12, 16), dtype=bool)
    sky[:3, :] = True
    sky[8:10, 4:8] = True

    filtered = postprocess_sky_components(sky, min_area_ratio=0.0, expand_px=0, top_connected=True)

    assert filtered[1, 1]
    assert not filtered[8, 5]


def test_sky_mask_process_image_merges_with_existing_mask(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    image_path = images / "frame_0001.png"
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    image[:8, :, 0] = 255
    cv2.imwrite(str(image_path), image)
    existing = np.full((16, 24), 255, dtype=np.uint8)
    existing[10:12, 5:7] = 0
    cv2.imwrite(str(masks / "frame_0001.png"), existing)

    error = process_image(
        image_path,
        images,
        masks,
        FakeSkySegmenter(),
        SkyMaskOptions(projection="normal", mode="direct", min_area_ratio=0.0, top_connected=True),
    )

    assert error is None
    merged = cv2.imread(str(masks / "frame_0001.png"), cv2.IMREAD_GRAYSCALE)
    assert merged is not None
    assert merged[2, 4] == 0
    assert merged[10, 5] == 0
    assert merged[14, 12] == 255


def test_sky_mask_output_path_preserves_subfolders(tmp_path: Path) -> None:
    images = tmp_path / "images"
    image = images / "nested" / "frame.jpg"
    masks = tmp_path / "masks"

    assert mask_output_path_for_image(image, images, masks) == masks / "nested" / "frame.png"


def test_sky_mask_resolve_model_source_prefers_local_models_dir(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "mask2former-swin-large-ade-semantic"
    model_dir.mkdir(parents=True)

    assert resolve_model_source(repo_root=tmp_path) == str(model_dir)


def test_sky_mask_auto_view_size_is_bounded() -> None:
    assert auto_view_size(1024, 512) == 512
    assert auto_view_size(8192, 4096) == 2048
