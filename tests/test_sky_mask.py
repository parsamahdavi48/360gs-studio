from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sky_mask import (
    BACKEND_SAM31,
    DEFAULT_SAM31_CHECKPOINT_NAME,
    DetectedRegionMasks,
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


def test_sky_postprocess_does_not_remove_non_sky_targets() -> None:
    class SplitSegmenter:
        def detect_prompt_masks(
            self,
            bgr: np.ndarray,
            options: SkyMaskOptions,
            *,
            sky_prompts: tuple[str, ...],
            other_prompts: tuple[str, ...],
        ) -> DetectedRegionMasks:
            del bgr, options, sky_prompts, other_prompts
            sky = np.zeros((12, 16), dtype=bool)
            sky[8:10, 4:8] = True
            other = np.zeros((12, 16), dtype=bool)
            other[9:11, 10:14] = True
            return DetectedRegionMasks(sky=sky, other=other)

    image = np.zeros((12, 16, 3), dtype=np.uint8)

    mask = detect_sky_mask(
        image,
        SplitSegmenter(),
        SkyMaskOptions(
            projection="normal",
            mode="direct",
            min_area_ratio=0.0,
            top_connected=True,
            sam_prompts=("sky", "person"),
        ),
    )

    assert mask[8, 5] == 255
    assert mask[9, 11] == 0


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


def test_sky_mask_replace_ignores_existing_mask(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    image_path = images / "frame_0001.png"
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    image[:8, :, 0] = 255
    cv2.imwrite(str(image_path), image)
    existing = np.zeros((16, 24), dtype=np.uint8)
    cv2.imwrite(str(masks / "frame_0001.png"), existing)

    error = process_image(
        image_path,
        images,
        masks,
        FakeSkySegmenter(),
        SkyMaskOptions(projection="normal", mode="direct", min_area_ratio=0.0, top_connected=True, replace=True),
    )

    assert error is None
    replaced = cv2.imread(str(masks / "frame_0001.png"), cv2.IMREAD_GRAYSCALE)
    assert replaced is not None
    assert replaced[2, 4] == 0
    assert replaced[14, 12] == 255


def test_sky_mask_output_path_preserves_subfolders(tmp_path: Path) -> None:
    images = tmp_path / "images"
    image = images / "nested" / "frame.jpg"
    masks = tmp_path / "masks"

    assert mask_output_path_for_image(image, images, masks) == masks / "nested" / "frame.png"


def test_sky_mask_resolve_model_source_prefers_local_models_dir(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "mask2former-swin-large-ade-semantic"
    model_dir.mkdir(parents=True)

    assert resolve_model_source(repo_root=tmp_path) == str(model_dir)


def test_sky_mask_resolve_model_source_prefers_local_sam31_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "models" / "sam3.1" / DEFAULT_SAM31_CHECKPOINT_NAME
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    assert resolve_model_source(repo_root=tmp_path, backend=BACKEND_SAM31) == str(checkpoint)


def test_sky_mask_resolve_model_source_accepts_sam31_model_dir_override(tmp_path: Path) -> None:
    model_dir = tmp_path / "sam3.1"
    model_dir.mkdir()

    assert resolve_model_source(model_dir=model_dir, backend=BACKEND_SAM31) == str(model_dir)


def test_sky_mask_auto_view_size_is_bounded() -> None:
    assert auto_view_size(1024, 512) == 512
    assert auto_view_size(8192, 4096) == 2048


def test_sky_mask_normal_quality_uses_tiles_without_pole_projection() -> None:
    class CountingSegmenter:
        def __init__(self) -> None:
            self.shapes: list[tuple[int, int]] = []

        def detect_sky(self, bgr: np.ndarray, options: SkyMaskOptions) -> np.ndarray:
            del options
            self.shapes.append(bgr.shape[:2])
            return np.zeros(bgr.shape[:2], dtype=bool)

    image = np.zeros((20, 40, 3), dtype=np.uint8)
    segmenter = CountingSegmenter()

    detect_sky_mask(
        image,
        segmenter,
        SkyMaskOptions(projection="normal", quality="high", min_area_ratio=0.0, top_connected=False),
    )

    assert segmenter.shapes[0] == (20, 40)
    assert len(segmenter.shapes) == 1 + 8
