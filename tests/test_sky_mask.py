from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import sky_mask
from sky_mask import (
    BACKEND_SAM31,
    DEFAULT_SAM31_CHECKPOINT_NAME,
    DetectedRegionMasks,
    SkyMaskOptions,
    auto_view_size,
    detect_sky_mask,
    mask_output_path_for_image,
    merge_with_existing,
    postprocess_sky_components,
    process_image,
    resolve_model_source,
    resume_state_path,
    run,
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


def test_sam31_subtract_prompt_removes_prompt_from_detection() -> None:
    class PromptSegmenter:
        def detect_prompt_masks(
            self,
            bgr: np.ndarray,
            options: SkyMaskOptions,
            *,
            sky_prompts: tuple[str, ...],
            other_prompts: tuple[str, ...],
        ) -> DetectedRegionMasks:
            del bgr, options, sky_prompts
            other = np.zeros((12, 16), dtype=bool)
            for prompt in other_prompts:
                if prompt == "person":
                    other[2:8, 2:10] = True
                if prompt == "pictogram":
                    other[4:8, 6:10] = True
            return DetectedRegionMasks(sky=np.zeros_like(other), other=other)

    image = np.zeros((12, 16, 3), dtype=np.uint8)

    mask = detect_sky_mask(
        image,
        PromptSegmenter(),
        SkyMaskOptions(
            projection="normal",
            mode="direct",
            min_area_ratio=0.0,
            top_connected=False,
            sam_prompts=("person",),
            sam_subtract_prompts=("pictogram",),
        ),
    )

    assert mask[3, 3] == 0
    assert mask[5, 8] == 255


def test_sky_mask_subtract_merge_restores_detected_existing_pixels(tmp_path: Path) -> None:
    mask_out = tmp_path / "frame.png"
    existing = np.full((8, 10), 255, dtype=np.uint8)
    existing[2:7, 2:8] = 0
    cv2.imwrite(str(mask_out), existing)
    detected = np.full((8, 10), 255, dtype=np.uint8)
    detected[3:5, 4:6] = 0

    merged = merge_with_existing(mask_out, detected, merge_mode="subtract")

    assert merged[3, 4] == 255
    assert merged[2, 2] == 0


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


def test_sky_mask_resume_state_keeps_completed_masks_after_oom_and_resumes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for idx in range(2):
        image = np.zeros((16, 24, 3), dtype=np.uint8)
        image[:8, :, 0] = 255
        cv2.imwrite(str(images / f"frame_{idx:04d}.png"), image)

    class OomAfterFirstSegmenter:
        def __init__(self) -> None:
            self.calls = 0

        def detect_sky(self, bgr: np.ndarray, options: SkyMaskOptions) -> np.ndarray:
            del options
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("CUDA out of memory while allocating test tensor")
            return bgr[:, :, 0] > 200

    first_segmenter = OomAfterFirstSegmenter()
    monkeypatch.setattr(sky_mask, "create_sky_segmenter", lambda *args, **kwargs: first_segmenter)
    monkeypatch.setattr(sky_mask, "resolve_model_source", lambda *args, **kwargs: "sam3.1_multiplex.pt")
    options = SkyMaskOptions(projection="normal", mode="direct", replace=True)

    first = run(images, masks, backend=BACKEND_SAM31, options=options, resume_state=True)

    assert not first.ok
    assert first.applied == 1
    assert first.failed == 1
    assert (masks / "frame_0000.png").is_file()
    assert not (masks / "frame_0001.png").is_file()
    assert resume_state_path(masks).is_file()

    class CountingSegmenter:
        def __init__(self) -> None:
            self.calls = 0

        def detect_sky(self, bgr: np.ndarray, options: SkyMaskOptions) -> np.ndarray:
            del options
            self.calls += 1
            return bgr[:, :, 0] > 200

    second_segmenter = CountingSegmenter()
    monkeypatch.setattr(sky_mask, "create_sky_segmenter", lambda *args, **kwargs: second_segmenter)

    second = run(images, masks, backend=BACKEND_SAM31, options=options, resume_state=True)

    assert second.ok
    assert second.resumed == 1
    assert second.applied == 1
    assert second_segmenter.calls == 1
    assert (masks / "frame_0000.png").is_file()
    assert (masks / "frame_0001.png").is_file()
    assert not resume_state_path(masks).exists()


def test_sky_mask_safe_batch_retries_smaller_chunks_after_oom(
    tmp_path: Path,
    monkeypatch,
) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for idx in range(3):
        cv2.imwrite(str(images / f"frame_{idx:04d}.png"), np.zeros((8, 12, 3), dtype=np.uint8))
    options = SkyMaskOptions(projection="normal", mode="direct", replace=True, sam_prompts=("sky",))
    args = SimpleNamespace(
        images=str(images),
        masks_dir=str(masks),
        backend=BACKEND_SAM31,
        model_dir=None,
        device="auto",
        projection="normal",
        quality="high",
        mode="direct",
        inference_size=1008,
        view_size=0,
        min_score=0.5,
        min_area_ratio=0.0,
        expand=0,
        merge_mode="replace",
        labels="sky",
        top_connected=False,
        no_top_connected=False,
        add_ext=False,
        replace=True,
        sam_prompt=["sky"],
        subtract_sam_prompt=[],
        resume_state=False,
        max_images=0,
        progress_offset=-1,
        progress_total=0,
    )
    monkeypatch.setattr(sky_mask, "resolve_model_source", lambda *args, **kwargs: "sam3.1_multiplex.pt")
    settings_hash = sky_mask._settings_fingerprint(
        backend=BACKEND_SAM31,
        model_source="sam3.1_multiplex.pt",
        device="auto",
        options=options,
    )
    image_files = sky_mask.iter_image_files(images)
    state_path = resume_state_path(masks)
    chunks: list[int] = []

    def mark_next(count: int) -> None:
        state = sky_mask._load_resume_state(state_path, settings_hash)
        pending = [
            image_path
            for image_path in image_files
            if not sky_mask._completed_record_matches(state, image_path, images, masks, options)
        ]
        for image_path in pending[:count]:
            mask_out = sky_mask.mask_output_path_for_image(image_path, images, masks)
            mask_out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_out), np.full((8, 12), 255, dtype=np.uint8))
            sky_mask._mark_resume_completed(state_path, state, image_path, images, masks, options)

    def fake_child(cmd: list[str]) -> tuple[int, bool]:
        chunk = int(cmd[cmd.index("--max-images") + 1])
        chunks.append(chunk)
        if len(chunks) == 1:
            mark_next(1)
            return 1, True
        mark_next(chunk)
        return 0, False

    monkeypatch.setattr(sky_mask, "_run_child_and_stream", fake_child)

    exit_code = sky_mask.run_safe_batch(args, options)

    assert exit_code == 0
    assert chunks == [3, 1, 1]
    assert all((masks / f"frame_{idx:04d}.png").is_file() for idx in range(3))
    assert not state_path.exists()


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
