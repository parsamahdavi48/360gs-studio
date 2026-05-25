from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import core.yolo_mask as yolo_mask


def _runtime_context(
    *,
    quality: str = "standard",
    projection: str = "equirect",
    level: int | None = None,
    expand: int = 0,
    bottom_tta_rotations: int | None = None,
) -> yolo_mask.YoloMaskRuntimeContext:
    recipe = yolo_mask.recipe_for(quality, projection)
    settings = yolo_mask.YoloMaskRuntimeSettings(
        class_ids=(0,),
        level=int(recipe.yolo_level if level is None else level),
        quality=quality,
        projection=projection,
        expand=expand,
        bottom_conf=recipe.bottom_conf,
        bottom_tta_rotations=(
            len(recipe.bottom_rotations) if bottom_tta_rotations is None else int(bottom_tta_rotations)
        ),
        bottom_model=recipe.bottom_model,
        bottom_filter=recipe.bottom_filter,
        recipe=recipe,
        profile_json=None,
    )
    return yolo_mask.create_runtime_context(settings)


@pytest.mark.parametrize(
    ("level", "projection", "expected"),
    [
        (0, "equirect", False),
        (1, "equirect", True),
        (2, "equirect", True),
        (1, "normal", False),
    ],
)
def test_bottom_redetection_condition(level: int, projection: str, expected: bool) -> None:
    assert yolo_mask.should_run_bottom_redetection(level, projection) is expected


def test_equirect_standard_level_runs_bottom_redetection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "images"
    output = tmp_path / "masks"
    source.mkdir()
    output.mkdir()
    image_path = source / "frame_0001.jpg"
    cv2.imwrite(str(image_path), np.full((8, 16, 3), 128, dtype=np.uint8))

    add_calls: list[tuple[int, int]] = []
    bottom_sizes: list[int] = []
    back_calls: list[tuple[int, int, int, int]] = []

    def fake_add_yolo_mask(img, mask, has_mask=0, **_kwargs):
        add_calls.append(img.shape[:2])
        return mask, has_mask

    def fake_detect_yolo_bboxes(img, **_kwargs):
        return [[0, 0, 2, 2]]

    def fake_add_sam_mask(img, mask, bboxes, has_mask=0, **_kwargs):
        add_calls.append(img.shape[:2])
        assert bboxes
        return np.full_like(mask, 255), has_mask + 1

    def fake_get_bottom_from_pano(img, size=1024):
        bottom_sizes.append(size)
        return np.zeros((size, size, 3), dtype=np.uint8)

    def fake_back_to_pano_from_bottom(bottom_img, pano_width, pano_height):
        back_calls.append((bottom_img.shape[1], bottom_img.shape[0], pano_width, pano_height))
        return np.full((pano_height, pano_width), 255, dtype=np.uint8)

    monkeypatch.setattr(yolo_mask, "add_yolo_mask", fake_add_yolo_mask)
    monkeypatch.setattr(yolo_mask, "detect_yolo_bboxes", fake_detect_yolo_bboxes)
    monkeypatch.setattr(yolo_mask, "add_sam_mask", fake_add_sam_mask)
    monkeypatch.setattr(yolo_mask, "get_bottom_from_pano", fake_get_bottom_from_pano)
    monkeypatch.setattr(yolo_mask, "back_to_pano_from_bottom", fake_back_to_pano_from_bottom)
    context = _runtime_context(projection="equirect", level=1, expand=0, bottom_tta_rotations=1)

    yolo_mask.process_file(str(source), str(output), image_path.name, add_ext=False, context=context)

    assert add_calls == [(8, 16), (4, 4)]
    assert bottom_sizes == [4]
    assert back_calls == [(4, 4, 16, 8)]
    written = cv2.imread(str(output / "frame_0001.png"), cv2.IMREAD_GRAYSCALE)
    assert written is not None
    assert np.all(written == 0)


@pytest.mark.parametrize(
    ("level", "projection"),
    [
        (1, "normal"),
    ],
)
def test_bottom_redetection_is_skipped_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    level: int,
    projection: str,
) -> None:
    source = tmp_path / "images"
    output = tmp_path / "masks"
    source.mkdir()
    output.mkdir()
    image_path = source / "frame_0001.jpg"
    cv2.imwrite(str(image_path), np.full((8, 16, 3), 128, dtype=np.uint8))

    add_calls: list[tuple[int, int]] = []

    def fake_add_yolo_mask(img, mask, has_mask=0, **_kwargs):
        add_calls.append(img.shape[:2])
        return mask, has_mask

    monkeypatch.setattr(yolo_mask, "add_yolo_mask", fake_add_yolo_mask)
    monkeypatch.setattr(
        yolo_mask,
        "get_bottom_from_pano",
        lambda *_args, **_kwargs: pytest.fail("bottom redetection must be skipped"),
    )
    context = _runtime_context(projection=projection, level=level, expand=0)

    yolo_mask.process_file(str(source), str(output), image_path.name, add_ext=False, context=context)

    assert add_calls == [(8, 16)]


def test_bottom_tta_runs_all_rotations_and_merges(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.full((8, 16, 3), 128, dtype=np.uint8)
    yolo_shapes: list[tuple[int, int]] = []
    sam_calls: list[tuple[int, int]] = []

    def fake_get_bottom_from_pano(_img, size=1024):
        return np.zeros((size, size, 3), dtype=np.uint8)

    def fake_detect_yolo_bboxes(img, **_kwargs):
        yolo_shapes.append(img.shape[:2])
        if len(yolo_shapes) in (2, 4):
            return [[1, 1, 3, 3]]
        return []

    def fake_add_sam_mask(img, mask, bboxes, has_mask=0, **_kwargs):
        sam_calls.append(img.shape[:2])
        if bboxes:
            mask[1:3, 1:3] = 255
            return mask, has_mask + 1
        return mask, has_mask

    def fake_back_to_pano_from_bottom(bottom_img, pano_width, pano_height):
        out = np.zeros((pano_height, pano_width), dtype=np.uint8)
        out[: bottom_img.shape[0], : bottom_img.shape[1]] = bottom_img
        return out

    monkeypatch.setattr(yolo_mask, "get_bottom_from_pano", fake_get_bottom_from_pano)
    monkeypatch.setattr(yolo_mask, "detect_yolo_bboxes", fake_detect_yolo_bboxes)
    monkeypatch.setattr(yolo_mask, "add_sam_mask", fake_add_sam_mask)
    monkeypatch.setattr(yolo_mask, "back_to_pano_from_bottom", fake_back_to_pano_from_bottom)
    context = _runtime_context(bottom_tta_rotations=4)

    bottom_mask, has_bottom = yolo_mask.detect_bottom_mask(image, pano_width=16, pano_height=8, context=context)

    assert yolo_shapes == [(4, 4), (4, 4), (4, 4), (4, 4)]
    assert sam_calls == [(4, 4)]
    assert has_bottom == 1
    assert bottom_mask is not None
    assert np.any(bottom_mask == 255)


def test_transform_bbox_from_rotated_bottom_maps_box_back_to_original_orientation() -> None:
    assert yolo_mask.transform_bbox_from_rotated_bottom([1, 2, 3, 5], 0, width=10, height=8) == [1, 2, 3, 5]
    assert yolo_mask.transform_bbox_from_rotated_bottom([1, 2, 3, 5], 90, width=10, height=8) == [2, 5, 5, 7]
    assert yolo_mask.transform_bbox_from_rotated_bottom([1, 2, 3, 5], 180, width=10, height=8) == [7, 3, 9, 6]
    assert yolo_mask.transform_bbox_from_rotated_bottom([1, 2, 3, 5], 270, width=10, height=8) == [5, 1, 8, 3]


def test_bottom_component_filter_removes_unreliable_components() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[45:62, 45:60] = 255
    mask[2:4, 2:4] = 255
    mask[80:82, 10:95] = 255

    filtered = yolo_mask.filter_bottom_mask_components(mask)

    assert np.any(filtered[45:62, 45:60] == 255)
    assert np.all(filtered[2:4, 2:4] == 0)
    assert np.all(filtered[80:82, 10:95] == 0)
