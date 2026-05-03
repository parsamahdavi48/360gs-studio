from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import yolo_mask


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
        if len(add_calls) == 2:
            return np.full_like(mask, 255), has_mask + 1
        return mask, has_mask

    def fake_get_bottom_from_pano(img, size=1024):
        bottom_sizes.append(size)
        return np.zeros((size, size, 3), dtype=np.uint8)

    def fake_back_to_pano_from_bottom(bottom_img, pano_width, pano_height):
        back_calls.append((bottom_img.shape[1], bottom_img.shape[0], pano_width, pano_height))
        return np.full((pano_height, pano_width), 255, dtype=np.uint8)

    monkeypatch.setattr(yolo_mask, "add_yolo_mask", fake_add_yolo_mask)
    monkeypatch.setattr(yolo_mask, "get_bottom_from_pano", fake_get_bottom_from_pano)
    monkeypatch.setattr(yolo_mask, "back_to_pano_from_bottom", fake_back_to_pano_from_bottom)
    monkeypatch.setattr(yolo_mask, "LEVEL", 1)
    monkeypatch.setattr(yolo_mask, "PROJECTION", "equirect")
    monkeypatch.setattr(yolo_mask, "EXPAND", 0)
    monkeypatch.setattr(yolo_mask, "BOTTOM_TTA_ROTATIONS", 1)

    yolo_mask.process_file(str(source), str(output), image_path.name, add_ext=False)

    assert add_calls == [(8, 16), (4, 4)]
    assert bottom_sizes == [4]
    assert back_calls == [(4, 4, 16, 8)]
    written = cv2.imread(str(output / "frame_0001.png"), cv2.IMREAD_GRAYSCALE)
    assert written is not None
    assert np.all(written == 0)


@pytest.mark.parametrize(
    ("level", "projection"),
    [
        (0, "equirect"),
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
    monkeypatch.setattr(yolo_mask, "LEVEL", level)
    monkeypatch.setattr(yolo_mask, "PROJECTION", projection)
    monkeypatch.setattr(yolo_mask, "EXPAND", 0)

    yolo_mask.process_file(str(source), str(output), image_path.name, add_ext=False)

    assert add_calls == [(8, 16)]


def test_bottom_tta_runs_all_rotations_and_merges(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.full((8, 16, 3), 128, dtype=np.uint8)
    calls: list[tuple[int, int]] = []

    def fake_get_bottom_from_pano(_img, size=1024):
        return np.zeros((size, size, 3), dtype=np.uint8)

    def fake_add_yolo_mask(img, mask, has_mask=0, **_kwargs):
        calls.append(img.shape[:2])
        if len(calls) in (2, 4):
            mask[1:3, 1:3] = 255
            return mask, has_mask + 1
        return mask, has_mask

    def fake_back_to_pano_from_bottom(bottom_img, pano_width, pano_height):
        out = np.zeros((pano_height, pano_width), dtype=np.uint8)
        out[: bottom_img.shape[0], : bottom_img.shape[1]] = bottom_img
        return out

    monkeypatch.setattr(yolo_mask, "get_bottom_from_pano", fake_get_bottom_from_pano)
    monkeypatch.setattr(yolo_mask, "add_yolo_mask", fake_add_yolo_mask)
    monkeypatch.setattr(yolo_mask, "back_to_pano_from_bottom", fake_back_to_pano_from_bottom)
    monkeypatch.setattr(yolo_mask, "BOTTOM_TTA_ROTATIONS", 4)

    bottom_mask, has_bottom = yolo_mask.detect_bottom_mask(image, pano_width=16, pano_height=8)

    assert calls == [(4, 4), (4, 4), (4, 4), (4, 4)]
    assert has_bottom == 2
    assert bottom_mask is not None
    assert np.any(bottom_mask == 255)


def test_temporal_bottom_propagation_merges_neighbor_masks(tmp_path: Path) -> None:
    final_paths = [tmp_path / f"frame_{idx:04d}.png" for idx in range(3)]
    for path in final_paths:
        cv2.imwrite(str(path), np.full((4, 4), 255, dtype=np.uint8))

    source_mask_path = tmp_path / "bottom_0000.png"
    source_mask = np.zeros((4, 4), dtype=np.uint8)
    source_mask[1:3, 1:3] = 255
    cv2.imwrite(str(source_mask_path), source_mask)

    results = [
        yolo_mask.ProcessResult(final_paths[0], "group", source_mask_path),
        yolo_mask.ProcessResult(final_paths[1], "group", None),
        yolo_mask.ProcessResult(final_paths[2], "group", None),
    ]

    updated = yolo_mask.apply_temporal_bottom_propagation(results, window=1)

    assert updated == 2
    propagated = cv2.imread(str(final_paths[1]), cv2.IMREAD_GRAYSCALE)
    untouched = cv2.imread(str(final_paths[2]), cv2.IMREAD_GRAYSCALE)
    assert propagated is not None
    assert untouched is not None
    assert np.all(propagated[1:3, 1:3] == 0)
    assert np.all(untouched == 255)
