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

    def fake_add_yolo_mask(img, mask, has_mask=0):
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

    def fake_add_yolo_mask(img, mask, has_mask=0):
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
