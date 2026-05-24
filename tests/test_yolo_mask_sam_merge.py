from __future__ import annotations

import numpy as np
import pytest

import core.yolo_mask as yolo_mask


class FakeMaskData:
    def __init__(self, data: np.ndarray):
        self._data = data

    def __len__(self) -> int:
        if self._data.ndim <= 2:
            return 1
        return len(self._data)

    def __iter__(self):
        if self._data.ndim <= 2:
            yield self
            return
        for item in self._data:
            yield FakeMaskData(item)

    def cpu(self) -> FakeMaskData:
        return self

    def numpy(self) -> np.ndarray:
        return self._data


class FakeMasks:
    def __init__(self, data: np.ndarray):
        self.data = FakeMaskData(data)


class FakeResult:
    def __init__(self, data: np.ndarray):
        self.masks = FakeMasks(data)


def test_add_sam_mask_merges_all_masks_in_one_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    masks = np.zeros((2, 4, 4), dtype=np.float32)
    masks[0, 1, 1] = 1.0
    masks[1, 2, 2] = 1.0

    def fake_sam(*_args, **_kwargs):
        return [FakeResult(masks)]

    monkeypatch.setattr(yolo_mask, "sam", fake_sam)
    monkeypatch.setattr(yolo_mask, "PROFILE", None)

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    base_mask = np.zeros((4, 4), dtype=np.uint8)
    merged, has_mask = yolo_mask.add_sam_mask(img, base_mask, [[0, 0, 1, 1], [2, 2, 3, 3]])

    assert has_mask == 1
    assert merged[1, 1] == 255
    assert merged[2, 2] == 255
    assert int(np.count_nonzero(merged)) == 2


def test_add_sam_mask_handles_single_mask_without_batch_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    masks = np.zeros((1, 4, 4), dtype=np.float32)
    masks[0, 1, 2] = 1.0

    def fake_sam(*_args, **_kwargs):
        return [FakeResult(masks)]

    monkeypatch.setattr(yolo_mask, "sam", fake_sam)
    monkeypatch.setattr(yolo_mask, "PROFILE", None)

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    base_mask = np.zeros((4, 4), dtype=np.uint8)
    merged, has_mask = yolo_mask.add_sam_mask(img, base_mask, [[0, 0, 1, 1]])

    assert has_mask == 1
    assert merged[1, 2] == 255
    assert int(np.count_nonzero(merged)) == 1
