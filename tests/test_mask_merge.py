from __future__ import annotations

import numpy as np

from core.mask_merge import merge_mask_arrays


def test_merge_mask_arrays_adds_exclusions_with_and_contract() -> None:
    existing = np.array([[255, 0], [255, 255]], dtype=np.uint8)
    candidate = np.array([[255, 255], [0, 255]], dtype=np.uint8)

    merged = merge_mask_arrays(existing, candidate, merge_mode="add")

    assert np.array_equal(merged, np.array([[255, 0], [0, 255]], dtype=np.uint8))


def test_merge_mask_arrays_subtract_restores_candidate_exclusions() -> None:
    existing = np.array([[0, 0], [255, 0]], dtype=np.uint8)
    candidate = np.array([[0, 255], [0, 255]], dtype=np.uint8)

    merged = merge_mask_arrays(existing, candidate, merge_mode="subtract")

    assert np.array_equal(merged, np.array([[255, 0], [255, 0]], dtype=np.uint8))
