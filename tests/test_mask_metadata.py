from gui.steps.mask_postprocess import mask_stats


def test_mask_stats_records_minimal_step3_ledger_state_without_reading_image(tmp_path) -> None:
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"not decoded by step3 ledger metadata")

    stats = mask_stats(mask)

    assert stats == {
        "readable": True,
        "pixel_stats": "skipped",
    }


def test_mask_stats_marks_missing_masks_unreadable(tmp_path) -> None:
    stats = mask_stats(tmp_path / "missing.png")

    assert stats == {
        "readable": False,
        "pixel_stats": "skipped",
    }
