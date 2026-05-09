from pathlib import Path

from PIL import Image

from gui.steps.mask_postprocess import mask_stats


def test_mask_stats_records_header_without_pixel_counts(tmp_path: Path) -> None:
    mask = tmp_path / "mask.png"
    Image.new("L", (12, 8), 255).save(mask)

    stats = mask_stats(mask)

    assert stats == {
        "readable": True,
        "width": 12,
        "height": 8,
        "mode": "L",
        "pixel_stats": "skipped",
    }
