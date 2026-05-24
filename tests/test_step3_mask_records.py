from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from core.scene_layout import mask_runs_path
from gui.steps.step3_mask_records import record_mask_outputs


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(64, 96, 128)).save(path)


def _write_mask(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=255).save(path)


def test_record_mask_outputs_writes_run_for_existing_masks(tmp_path: Path) -> None:
    scene = tmp_path
    image = scene / "images" / "frame.jpg"
    mask = scene / "masks" / "frame.png"
    missing_image = scene / "images" / "missing.jpg"
    _write_image(image, (16, 8))
    _write_image(missing_image, (16, 8))
    _write_mask(mask, (16, 8))

    record_mask_outputs(
        scene,
        [image, missing_image],
        mode="batch",
        settings={"quality": "test"},
        phases=["yolo"],
        mask_path_for_image=lambda path: scene / "masks" / f"{path.stem}.png",
        run_id="mask_test",
    )

    runs = json.loads(mask_runs_path(scene).read_text(encoding="utf-8"))["runs"]
    assert runs[0]["id"] == "mask_test"
    assert runs[0]["image_count"] == 2
    assert runs[0]["mask_count"] == 1
    assert runs[0]["generated"][0]["image"] == "images/frame.jpg"
    assert runs[0]["generated"][0]["mask"] == "masks/frame.png"
