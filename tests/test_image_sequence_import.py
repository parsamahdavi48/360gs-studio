from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from core.image_sequence_import import import_image_sequence_folder
from core.scene_layout import selected_frames_path, source_image_sets_path
from core.scene_project import load_json


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(100, 110, 120)).save(path)


def test_import_image_sequence_copies_images_and_registers_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    scene = tmp_path / "scene"
    _write_image(source / "img_0001.jpg", (64, 32))
    _write_image(source / "img_0002.png", (32, 32))

    result = import_image_sequence_folder(source, scene, prefix="take")

    assert result.image_count == 2
    assert (scene / "images" / "take_0001.jpg").is_file()
    assert (scene / "images" / "take_0002.png").is_file()
    assert result.output_files == ("images/take_0001.jpg", "images/take_0002.png")

    with selected_frames_path(scene).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["output_file"] for row in rows] == ["images/take_0001.jpg", "images/take_0002.png"]
    assert {row["source_type"] for row in rows} == {"image_sequence"}
    assert {row["source_session"] for row in rows} == {result.import_id}

    data = load_json(source_image_sets_path(scene))
    image_sets = data["image_sets"]
    assert image_sets[0]["source_type"] == "image_sequence"
    assert image_sets[0]["projection"] == "mixed"
    assert image_sets[0]["files"][0]["scene_path"] == "images/take_0001.jpg"


def test_import_image_sequence_allocates_unique_prefix(tmp_path: Path) -> None:
    source = tmp_path / "source"
    scene = tmp_path / "scene"
    _write_image(source / "img_0001.jpg", (64, 32))
    (scene / "images").mkdir(parents=True)
    _write_image(scene / "images" / "take_0001.jpg", (64, 32))

    result = import_image_sequence_folder(source, scene, prefix="take")

    assert result.output_files == ("images/take_seq2_0001.jpg",)
    assert (scene / "images" / "take_seq2_0001.jpg").is_file()
