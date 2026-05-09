from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from core.scene_import import import_scene
from core.scene_layout import (
    mask_runs_path,
    scene_imports_path,
    selected_frames_path,
    source_image_sets_path,
    step4_dataset_runs_path,
    step4_export_settings_path,
)


def _write_image(path: Path, size: tuple[int, int] = (64, 32), color: tuple[int, int, int] = (0, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_mask(path: Path, size: tuple[int, int] = (64, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, 255).save(path)


def _write_transforms(path: Path, image_name: str = "frame_0001_px.jpg") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "camera_model": "SIMPLE_PINHOLE",
                "w": 64,
                "h": 64,
                "fl_x": 32.0,
                "fl_y": 32.0,
                "cx": 32.0,
                "cy": 32.0,
                "frames": [
                    {
                        "file_path": f"images/{image_name}",
                        "transform_matrix": np.eye(4).tolist(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_scene_import_registers_existing_scene_assets(tmp_path: Path) -> None:
    scene = tmp_path
    _write_image(scene / "images" / "frame_0001.jpg")
    _write_mask(scene / "masks" / "frame_0001.png")
    _write_image(scene / "output" / "images" / "frame_0001_px.jpg", size=(64, 64))
    _write_mask(scene / "output" / "masks" / "frame_0001_px.png", size=(64, 64))
    _write_transforms(scene / "output" / "transforms.json")

    result = import_scene(scene)

    assert result.status == "ok"
    assert result.image_count == 1
    assert result.mask_count == 1
    assert result.output_shape == "projected"

    rows = _read_csv(selected_frames_path(scene))
    assert len(rows) == 1
    assert rows[0]["analysis_pipeline"] == "external_import"
    assert rows[0]["status"] == "ok"
    assert rows[0]["decision"] == "keep"

    image_sets = json.loads(source_image_sets_path(scene).read_text(encoding="utf-8"))["image_sets"]
    assert len(image_sets) == 1
    assert image_sets[0]["origin"]["kind"] == "external_import"
    assert image_sets[0]["registration_mode"] == "in_place"

    mask_runs = json.loads(mask_runs_path(scene).read_text(encoding="utf-8"))["runs"]
    assert len(mask_runs) == 1
    assert mask_runs[0]["mode"] == "external_import"

    settings = json.loads(step4_export_settings_path(scene).read_text(encoding="utf-8"))
    assert settings["origin"]["kind"] == "external_import"
    assert settings["portable_output"] == {
        "root": "output",
        "dataset_kind": "projection_views",
        "active": True,
    }
    assert settings["output_shape"] == "projected"

    dataset_runs = json.loads(step4_dataset_runs_path(scene).read_text(encoding="utf-8"))["runs"]
    assert len(dataset_runs) == 1
    assert dataset_runs[0]["route"] == "external_import"


def test_scene_import_rescan_replaces_previous_external_metadata(tmp_path: Path) -> None:
    scene = tmp_path
    _write_image(scene / "images" / "frame_0001.jpg")

    first = import_scene(scene)
    assert len(_read_csv(selected_frames_path(scene))) == 1

    (scene / "images" / "frame_0001.jpg").unlink()
    _write_image(scene / "images" / "frame_0002.jpg")
    second = import_scene(scene)

    rows = _read_csv(selected_frames_path(scene))
    assert len(rows) == 1
    assert rows[0]["output_file"] == "images/frame_0002.jpg"
    assert rows[0]["import_id"] == second.import_id
    assert first.import_id != second.import_id
    assert second.backup_dir is not None
    assert (second.backup_dir / "frames" / "selected_frames.csv").is_file()

    image_sets = json.loads(source_image_sets_path(scene).read_text(encoding="utf-8"))["image_sets"]
    assert [item["id"] for item in image_sets] == [f"imageset_{second.import_id}"]

    imports = json.loads(scene_imports_path(scene).read_text(encoding="utf-8"))
    assert imports["active_import_id"] == second.import_id
    assert [record["id"] for record in imports["imports"]] == [first.import_id, second.import_id]
    assert imports["imports"][-1]["mode"] == "rescan_replace"
    assert imports["imports"][-1]["replaces_import_id"] == first.import_id


def test_scene_import_records_warnings_without_blocking(tmp_path: Path) -> None:
    scene = tmp_path
    _write_image(scene / "images" / "frame_0001.jpg", size=(64, 32))
    _write_mask(scene / "masks" / "frame_0001.png", size=(32, 32))
    _write_image(scene / "output" / "images" / "frame_0001_px.jpg", size=(64, 64))
    _write_transforms(scene / "output" / "transforms.json", image_name="missing.jpg")

    result = import_scene(scene)

    assert result.status == "warning"
    assert result.mask_count == 0
    assert any("size mismatch" in warning for warning in result.warnings)
    assert any("missing images" in warning for warning in result.warnings)
    record = json.loads(scene_imports_path(scene).read_text(encoding="utf-8"))["imports"][-1]
    assert record["validation"]["warnings"] == list(result.warnings)
