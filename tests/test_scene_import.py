from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.artifact_registry import load_artifacts
from core.scene_import import import_scene
from core.scene_import_contracts import SceneImportCancelled, SceneImportCancelToken, SceneImportOptions
from core.scene_layout import (
    mask_runs_path,
    scene_imports_path,
    selected_frames_path,
    source_image_sets_path,
    step4_dataset_runs_path,
    step4_export_settings_path,
)
from core.workflow_artifacts import latest_dataset_root, register_dataset_artifact


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
    stats = mask_runs[0]["generated"][0]["stats"]
    assert stats["readable"] is True
    assert stats["width"] == 64
    assert stats["height"] == 32
    assert stats["pixel_stats"] == "skipped"
    assert "black_pixels" not in stats

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


def test_scene_import_registers_route_specific_output_dataset(tmp_path: Path) -> None:
    scene = tmp_path
    output = scene / "output" / "metashape_cubemap"
    _write_image(scene / "images" / "frame_0001.jpg")
    _write_image(output / "images" / "frame_0001_px.jpg", size=(64, 64))
    _write_transforms(output / "transforms.json")

    result = import_scene(scene)

    assert result.status == "ok"
    assert result.output_shape == "projected"
    assert result.output_image_count == 1

    settings = json.loads(step4_export_settings_path(scene).read_text(encoding="utf-8"))
    assert settings["output_dir"] == str(output)
    assert settings["portable_output"] == {
        "root": "output/metashape_cubemap",
        "dataset_kind": "projection_views",
        "active": True,
    }
    assert settings["registered_assets"]["images_dir"] == "output/metashape_cubemap/images"
    assert settings["registered_assets"]["transforms_json"] == "output/metashape_cubemap/transforms.json"

    dataset_runs = json.loads(step4_dataset_runs_path(scene).read_text(encoding="utf-8"))["runs"]
    assert dataset_runs[0]["dataset_root"] == "output/metashape_cubemap"
    assert dataset_runs[0]["artifacts"]["root"] == "output/metashape_cubemap"

    artifacts = load_artifacts(scene, "dataset")
    assert len(artifacts) == 1
    assert artifacts[0].id == f"dataset_{result.import_id}"
    assert artifacts[0].root == "output/metashape_cubemap"
    assert artifacts[0].metadata["origin"]["kind"] == "external_import"
    assert latest_dataset_root(scene) == output


def test_scene_import_prefers_existing_settings_output_root(tmp_path: Path) -> None:
    scene = tmp_path
    metashape_output = scene / "output" / "metashape_cubemap"
    spheresfm_output = scene / "output" / "colmap_equirect_cubemap"
    _write_image(scene / "images" / "frame_0001.jpg")
    _write_image(metashape_output / "images" / "frame_0001_px.jpg", size=(64, 64))
    _write_transforms(metashape_output / "transforms.json")
    _write_image(spheresfm_output / "images" / "frame_0001_px.jpg", size=(64, 64), color=(10, 0, 0))
    _write_transforms(spheresfm_output / "transforms.json")
    step4_export_settings_path(scene).parent.mkdir(parents=True, exist_ok=True)
    step4_export_settings_path(scene).write_text(
        json.dumps(
            {
                "output_dir": str(spheresfm_output),
                "portable_output": {"root": "output/colmap_equirect_cubemap"},
            }
        ),
        encoding="utf-8",
    )

    result = import_scene(scene)

    assert result.status == "ok"
    settings = json.loads(step4_export_settings_path(scene).read_text(encoding="utf-8"))
    assert settings["output_dir"] == str(spheresfm_output)
    assert settings["portable_output"]["root"] == "output/colmap_equirect_cubemap"


def test_scene_import_registered_dataset_artifact_tracks_imported_output(tmp_path: Path) -> None:
    scene = tmp_path
    stale = scene / "output" / "old_dataset"
    stale.mkdir(parents=True)
    (stale / "transforms.json").write_text(json.dumps({"camera_model": "SIMPLE_PINHOLE", "frames": []}), encoding="utf-8")
    assert register_dataset_artifact(scene, artifact_id="dataset_old", root=stale) is not None

    output = scene / "output" / "metashape_cubemap"
    _write_image(output / "images" / "frame_0001_px.jpg", size=(64, 64))
    _write_transforms(output / "transforms.json")

    result = import_scene(scene)

    assert result.status == "ok"
    assert latest_dataset_root(scene) == output
    artifacts = load_artifacts(scene, "dataset")
    assert [artifact.id for artifact in artifacts] == ["dataset_old", f"dataset_{result.import_id}"]


def test_scene_import_removes_external_dataset_artifact_when_output_is_absent(tmp_path: Path) -> None:
    scene = tmp_path
    _write_image(scene / "images" / "frame_0001.jpg")
    output = scene / "output" / "metashape_cubemap"
    _write_image(output / "images" / "frame_0001_px.jpg", size=(64, 64))
    _write_transforms(output / "transforms.json")
    first = import_scene(scene)
    assert [artifact.id for artifact in load_artifacts(scene, "dataset")] == [f"dataset_{first.import_id}"]

    (output / "images" / "frame_0001_px.jpg").unlink()
    (output / "transforms.json").unlink()
    second = import_scene(scene)

    assert second.status == "ok"
    assert load_artifacts(scene, "dataset") == []


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


def test_scene_import_cancel_before_apply_leaves_metadata_unchanged(tmp_path: Path) -> None:
    scene = tmp_path
    _write_image(scene / "images" / "frame_0001.jpg")
    token = SceneImportCancelToken()

    def progress(message: str) -> None:
        if "build source metadata" in message:
            token.request_cancel()

    with pytest.raises(SceneImportCancelled):
        import_scene(scene, cancel_token=token, progress_callback=progress)

    assert not selected_frames_path(scene).exists()
    assert not source_image_sets_path(scene).exists()
    assert not scene_imports_path(scene).exists()


def test_scene_import_samples_large_output_image_validation(tmp_path: Path, monkeypatch) -> None:
    scene = tmp_path
    output_images = scene / "output" / "images"
    output_images.mkdir(parents=True)
    frames = []
    for index in range(200):
        name = f"frame_{index:04d}.jpg"
        (output_images / name).write_bytes(b"placeholder")
        frames.append({"file_path": f"images/{name}", "transform_matrix": np.eye(4).tolist()})
    (scene / "output" / "transforms.json").write_text(
        json.dumps({"camera_model": "SIMPLE_PINHOLE", "frames": frames}),
        encoding="utf-8",
    )

    calls: list[Path] = []

    def fake_image_size(path: Path) -> tuple[int, int]:
        calls.append(path)
        return (64, 64)

    monkeypatch.setattr("core.scene_import_outputs.image_size", fake_image_size)

    result = import_scene(scene, options=SceneImportOptions(output_validation_sample_limit=10))

    assert result.status == "ok"
    assert result.output_image_count == 200
    assert len(calls) == 10
    record = json.loads(scene_imports_path(scene).read_text(encoding="utf-8"))["imports"][-1]
    assert record["validation"]["output_image_sample_count"] == 10
    assert record["validation"]["output_image_sample_limit"] == 10


def test_scene_import_empty_mask_dir_skips_per_image_mask_checks(tmp_path: Path, monkeypatch) -> None:
    scene = tmp_path
    (scene / "masks").mkdir()
    for index in range(8):
        _write_image(scene / "images" / f"frame_{index:04d}.jpg")

    def fail_image_size(path: Path) -> tuple[int, int]:
        raise AssertionError(f"unexpected mask size check: {path}")

    monkeypatch.setattr("core.scene_import_sources.image_size", fail_image_size)

    result = import_scene(scene)

    assert result.status == "ok"
    assert result.mask_count == 0
    assert not any("masks/ missing matching files" in warning for warning in result.warnings)


def test_scene_import_mask_metadata_reuses_source_headers_and_skips_pixel_stats(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene = tmp_path
    _write_image(scene / "images" / "frame_0001.jpg")
    _write_mask(scene / "masks" / "frame_0001.png")

    def fail_image_size(path: Path) -> tuple[int, int]:
        raise AssertionError(f"unexpected image size read: {path}")

    monkeypatch.setattr("core.scene_import_sources.image_size", fail_image_size)

    result = import_scene(scene)

    assert result.status == "ok"
    assert result.mask_count == 1
    mask_runs = json.loads(mask_runs_path(scene).read_text(encoding="utf-8"))["runs"]
    stats = mask_runs[0]["generated"][0]["stats"]
    assert stats == {
        "readable": True,
        "width": 64,
        "height": 32,
        "mode": "L",
        "pixel_stats": "skipped",
    }
