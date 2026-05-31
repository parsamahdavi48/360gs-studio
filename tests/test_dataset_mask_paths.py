from __future__ import annotations

import json
from pathlib import Path

from core.dataset_mask_paths import attach_nerf_mask_paths, clear_nerf_mask_paths


def test_attach_nerf_mask_paths_matches_dataset_images(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "images" / "cam").mkdir(parents=True)
    (dataset / "masks" / "cam").mkdir(parents=True)
    (dataset / "images" / "cam" / "frame_0001.jpg").write_text("image", encoding="utf-8")
    (dataset / "masks" / "cam" / "frame_0001.png").write_text("mask", encoding="utf-8")
    transforms = dataset / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "frames": [
                    {"file_path": "images/cam/frame_0001.jpg"},
                    {"file_path": "images/cam/frame_0002.jpg", "mask_path": "masks/old.png"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = attach_nerf_mask_paths(dataset_root=dataset)

    data = json.loads(transforms.read_text(encoding="utf-8"))
    assert result.frame_count == 2
    assert result.mask_path_count == 1
    assert result.missing_mask_count == 1
    assert data["frames"][0]["mask_path"] == "masks/cam/frame_0001.png"
    assert "mask_path" not in data["frames"][1]


def test_clear_nerf_mask_paths_removes_existing_entries(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    transforms = dataset / "transforms_lichtfeld.json"
    transforms.write_text(
        json.dumps({"frames": [{"file_path": "images/frame.jpg", "mask_path": "masks/frame.png"}]}),
        encoding="utf-8",
    )

    result = clear_nerf_mask_paths(dataset_root=dataset)

    data = json.loads(transforms.read_text(encoding="utf-8"))
    assert result.frame_count == 1
    assert result.missing_mask_count == 1
    assert "mask_path" not in data["frames"][0]
