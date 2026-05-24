from __future__ import annotations

from pathlib import Path

import numpy as np

from core.dataset_writer_colmap import ColmapCamera, ColmapImage, write_colmap_text_dataset
from core.dataset_writer_nerf import write_nerf_json_ply_dataset


def test_colmap_writer_writes_sparse_contract(tmp_path: Path) -> None:
    camera = ColmapCamera(1, "PINHOLE", 64, 48, (50.0, 51.0, 32.0, 24.0))
    image = ColmapImage(1, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 1, "frame.jpg")

    result = write_colmap_text_dataset(tmp_path, [camera], [image])

    assert result.sparse_dir == tmp_path / "sparse" / "0"
    assert (result.sparse_dir / "cameras.txt").is_file()
    assert "PINHOLE" in (result.sparse_dir / "cameras.txt").read_text(encoding="utf-8")
    assert "frame.jpg" in (result.sparse_dir / "images.txt").read_text(encoding="utf-8")
    assert (result.sparse_dir / "points3D.txt").is_file()


def test_nerf_writer_writes_transforms_and_manifest(tmp_path: Path) -> None:
    result = write_nerf_json_ply_dataset(
        tmp_path,
        {"frames": [{"file_path": "images/a.jpg", "transform_matrix": np.eye(4).tolist()}]},
    )

    assert result.transforms_json == tmp_path / "transforms.json"
    assert result.frame_count == 1
    assert (tmp_path / "stechdrive_dataset_manifest.json").is_file()
