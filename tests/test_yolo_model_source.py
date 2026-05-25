from __future__ import annotations

from pathlib import Path

from core.yolo_mask import model_source


def test_model_source_prefers_models_ultralytics(tmp_path: Path) -> None:
    root_file = tmp_path / "yolo26l.pt"
    root_file.write_bytes(b"root")
    preferred = tmp_path / "models" / "ultralytics" / "yolo26l.pt"
    preferred.parent.mkdir(parents=True)
    preferred.write_bytes(b"preferred")

    assert Path(model_source(tmp_path, "yolo26l.pt")) == preferred


def test_model_source_ignores_repo_root_model_file(tmp_path: Path) -> None:
    root_file = tmp_path / "sam2.1_l.pt"
    root_file.write_bytes(b"root")

    assert model_source(tmp_path, "sam2.1_l.pt") == "sam2.1_l.pt"


def test_model_source_returns_model_name_when_local_file_missing(tmp_path: Path) -> None:
    assert model_source(tmp_path, "yolo26m.pt") == "yolo26m.pt"
