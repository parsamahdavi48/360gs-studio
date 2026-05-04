from __future__ import annotations

from pathlib import Path

from yolo_mask import model_source


def test_model_source_prefers_models_ultralytics(tmp_path: Path) -> None:
    legacy = tmp_path / "yolo26l.pt"
    legacy.write_bytes(b"legacy")
    preferred = tmp_path / "models" / "ultralytics" / "yolo26l.pt"
    preferred.parent.mkdir(parents=True)
    preferred.write_bytes(b"preferred")

    assert Path(model_source(tmp_path, "yolo26l.pt")) == preferred


def test_model_source_uses_legacy_root_fallback(tmp_path: Path) -> None:
    legacy = tmp_path / "sam2.1_l.pt"
    legacy.write_bytes(b"legacy")

    assert Path(model_source(tmp_path, "sam2.1_l.pt")) == legacy


def test_model_source_returns_model_name_when_local_file_missing(tmp_path: Path) -> None:
    assert model_source(tmp_path, "yolo26m.pt") == "yolo26m.pt"
