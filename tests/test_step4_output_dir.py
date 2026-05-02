import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from gui.steps.step4_cubemap import CubemapStep


def _app():
    return QApplication.instance() or QApplication([])


def _ready_step(scene: Path) -> CubemapStep:
    _app()
    scene.mkdir(exist_ok=True)
    (scene / "pointcloud.ply").write_text("ply\n", encoding="utf-8")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(scene))
    step.preprocess_cb.setChecked(False)
    return step


def test_cubemap_step_uses_fixed_output_folder_label(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)

    assert not hasattr(step, "output_browse")
    assert step.output_path_label.text() == str(tmp_path / "output")

    cmd = step._build_cubemap_cmd()

    assert cmd[3] == str(tmp_path)
    assert cmd[4] == str(tmp_path / "output")


def test_cubemap_build_cancel_keeps_existing_output(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    old_file = output / "old.txt"
    old_file.write_text("old", encoding="utf-8")
    (tmp_path / "images").mkdir()
    step = _ready_step(tmp_path)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)

    commands = step.build_commands()

    assert commands == []
    assert old_file.is_file()
    assert not (output / "views_config.json").exists()


def test_cubemap_build_resets_existing_output_when_confirmed(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    nested = output / "nested"
    nested.mkdir(parents=True)
    old_file = output / "old.txt"
    nested_file = nested / "old_nested.txt"
    old_file.write_text("old", encoding="utf-8")
    nested_file.write_text("old", encoding="utf-8")
    step = _ready_step(tmp_path)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["cubemap"]
    assert not old_file.exists()
    assert not nested.exists()
    assert (output / "views_config.json").is_file()


def test_cubemap_build_validates_before_resetting_output(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    old_file = output / "old.txt"
    old_file.write_text("old", encoding="utf-8")
    (tmp_path / "images").mkdir()
    step = _ready_step(tmp_path)
    step.preprocess_cb.setChecked(True)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("confirmation should not open")),
    )

    with pytest.raises(ValueError, match="XML"):
        step.build_commands()

    assert old_file.is_file()
