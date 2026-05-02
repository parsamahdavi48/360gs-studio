import os
import math
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
    assert not hasattr(step, "json_name_edit")
    assert not hasattr(step, "mask_browse")
    assert not hasattr(step, "mask_from_alpha_cb")
    assert not hasattr(step, "no_image_cb")
    assert not hasattr(step, "no_transform_cb")
    assert not hasattr(step, "duplicate_cb")
    assert hasattr(step, "invert_masks_cb")
    assert not step.output_path_label.wordWrap()
    assert step.output_path_label.full_text() == str(tmp_path / "output")
    assert step.scale_combo.itemText(0) == "Full (Quality)"
    assert step.scale_combo.itemText(1) == "Normal"
    assert step.scale_combo.itemText(2) == "Half (Light)"
    assert float(step.scale_combo.itemData(1)) == pytest.approx(2.0 / math.pi)
    assert float(step.scale_combo.currentData()) == 1.0

    cmd = step._build_cubemap_cmd()

    assert cmd[3] == str(tmp_path)
    assert cmd[4] == str(tmp_path / "output")
    assert "--json" not in cmd
    assert "--mask_dir" not in cmd
    assert "--mask_from_alpha" not in cmd
    assert "--no_image" not in cmd
    assert "--duplicate" not in cmd
    assert "--no_transform" in cmd

    step.scale_combo.setCurrentIndex(1)
    normal_cmd = step._build_cubemap_cmd()
    normal_scale = float(normal_cmd[normal_cmd.index("--output_scale") + 1])
    assert normal_scale == pytest.approx(2.0 / math.pi, rel=1e-5)


def test_cubemap_step_keeps_mask_inversion_as_advanced_option(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    step.invert_masks_cb.setChecked(True)

    cmd = step._build_cubemap_cmd()

    assert "--invert_masks" in cmd


def test_cubemap_preview_uses_scene_mask_folder(tmp_path: Path, monkeypatch) -> None:
    step = _ready_step(tmp_path)
    captured: dict[str, str] = {}

    def fake_render(_views, mask_dir: str) -> None:
        captured["mask_dir"] = mask_dir

    monkeypatch.setattr(step.preview, "render", fake_render)

    step._render_preview()

    assert captured["mask_dir"] == str(tmp_path / "masks")


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
