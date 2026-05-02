import json
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
    assert not hasattr(step, "no_transform_cb")
    assert not hasattr(step, "duplicate_cb")
    assert not hasattr(step, "ms_images_browse")
    assert hasattr(step, "ms_use_ply_cb")
    assert hasattr(step, "axis_transform_combo")
    assert hasattr(step, "invert_masks_cb")
    assert hasattr(step, "no_image_cb")
    assert not step.no_image_cb.isChecked()
    assert step.export_method_combo.currentData() == "metashape"
    assert not step.output_path_label.wordWrap()
    assert step.output_path_label.full_text() == str(tmp_path / "output")
    assert step.ms_images_path_label.full_text() == str(tmp_path / "images")
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
    assert step.axis_transform_combo.currentData() == "none"
    assert step.ms_use_ply_cb.isChecked()

    step.scale_combo.setCurrentIndex(1)
    normal_cmd = step._build_cubemap_cmd()
    normal_scale = float(normal_cmd[normal_cmd.index("--output_scale") + 1])
    assert normal_scale == pytest.approx(2.0 / math.pi, rel=1e-5)


def test_colmap_export_method_uses_image_only_conversion(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    (images / "frame_0001.jpg").write_bytes(b"dummy")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    colmap_idx = step.export_method_combo.findData("colmap")
    assert colmap_idx >= 0

    step.export_method_combo.setCurrentIndex(colmap_idx)

    assert not step.metashape_section.isVisible()
    commands = step.build_commands()
    assert [phase for phase, _cmd in commands] == ["colmap_export"]
    cmd = commands[0][1]
    assert "--image-only" in cmd
    assert "--no_transform" not in cmd
    assert "--brush" not in cmd
    assert "--no_image" not in cmd


def test_colmap_export_method_validates_images_before_resetting_output(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    old_file = output / "old.txt"
    old_file.write_text("old", encoding="utf-8")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    colmap_idx = step.export_method_combo.findData("colmap")
    assert colmap_idx >= 0
    step.export_method_combo.setCurrentIndex(colmap_idx)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("confirmation should not open")),
    )

    with pytest.raises(ValueError, match="画像フォルダ"):
        step.build_commands()

    assert old_file.is_file()


def test_colmap_export_finalize_writes_export_method_settings(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    (images / "frame_0001.jpg").write_bytes(b"dummy")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    colmap_idx = step.export_method_combo.findData("colmap")
    assert colmap_idx >= 0
    step.export_method_combo.setCurrentIndex(colmap_idx)

    commands = step.build_commands()
    assert [phase for phase, _cmd in commands] == ["colmap_export"]
    step._finalize_bundle()

    settings_path = tmp_path / "output" / "stechdrive_export_settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["export_method"] == "colmap"
    assert settings["conversion"]["no_image"] is False
    assert settings["conversion"]["export_colmap"] is False


def test_metashape_import_uses_scene_images_and_lf_ply(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    (tmp_path / "images").mkdir()
    (tmp_path / "metashape.xml").write_text("<root />", encoding="utf-8")
    (tmp_path / "metashape.ply").write_text("ply\n", encoding="utf-8")
    step.ms_ply_browse.set_text(str(tmp_path / "metashape.ply"))
    step.preprocess_cb.setChecked(True)

    cmd = step._build_preprocess_cmd()

    assert cmd[cmd.index("--images") + 1] == str(tmp_path / "images")
    assert cmd[cmd.index("--xml") + 1] == str(tmp_path / "metashape.xml")
    assert cmd[cmd.index("--ply") + 1] == str(tmp_path / "metashape.ply")


def test_postshot_does_not_use_lichtfeld_pointcloud(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    idx = step.profile_combo.findData("postshot")
    assert idx >= 0
    step.profile_combo.setCurrentIndex(idx)

    assert step._resolve_ply_source() is None
    with pytest.raises(ValueError, match="pointcloud"):
        step.build_commands()


def test_profile_presets_sync_manual_axis_controls(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)

    postshot_idx = step.profile_combo.findData("postshot")
    brush_idx = step.profile_combo.findData("brush")
    lichtfeld_idx = step.profile_combo.findData("lichtfeld")
    assert postshot_idx >= 0
    assert brush_idx >= 0
    assert lichtfeld_idx >= 0

    step.profile_combo.setCurrentIndex(postshot_idx)
    assert step.axis_transform_combo.currentData() == "postshot"
    assert not step.ms_use_ply_cb.isChecked()
    assert step.ms_scale_edit.text() == "1.0"
    assert not step.ms_no_fix_rot_cb.isChecked()

    step.profile_combo.setCurrentIndex(brush_idx)
    assert step.axis_transform_combo.currentData() == "brush"
    assert not step.ms_use_ply_cb.isChecked()
    assert step.ms_scale_edit.text() == "1.0"
    assert not step.ms_no_fix_rot_cb.isChecked()

    step.profile_combo.setCurrentIndex(lichtfeld_idx)
    assert step.axis_transform_combo.currentData() == "none"
    assert step.ms_use_ply_cb.isChecked()
    assert step.ms_scale_edit.text() == "1.0"
    assert not step.ms_no_fix_rot_cb.isChecked()


def test_manual_axis_change_switches_to_custom_profile(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    (tmp_path / "metashape.ply").write_text("ply\n", encoding="utf-8")
    postshot_idx = step.profile_combo.findData("postshot")
    assert postshot_idx >= 0
    step.profile_combo.setCurrentIndex(postshot_idx)

    brush_axis = step.axis_transform_combo.findData("brush")
    assert brush_axis >= 0
    step.axis_transform_combo.setCurrentIndex(brush_axis)

    assert step.profile_combo.currentData() == "custom"
    cmd = step._build_cubemap_cmd()
    assert "--brush" in cmd
    assert "--no_transform" not in cmd


def test_manual_metashape_ply_toggle_switches_to_custom_profile(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    (tmp_path / "images").mkdir()
    (tmp_path / "metashape.xml").write_text("<root />", encoding="utf-8")
    (tmp_path / "metashape.ply").write_text("ply\n", encoding="utf-8")
    step.ms_ply_browse.set_text(str(tmp_path / "metashape.ply"))
    postshot_idx = step.profile_combo.findData("postshot")
    assert postshot_idx >= 0
    step.profile_combo.setCurrentIndex(postshot_idx)

    step.ms_use_ply_cb.setChecked(True)
    step.preprocess_cb.setChecked(True)

    assert step.profile_combo.currentData() == "custom"
    cmd = step._build_preprocess_cmd()
    assert cmd[cmd.index("--ply") + 1] == str(tmp_path / "metashape.ply")


def test_manual_metashape_import_detail_change_switches_to_custom_profile(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    lichtfeld_idx = step.profile_combo.findData("lichtfeld")
    postshot_idx = step.profile_combo.findData("postshot")
    assert lichtfeld_idx >= 0
    assert postshot_idx >= 0

    step.profile_combo.setCurrentIndex(lichtfeld_idx)
    step.ms_no_fix_rot_cb.setChecked(True)
    assert step.profile_combo.currentData() == "custom"

    step.profile_combo.setCurrentIndex(postshot_idx)
    step.ms_scale_edit.setText("2.0")
    step._on_profile_option_changed()
    assert step.profile_combo.currentData() == "custom"


def test_postshot_accepts_raw_ply_with_custom_name(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    raw_ply = tmp_path / "raw_scan.ply"
    raw_ply.write_text("ply\n", encoding="utf-8")
    idx = step.profile_combo.findData("postshot")
    assert idx >= 0
    step.profile_combo.setCurrentIndex(idx)

    assert step._resolve_ply_source() == raw_ply


def test_lichtfeld_requires_converted_pointcloud_without_preprocess(tmp_path: Path) -> None:
    _app()
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "metashape.ply").write_text("ply\n", encoding="utf-8")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step.preprocess_cb.setChecked(False)

    assert step._resolve_ply_source() is None
    with pytest.raises(ValueError, match="pointcloud"):
        step.build_commands()


def test_cubemap_step_keeps_mask_inversion_as_advanced_option(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    step.invert_masks_cb.setChecked(True)

    cmd = step._build_cubemap_cmd()

    assert "--invert_masks" in cmd


def test_cubemap_step_can_skip_image_and_mask_conversion(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    step.no_image_cb.setChecked(True)

    cmd = step._build_cubemap_cmd()

    assert "--no_image" in cmd


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


def test_cubemap_no_image_preserves_existing_output(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    old_file = output / "old_render.png"
    old_file.write_text("old", encoding="utf-8")
    old_mask_dir = output / "masks"
    old_mask_dir.mkdir()
    old_mask = old_mask_dir / "old_mask.png"
    old_mask.write_text("mask", encoding="utf-8")
    old_settings = output / "stechdrive_export_settings.json"
    old_settings.write_text('{"old": true}\n', encoding="utf-8")
    step = _ready_step(tmp_path)
    step.no_image_cb.setChecked(True)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("confirmation should not open")),
    )

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["cubemap"]
    assert "--no_image" in commands[0][1]
    assert old_file.is_file()
    assert old_mask.is_file()
    assert old_settings.read_text(encoding="utf-8") == '{"old": true}\n'
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


def test_cubemap_finalize_writes_export_settings(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    commands = step.build_commands()
    assert [phase for phase, _cmd in commands] == ["cubemap"]
    step._finalize_bundle()

    settings_path = tmp_path / "output" / "stechdrive_export_settings.json"
    assert settings_path.is_file()
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["app"] == "stechdrive-3dgs-utils"
    assert settings["settings_version"] == 1
    assert settings["export_method"] == "metashape"
    assert settings["target_profile"] == "lichtfeld"
    assert settings["effective_profile"] == "lichtfeld"
    assert settings["axis_transform"] == "none"
    assert settings["fov"] == 90.0
    assert settings["image_size"]["scale"] == 1.0
    assert settings["conversion"]["yaw_offset_per_frame"] == 30.0
    assert settings["conversion"]["output_format"] == "auto"
    assert settings["conversion"]["output_bit_depth"] == "8"
    assert settings["conversion"]["no_image"] is False
    assert settings["metashape_import"]["use_ply"] is True
    assert settings["output_files"]["settings"] == "stechdrive_export_settings.json"
    assert settings["view_config"]["views"]
    assert settings["views_config_path"] == "views_config.json"
    assert settings["views_config_snapshot"] == json.loads(
        (tmp_path / "output" / "views_config.json").read_text(encoding="utf-8")
    )
