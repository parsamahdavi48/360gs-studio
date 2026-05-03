import json
import os
import math
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from gui.steps.step4_cubemap import CubemapStep
from transforms_to_colmap import read_ply_points


def _app():
    return QApplication.instance() or QApplication([])


def _ready_step(scene: Path, *, metashape_inputs: bool = False) -> CubemapStep:
    _app()
    scene.mkdir(exist_ok=True)
    _write_ascii_ply(scene / "pointcloud.ply", [(0.0, 0.0, 0.0)])
    if metashape_inputs:
        (scene / "images").mkdir(exist_ok=True)
        (scene / "metashape.xml").write_text("<root />", encoding="utf-8")
        _write_ascii_ply(scene / "metashape.ply", [(1.0, 2.0, 3.0)])
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(scene))
    return step


def _write_ascii_ply(path: Path, points: list[tuple[float, float, float]]) -> None:
    rows = "\n".join(f"{x:g} {y:g} {z:g}" for x, y, z in points)
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        f"{rows}\n",
        encoding="ascii",
    )


def _is_descendant(widget, ancestor) -> bool:
    current = widget
    while current is not None:
        if current is ancestor:
            return True
        current = current.parentWidget()
    return False


def test_cubemap_step_uses_fixed_output_folder_label(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)

    assert not hasattr(step, "output_browse")
    assert not hasattr(step, "json_name_edit")
    assert not hasattr(step, "mask_browse")
    assert not hasattr(step, "mask_from_alpha_cb")
    assert not hasattr(step, "no_transform_cb")
    assert not hasattr(step, "duplicate_cb")
    assert not hasattr(step, "ms_images_browse")
    assert not hasattr(step, "preprocess_cb")
    assert not hasattr(step, "no_image_cb")
    assert not hasattr(step.view_config, "cube6_drop_top")
    assert not hasattr(step.view_config, "cube6_drop_bottom")
    assert hasattr(step, "ms_use_ply_cb")
    assert hasattr(step, "axis_transform_combo")
    assert hasattr(step, "invert_masks_cb")
    assert not hasattr(step, "export_method_combo")
    assert not hasattr(step.view_config, "pitch_edit")
    assert not hasattr(step.view_config, "apply_btn")
    assert set(step.export_method_buttons) == {"metashape", "colmap"}
    assert step.export_images_cb.isChecked()
    assert step.export_masks_cb.isChecked()
    assert not _is_descendant(step.export_targets_row, step.advanced_output_section)
    assert _is_descendant(step.view_config.settings_widget, step.advanced_output_section)
    assert _is_descendant(step.view_config.grid_section, step.advanced_output_section)
    assert _is_descendant(step.output_details_section, step.advanced_output_section)
    assert not _is_descendant(step.export_summary_label, step.advanced_output_section)
    assert not _is_descendant(step.export_summary_label, step.view_config.settings_widget)
    settings_layout = step.view_config.settings_widget.layout()
    custom_index = settings_layout.indexOf(step.view_config.custom_controls_widget)
    grid_index = settings_layout.indexOf(step.view_config.grid_section)
    assert custom_index + 1 == grid_index
    assert step.export_summary_label.text() == step.view_config.summary_text()
    cube6_views = step.view_config.collect_views(include_disabled=True)
    assert {v["name"] for v in cube6_views} == {"px", "nx", "pz", "nz", "top", "bottom"}
    assert all(v["enabled"] for v in cube6_views)
    assert step._export_method() == "metashape"
    assert step.export_method_buttons["metashape"].isChecked()
    assert not step.output_path_label.wordWrap()
    assert step.output_path_label.full_text() == str(tmp_path / "output")
    assert step.ms_images_path_label.full_text() == str(tmp_path / "images")
    assert step.scale_combo.itemText(0) == "Full"
    assert step.scale_combo.itemText(1) == "Normal"
    assert step.scale_combo.itemText(2) == "Half"
    assert float(step.scale_combo.itemData(1)) == pytest.approx(2.0 / math.pi)
    assert float(step.scale_combo.currentData()) == 1.0

    cmd = step._build_cubemap_cmd()

    assert cmd[3] == str(tmp_path)
    assert cmd[4] == str(tmp_path / "output")
    assert "--json" not in cmd
    assert "--mask_dir" not in cmd
    assert "--mask_from_alpha" not in cmd
    assert "--no_image" not in cmd
    assert "--skip-images" not in cmd
    assert "--skip-masks" not in cmd
    assert "--duplicate" not in cmd
    assert "--no_transform" in cmd
    assert step.axis_transform_combo.currentData() == "none"
    assert step.ms_use_ply_cb.isChecked()

    step.scale_combo.setCurrentIndex(1)
    normal_cmd = step._build_cubemap_cmd()
    normal_scale = float(normal_cmd[normal_cmd.index("--output_scale") + 1])
    assert normal_scale == pytest.approx(2.0 / math.pi, rel=1e-5)


def test_cubemap_step_does_not_count_repo_images_without_scene_dir() -> None:
    _app()
    step = CubemapStep(Path.cwd())

    assert step.scene_dir == ""
    assert step._count_input_images() == 0


def test_custom_grid_defaults_to_three_pitch_rows_all_enabled() -> None:
    _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    assert idx >= 0
    step.view_config.view_mode_combo.setCurrentIndex(idx)

    assert step.view_config.pitch_rows_combo.currentText() == "3"
    assert step.view_config.pitch_values() == [-45.0, 0.0, 45.0]
    assert step.view_config.pitch_rows_text() == "-45,0,45"
    assert step.view_config.yaw_slots_combo.currentText() == "6"
    views = step.view_config.collect_views(include_disabled=True)

    assert len(views) == 18
    assert sum(1 for view in views if view["enabled"]) == 18
    assert {view["pitch"] for view in views} == {-45.0, 0.0, 45.0}


def test_custom_grid_pitch_rows_are_limited_to_five() -> None:
    _app()
    step = CubemapStep(Path.cwd())

    assert [step.view_config.pitch_rows_combo.itemText(i) for i in range(step.view_config.pitch_rows_combo.count())] == [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
    step.view_config.pitch_rows_combo.setCurrentText("5")
    assert step.view_config._parse_pitches() == [-60.0, -30.0, 0.0, 30.0, 60.0]


def test_custom_grid_controls_apply_immediately_and_keep_pitch_unique() -> None:
    _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)

    step.view_config.yaw_slots_combo.setCurrentText("8")
    assert len(step.view_config.collect_views(include_disabled=True)) == 24

    step.view_config.pitch_rows_combo.setCurrentText("5")
    assert len(step.view_config.collect_views(include_disabled=True)) == 40
    assert sum(1 for view in step.view_config.collect_views(include_disabled=True) if view["enabled"]) == 40

    step.view_config.pitch_rows[0]["pitch_edit"].setValue(0.0)
    pitches = step.view_config.pitch_values()
    assert len(pitches) == len({_pitch_key for _pitch_key in pitches})
    assert pitches[0] != 0.0


def test_cubemap_yaw_numeric_fields_are_clamped_and_used(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)

    step.view_config.yaw_offset_edit.setValue(999.0)
    step.yaw_per_frame_edit.setValue(-999.0)

    assert step.view_config.yaw_offset() == 180.0
    assert step.yaw_per_frame_edit.value() == -180.0

    cmd = step._build_cubemap_cmd()

    assert cmd[cmd.index("--yaw-offset-per-frame") + 1] == "-180"
    views = json.loads((tmp_path / "output" / "views_config.json").read_text(encoding="utf-8"))["views"]
    assert any(view["yaw"] == -90.0 for view in views)


def test_colmap_export_method_uses_image_only_conversion(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    (images / "frame_0001.jpg").write_bytes(b"dummy")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    step._set_export_method("colmap")

    assert not step.metashape_section.isVisible()
    assert step.export_method_buttons["colmap"].isChecked()
    commands = step.build_commands()
    assert [phase for phase, _cmd in commands] == ["colmap_rig_export"]
    cmd = commands[0][1]
    assert "--image-only" in cmd
    assert "--colmap-rig" in cmd
    assert cmd[cmd.index("--yaw-offset-per-frame") + 1] == "0"
    assert "--no_transform" not in cmd
    assert "--brush" not in cmd
    assert "--no_image" not in cmd
    assert "--skip-images" not in cmd
    assert "--skip-masks" not in cmd


def test_colmap_export_method_validates_images_before_resetting_output(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    old_file = output / "old.txt"
    old_file.write_text("old", encoding="utf-8")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("colmap")
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
    step._set_export_method("colmap")

    commands = step.build_commands()
    assert [phase for phase, _cmd in commands] == ["colmap_rig_export"]
    step._finalize_bundle()

    settings_path = tmp_path / "output" / "stechdrive_export_settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["export_method"] == "colmap"
    assert settings["conversion"]["no_image"] is False
    assert settings["conversion"]["export_colmap"] is False
    assert settings["conversion"]["yaw_offset_per_frame"] == 0.0
    assert settings["colmap_rig"]["enabled"] is True
    assert settings["colmap_rig"]["dir"] == str(tmp_path / "output" / "colmap_rig")


def test_colmap_export_can_queue_colmap_sfm_with_custom_executable(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    (images / "frame_0001.jpg").write_bytes(b"dummy")
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("colmap")
    step.run_colmap_cb.setChecked(True)
    step.colmap_exec_browse.set_text(str(fake_colmap))

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == [
        "colmap_rig_export",
        "colmap_feature",
        "colmap_rig_config",
        "colmap_match",
        "colmap_mapper",
    ]
    assert commands[1][1][0] == str(fake_colmap)
    assert commands[1][1][1] == "feature_extractor"
    assert "--ImageReader.single_camera_per_folder" in commands[1][1]
    assert commands[2][1][1] == "rig_configurator"
    assert commands[3][1][1] == "sequential_matcher"
    assert commands[4][1][1] == "global_mapper"


def test_colmap_export_can_queue_colmap_global_mapper(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    (images / "frame_0001.jpg").write_bytes(b"dummy")
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("colmap")
    step.run_colmap_cb.setChecked(True)
    step.colmap_exec_browse.set_text(str(fake_colmap))
    idx = step.colmap_mapper_combo.findData("global")
    assert idx >= 0
    step.colmap_mapper_combo.setCurrentIndex(idx)

    commands = step.build_commands()

    assert commands[-1][1][0] == str(fake_colmap)
    assert commands[-1][1][1] == "global_mapper"


def test_colmap_user_preferences_restore_executable_and_pipeline_choices(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _app()
    settings_path = tmp_path / "settings.json"
    monkeypatch.setenv("STECHDRIVE_USER_SETTINGS_PATH", str(settings_path))

    first = CubemapStep(Path.cwd())
    first.enable_user_preferences()
    first.colmap_exec_browse.set_text(str(tmp_path / "colmap.exe"))
    first.glomap_exec_browse.set_text(str(tmp_path / "glomap.exe"))
    matcher_idx = first.colmap_matcher_combo.findData("exhaustive")
    mapper_idx = first.colmap_mapper_combo.findData("incremental")
    assert matcher_idx >= 0
    assert mapper_idx >= 0
    first.colmap_matcher_combo.setCurrentIndex(matcher_idx)
    first.colmap_mapper_combo.setCurrentIndex(mapper_idx)

    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert stored["step4_colmap"]["colmap_executable"].endswith("colmap.exe")
    assert stored["step4_colmap"]["glomap_executable"].endswith("glomap.exe")
    assert stored["step4_colmap"]["matcher"] == "exhaustive"
    assert stored["step4_colmap"]["mapper"] == "incremental"

    second = CubemapStep(Path.cwd())
    second.enable_user_preferences()

    assert second.colmap_exec_browse.text().endswith("colmap.exe")
    assert second.glomap_exec_browse.text().endswith("glomap.exe")
    assert second.colmap_matcher_combo.currentData() == "exhaustive"
    assert second.colmap_mapper_combo.currentData() == "incremental"


def test_metashape_import_uses_scene_images_and_lf_ply(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    (tmp_path / "images").mkdir()
    (tmp_path / "metashape.xml").write_text("<root />", encoding="utf-8")
    (tmp_path / "metashape.ply").write_text("ply\n", encoding="utf-8")
    step.ms_ply_browse.set_text(str(tmp_path / "metashape.ply"))

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


def test_lichtfeld_import_requires_raw_ply_when_ply_enabled(tmp_path: Path) -> None:
    _app()
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "images").mkdir()
    (tmp_path / "metashape.xml").write_text("<root />", encoding="utf-8")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step._resolve_ply_source() is None
    with pytest.raises(ValueError, match="PLY"):
        step.build_commands()


def test_cubemap_step_keeps_mask_inversion_as_advanced_option(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    step.invert_masks_cb.setChecked(True)

    cmd = step._build_cubemap_cmd()

    assert "--invert_masks" in cmd


def test_cubemap_step_can_skip_image_and_mask_conversion(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    step.export_images_cb.setChecked(False)
    step.export_masks_cb.setChecked(False)

    cmd = step._build_cubemap_cmd()

    assert "--skip-images" in cmd
    assert "--skip-masks" in cmd


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
    step = _ready_step(tmp_path, metashape_inputs=True)
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
    step = _ready_step(tmp_path, metashape_inputs=True)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["metashape", "cubemap"]
    assert not old_file.exists()
    assert not nested.exists()
    assert (output / "views_config.json").is_file()


def test_cubemap_mask_only_preserves_existing_images(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    old_image_dir = output / "images"
    old_image_dir.mkdir()
    old_file = old_image_dir / "old_render.png"
    old_file.write_text("old", encoding="utf-8")
    old_mask_dir = output / "masks"
    old_mask_dir.mkdir()
    old_mask = old_mask_dir / "old_mask.png"
    old_mask.write_text("mask", encoding="utf-8")
    old_settings = output / "stechdrive_export_settings.json"
    old_settings.write_text('{"old": true}\n', encoding="utf-8")
    step = _ready_step(tmp_path, metashape_inputs=True)
    step.export_images_cb.setChecked(False)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["metashape", "cubemap"]
    assert "--skip-images" in commands[1][1]
    assert "--skip-masks" not in commands[1][1]
    assert old_file.is_file()
    assert not old_mask.exists()
    assert old_settings.read_text(encoding="utf-8") == '{"old": true}\n'
    assert (output / "views_config.json").is_file()


def test_cubemap_build_validates_before_resetting_output(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    old_file = output / "old.txt"
    old_file.write_text("old", encoding="utf-8")
    (tmp_path / "images").mkdir()
    step = _ready_step(tmp_path)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("confirmation should not open")),
    )

    with pytest.raises(ValueError, match="XML"):
        step.build_commands()

    assert old_file.is_file()


def test_cubemap_finalize_writes_export_settings(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    commands = step.build_commands()
    assert [phase for phase, _cmd in commands] == ["metashape", "cubemap"]
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
    assert settings["conversion"]["write_images"] is True
    assert settings["conversion"]["write_masks"] is True
    assert settings["view_config"]["cube6_drop_top"] is False
    assert settings["view_config"]["cube6_drop_bottom"] is False
    assert settings["metashape_import"]["use_ply"] is True
    assert settings["output_files"]["settings"] == "stechdrive_export_settings.json"
    assert settings["view_config"]["views"]
    assert settings["views_config_path"] == "views_config.json"
    assert settings["views_config_snapshot"] == json.loads(
        (tmp_path / "output" / "views_config.json").read_text(encoding="utf-8")
    )


def test_lichtfeld_finalize_applies_final_orientation_correction(tmp_path: Path, monkeypatch) -> None:
    _write_ascii_ply(tmp_path / "pointcloud.ply", [(1.0, 2.0, 3.0)])
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    monkeypatch.setattr(CubemapStep, "_transform_ply_with_open3d", staticmethod(lambda _path, _matrix: False))

    output = tmp_path / "output"
    output.mkdir()
    (output / "transforms.json").write_text(
        json.dumps(
            {
                "camera_model": "SIMPLE_PINHOLE",
                "frames": [{"file_path": "images/frame_front.jpg", "transform_matrix": np.eye(4).tolist()}],
            }
        ),
        encoding="utf-8",
    )

    step._finalize_bundle()

    data = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
    corrected = np.array(data["frames"][0]["transform_matrix"])
    expected = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(corrected, expected)

    points, _colors = read_ply_points(output / "pointcloud.ply")
    assert np.allclose(points[0], [3.0, -2.0, 1.0])

    settings = json.loads((output / "stechdrive_export_settings.json").read_text(encoding="utf-8"))
    assert settings["postprocess"]["lichtfeld_final_orientation_correction"] is True
