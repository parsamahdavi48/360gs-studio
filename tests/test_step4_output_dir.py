import json
import math
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QMessageBox

import gui.steps.step4_cubemap as step4_cubemap
from gui import i18n
from gui.steps.step4_cubemap import CubemapStep
from scene_layout import step4_export_settings_path, step4_views_config_path
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


def _write_test_image(path: Path, size: tuple[int, int] = (64, 32)) -> None:
    Image.new("RGB", size, (0, 0, 0)).save(path)


def _write_spheresfm_sparse_stub(scene: Path) -> Path:
    sparse_model = scene / "output" / "spheresfm" / "sparse" / "0"
    sparse_model.mkdir(parents=True, exist_ok=True)
    (sparse_model / "cameras.txt").write_text("# cameras\n", encoding="ascii")
    (sparse_model / "images.txt").write_text("# images\n", encoding="ascii")
    (sparse_model / "points3D.txt").write_text("# points\n", encoding="ascii")
    return sparse_model


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
    assert not hasattr(step.view_config, "yaw_slots_combo")
    assert not hasattr(step.view_config, "pitch_rows_combo")
    assert not hasattr(step.view_config, "custom_controls_widget")
    assert hasattr(step, "ms_use_ply_cb")
    assert hasattr(step, "axis_transform_combo")
    assert hasattr(step, "invert_masks_cb")
    assert not hasattr(step, "export_method_combo")
    assert not hasattr(step.view_config, "pitch_edit")
    assert not hasattr(step.view_config, "apply_btn")
    assert set(step.export_method_buttons) == {"metashape", "colmap", "spheresfm"}
    assert step.export_images_cb.isChecked()
    assert step.export_masks_cb.isChecked()
    assert step.output_shape_combo.currentData() == "projected"
    assert step.view_export_tab_index == 0
    assert step.metashape_tab_index == 1
    assert step.colmap_tab_index == 2
    assert step.settings_tabs.tabText(0) == i18n.t("STEP4_TAB_VIEW_EXPORT")
    assert step.settings_tabs.tabText(step.metashape_tab_index) == i18n.t("STEP4_TAB_METASHAPE")
    assert step.settings_tabs.tabText(step.view_export_tab_index) == i18n.t("STEP4_TAB_VIEW_EXPORT")
    assert step.settings_tabs.tabText(step.colmap_tab_index) == i18n.t("STEP4_TAB_COLMAP")
    assert step.settings_tabs.isTabVisible(step.metashape_tab_index)
    assert not step.settings_tabs.isTabVisible(step.colmap_tab_index)
    assert not _is_descendant(step.export_targets_row, step.advanced_output_section)
    assert _is_descendant(step.view_config.settings_widget, step.advanced_output_section)
    assert _is_descendant(step.view_config.grid_section, step.advanced_output_section)
    assert _is_descendant(step.view_config.all_on_btn, step.view_config.grid_controls_widget)
    assert _is_descendant(step.view_config.all_off_btn, step.view_config.grid_controls_widget)
    assert not _is_descendant(step.view_config.all_on_btn, step.view_config.grid_widget)
    assert not _is_descendant(step.view_config.all_off_btn, step.view_config.grid_widget)
    assert _is_descendant(step.view_config.pitch_add_btn, step.view_config.pitch_controls_widget)
    assert _is_descendant(step.view_config.pitch_count_label, step.view_config.pitch_controls_widget)
    assert not _is_descendant(step.view_config.pitch_add_btn, step.view_config.grid_controls_widget)
    assert _is_descendant(step.output_details_section, step.advanced_output_section)
    assert not _is_descendant(step.export_summary_label, step.advanced_output_section)
    assert not _is_descendant(step.export_summary_label, step.view_config.settings_widget)
    assert step.export_summary_label.text() == step.view_config.summary_text()
    cube6_views = step.view_config.collect_views(include_disabled=True)
    cube6_enabled = [v for v in cube6_views if v["enabled"]]
    assert step.view_config.view_mode() == "cube6"
    assert step.view_config.yaw_slot_count() == 4
    assert step.view_config.pitch_values() == [-90.0, 0.0, 90.0]
    assert len(cube6_views) == 12
    assert {v["name"] for v in cube6_enabled} == {"px", "nx", "pz", "nz", "top", "bottom"}
    assert {v["slot"] for v in cube6_enabled if v["name"] in {"top", "bottom"}} == {3}
    assert sum(1 for v in cube6_views if not v["enabled"]) == 6
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


def test_export_method_switch_keeps_view_export_tab_leftmost() -> None:
    _app()
    step = CubemapStep(Path.cwd())

    assert step.settings_tabs.tabText(0) == i18n.t("STEP4_TAB_VIEW_EXPORT")
    assert step.settings_tabs.currentIndex() == step.view_export_tab_index

    step.settings_tabs.setCurrentIndex(step.metashape_tab_index)
    step._set_export_method("colmap")

    assert step.settings_tabs.tabText(0) == i18n.t("STEP4_TAB_VIEW_EXPORT")
    assert step.settings_tabs.isTabVisible(step.view_export_tab_index)
    assert not step.settings_tabs.isTabVisible(step.metashape_tab_index)
    assert step.settings_tabs.isTabVisible(step.colmap_tab_index)
    assert step.settings_tabs.currentIndex() == step.colmap_tab_index

    step.settings_tabs.setCurrentIndex(step.view_export_tab_index)
    step._set_export_method("metashape")

    assert step.settings_tabs.tabText(0) == i18n.t("STEP4_TAB_VIEW_EXPORT")
    assert step.settings_tabs.currentIndex() == step.view_export_tab_index
    assert step.settings_tabs.isTabVisible(step.metashape_tab_index)
    assert not step.settings_tabs.isTabVisible(step.colmap_tab_index)


def test_cubemap_step_does_not_count_repo_images_without_scene_dir() -> None:
    _app()
    step = CubemapStep(Path.cwd())

    assert step.scene_dir == ""
    assert step._count_input_images() == 0


def test_cubemap_step_does_not_use_current_directory_without_scene_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    (tmp_path / "pointcloud.ply").write_text("ply\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    step = CubemapStep(Path.cwd())

    step.on_activated()

    assert step.preview.current_image_path() is None
    assert step._count_input_images() == 0
    assert step._resolve_ply_source() is None
    with pytest.raises(ValueError, match=i18n.t("SCENE_REQUIRED_ACTION_HINT")):
        step._output_dir()
    with pytest.raises(ValueError, match=i18n.t("SCENE_REQUIRED_ACTION_HINT")):
        step._mask_dir()
    with pytest.raises(ValueError, match=i18n.t("SCENE_REQUIRED_ACTION_HINT")):
        step._metashape_images_dir()


def test_cubemap_step_refreshes_preview_when_activated_after_extraction(tmp_path: Path) -> None:
    _app()
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step.preview.current_image_path() is None

    images = tmp_path / "images"
    images.mkdir()
    image_path = images / "frame_0001.jpg"
    _write_test_image(image_path)

    step.on_activated()

    assert step.preview.current_image_path() == image_path
    assert step.preview.image_label._source_pixmap is not None
    assert step._count_input_images() == 1
    assert i18n.t("OUTPUT_IMAGE_COUNT_FORMAT").format(count=6) in step.view_config.summary_text()


def test_cubemap_step_projection_toggle_is_in_preview_header() -> None:
    _app()
    step = CubemapStep(Path.cwd())

    assert not step.preview.isAncestorOf(step.preview.projection_toggle_btn)
    assert step.preview.projection_toggle_btn.parentWidget().objectName() == "workPane"


def test_custom_grid_defaults_to_three_pitch_rows_all_enabled() -> None:
    _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    assert idx >= 0
    step.view_config.view_mode_combo.setCurrentIndex(idx)

    assert step.view_config.pitch_row_count() == 3
    assert step.view_config.pitch_values() == [-45.0, 0.0, 45.0]
    assert step.view_config.pitch_rows_text() == "-45,0,45"
    assert step.view_config.yaw_slot_count() == 6
    views = step.view_config.collect_views(include_disabled=True)

    assert len(views) == 18
    assert sum(1 for view in views if view["enabled"]) == 18
    assert {view["pitch"] for view in views} == {-45.0, 0.0, 45.0}


def test_editing_cube6_grid_switches_to_custom_without_resetting_grid() -> None:
    _app()
    step = CubemapStep(Path.cwd())

    assert step.view_config.view_mode() == "cube6"
    first_enabled = next(
        view
        for view in step.view_config.collect_views(include_disabled=True)
        if view["name"] == "px"
    )
    middle_row_index = step.view_config.pitch_values().index(0.0)
    step.view_config.pitch_rows[middle_row_index]["checks"][first_enabled["slot"]].setChecked(False)

    assert step.view_config.view_mode() == "custom_views"
    assert step.view_config.yaw_slot_count() == 4
    assert step.view_config.pitch_values() == [-90.0, 0.0, 90.0]
    assert sum(1 for view in step.view_config.collect_views(include_disabled=True) if view["enabled"]) == 5


def test_custom_grid_pitch_rows_are_limited_to_five() -> None:
    _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)

    step.view_config.set_pitch_row_count(99)
    assert step.view_config.pitch_row_count() == 5
    assert step.view_config._parse_pitches()[:3] == [-45.0, 0.0, 45.0]
    assert len(step.view_config._parse_pitches()) == len(set(step.view_config._parse_pitches()))

    step.view_config.set_pitch_row_count(0)
    assert step.view_config.pitch_row_count() == 1


def test_custom_grid_controls_apply_immediately_and_keep_pitch_unique() -> None:
    _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)

    step.view_config.set_yaw_slot_count(8)
    assert len(step.view_config.collect_views(include_disabled=True)) == 24

    step.view_config.set_pitch_row_count(5)
    assert len(step.view_config.collect_views(include_disabled=True)) == 40
    assert sum(1 for view in step.view_config.collect_views(include_disabled=True) if view["enabled"]) == 40

    step.view_config.pitch_rows[0]["pitch_edit"].setValue(0.0)
    pitches = step.view_config.pitch_values()
    assert len(pitches) == len({_pitch_key for _pitch_key in pitches})
    assert pitches[0] != 0.0


def test_custom_grid_resize_preserves_existing_rows_and_checks() -> None:
    _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)

    step.view_config.pitch_rows[0]["pitch_edit"].setValue(-12.0)
    step.view_config.pitch_rows[0]["checks"][0].setChecked(False)
    step.view_config.pitch_rows[1]["checks"][4].setChecked(False)
    pitches_before = step.view_config.pitch_values()

    step.view_config.set_yaw_slot_count(8)

    assert step.view_config.pitch_values() == pitches_before
    assert not step.view_config.pitch_rows[0]["checks"][0].isChecked()
    assert not step.view_config.pitch_rows[1]["checks"][4].isChecked()
    assert all(row["checks"][6].isChecked() and row["checks"][7].isChecked() for row in step.view_config.pitch_rows)

    step.view_config.set_yaw_slot_count(4)

    assert step.view_config.pitch_values() == pitches_before
    assert not step.view_config.pitch_rows[0]["checks"][0].isChecked()
    assert len(step.view_config.pitch_rows[0]["checks"]) == 4

    step.view_config.add_pitch_row()

    assert step.view_config.pitch_values()[:3] == pitches_before
    assert step.view_config.pitch_row_count() == 4
    assert all(cb.isChecked() for cb in step.view_config.pitch_rows[-1]["checks"])

    removed_pitch = step.view_config.pitch_values()[1]
    step.view_config.remove_pitch_row(1)

    assert removed_pitch not in step.view_config.pitch_values()
    assert step.view_config.pitch_values()[0] == pitches_before[0]
    assert step.view_config.pitch_row_count() == 3


def test_pitch_row_controls_are_packed_left_without_clipping() -> None:
    app = _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)
    step.view_config.pitch_rows[0]["pitch_edit"].setValue(-90.0)

    step.resize(720, 720)
    step.show()
    app.processEvents()

    first_row = step.view_config.pitch_rows[0]
    delete_btn = first_row["delete_btn"]
    pitch_edit = first_row["pitch_edit"]
    gap = pitch_edit.geometry().left() - delete_btn.geometry().right() - 1
    required_width = pitch_edit.fontMetrics().horizontalAdvance("-999.9") + 22

    assert gap <= 4
    assert pitch_edit.width() >= required_width

    step.close()


def test_yaw_slots_share_remaining_grid_width() -> None:
    app = _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)
    step.view_config.set_yaw_slot_count(4)

    step.resize(720, 720)
    step.show()
    app.processEvents()

    first_row = step.view_config.pitch_rows[0]
    pitch_cell = first_row["pitch_edit"].parentWidget()
    grid = step.view_config.grid_widget
    checkboxes = first_row["checks"]
    centers = [
        cb.mapTo(grid, QPoint(0, 0)).x() + cb.width() / 2.0
        for cb in checkboxes
    ]
    gaps = [b - a for a, b in zip(centers, centers[1:], strict=False)]
    pitch_width = pitch_cell.width()
    available_width = grid.width() - pitch_width
    expected_first_center = pitch_width + available_width / (2.0 * len(checkboxes))

    assert max(gaps) - min(gaps) <= 2.0
    assert abs(centers[0] - expected_first_center) <= 4.0

    step.close()


def test_custom_grid_bulk_selection_emits_single_change() -> None:
    _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)
    step.view_config.set_yaw_slot_count(8)
    step.view_config.set_pitch_row_count(5)

    emitted = 0

    def on_changed() -> None:
        nonlocal emitted
        emitted += 1

    step.view_config.views_changed.connect(on_changed)

    step.view_config._all_off()

    assert emitted == 1
    assert sum(1 for view in step.view_config.collect_views(include_disabled=True) if view["enabled"]) == 0

    step.view_config._all_off()
    assert emitted == 1

    step.view_config._all_on()
    assert emitted == 2
    assert sum(1 for view in step.view_config.collect_views(include_disabled=True) if view["enabled"]) == 40


def test_view_selection_reuses_cached_input_image_count(tmp_path: Path, monkeypatch) -> None:
    _app()
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    _write_test_image(images / "frame_0002.jpg")
    step.on_activated()

    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)
    monkeypatch.setattr(step.preview, "render", lambda _views, _mask_dir: None)

    def fail_rescan() -> int:
        raise AssertionError("unexpected image rescan")

    monkeypatch.setattr(step, "_count_input_images", fail_rescan)

    first_check = step.view_config.pitch_rows[0]["checks"][0]
    first_check.setChecked(False)

    assert i18n.t("OUTPUT_IMAGE_COUNT_FORMAT").format(count=34) in step.view_config.summary_text()


def test_custom_grid_hover_marks_view_and_rerenders_preview(tmp_path: Path, monkeypatch) -> None:
    step = _ready_step(tmp_path)
    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)
    captured: list[list[dict]] = []

    def fake_render(views: list[dict], _mask_dir: str) -> None:
        captured.append(views)

    monkeypatch.setattr(step.preview, "render", fake_render)

    step.view_config._on_view_hover(0, 2, True)

    assert captured
    highlighted = [view for view in captured[-1] if view.get("highlighted")]
    assert len(highlighted) == 1
    assert highlighted[0]["name"] == "pitm45_s2"

    step.view_config._on_view_hover(0, 2, False)
    assert not any(view.get("highlighted") for view in captured[-1])


def test_custom_grid_yaw_labels_are_compact_without_degree_mark() -> None:
    _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)
    step.view_config.set_yaw_slot_count(7)

    texts = [label.text() for label in step.view_config.yaw_slot_labels]
    views = step.view_config.collect_views(include_disabled=True)

    assert "S0\n45.0" in texts
    assert "S1\n96.4" in texts
    assert all("°" not in text for text in texts)
    assert all(label.toolTip().endswith("°") for label in step.view_config.yaw_slot_labels)
    assert step.view_config.yaw_slot_labels[1].toolTip() == "96.4°"
    assert all("8pt" in label.styleSheet() for label in step.view_config.yaw_slot_labels)
    assert any(view["slot"] == 1 and view["yaw"] == pytest.approx(45.0 + 360.0 / 7.0) for view in views)


def test_cubemap_yaw_numeric_fields_are_clamped_and_used(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)

    step.view_config.yaw_offset_edit.setValue(999.0)
    step.yaw_per_frame_edit.setValue(-999.0)

    assert step.view_config.yaw_offset() == 180.0
    assert step.yaw_per_frame_edit.value() == -180.0

    cmd = step._build_cubemap_cmd()

    assert cmd[cmd.index("--yaw-offset-per-frame") + 1] == "-180"
    views = json.loads(step4_views_config_path(tmp_path).read_text(encoding="utf-8"))["views"]
    assert any(view["yaw"] == -90.0 for view in views)


def test_colmap_export_method_uses_image_only_conversion(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
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
    _write_test_image(images / "frame_0001.jpg")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("colmap")

    commands = step.build_commands()
    assert [phase for phase, _cmd in commands] == ["colmap_rig_export"]
    step._finalize_bundle()

    settings_path = step4_export_settings_path(tmp_path)
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["export_method"] == "colmap"
    assert settings["conversion"]["no_image"] is False
    assert settings["conversion"]["export_colmap"] is False
    assert settings["conversion"]["yaw_offset_per_frame"] == 0.0
    assert settings["colmap_rig"]["enabled"] is True
    assert settings["colmap_rig"]["dir"] == str(tmp_path / "output" / "colmap_rig")
    assert settings["colmap_rig"]["project_dir"] == str(tmp_path / "output" / "colmap_rig")

    manifest_path = tmp_path / "output" / "colmap_rig" / "stechdrive_colmap_project.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["export_type"] == "colmap_project"
    assert manifest["project_dir"] == str(tmp_path / "output" / "colmap_rig")
    assert manifest["ready_for_import"] is False


def test_colmap_export_manifest_marks_sparse_project_ready(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    sparse_model = tmp_path / "output" / "colmap_rig" / "sparse" / "0"
    sparse_model.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse_model / name).write_bytes(b"model")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("colmap")

    step._finalize_bundle()

    manifest = json.loads(
        (tmp_path / "output" / "colmap_rig" / "stechdrive_colmap_project.json").read_text(encoding="utf-8")
    )
    assert manifest["ready_for_import"] is True
    assert manifest["sparse_model_dir"] == "sparse/0"


def test_colmap_export_method_displays_colmap_project_folder(tmp_path: Path) -> None:
    _app()
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step.output_path_label.full_text() == str(tmp_path / "output")

    step._set_export_method("colmap")

    assert step.output_path_label.full_text() == str(tmp_path / "output" / "colmap_rig")


def test_colmap_export_can_queue_colmap_sfm_with_custom_executable(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
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
    assert commands[1][1][commands[1][1].index("--ImageReader.camera_params") + 1] == "16,16,15.5,15.5"
    assert commands[2][1][1] == "rig_configurator"
    assert commands[3][1][1] == "sequential_matcher"
    assert commands[4][1][1] == "global_mapper"


def test_colmap_export_can_queue_colmap_global_mapper(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
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


def test_spheresfm_method_can_queue_3dgut_export_without_projection_views(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    _write_test_image(masks / "frame_0001.png")
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("spheresfm")
    step._set_combo_data(step.spheresfm_output_shape_combo, "equirect_3dgut")
    step.spheresfm_exec_browse.set_text(str(fake_colmap))

    assert step.output_path_label.full_text() == str(tmp_path)
    assert not step.export_targets_row.isEnabled()
    assert not step.view_config.settings_widget.isEnabled()

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == [
        "spheresfm_preflight",
        "spheresfm_prepare",
        "spheresfm_database",
        "spheresfm_feature",
        "spheresfm_match",
        "spheresfm_mapper",
        "spheresfm_transforms",
    ]
    assert commands[0][1][commands[0][1].index("--colmap") + 1] == str(fake_colmap)
    assert commands[3][1][commands[3][1].index("--ImageReader.camera_model") + 1] == "SPHERE"
    assert commands[3][1][commands[3][1].index("--ImageReader.camera_params") + 1] == "1,32,16"
    assert commands[3][1][commands[3][1].index("--ImageReader.mask_path") + 1] == str(
        tmp_path / "output" / "spheresfm" / "masks_colmap"
    )
    assert commands[4][1][commands[4][1].index("--SequentialMatching.overlap") + 1] == "10"
    assert commands[5][1][commands[5][1].index("--Mapper.sphere_camera") + 1] == "1"
    assert commands[5][1][commands[5][1].index("--Mapper.multiple_models") + 1] == "0"
    assert commands[5][1][commands[5][1].index("--Mapper.ba_global_max_num_iterations") + 1] == "33"
    assert commands[6][1][3] == str(tmp_path / "output" / "spheresfm" / "sparse")
    assert commands[6][1][4] == str(tmp_path)
    assert commands[6][1][commands[6][1].index("--image-path-mode") + 1] == "relative-to-output"


def test_spheresfm_method_can_queue_projected_cubemap_export(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    _write_test_image(masks / "frame_0001.png")
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("spheresfm")
    step.spheresfm_exec_browse.set_text(str(fake_colmap))

    assert step.output_path_label.full_text() == str(tmp_path / "output")
    assert step.export_targets_row.isEnabled()
    assert step.view_config.settings_widget.isEnabled()

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == [
        "spheresfm_preflight",
        "spheresfm_prepare",
        "spheresfm_database",
        "spheresfm_feature",
        "spheresfm_match",
        "spheresfm_mapper",
        "spheresfm_transforms",
        "spheresfm_cubemap",
    ]
    transform_cmd = commands[6][1]
    cubemap_cmd = commands[7][1]
    assert transform_cmd[4] == str(tmp_path / "output" / "spheresfm" / "equirect")
    assert transform_cmd[transform_cmd.index("--image-path-mode") + 1] == "relative"
    assert cubemap_cmd[3] == str(tmp_path / "output" / "spheresfm" / "equirect")
    assert cubemap_cmd[4] == str(tmp_path / "output")
    assert cubemap_cmd[cubemap_cmd.index("--views-json") + 1] == str(step4_views_config_path(tmp_path))
    assert cubemap_cmd[cubemap_cmd.index("--image-dir") + 1] == str(images)
    assert cubemap_cmd[cubemap_cmd.index("--mask_dir") + 1] == str(masks)
    assert "--no_transform" in cubemap_cmd


def test_spheresfm_method_can_queue_sfm_only(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    _write_test_image(masks / "frame_0001.png")
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("spheresfm")
    step._set_combo_data(step.spheresfm_run_scope_combo, "sfm_only")
    step.spheresfm_exec_browse.set_text(str(fake_colmap))

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == [
        "spheresfm_preflight",
        "spheresfm_prepare",
        "spheresfm_database",
        "spheresfm_feature",
        "spheresfm_match",
        "spheresfm_mapper",
    ]
    assert not step.settings_tabs.isTabEnabled(step.spheresfm_convert_tab_index)


def test_spheresfm_convert_only_requires_existing_sparse(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("spheresfm")
    step._set_combo_data(step.spheresfm_run_scope_combo, "convert_only")

    with pytest.raises(ValueError, match="sparse"):
        step.build_commands()


def test_spheresfm_convert_only_queues_3dgut_without_colmap_binary(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    sparse_model = _write_spheresfm_sparse_stub(tmp_path)

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("spheresfm")
    step._set_combo_data(step.spheresfm_run_scope_combo, "convert_only")
    step._set_combo_data(step.spheresfm_output_shape_combo, "equirect_3dgut")

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["spheresfm_transforms"]
    assert commands[0][1][3] == str(tmp_path / "output" / "spheresfm" / "sparse")
    assert commands[0][1][4] == str(tmp_path)
    assert sparse_model.is_dir()


def test_spheresfm_open_gui_warns_when_selected_binary_has_no_gui_support(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    sparse_model = _write_spheresfm_sparse_stub(tmp_path)
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")
    warnings: list[tuple[str, str]] = []

    class FakeGuiLessQProcess:
        MergedChannels = object()
        NormalExit = object()

        def __init__(self, parent=None) -> None:
            self.parent = parent
            self.program = ""
            self.arguments: list[str] = []

        def setProgram(self, program: str) -> None:
            self.program = program

        def setArguments(self, arguments: list[str]) -> None:
            self.arguments = arguments

        def setProcessChannelMode(self, _mode) -> None:
            pass

        def start(self) -> None:
            pass

        def waitForStarted(self, _msecs: int) -> bool:
            return True

        def waitForFinished(self, _msecs: int) -> bool:
            return True

        def readAllStandardOutput(self) -> bytes:
            return (
                b"ERROR: Cannot start colmap GUI; colmap was built without GUI support "
                b"or QT dependency is missing."
            )

        def readAllStandardError(self) -> bytes:
            return b""

        def errorString(self) -> str:
            return ""

        def exitStatus(self):
            return self.NormalExit

        def exitCode(self) -> int:
            return 1

    monkeypatch.setattr(step4_cubemap, "QProcess", FakeGuiLessQProcess)
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, title, text: warnings.append((title, text)))

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step.spheresfm_exec_browse.set_text(str(fake_colmap))

    step._open_spheresfm_result()

    assert len(warnings) == 1
    assert warnings[0][0] == i18n.t("SPHERESFM_OPEN_GUI")
    assert str(fake_colmap) in warnings[0][1]
    assert str(sparse_model) in warnings[0][1]
    assert "Qt GUI" in warnings[0][1]


def test_spheresfm_3dgut_convert_only_confirms_scene_root_outputs(tmp_path: Path, monkeypatch) -> None:
    _app()
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    _write_test_image(masks / "frame_0001.png")
    sparse_model = _write_spheresfm_sparse_stub(tmp_path)
    transforms = tmp_path / "transforms.json"
    pointcloud = tmp_path / "pointcloud.ply"
    transforms.write_text("old", encoding="utf-8")
    pointcloud.write_text("old", encoding="utf-8")
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("spheresfm")
    step._set_combo_data(step.spheresfm_run_scope_combo, "convert_only")
    step._set_combo_data(step.spheresfm_output_shape_combo, "equirect_3dgut")

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["spheresfm_transforms"]
    assert not transforms.exists()
    assert not pointcloud.exists()
    assert sparse_model.is_dir()
    assert images.is_dir()
    assert masks.is_dir()


def test_spheresfm_convert_only_resets_conversion_outputs_only(tmp_path: Path, monkeypatch) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    sparse_model = _write_spheresfm_sparse_stub(tmp_path)
    database = tmp_path / "output" / "spheresfm" / "database.db"
    database.write_text("db", encoding="utf-8")
    old_equirect = tmp_path / "output" / "spheresfm" / "equirect" / "old.txt"
    old_views = step4_views_config_path(tmp_path)
    old_images = tmp_path / "output" / "images" / "old.jpg"
    old_masks = tmp_path / "output" / "masks" / "old.png"
    old_equirect.parent.mkdir(parents=True)
    old_views.parent.mkdir(parents=True)
    old_images.parent.mkdir(parents=True)
    old_masks.parent.mkdir(parents=True)
    old_equirect.write_text("old", encoding="utf-8")
    old_views.write_text("old", encoding="utf-8")
    old_images.write_text("old", encoding="utf-8")
    old_masks.write_text("old", encoding="utf-8")
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("spheresfm")
    step._set_combo_data(step.spheresfm_run_scope_combo, "convert_only")

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["spheresfm_transforms", "spheresfm_cubemap"]
    assert not old_equirect.exists()
    assert old_views.is_file()
    assert old_views.read_text(encoding="utf-8") != "old"
    assert not old_images.exists()
    assert not old_masks.exists()
    assert sparse_model.is_dir()
    assert database.is_file()


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


def test_spheresfm_user_preferences_migrate_feature_preset_to_quality(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _app()
    settings_path = tmp_path / "settings.json"
    monkeypatch.setenv("STECHDRIVE_USER_SETTINGS_PATH", str(settings_path))
    settings_path.write_text(
        json.dumps({"step4_colmap": {"spheresfm_feature_preset": "robust"}}),
        encoding="utf-8",
    )

    step = CubemapStep(Path.cwd())
    step.enable_user_preferences()

    assert step.spheresfm_quality_combo.currentData() == "quality"


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


def test_lichtfeld_3dgut_direct_mode_runs_metashape_only_and_disables_view_export(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    old_file = output / "old.txt"
    old_file.write_text("old", encoding="utf-8")
    step = _ready_step(tmp_path, metashape_inputs=True)
    _write_test_image(tmp_path / "images" / "frame_0001.jpg")
    step.on_activated()
    step.export_images_cb.setChecked(False)

    direct_idx = step.output_shape_combo.findData("equirect_3dgut")
    assert direct_idx >= 0
    step.output_shape_combo.setCurrentIndex(direct_idx)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("confirmation should not open")),
    )

    assert step._uses_direct_equirect_output()
    assert step._effective_profile() == "lichtfeld"
    assert step.axis_transform_combo.currentData() == "none"
    assert step.ms_use_ply_cb.isChecked()
    assert step.output_path_label.full_text() == str(tmp_path)
    assert not step.settings_tabs.isTabEnabled(step.view_export_tab_index)
    assert not step.export_targets_row.isEnabled()
    assert step.export_images_cb.isChecked()
    assert step.export_masks_cb.isChecked()
    assert not step.output_details_section.isEnabled()
    assert not step.export_colmap_cb.isEnabled()
    assert "元画像" in step.view_config.summary_text()

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["metashape"]
    assert "--ply" in commands[0][1]
    assert old_file.is_file()
    assert not step4_views_config_path(tmp_path).exists()


def test_lichtfeld_3dgut_direct_mode_restores_projection_export_targets(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    step.export_images_cb.setChecked(False)
    step.export_masks_cb.setChecked(True)

    direct_idx = step.output_shape_combo.findData("equirect_3dgut")
    projected_idx = step.output_shape_combo.findData("projected")
    assert direct_idx >= 0
    assert projected_idx >= 0
    step.output_shape_combo.setCurrentIndex(direct_idx)
    step.output_shape_combo.setCurrentIndex(projected_idx)

    assert step.export_images_cb.isChecked() is False
    assert step.export_masks_cb.isChecked() is True
    assert step.settings_tabs.isTabEnabled(step.view_export_tab_index)
    assert step.output_path_label.full_text() == str(tmp_path / "output")


def test_switching_profile_away_from_lichtfeld_exits_3dgut_direct_mode(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    direct_idx = step.output_shape_combo.findData("equirect_3dgut")
    postshot_idx = step.profile_combo.findData("postshot")
    assert direct_idx >= 0
    assert postshot_idx >= 0

    step.output_shape_combo.setCurrentIndex(direct_idx)
    step.profile_combo.setCurrentIndex(postshot_idx)

    assert step.output_shape_combo.currentData() == "projected"
    assert step._uses_direct_equirect_output() is False
    assert step.settings_tabs.isTabEnabled(step.view_export_tab_index)


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
    assert not step4_views_config_path(tmp_path).exists()


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
    assert step4_views_config_path(tmp_path).is_file()


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
    old_settings = step4_export_settings_path(tmp_path)
    old_settings.parent.mkdir(parents=True, exist_ok=True)
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
    assert step4_views_config_path(tmp_path).is_file()


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

    settings_path = step4_export_settings_path(tmp_path)
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
    assert settings["output_files"]["settings"] == "_stechdrive/export_settings.json"
    assert settings["view_config"]["views"]
    assert settings["views_config_path"] == "_stechdrive/views_config.json"
    assert settings["views_config_snapshot"] == json.loads(
        step4_views_config_path(tmp_path).read_text(encoding="utf-8")
    )


def test_lichtfeld_3dgut_finalize_writes_scene_dataset_settings_and_correction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    direct_idx = step.output_shape_combo.findData("equirect_3dgut")
    assert direct_idx >= 0
    step.output_shape_combo.setCurrentIndex(direct_idx)
    monkeypatch.setattr(CubemapStep, "_transform_ply_with_open3d", staticmethod(lambda _path, _matrix: False))
    _write_ascii_ply(tmp_path / "pointcloud.ply", [(1.0, 2.0, 3.0)])
    (tmp_path / "transforms.json").write_text(
        json.dumps(
            {
                "camera_model": "EQUIRECTANGULAR",
                "frames": [{"file_path": "images/frame_0001.jpg", "transform_matrix": np.eye(4).tolist()}],
            }
        ),
        encoding="utf-8",
    )

    step._finalize_bundle()

    assert not step4_export_settings_path(tmp_path / "output").exists()
    settings = json.loads(step4_export_settings_path(tmp_path).read_text(encoding="utf-8"))
    assert settings["export_method"] == "metashape"
    assert settings["output_shape"] == "equirect_3dgut"
    assert settings["output_dir"] == str(tmp_path)
    assert settings["views_config_path"] == ""
    assert settings["views_config_snapshot"] is None
    assert settings["conversion"]["no_image"] is True
    assert settings["conversion"]["write_images"] is False
    assert settings["conversion"]["write_masks"] is False
    assert settings["conversion"]["uses_source_images"] is True
    assert settings["output_files"]["pointcloud"] == "pointcloud.ply"

    data = json.loads((tmp_path / "transforms.json").read_text(encoding="utf-8"))
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
    points, _colors = read_ply_points(tmp_path / "pointcloud.ply")
    assert np.allclose(points[0], [3.0, -2.0, 1.0])


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

    settings = json.loads(step4_export_settings_path(tmp_path).read_text(encoding="utf-8"))
    assert settings["postprocess"]["lichtfeld_final_orientation_correction"] is True
