import os
import subprocess
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QWidget

from gui import i18n
from gui.common.drag_spinbox import DragDoubleSpinBox, DragSpinBox
from gui.steps.base_step import SETTINGS_PANE_MARGINS, SETTINGS_PANE_WIDTH
from gui.steps.step1_extract import ExtractStep
from gui.steps.step3_mask import MaskStep
from gui.steps.step4_cubemap import CubemapStep


def _app():
    return QApplication.instance() or QApplication([])


def _label(widget: QWidget, text: str) -> QLabel:
    for child in widget.findChildren(QLabel):
        if child.text() == text:
            return child
    raise AssertionError(f"label not found: {text}")


def test_i18n_tips_are_wrapped() -> None:
    for key in i18n._tips:
        lines = i18n.tip(key).splitlines()
        assert lines, key
        assert all(len(line) <= i18n._TOOLTIP_WRAP_WIDTH + 2 for line in lines), key
        assert all(not line.startswith(tuple(i18n._JA_FORBIDDEN_LINE_START)) for line in lines), key


def test_i18n_tips_are_wrapped_in_english() -> None:
    script = textwrap.dedent(
        """
        import os
        os.environ["STUDIO_LANG"] = "en"
        from gui import i18n

        for key in i18n._tips:
            lines = i18n.tip(key).splitlines()
            assert lines, key
            assert all(len(line) <= i18n._TOOLTIP_WRAP_WIDTH for line in lines), key
        """
    )
    env = os.environ.copy()
    env["STUDIO_LANG"] = "en"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_extract_numeric_labels_share_field_tooltips() -> None:
    _app()
    step = ExtractStep(Path.cwd())

    assert _label(step, f"{i18n.t('MODE_FIXED_SHORT')}:").toolTip() == i18n.tip("MODE_FIXED")
    assert _label(step, f"{i18n.t('INTERVAL_SHORT')}:").toolTip() == i18n.tip("INTERVAL")
    assert _label(step, f"{i18n.t('MIN_GAP_SHORT')}:").toolTip() == i18n.tip("MIN_GAP")
    assert _label(step, f"{i18n.t('MAX_GAP_SHORT')}:").toolTip() == i18n.tip("MAX_GAP")
    assert step.smart_fixed_cb.toolTip() == i18n.tip("FIXED_SMART")
    assert step.quick_extract_cb.toolTip() == i18n.tip("QUICK_EXTRACT")
    assert _label(step, i18n.t("QUALITY_MIN_SCORE")).toolTip() == i18n.tip("QUALITY_MIN_SCORE")
    assert _label(step, i18n.t("QUALITY_MIN_IMPROVEMENT")).toolTip() == i18n.tip("QUALITY_MIN_IMPROVEMENT")


def test_extract_compact_mode_rows_enable_matching_fields() -> None:
    _app()
    step = ExtractStep(Path.cwd())

    assert step.interval_edit.isEnabled()
    assert step.smart_fixed_cb.isChecked()
    assert not step.quick_extract_cb.isChecked()
    assert step.min_gap_edit.isEnabled()
    assert step.max_gap_edit.isEnabled()

    step.smart_fixed_cb.setChecked(False)

    assert step.interval_edit.isEnabled()
    assert not step.min_gap_edit.isEnabled()
    assert not step.max_gap_edit.isEnabled()

    step.smart_fixed_cb.setChecked(True)
    step.quick_extract_cb.setChecked(True)

    assert step.interval_edit.isEnabled()
    assert step.quick_extract_cb.isEnabled()
    assert not step.smart_fixed_cb.isChecked()
    assert not step.smart_fixed_cb.isEnabled()
    assert not step.min_gap_edit.isEnabled()
    assert not step.max_gap_edit.isEnabled()
    assert not step.analysis_width_edit.isEnabled()
    assert not step.quality_min_score_edit.isEnabled()
    assert not step.quality_min_improvement_edit.isEnabled()

    step.quick_extract_cb.setChecked(False)

    assert step.smart_fixed_cb.isChecked()
    assert step.smart_fixed_cb.isEnabled()
    assert step.min_gap_edit.isEnabled()
    assert step.max_gap_edit.isEnabled()
    assert step.analysis_width_edit.isEnabled()
    assert step.quality_min_score_edit.isEnabled()
    assert step.quality_min_improvement_edit.isEnabled()


def test_extract_mode_block_preserves_right_padding() -> None:
    _app()
    step = ExtractStep(Path.cwd())

    content_width = SETTINGS_PANE_WIDTH - SETTINGS_PANE_MARGINS[2]
    assert step.mode_panel.sizeHint().width() <= content_width
    assert step.fixed_interval_row.sizeHint().width() <= content_width
    assert step.smart_interval_row.sizeHint().width() <= content_width
    assert step.extract_action_row.sizeHint().width() <= content_width


def test_extract_mode_block_preserves_right_padding_in_english() -> None:
    script = textwrap.dedent(
        """
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from pathlib import Path
        from PySide6.QtWidgets import QApplication
        from gui.steps.base_step import SETTINGS_PANE_MARGINS, SETTINGS_PANE_WIDTH
        from gui.steps.step1_extract import ExtractStep

        app = QApplication.instance() or QApplication([])
        step = ExtractStep(Path.cwd())
        content_width = SETTINGS_PANE_WIDTH - SETTINGS_PANE_MARGINS[2]
        assert step.mode_panel.sizeHint().width() <= content_width
        assert step.fixed_interval_row.sizeHint().width() <= content_width
        assert step.smart_interval_row.sizeHint().width() <= content_width
        assert step.extract_action_row.sizeHint().width() <= content_width
        """
    )
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["STUDIO_LANG"] = "en"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_extract_mode_numbers_are_draggable_and_clamped() -> None:
    _app()
    step = ExtractStep(Path.cwd())

    for field in (step.interval_edit, step.min_gap_edit, step.max_gap_edit):
        assert isinstance(field, DragDoubleSpinBox)
    assert isinstance(step.jpg_quality_edit, DragSpinBox)

    assert step.interval_edit.suffix() == f" {i18n.t('SECONDS_SUFFIX')}"
    assert step.min_gap_edit.suffix() == f" {i18n.t('SECONDS_SUFFIX')}"
    assert step.max_gap_edit.suffix() == f" {i18n.t('SECONDS_SUFFIX')}"
    assert step.min_gap_edit.minimumWidth() >= step.min_gap_edit.sizeHint().width()
    assert step.max_gap_edit.minimumWidth() >= step.max_gap_edit.sizeHint().width()
    assert step.jpg_quality_edit.minimum() == 1
    assert step.jpg_quality_edit.maximum() == 31
    assert step.jpg_quality_edit.value() == 2

    step.min_gap_edit.setValue(3.0)
    assert step.max_gap_edit.value() == 3.0

    step.max_gap_edit.setValue(1.0)
    assert step.min_gap_edit.value() == 1.0


def test_extract_command_uses_drag_spinbox_values(tmp_path: Path) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")

    step = ExtractStep(Path.cwd())
    step.video_browse.line_edit.blockSignals(True)
    step.video_browse.set_text(str(video))
    step.video_browse.line_edit.blockSignals(False)
    step.set_scene_dir(str(tmp_path))

    step.interval_edit.setValue(1.25)
    step.jpg_quality_edit.setValue(4)
    fixed_cmd = step._build_extract_cmd()
    assert fixed_cmd[fixed_cmd.index("--interval-sec") + 1] == "1.25"
    assert fixed_cmd[fixed_cmd.index("--jpg-quality") + 1] == "4"
    assert "--fixed-smart" in fixed_cmd
    assert fixed_cmd[fixed_cmd.index("--thin-motion-threshold") + 1] == "0"

    step.min_gap_edit.setValue(0.5)
    step.max_gap_edit.setValue(3.0)
    smart_cmd = step._build_extract_cmd()
    assert smart_cmd[smart_cmd.index("--min-gap-sec") + 1] == "0.5"
    assert smart_cmd[smart_cmd.index("--max-gap-sec") + 1] == "3"

    step.smart_fixed_cb.setChecked(False)
    plain_cmd = step._build_extract_cmd()
    assert "--fixed-smart" not in plain_cmd
    assert "--min-gap-sec" not in plain_cmd
    assert "--max-gap-sec" not in plain_cmd
    assert plain_cmd[plain_cmd.index("--thin-motion-threshold") + 1] == "0"

    step.quick_extract_cb.setChecked(True)
    quick_cmd = step._build_extract_cmd()
    assert "--quick-extract" in quick_cmd
    assert "--fixed-smart" not in quick_cmd
    assert "--min-gap-sec" not in quick_cmd
    assert "--max-gap-sec" not in quick_cmd
    assert "--analysis-width" not in quick_cmd
    assert "--quality-min-score" not in quick_cmd
    assert "--quality-min-improvement" not in quick_cmd
    assert quick_cmd[quick_cmd.index("--thin-motion-threshold") + 1] == "0"


def test_mask_numeric_labels_share_field_tooltips() -> None:
    _app()
    step = MaskStep(Path.cwd())

    assert step.projection_label.toolTip() == i18n.tip("MASK_IMAGE_TYPE")
    assert step.projection_buttons["equirect"].toolTip() == i18n.tip("MASK_IMAGE_TYPE_EQUIRECT")
    assert step.projection_buttons["normal"].toolTip() == i18n.tip("MASK_IMAGE_TYPE_NORMAL")
    assert step.person_backend_label.toolTip() == i18n.tip("PERSON_MODEL")
    assert step.person_backend_combo.toolTip() == i18n.tip("PERSON_MODEL")
    assert step.yolo_level_label.toolTip() == i18n.tip("MASK_QUALITY")
    assert _label(step, i18n.t("YOLO_EXPAND_COMPACT")).toolTip() == i18n.tip("YOLO_EXPAND")
    assert step.yolo_bottom_settings_row.isHidden()
    assert step.stitch_boundary_width_label.toolTip() == i18n.tip("STITCH_BOUNDARY_WIDTH")
    assert step.overexp_threshold_label.toolTip() == i18n.tip("OVEREXPOSURE_THRESHOLD")
    assert step.sky_min_area_edit.toolTip() == i18n.tip("SKY_MIN_AREA")
    assert step.sky_top_connected_cb.toolTip() == i18n.tip("SKY_TOP_CONNECTED")
    assert step.mask_settings_tabs.count() == 2
    assert [step.mask_settings_tabs.tabText(i) for i in range(step.mask_settings_tabs.count())] == [
        i18n.t("MASK_TAB_YOLO"),
        i18n.t("MASK_TAB_OPTIONS"),
    ]


def test_mask_yolo_compact_row_preserves_width_in_english() -> None:
    script = textwrap.dedent(
        """
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from pathlib import Path
        from PySide6.QtWidgets import QApplication
        from gui.steps.base_step import SETTINGS_PANE_MARGINS, SETTINGS_PANE_WIDTH
        from gui.steps.step3_mask import MaskStep

        app = QApplication.instance() or QApplication([])
        step = MaskStep(Path.cwd())
        content_width = SETTINGS_PANE_WIDTH - SETTINGS_PANE_MARGINS[2]
        assert step.person_backend_combo.itemText(0) == "YOLO/SAM2.1"
        assert step.person_backend_combo.itemText(1) == "Mask2Former"
        assert step.person_backend_combo.itemText(2) == "SAM3.1"
        assert step.yolo_settings_row.sizeHint().width() <= content_width
        assert step.yolo_level_combo.itemText(0) == "Standard"
        assert step.yolo_level_combo.itemText(1) == "High"
        assert step.yolo_level_combo.itemText(2) == "Best"
        assert step.yolo_bottom_settings_row.isHidden()
        """
    )
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["STUDIO_LANG"] = "en"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_cubemap_labels_share_field_tooltips() -> None:
    _app()
    step = CubemapStep(Path.cwd())

    assert _label(step, i18n.t("EXPORT_METHOD_COMPACT")).toolTip() == i18n.tip("EXPORT_METHOD")
    assert _label(step, i18n.t("EXPORT_TARGETS")).toolTip() == i18n.tip("EXPORT_TARGETS")
    assert step.export_method_buttons["metashape"].toolTip() == i18n.tip("METHOD_METASHAPE_IMPORT")
    assert step.export_method_buttons["colmap"].toolTip() == i18n.tip("METHOD_COLMAP_EXPORT")
    assert _label(step, i18n.t("AXIS_TRANSFORM")).toolTip() == i18n.tip("AXIS_TRANSFORM")
    assert _label(step, i18n.t("YAW_OFFSET_PER_FRAME")).toolTip() == i18n.t("YAW_OFFSET_PER_FRAME_HINT")
    assert _label(step, i18n.OUTPUT_SCALE + ":").toolTip() == i18n.tip("OUTPUT_SCALE")
    assert _label(step, i18n.METASHAPE_XML).toolTip() == i18n.tip("MS_XML")
    assert _label(step, i18n.METASHAPE_PLY).toolTip() == i18n.tip("MS_PLY")
    assert step.ms_scale_label.toolTip() == i18n.tip("SCALE_FACTOR")
    assert step.ms_use_ply_cb.toolTip() == i18n.tip("MS_USE_PLY")
    assert step.export_images_cb.toolTip() == i18n.tip("EXPORT_IMAGES")
    assert step.export_masks_cb.toolTip() == i18n.tip("EXPORT_MASKS")
    assert step.view_config.grid_section.toolTip() == i18n.tip("VIEW_SELECTION_SECTION")
    assert step.view_config.grid_title_label.toolTip() == i18n.tip("VIEW_SELECTION_SECTION")
    assert step.view_config.yaw_add_btn.toolTip() == i18n.t("YAW_SLOT_ADD")
    assert step.view_config.yaw_remove_btn.toolTip() == i18n.t("YAW_SLOT_REMOVE")
    assert step.view_config.pitch_add_btn.toolTip() == i18n.t("PITCH_ROW_ADD")
    assert step.view_config.pitch_delete_buttons[0].toolTip() == i18n.t("PITCH_ROW_REMOVE")
    assert step.output_format_label.toolTip() == i18n.tip("OUTPUT_FORMAT")
    assert step.output_format_combo.toolTip() == i18n.tip("OUTPUT_FORMAT")
    assert step.output_bit_depth_label.toolTip() == i18n.tip("OUTPUT_BIT_DEPTH")
    assert step.output_bit_depth_combo.toolTip() == i18n.tip("OUTPUT_BIT_DEPTH")
    assert step.jpg_quality_label.toolTip() == i18n.tip("JPG_QUALITY")
    assert step.jpg_quality_edit.toolTip() == i18n.tip("JPG_QUALITY")
    assert isinstance(step.view_config.yaw_offset_edit, DragDoubleSpinBox)
    assert step.view_config.yaw_offset_edit.minimum() == -180.0
    assert step.view_config.yaw_offset_edit.maximum() == 180.0
    assert isinstance(step.yaw_per_frame_edit, DragDoubleSpinBox)
    assert step.yaw_per_frame_edit.minimum() == -180.0
    assert step.yaw_per_frame_edit.maximum() == 180.0


def test_cubemap_profile_option_rows_preserve_width_in_english_and_japanese() -> None:
    script = textwrap.dedent(
        """
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from pathlib import Path
        from PySide6.QtWidgets import QApplication
        from gui.steps.base_step import SETTINGS_PANE_MARGINS, SETTINGS_PANE_WIDTH
        from gui.steps.step4_cubemap import CubemapStep

        app = QApplication.instance() or QApplication([])
        step = CubemapStep(Path.cwd())
        idx = step.view_config.view_mode_combo.findData("custom_views")
        step.view_config.view_mode_combo.setCurrentIndex(idx)
        step.view_config.set_yaw_slot_count(8)
        step.view_config.set_pitch_row_count(5)
        content_width = SETTINGS_PANE_WIDTH - SETTINGS_PANE_MARGINS[2]
        grid_available_width = content_width - 12
        assert step.export_method_row.sizeHint().width() <= content_width
        assert step.metashape_import_options_row.sizeHint().width() <= content_width
        assert step.view_config.angle_row.sizeHint().width() <= content_width
        assert step.view_config.grid_controls_widget.sizeHint().width() <= content_width
        assert step.view_config.grid_widget.sizeHint().width() <= grid_available_width
        assert step.yaw_per_frame_row.sizeHint().width() <= content_width
        assert step.output_details_section.sizeHint().width() <= content_width
        """
    )
    for lang in ("en", "ja"):
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["STUDIO_LANG"] = lang

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"lang={lang}\n{result.stdout}{result.stderr}"
