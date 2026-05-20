import os
import subprocess
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QRadioButton, QToolButton, QWidget

from gui import i18n, theme
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


def test_offscreen_theme_loads_windows_japanese_fonts() -> None:
    if not Path("C:/Windows/Fonts/meiryo.ttc").exists():
        return

    script = textwrap.dedent(
        """
        import os

        os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from PySide6.QtGui import QFontDatabase
        from PySide6.QtWidgets import QApplication

        from gui.theme import apply_theme

        app = QApplication([])
        assert len(QFontDatabase.families()) == 0
        apply_theme(app)
        families = set(QFontDatabase.families())
        assert "Meiryo UI" in families
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_i18n_tips_are_wrapped() -> None:
    for key in i18n._tips:
        lines = i18n.tip(key).splitlines()
        assert lines, key
        assert all(len(line) <= i18n._TOOLTIP_WRAP_WIDTH + 2 for line in lines), key
        assert all(not line.startswith(tuple(i18n._JA_FORBIDDEN_LINE_START)) for line in lines), key
        assert all(line.strip() not in i18n._PATH_SEPARATORS for line in lines), key


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
            assert all(not line.startswith(tuple(i18n._PATH_SEPARATORS)) for line in lines), key
            assert all(line.strip() not in i18n._PATH_SEPARATORS for line in lines), key
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


def test_disabled_checked_controls_keep_selected_indicator_style() -> None:
    disabled_rule = "QCheckBox::indicator:disabled, QRadioButton::indicator:disabled"
    checked_disabled_rule = "QCheckBox::indicator:checked:disabled, QRadioButton::indicator:checked:disabled"

    assert disabled_rule in theme.QSS
    assert checked_disabled_rule in theme.QSS
    assert theme.QSS.index(checked_disabled_rule) > theme.QSS.index(disabled_rule)
    rule_body = theme.QSS.split(checked_disabled_rule, 1)[1].split("}", 1)[0]
    assert theme.ACCENT_PRESSED in rule_body


def test_lfs_mask_mode_tooltip_guides_app_generated_masks() -> None:
    ja_tip = i18n._TIPS_JA["LFS_MASK_MODE"]
    en_tip = i18n._TIPS_EN["LFS_MASK_MODE"]

    assert "通常はNone" not in ja_tip
    assert "標準のマスク値は白=使用、黒=背景/除外対象" in ja_tip
    assert "Ignore" in ja_tip
    assert "色の学習から除外" in ja_tip
    assert "アルファ値に生成結果の透明度を合わせる" in ja_tip
    assert "Standard mask values are white=used and black=background/exclusion target" in en_tip
    assert "exclude black from color training" in en_tip
    assert "match rendered opacity to alpha values" in en_tip


def test_step4_tabs_fit_fixed_settings_pane_without_scroll_buttons() -> None:
    script = textwrap.dedent(
        """
        import os
        from pathlib import Path

        os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from PySide6.QtWidgets import QApplication

        from gui.theme import apply_theme
        from gui.steps.base_step import SETTINGS_PANE_WIDTH
        from gui.steps.step4_cubemap import CubemapStep

        app = QApplication([])
        apply_theme(app)
        step = CubemapStep(Path.cwd())
        step.resize(1280, 920)
        step.show()
        app.processEvents()

        tab_bar = step.settings_tabs.tabBar()
        tab_widths = [tab_bar.tabRect(i).width() for i in range(tab_bar.count())]
        assert not tab_bar.usesScrollButtons()
        assert sum(tab_widths) <= SETTINGS_PANE_WIDTH, (tab_widths, SETTINGS_PANE_WIDTH)
        """
    )

    for lang in ("ja", "en"):
        env = os.environ.copy()
        env["STUDIO_LANG"] = lang
        env["QT_QPA_PLATFORM"] = "offscreen"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_step4_route_buttons_stay_inside_fixed_settings_pane() -> None:
    script = textwrap.dedent(
        """
        import os
        from pathlib import Path

        os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication

        from gui.theme import apply_theme
        from gui.steps.base_step import SETTINGS_PANE_MARGINS, SETTINGS_PANE_WIDTH
        from gui.steps.step4_cubemap import CubemapStep

        app = QApplication([])
        apply_theme(app)
        step = CubemapStep(Path.cwd())
        step.resize(1280, 920)
        step.show()
        app.processEvents()

        content_width = SETTINGS_PANE_WIDTH - SETTINGS_PANE_MARGINS[2]
        row_widths = []
        for method in ("spheresfm", "colmap", "metashape", "spheresfm"):
            step.settings_tabs.setCurrentIndex(step.input_tab_index)
            QTest.mouseClick(step.export_method_buttons[method], Qt.LeftButton)
            app.processEvents()
            settings_pane = step.export_method_row.parentWidget()
            while settings_pane is not None and settings_pane.width() != SETTINGS_PANE_WIDTH:
                settings_pane = settings_pane.parentWidget()
            row_widths.append(step.export_method_row.width())
            assert settings_pane is not None
            assert step.export_method_row.width() <= content_width
            row_pos = step.export_method_row.mapTo(settings_pane, QPoint(0, 0))
            assert row_pos.x() + step.export_method_row.width() < SETTINGS_PANE_WIDTH

        assert len(set(row_widths)) == 1
        """
    )

    for lang in ("ja", "en"):
        env = os.environ.copy()
        env["STUDIO_LANG"] = lang
        env["QT_QPA_PLATFORM"] = "offscreen"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"lang={lang}\n{result.stdout}{result.stderr}"


def test_step4_route_and_training_selectors_use_radio_buttons() -> None:
    _app()
    step = CubemapStep(Path.cwd())

    layout = step.export_method_row.layout()
    margins = layout.contentsMargins()
    assert step.export_method_row.objectName() == "radioOptionRow"
    assert layout.spacing() == 10
    assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == (0, 0, 0, 0)

    assert all(isinstance(button, QRadioButton) for button in step.export_method_buttons.values())
    assert all(button.objectName() == "optionRadio" for button in step.export_method_buttons.values())
    assert all(
        isinstance(button, QRadioButton) for button in step.training_backend_selector.primary_backend_buttons.values()
    )
    assert all(
        button.objectName() == "optionRadio"
        for button in step.training_backend_selector.primary_backend_buttons.values()
    )
    assert isinstance(step.training_backend_other_button, QRadioButton)
    assert step.training_backend_other_button.objectName() == "optionRadio"
    assert isinstance(step.training_backend_other_menu_button, QToolButton)
    assert step.training_backend_other_menu_button.objectName() == "optionMenuArrow"
    assert set(step.training_backend_buttons) == {"lichtfeld", "postshot"}
    assert set(step.training_backend_selector.primary_backend_buttons) == {"lichtfeld", "postshot"}
    assert set(step.training_backend_selector.other_backend_actions) == {"custom"}
    assert step.training_backend_buttons["lichtfeld"].isChecked()
    assert step.training_backend_other_button.text() == i18n.t("TRAINING_BACKEND_OTHER")
    assert not hasattr(step, "training_backend_other_row")
    assert not step.training_backend_other_button.isChecked()

    step._set_training_backend("custom")
    assert step.training_backend_other_button.isChecked()
    assert step.training_backend_other_button.text() == i18n.t("TRAINING_BACKEND_CUSTOM_SHORT")
    assert step.training_backend_selector.other_backend_actions["custom"].isChecked()

    step._set_training_backend("lichtfeld")
    step.training_backend_selector.other_backend_actions["custom"].trigger()
    assert step._training_backend() == "custom"
    assert step.training_backend_other_button.isChecked()


def test_step4_scene_preview_launch_uses_icon_button() -> None:
    _app()
    step = CubemapStep(Path.cwd())

    assert isinstance(step.scene_preview_btn, QToolButton)
    assert step.scene_preview_btn.objectName() == "iconToolButton"
    assert step.scene_preview_btn.text() == ""
    assert step.scene_preview_btn.accessibleName() == i18n.t("SCENE_PREVIEW_OPEN")
    assert not step.scene_preview_btn.icon().isNull()


def test_step5_japanese_training_copy_uses_learning_step_wording() -> None:
    script = textwrap.dedent(
        """
        import os

        os.environ["STUDIO_LANG"] = "ja"

        from gui import i18n

        visible_keys = [
            "STEP5_TITLE",
            "STEP5_NAV",
            "LAUNCH",
            "PHASE_TRAINING_LICHTFELD",
            "PHASE_TRAINING_POSTSHOT",
            "PHASE_TRAINING_CUSTOM",
            "TRAINING_EXEC_NOT_FOUND",
            "TRAINING_REQUIRES_DATASET_OUTPUT",
            "TRAINING_OUTPUT",
        ]
        tip_keys = [
            "LAUNCH_TRAINING",
            "TRAINING_BACKEND_LICHTFELD",
            "TRAINING_BACKEND_POSTSHOT",
            "TRAINING_EXECUTABLE",
            "TRAINING_DATASET",
            "TRAINING_OUTPUT",
            "TRAINING_HEADLESS",
            "LFS_STRATEGY",
            "LFS_ITERATIONS",
            "LFS_OUTPUT_PLY_NAME",
            "LFS_STEPS_SCALER",
            "POSTSHOT_KSTEPS",
        ]

        assert i18n.t("STEP4_TITLE") == "4. SfM"
        assert i18n.t("STEP5_TITLE") == "5. データセット"
        assert i18n.t("STEP6_TITLE") == "6. 学習"
        assert i18n.t("LAUNCH") == "起動"
        assert i18n.t("TRAINING_OUTPUT") == "出力先"
        assert all("書き出し後" not in i18n.t(key) for key in visible_keys)
        assert all("書き出し後" not in i18n.tip(key) for key in tip_keys)
        """
    )

    env = os.environ.copy()
    env["STUDIO_LANG"] = "ja"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_step4_scrolls_tab_content_not_whole_settings_pane() -> None:
    script = textwrap.dedent(
        """
        import os

        os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication, QScrollArea

        from gui import i18n
        from gui.app import MainWindow
        from gui.theme import apply_theme

        app = QApplication([])
        apply_theme(app)
        window = MainWindow()
        window.resize(1280, 920)
        window.show()
        window._set_current_step(4)
        window.dataset_step.show_tool("metashape_dataset")
        app.processEvents()

        step = window.step4
        assert set(window.dataset_step.card_grid.buttons) == {
            "metashape_dataset",
            "spheresfm_dataset",
            "realityscan_lfs",
            "scale",
            "colmap_text_model",
        }
        assert window.step_header.text() == i18n.t("DATASET_TOOL_METASHAPE_TITLE")
        assert step._export_method() == "metashape"
        assert not step.export_method_row.isVisible()
        assert step.findChildren(QScrollArea, "settingsScroll") == []

        tab_scrolls = [step.settings_tabs.widget(index) for index in range(step.settings_tabs.count())]
        assert len(tab_scrolls) == 3
        assert all(isinstance(scroll, QScrollArea) for scroll in tab_scrolls)
        assert all(scroll.objectName() == "step4TabScroll" for scroll in tab_scrolls)
        assert all(scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff for scroll in tab_scrolls)
        assert all(scroll.verticalScrollBarPolicy() == Qt.ScrollBarAsNeeded for scroll in tab_scrolls)

        parent = step.export_method_row.parentWidget()
        found_route_scroll = False
        while parent is not None:
            found_route_scroll = found_route_scroll or isinstance(parent, QScrollArea)
            parent = parent.parentWidget()
        assert found_route_scroll

        assert step.pipeline_stage_intent("sfm") is True
        assert step.pipeline_stage_intent("conversion") is True
        assert step.settings_tabs.currentIndex() == step.input_tab_index
        assert window.run_btn.text().strip() == i18n.t("DATASET_RUN_METASHAPE")

        window.dataset_step.show_tool("spheresfm_dataset")
        app.processEvents()
        assert window.step_header.text() == i18n.t("DATASET_TOOL_SPHERESFM_TITLE")
        assert step._export_method() == "spheresfm"
        assert step.pipeline_stage_intent("sfm") is False
        assert step.pipeline_stage_intent("conversion") is True
        assert not step.spheresfm_section.isVisible()
        assert window.run_btn.text().strip() == i18n.t("DATASET_RUN_SPHERESFM")

        step.set_pipeline_stage_intent("conversion", False)
        assert step.pipeline_stage_intent("conversion") is False
        window.dataset_step.show_tool("metashape_dataset")
        app.processEvents()
        assert step.pipeline_stage_intent("conversion") is True

        window._set_current_step(5)
        app.processEvents()
        assert window.run_btn.text().strip() == i18n.t("LAUNCH")
        assert window.step5.dataset_step is step

        step._set_training_backend("lichtfeld")
        app.processEvents()
        assert step.training_settings_scroll.verticalScrollBar().maximum() == 0
        step._set_training_backend("postshot")
        app.processEvents()
        assert step.training_settings_scroll.verticalScrollBar().maximum() == 0
        step._set_training_backend("custom")
        app.processEvents()
        assert step.training_settings_scroll.verticalScrollBar().maximum() == 0
        for row in (step.training_backend_row, step.training_run_options_row, step.training_common_fields_widget):
            parent = row.parentWidget()
            while parent is not None:
                assert not isinstance(parent, QScrollArea)
                parent = parent.parentWidget()
        parent = step.training_options_stack.parentWidget()
        found_training_options_scroll = False
        while parent is not None:
            found_training_options_scroll = found_training_options_scroll or parent is step.training_settings_scroll
            parent = parent.parentWidget()
        assert found_training_options_scroll
        """
    )

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
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

    assert _label(step, f"{i18n.t('EXTRACT_INTERVAL')}:").toolTip() == i18n.tip("INTERVAL")
    assert _label(step, f"{i18n.t('MIN_GAP_SHORT')}:").toolTip() == i18n.tip("MIN_GAP")
    assert _label(step, f"{i18n.t('MAX_GAP_SHORT')}:").toolTip() == i18n.tip("MAX_GAP")
    assert step.smart_fixed_cb.toolTip() == i18n.tip("FIXED_SMART")
    assert step.quick_extract_cb.toolTip() == i18n.tip("QUICK_EXTRACT")


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
    assert not step.pair_motion_profile_combo.isEnabled()

    step.quick_extract_cb.setChecked(False)

    assert step.smart_fixed_cb.isChecked()
    assert step.smart_fixed_cb.isEnabled()
    assert step.min_gap_edit.isEnabled()
    assert step.max_gap_edit.isEnabled()
    assert step.analysis_width_edit.isEnabled()
    assert step.pair_motion_profile_combo.isEnabled()


def test_extract_mode_block_preserves_right_padding() -> None:
    _app()
    step = ExtractStep(Path.cwd())

    content_width = SETTINGS_PANE_WIDTH - SETTINGS_PANE_MARGINS[2]
    assert step.mode_panel.sizeHint().width() <= content_width
    assert step.fixed_interval_row.sizeHint().width() <= content_width
    assert step.smart_interval_row.sizeHint().width() <= content_width
    assert step.extract_action_row.sizeHint().width() <= content_width


def test_pair_motion_profile_combo_has_tooltip() -> None:
    _app()
    step = ExtractStep(Path.cwd())

    assert _label(step, i18n.t("PAIR_MOTION_PROFILE")).toolTip() == i18n.tip("PAIR_MOTION_PROFILE")
    assert step.pair_motion_profile_combo.toolTip() == i18n.tip("PAIR_MOTION_PROFILE")
    assert [step.pair_motion_profile_combo.itemData(i) for i in range(step.pair_motion_profile_combo.count())] == [
        "walk_standard",
        "walk_close",
        "walk_wide",
        "drone_distant",
    ]


def test_pair_motion_profile_combo_applies_interval_presets() -> None:
    _app()
    step = ExtractStep(Path.cwd())

    step.pair_motion_profile_combo.setCurrentIndex(1)
    assert step.interval_edit.value() == 1.0
    assert step.min_gap_edit.value() == 0.5
    assert step.max_gap_edit.value() == 2.5

    step.pair_motion_profile_combo.setCurrentIndex(2)
    assert step.interval_edit.value() == 3.0
    assert step.min_gap_edit.value() == 1.5
    assert step.max_gap_edit.value() == 7.0

    step.pair_motion_profile_combo.setCurrentIndex(3)
    assert step.interval_edit.value() == 3.0
    assert step.min_gap_edit.value() == 1.5
    assert step.max_gap_edit.value() == 8.0


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
    assert step.interval_edit.value() == 1.5
    assert step.min_gap_edit.value() == 0.8
    assert step.max_gap_edit.value() == 4.0

    step.max_gap_edit.setValue(2.0)
    step.min_gap_edit.setValue(3.0)
    assert step.interval_edit.value() == 3.0
    assert step.max_gap_edit.value() == 3.0

    step.max_gap_edit.setValue(1.0)
    assert step.interval_edit.value() == 1.0
    assert step.min_gap_edit.value() == 1.0

    step.interval_edit.setValue(0.5)
    assert step.min_gap_edit.value() == 0.5
    assert step.max_gap_edit.value() == 1.0

    step.interval_edit.setValue(4.0)
    assert step.min_gap_edit.value() == 0.5
    assert step.max_gap_edit.value() == 4.0


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

    step.min_gap_edit.setValue(0.5)
    step.max_gap_edit.setValue(3.0)
    smart_cmd = step._build_extract_cmd()
    assert smart_cmd[smart_cmd.index("--pair-motion-profile") + 1] == "walk_standard"
    assert smart_cmd[smart_cmd.index("--min-gap-sec") + 1] == "0.5"
    assert smart_cmd[smart_cmd.index("--max-gap-sec") + 1] == "3"

    step.pair_motion_profile_combo.setCurrentIndex(1)
    close_cmd = step._build_extract_cmd()
    assert close_cmd[close_cmd.index("--pair-motion-profile") + 1] == "walk_close"

    step.smart_fixed_cb.setChecked(False)
    plain_cmd = step._build_extract_cmd()
    assert "--fixed-smart" not in plain_cmd
    assert "--min-gap-sec" not in plain_cmd
    assert "--max-gap-sec" not in plain_cmd

    step.quick_extract_cb.setChecked(True)
    quick_cmd = step._build_extract_cmd()
    assert "--quick-extract" in quick_cmd
    assert "--fixed-smart" not in quick_cmd
    assert "--min-gap-sec" not in quick_cmd
    assert "--max-gap-sec" not in quick_cmd
    assert "--analysis-width" not in quick_cmd
    assert "--quality-min-score" not in quick_cmd
    assert "--quality-min-improvement" not in quick_cmd
    assert "--pair-motion-profile" not in quick_cmd


def test_mask_numeric_labels_share_field_tooltips() -> None:
    _app()
    step = MaskStep(Path.cwd())

    assert i18n.t("MASK_IMAGE_TYPE_EQUIRECT") in step.projection_label.toolTip()
    assert not hasattr(step, "projection_buttons")
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
        step.sam_apply_mode_label.setVisible(True)
        step.sam_apply_mode_combo.setVisible(True)
        assert step.yolo_settings_row.sizeHint().width() <= content_width
        assert step.yolo_level_label.text() == "Qual."
        assert step.yolo_expand_label.text() == "Exp."
        assert step.sam_apply_mode_label.text() == "Op."
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

    assert step.export_method_label.toolTip() == i18n.tip("EXPORT_METHOD")
    assert step.export_targets_row.toolTip() == i18n.tip("EXPORT_TARGETS")
    assert all(child.text() != i18n.t("EXPORT_TARGETS") for child in step.output_tab.findChildren(QLabel))
    assert step.export_method_buttons["metashape"].toolTip() == i18n.tip("METHOD_METASHAPE_IMPORT")
    assert step.export_method_buttons["colmap"].toolTip() == i18n.tip("METHOD_COLMAP_EXPORT")
    assert step.export_method_buttons["spheresfm"].toolTip() == i18n.tip("METHOD_SPHERESFM")
    assert step.training_backend_label.toolTip() == i18n.tip("TRAINING_BACKEND_LICHTFELD")
    assert step.training_backend_buttons["lichtfeld"].toolTip() == i18n.tip("TRAINING_BACKEND_LICHTFELD")
    step._set_training_backend("postshot")
    assert step.training_backend_label.toolTip() == i18n.tip("TRAINING_BACKEND_POSTSHOT")
    assert step.training_backend_buttons["postshot"].isChecked()
    step._set_training_backend("custom")
    assert step.training_backend_label.toolTip() == i18n.tip("TRAINING_BACKEND_CUSTOM")
    assert step.training_backend_other_button.isChecked()
    assert step.training_backend_other_button.text() == i18n.t("TRAINING_BACKEND_CUSTOM_SHORT")
    assert step.training_backend_selector.other_backend_actions["custom"].isChecked()
    assert step.training_backend_other_button.toolTip() == i18n.tip("TRAINING_BACKEND_CUSTOM")
    assert step.training_backend_other_menu_button.toolTip() == i18n.tip("TRAINING_BACKEND_OTHER")
    training_widget = step.training_section
    assert _label(training_widget, i18n.t("TRAINING_EXECUTABLE")).toolTip() == i18n.tip("TRAINING_EXECUTABLE")
    assert _label(training_widget, i18n.t("TRAINING_DATASET")).toolTip() == i18n.tip("TRAINING_DATASET")
    assert _label(training_widget, i18n.t("TRAINING_OUTPUT")).toolTip() == i18n.tip("TRAINING_OUTPUT")
    assert step.training_headless_cb.toolTip() == i18n.tip("TRAINING_HEADLESS")
    assert _label(training_widget, i18n.t("LFS_STRATEGY")).toolTip() == i18n.tip("LFS_STRATEGY")
    assert _label(training_widget, i18n.t("LFS_ITERATIONS")).toolTip() == i18n.tip("LFS_ITERATIONS")
    assert _label(training_widget, i18n.t("LFS_MAX_GAUSSIANS")).toolTip() == i18n.tip("LFS_MAX_GAUSSIANS")
    assert _label(training_widget, i18n.t("LFS_OUTPUT_PLY_NAME")).toolTip() == i18n.tip("LFS_OUTPUT_PLY_NAME")
    assert _label(training_widget, i18n.t("LFS_MASK_MODE")).toolTip() == i18n.tip("LFS_MASK_MODE")
    assert _label(training_widget, i18n.t("LFS_INVERT_MASKS")).toolTip() == i18n.tip("LFS_INVERT_MASKS")
    assert step.lfs_invert_masks_cb.toolTip() == i18n.tip("LFS_INVERT_MASKS")
    for field, key in (
        (step.lfs_sparsity_cb, "LFS_SPARSITY"),
        (step.lfs_gut_cb, "LFS_GUT"),
        (step.lfs_undistort_cb, "LFS_UNDISTORT"),
        (step.lfs_mip_filter_cb, "LFS_MIP_FILTER"),
        (step.lfs_ppisp_cb, "LFS_PPISP"),
    ):
        assert field.toolTip() == i18n.tip(key)
    for field, key in (
        (step.lfs_mask_threshold_edit, "LFS_MASK_THRESHOLD"),
        (step.lfs_use_alpha_as_mask_cb, "LFS_USE_ALPHA_AS_MASK"),
        (step.lfs_mask_opacity_penalty_weight_edit, "LFS_MASK_OPACITY_PENALTY_WEIGHT"),
        (step.lfs_mask_opacity_penalty_power_edit, "LFS_MASK_OPACITY_PENALTY_POWER"),
        (step.lfs_ppisp_freeze_from_sidecar_cb, "LFS_PPISP_FREEZE_FROM_SIDECAR"),
        (step.lfs_ppisp_sidecar_browse, "LFS_PPISP_SIDECAR_PATH"),
        (step.lfs_ppisp_use_controller_cb, "LFS_PPISP_USE_CONTROLLER"),
        (step.lfs_ppisp_controller_activation_step_edit, "LFS_PPISP_CONTROLLER_ACTIVATION_STEP"),
        (step.lfs_ppisp_controller_lr_edit, "LFS_PPISP_CONTROLLER_LR"),
        (step.lfs_ppisp_freeze_gaussians_on_distill_cb, "LFS_PPISP_FREEZE_GAUSSIANS_ON_DISTILL"),
        (step.lfs_dataset_resize_factor_combo, "LFS_RESIZE_FACTOR"),
        (step.lfs_dataset_max_width_edit, "LFS_MAX_WIDTH"),
        (step.lfs_dataset_cpu_cache_cb, "LFS_CPU_CACHE"),
        (step.lfs_dataset_fs_cache_cb, "LFS_FS_CACHE"),
        (step.lfs_dataset_test_every_edit, "LFS_TEST_EVERY"),
    ):
        assert field.toolTip() == i18n.tip(key)
        assert _label(training_widget, i18n.t(key)).toolTip() == i18n.tip(key)
    for key, section in step.lfs_advanced_sections.items():
        assert section.toolTip() == i18n.tip(key)
        assert section.toggle_button.toolTip() == i18n.tip(key)
    for key, field in step.lfs_advanced_edits.items():
        tip_key = f"LFS_{key.upper()}"
        assert field.toolTip() == i18n.tip(tip_key)
        assert _label(training_widget, i18n.t(tip_key)).toolTip() == i18n.tip(tip_key)
    for key, field in step.lfs_advanced_checks.items():
        tip_key = f"LFS_{key.upper()}"
        assert field.toolTip() == i18n.tip(tip_key)
        assert _label(training_widget, i18n.t(tip_key)).toolTip() == i18n.tip(tip_key)
    assert _label(training_widget, i18n.t("POSTSHOT_PROJECT_NAME")).toolTip() == i18n.tip("POSTSHOT_PROJECT_NAME")
    assert _label(training_widget, i18n.t("CUSTOM_TRAINING_ARGS")).toolTip() == i18n.tip("CUSTOM_TRAINING_ARGS")
    assert step.colmap_repo_link.openExternalLinks()
    assert step.colmap_repo_link.toolTip() == i18n.tip("COLMAP_REPOSITORY_LINK")
    assert i18n.t("COLMAP_REPOSITORY_LINK") in step.colmap_repo_link.text()
    assert not hasattr(step, "run_colmap_cb")
    assert step.spheresfm_repo_link.openExternalLinks()
    assert step.spheresfm_repo_link.toolTip() == i18n.tip("SPHERESFM_REPOSITORY_LINK")
    assert i18n.t("SPHERESFM_REPOSITORY_LINK") in step.spheresfm_repo_link.text()
    assert not hasattr(step, "spheresfm_run_scope_combo")
    assert step.axis_transform_combo.toolTip() == i18n.tip("AXIS_TRANSFORM")
    assert step.axis_transform_combo.isHidden()
    assert step.axis_transform_label is not None
    assert step.axis_transform_label.isHidden()
    assert step.spheresfm_axis_transform_combo.toolTip() == i18n.tip("SPHERESFM_AXIS_TRANSFORM")
    assert step.spheresfm_axis_transform_combo.isHidden()
    assert step.spheresfm_axis_transform_label is not None
    assert step.spheresfm_axis_transform_label.isHidden()
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
        assert step.view_config.angle_row.indexOf(step.yaw_per_frame_edit) >= 0
        assert step.view_config.angle_row.indexOf(step.scale_combo) < 0
        assert step.output_scale_row.layout().indexOf(step.scale_combo) >= 0
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
