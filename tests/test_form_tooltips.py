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


def test_extract_numeric_labels_share_field_tooltips() -> None:
    _app()
    step = ExtractStep(Path.cwd())

    assert _label(step, f"{i18n.t('MODE_FIXED_SHORT')}:").toolTip() == i18n.tip("MODE_FIXED")
    assert _label(step, f"{i18n.t('INTERVAL_SHORT')}:").toolTip() == i18n.tip("INTERVAL")
    assert _label(step, f"{i18n.t('MIN_GAP_SHORT')}:").toolTip() == i18n.tip("MIN_GAP")
    assert _label(step, f"{i18n.t('MAX_GAP_SHORT')}:").toolTip() == i18n.tip("MAX_GAP")
    assert step.smart_fixed_cb.toolTip() == i18n.tip("FIXED_SMART")
    assert _label(step, i18n.t("QUALITY_MIN_SCORE")).toolTip() == i18n.tip("QUALITY_MIN_SCORE")
    assert _label(step, i18n.t("QUALITY_MIN_IMPROVEMENT")).toolTip() == i18n.tip("QUALITY_MIN_IMPROVEMENT")


def test_extract_compact_mode_rows_enable_matching_fields() -> None:
    _app()
    step = ExtractStep(Path.cwd())

    assert step.interval_edit.isEnabled()
    assert step.smart_fixed_cb.isChecked()
    assert step.min_gap_edit.isEnabled()
    assert step.max_gap_edit.isEnabled()

    step.smart_fixed_cb.setChecked(False)

    assert step.interval_edit.isEnabled()
    assert not step.min_gap_edit.isEnabled()
    assert not step.max_gap_edit.isEnabled()


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


def test_mask_numeric_labels_share_field_tooltips() -> None:
    _app()
    step = MaskStep(Path.cwd())

    assert _label(step, i18n.YOLO_EXPAND).toolTip() == i18n.tip("YOLO_EXPAND")
    assert _label(step, i18n.STITCH_BOUNDARY_WIDTH).toolTip() == i18n.tip("STITCH_BOUNDARY_WIDTH")
    assert _label(step, i18n.OVEREXPOSURE_THRESHOLD).toolTip() == i18n.tip("OVEREXPOSURE_THRESHOLD")


def test_cubemap_labels_share_field_tooltips() -> None:
    _app()
    step = CubemapStep(Path.cwd())

    assert _label(step, i18n.t("YAW_OFFSET_PER_FRAME")).toolTip() == i18n.t("YAW_OFFSET_PER_FRAME_HINT")
    assert _label(step, i18n.OUTPUT_SCALE + ":").toolTip() == i18n.tip("OUTPUT_SCALE")
