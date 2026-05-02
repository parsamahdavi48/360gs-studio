import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QWidget

from gui import i18n
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

    assert _label(step, i18n.t("QUALITY_MIN_SCORE")).toolTip() == i18n.tip("QUALITY_MIN_SCORE")
    assert _label(step, i18n.t("QUALITY_MIN_IMPROVEMENT")).toolTip() == i18n.tip("QUALITY_MIN_IMPROVEMENT")
    assert _label(step, i18n.t("THIN_MOTION_THRESHOLD")).toolTip() == i18n.tip("THIN_MOTION_THRESHOLD")


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
