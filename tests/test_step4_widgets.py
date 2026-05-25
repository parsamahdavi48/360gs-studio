from __future__ import annotations

import os

from PySide6.QtWidgets import QApplication

from gui.steps.step4_widgets import make_output_image_controls


def _app() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def test_output_image_controls_expose_stable_export_contract() -> None:
    app = _app()
    controls = make_output_image_controls()

    assert [controls.output_format_combo.itemData(index) for index in range(controls.output_format_combo.count())] == [
        "auto",
        "jpg",
        "png",
        "tiff",
        "webp",
    ]
    assert [controls.output_bit_depth_combo.itemData(index) for index in range(controls.output_bit_depth_combo.count())] == [
        "8",
        "source",
    ]
    assert controls.jpg_quality_edit.text() == "95"
    assert controls.invert_masks_cb.isChecked() is False
    assert controls.output_format_combo.toolTip()
    assert controls.output_bit_depth_combo.toolTip()
    assert controls.invert_masks_cb.toolTip()
    assert controls.jpg_quality_edit.toolTip()

    controls.widget.deleteLater()
    app.processEvents()
