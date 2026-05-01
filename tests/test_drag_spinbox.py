import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.common.drag_spinbox import DragDoubleSpinBox, DragSpinBox


def _app():
    return QApplication.instance() or QApplication([])


def test_drag_spinbox_constructor_applies_numeric_options():
    _app()

    box = DragSpinBox(minimum=-2, maximum=10, step=2, value=4, suffix=" px", drag_pixels_per_step=4.0)

    assert box.minimum() == -2
    assert box.maximum() == 10
    assert box.singleStep() == 2
    assert box.value() == 4
    assert box.suffix() == " px"

    box.setValue(99)
    assert box.value() == 10


def test_drag_double_spinbox_constructor_applies_numeric_options():
    _app()

    box = DragDoubleSpinBox(minimum=0.0, maximum=30.0, step=0.5, decimals=1, value=7.5)

    assert box.minimum() == 0.0
    assert box.maximum() == 30.0
    assert box.singleStep() == 0.5
    assert box.decimals() == 1
    assert box.value() == 7.5

    box.setValue(-1.0)
    assert box.value() == 0.0
