"""Mouse-draggable numeric spin boxes."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtWidgets import QAbstractSpinBox, QDoubleSpinBox, QSpinBox, QWidget


class _DragSpinMixin:
    def _init_drag(
        self,
        *,
        drag_pixels_per_step: float = 8.0,
        fine_multiplier: float = 0.2,
        coarse_multiplier: float = 5.0,
        hide_buttons: bool = True,
    ) -> None:
        self._drag_start_pos: QPoint | None = None
        self._drag_start_value = 0.0
        self._dragging = False
        self._pixels_per_step = float(drag_pixels_per_step)
        self._fine_multiplier = float(fine_multiplier)
        self._coarse_multiplier = float(coarse_multiplier)

        if hide_buttons:
            self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.lineEdit().installEventFilter(self)
        self.lineEdit().setCursor(Qt.SizeHorCursor)

    def setDragPixelsPerStep(self, pixels: float) -> None:
        self._pixels_per_step = max(1.0, float(pixels))

    def setDragStepMultipliers(self, *, fine: float | None = None, coarse: float | None = None) -> None:
        if fine is not None:
            self._fine_multiplier = float(fine)
        if coarse is not None:
            self._coarse_multiplier = float(coarse)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.lineEdit():
            if event.type() == QEvent.MouseButtonPress:
                return self._on_mouse_press(event)
            if event.type() == QEvent.MouseMove:
                return self._on_mouse_move(event)
            if event.type() == QEvent.MouseButtonRelease:
                return self._on_mouse_release(event)
        return super().eventFilter(watched, event)

    def _on_mouse_press(self, event) -> bool:
        if event.button() != Qt.LeftButton:
            return False
        self._drag_start_pos = event.globalPosition().toPoint()
        self._drag_start_value = self.value()
        self._dragging = False
        self.setFocus(Qt.MouseFocusReason)
        event.accept()
        return True

    def _on_mouse_move(self, event) -> bool:
        if self._drag_start_pos is None or not (event.buttons() & Qt.LeftButton):
            return False

        dx = event.globalPosition().toPoint().x() - self._drag_start_pos.x()
        if abs(dx) < 2 and not self._dragging:
            return True

        self._dragging = True
        step = self.singleStep()
        if self._pixels_per_step <= 0:
            self._pixels_per_step = 1.0
        if event.modifiers() & Qt.ShiftModifier:
            step *= self._fine_multiplier
        elif event.modifiers() & Qt.ControlModifier:
            step *= self._coarse_multiplier

        self._set_dragged_value(self._drag_start_value + (dx / self._pixels_per_step) * step)
        event.accept()
        return True

    def _on_mouse_release(self, event) -> bool:
        if event.button() != Qt.LeftButton:
            return False
        if not self._dragging:
            self.lineEdit().selectAll()
        self._drag_start_pos = None
        self._dragging = False
        event.accept()
        return True

    def _set_dragged_value(self, value: float) -> None:
        self.setValue(value)


class DragSpinBox(_DragSpinMixin, QSpinBox):
    """Integer number field that can be adjusted by horizontal mouse drag."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        minimum: int | None = None,
        maximum: int | None = None,
        step: int | None = None,
        value: int | None = None,
        suffix: str = "",
        drag_pixels_per_step: float = 8.0,
        fine_multiplier: float = 0.2,
        coarse_multiplier: float = 5.0,
        hide_buttons: bool = True,
    ) -> None:
        super().__init__(parent)
        if minimum is not None or maximum is not None:
            self.setRange(
                minimum if minimum is not None else self.minimum(),
                maximum if maximum is not None else self.maximum(),
            )
        if step is not None:
            self.setSingleStep(step)
        if suffix:
            self.setSuffix(suffix)
        if value is not None:
            self.setValue(value)
        self._init_drag(
            drag_pixels_per_step=drag_pixels_per_step,
            fine_multiplier=fine_multiplier,
            coarse_multiplier=coarse_multiplier,
            hide_buttons=hide_buttons,
        )

    def _set_dragged_value(self, value: float) -> None:
        self.setValue(int(round(value)))


class DragDoubleSpinBox(_DragSpinMixin, QDoubleSpinBox):
    """Floating-point number field that can be adjusted by horizontal mouse drag."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        step: float | None = None,
        decimals: int | None = None,
        value: float | None = None,
        suffix: str = "",
        drag_pixels_per_step: float = 8.0,
        fine_multiplier: float = 0.2,
        coarse_multiplier: float = 5.0,
        hide_buttons: bool = True,
    ) -> None:
        super().__init__(parent)
        if minimum is not None or maximum is not None:
            self.setRange(
                minimum if minimum is not None else self.minimum(),
                maximum if maximum is not None else self.maximum(),
            )
        if decimals is not None:
            self.setDecimals(decimals)
        if step is not None:
            self.setSingleStep(step)
        if suffix:
            self.setSuffix(suffix)
        if value is not None:
            self.setValue(value)
        self._init_drag(
            drag_pixels_per_step=drag_pixels_per_step,
            fine_multiplier=fine_multiplier,
            coarse_multiplier=coarse_multiplier,
            hide_buttons=hide_buttons,
        )
