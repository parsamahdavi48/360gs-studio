"""Compact Step 4 SfM route selector."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QRadioButton, QWidget

from gui import i18n
from gui.steps.sfm_route_specs import (
    DEFAULT_SFM_ROUTE,
    SfmRouteSpec,
    get_sfm_route_spec,
    normalize_sfm_route,
    sfm_route_specs,
)


class SfmRouteSelector(QWidget):
    """Shows the available Step 4 routes as an exclusive radio group."""

    route_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("radioOptionRow")
        self._route = DEFAULT_SFM_ROUTE
        self.route_buttons: dict[str, QRadioButton] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.route_group = QButtonGroup(self)
        self.route_group.setExclusive(True)
        for spec in sfm_route_specs():
            btn = self._make_route_button(spec)
            btn.clicked.connect(lambda _checked=False, route_id=spec.route_id: self.set_route(route_id, emit=True))
            self.route_group.addButton(btn)
            self.route_buttons[spec.route_id] = btn
            layout.addWidget(btn)
        layout.addStretch()

        self.set_route(DEFAULT_SFM_ROUTE)

    @staticmethod
    def _make_route_button(spec: SfmRouteSpec) -> QRadioButton:
        btn = QRadioButton(i18n.t(spec.label_key))
        btn.setObjectName("optionRadio")
        btn.setToolTip(i18n.tip(spec.tooltip_key))
        return btn

    def current_route(self) -> str:
        return self._route

    def display_name(self, route_id: str | None = None) -> str:
        spec = get_sfm_route_spec(route_id or self._route)
        return i18n.t(spec.label_key)

    def set_route(self, route_id: str | None, *, emit: bool = False) -> None:
        route = normalize_sfm_route(route_id)
        changed = route != self._route
        self._route = route
        btn = self.route_buttons.get(route)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        if changed and emit:
            self.route_changed.emit(route)
