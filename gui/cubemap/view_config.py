"""Pitch/Yaw グリッド、ビュー設定"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui import i18n
from gui.common.collapsible_section import CollapsibleSection
from gui.common.drag_spinbox import DragDoubleSpinBox
from gui.common.icons import deselect_all_icon, select_all_icon

_MIN_YAW_SLOTS = 4
_MAX_YAW_SLOTS = 8
_DEFAULT_YAW_SLOTS = 6
_MIN_PITCH_ROWS = 1
_MAX_PITCH_ROWS = 5
_DEFAULT_PITCH_ROWS = 3
_MIN_PITCH_DEG = -90.0
_MAX_PITCH_DEG = 90.0
_WARN_ENABLED_VIEWS = 24
_BLOCK_ENABLED_VIEWS = 40

VIEW_MODE_CUSTOM = "custom_views"
VIEW_MODE_CUBE6 = "cube6"


def _normalize_angle(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _angle_token(angle: float) -> str:
    sign = "p" if angle >= 0 else "m"
    value = abs(angle)
    token = f"{value:g}".replace(".", "d")
    return f"{sign}{token}"


def _default_pitches_for_count(count: int) -> list[float]:
    count = max(_MIN_PITCH_ROWS, min(_MAX_PITCH_ROWS, int(count)))
    defaults = {
        1: [0.0],
        2: [-45.0, 45.0],
        3: [-45.0, 0.0, 45.0],
        4: [-60.0, -20.0, 20.0, 60.0],
        5: [-60.0, -30.0, 0.0, 30.0, 60.0],
    }
    return defaults[count][:]


def _pitch_key(value: float) -> str:
    return f"{value:.1f}"


def _clamp_pitch(value: float) -> float:
    return max(_MIN_PITCH_DEG, min(_MAX_PITCH_DEG, round(float(value), 1)))


class ViewConfigWidget(QWidget):
    """ビュー選択グリッドウィジェット。views_changed シグナルで変更を通知。"""

    views_changed = Signal()
    hovered_view_changed = Signal(object)
    summary_changed = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        show_settings: bool = True,
        show_summary: bool = True,
    ) -> None:
        super().__init__(parent)
        self.pitch_rows: list[dict] = []
        self.yaw_slot_labels: list[QLabel] = []
        self._output_count_text = ""
        self._show_settings = show_settings
        self._show_summary = show_summary
        self._rebuilding_grid = False
        self._normalizing_pitch = False
        self._hovered_view_name: str | None = None

        self._build_ui()
        self._apply_pitch_rows()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.settings_widget = QWidget()
        ctrl = QVBoxLayout(self.settings_widget)
        ctrl.setContentsMargins(0, 0, 0, 0)
        ctrl.setSpacing(6)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.setToolTip(i18n.tip("VIEW_MODE"))
        self.view_mode_combo.addItem(i18n.t("CUSTOM_GRID"), VIEW_MODE_CUSTOM)
        self.view_mode_combo.addItem(i18n.t("CUBE6_LABEL"), VIEW_MODE_CUBE6)
        self.view_mode_combo.setMinimumWidth(180)
        self.view_mode_combo.currentIndexChanged.connect(self._on_view_mode_changed)
        mode_label = QLabel(i18n.t("VIEW_MODE_LABEL"))
        mode_label.setToolTip(i18n.tip("VIEW_MODE"))
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.view_mode_combo, stretch=1)
        ctrl.addLayout(mode_row)

        angle_row = QHBoxLayout()
        angle_row.setSpacing(8)
        self.angle_row = angle_row
        self.yaw_offset_edit = DragDoubleSpinBox(
            minimum=-180.0,
            maximum=180.0,
            step=1.0,
            decimals=1,
            value=45.0,
            drag_pixels_per_step=6.0,
        )
        self.yaw_offset_edit.setToolTip(i18n.tip("YAW_OFFSET"))
        self.yaw_offset_edit.setFixedWidth(76)
        self.yaw_offset_edit.valueChanged.connect(self._on_params_changed)
        yaw_offset_label = QLabel(i18n.t("YAW_OFFSET_LABEL"))
        yaw_offset_label.setToolTip(i18n.tip("YAW_OFFSET"))
        angle_row.addWidget(yaw_offset_label)
        angle_row.addWidget(self.yaw_offset_edit)
        ctrl.addLayout(angle_row)

        self.extra_controls_layout = QVBoxLayout()
        self.extra_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.extra_controls_layout.setSpacing(6)
        ctrl.addLayout(self.extra_controls_layout)

        self.custom_controls_widget = QWidget()
        custom_row = QHBoxLayout(self.custom_controls_widget)
        custom_row.setContentsMargins(0, 0, 0, 0)
        custom_row.setSpacing(8)
        self.yaw_slots_combo = QComboBox()
        self.yaw_slots_combo.setToolTip(i18n.tip("YAW_SLOTS"))
        self.yaw_slots_combo.addItems([str(v) for v in range(_MIN_YAW_SLOTS, _MAX_YAW_SLOTS + 1)])
        self.yaw_slots_combo.setCurrentText(str(_DEFAULT_YAW_SLOTS))
        self.yaw_slots_combo.setFixedWidth(56)
        self.yaw_slots_combo.currentTextChanged.connect(lambda _: self._apply_pitch_rows(reset_to_defaults=False))
        self.yaw_slots_label = QLabel(i18n.t("YAW_SLOTS_LABEL"))
        self.yaw_slots_label.setToolTip(i18n.tip("YAW_SLOTS"))
        custom_row.addWidget(self.yaw_slots_label)
        custom_row.addWidget(self.yaw_slots_combo)

        self.pitch_rows_combo = QComboBox()
        self.pitch_rows_combo.setToolTip(i18n.tip("PITCH_ROWS"))
        self.pitch_rows_combo.addItems([str(v) for v in range(_MIN_PITCH_ROWS, _MAX_PITCH_ROWS + 1)])
        self.pitch_rows_combo.setCurrentText(str(_DEFAULT_PITCH_ROWS))
        self.pitch_rows_combo.setFixedWidth(56)
        self.pitch_rows_combo.currentTextChanged.connect(lambda _: self._apply_pitch_rows(reset_to_defaults=True))
        self.pitch_label = QLabel(i18n.t("PITCH_ROWS_LABEL"))
        self.pitch_label.setToolTip(i18n.tip("PITCH_ROWS"))
        custom_row.addWidget(self.pitch_label)
        custom_row.addWidget(self.pitch_rows_combo)
        custom_row.addStretch()
        ctrl.addWidget(self.custom_controls_widget)

        # ビュー選択グリッド
        self.grid_section = CollapsibleSection(i18n.t("VIEW_SELECTION_SECTION"), expanded=False)
        self.grid_section.setToolTip(i18n.tip("VIEW_SELECTION_SECTION"))
        self.grid_section.toggle_button.setToolTip(i18n.tip("VIEW_SELECTION_SECTION"))

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setHorizontalSpacing(3)
        self.grid_layout.setVerticalSpacing(4)
        self.grid_section.content_layout.addWidget(self.grid_widget)
        ctrl.addWidget(self.grid_section)

        summary_row = QHBoxLayout()
        self.selected_label = QLabel(f"{i18n.t('SELECTED_VIEWS')}: 0")
        self.selected_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        summary_row.addStretch()
        summary_row.addWidget(self.selected_label)
        if self._show_summary:
            ctrl.addLayout(summary_row)

        if self._show_settings:
            layout.addWidget(self.settings_widget)

        cube6_index = self.view_mode_combo.findData(VIEW_MODE_CUBE6)
        if cube6_index >= 0:
            self.view_mode_combo.setCurrentIndex(cube6_index)
        self._on_view_mode_changed(self.view_mode_combo.currentIndex())

    # -- public API --

    def view_mode(self) -> str:
        return self.view_mode_combo.currentData() or VIEW_MODE_CUSTOM

    def yaw_offset(self) -> float:
        return float(self.yaw_offset_edit.value())

    def yaw_slot_count(self) -> int:
        try:
            return int(self.yaw_slots_combo.currentText())
        except Exception:
            return _DEFAULT_YAW_SLOTS

    def pitch_row_count(self) -> int:
        try:
            return int(self.pitch_rows_combo.currentText())
        except Exception:
            return _DEFAULT_PITCH_ROWS

    def pitch_values(self) -> list[float]:
        if self.pitch_rows:
            return [float(row["pitch"]) for row in self.pitch_rows]
        return _default_pitches_for_count(self.pitch_row_count())

    def pitch_rows_text(self) -> str:
        return ",".join(f"{pitch:g}" for pitch in self.pitch_values())

    def hovered_view_name(self) -> str | None:
        return self._hovered_view_name

    def collect_views(self, include_disabled: bool = False) -> list[dict]:
        offset = self.yaw_offset()
        if self.view_mode() == VIEW_MODE_CUBE6:
            views = self._cube6_views(offset)
            return views if include_disabled else [v for v in views if v["enabled"]]

        step = 360.0 / float(self.yaw_slot_count())
        views = []
        for row in self.pitch_rows:
            pitch = float(row["pitch"])
            for slot, cb in enumerate(row["checks"]):
                enabled = cb.isChecked()
                if not include_disabled and not enabled:
                    continue
                name = f"pit{_angle_token(pitch)}_s{slot}"
                views.append({
                    "name": name,
                    "yaw": float(offset + slot * step),
                    "pitch": pitch,
                    "enabled": enabled,
                    "slot": slot,
                    "label": f"p{pitch:g}/s{slot}",
                    "highlighted": name == self._hovered_view_name,
                })
        return views

    def set_output_count_text(self, text: str) -> None:
        self._output_count_text = text
        self._update_selected_label()

    def summary_text(self) -> str:
        return self.selected_label.text()

    # -- internal --

    def _on_view_mode_changed(self, _index: int) -> None:
        is_custom = self.view_mode() == VIEW_MODE_CUSTOM
        self.custom_controls_widget.setVisible(is_custom)
        self.grid_widget.setVisible(is_custom)
        self.grid_section.setVisible(is_custom)
        self._on_selection_changed()

    def _on_params_changed(self, *_args) -> None:
        self._update_yaw_labels()
        self._update_selected_label()
        self.views_changed.emit()

    def _on_selection_changed(self, *_args) -> None:
        if self._rebuilding_grid:
            return
        self._update_selected_label()
        self.views_changed.emit()

    def _parse_pitches(self) -> list[float]:
        return self.pitch_values()

    def _clear_grid(self) -> None:
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _apply_pitch_rows(self, *, reset_to_defaults: bool = False) -> None:
        old = {}
        for row in self.pitch_rows:
            key = _pitch_key(float(row["pitch"]))
            old[key] = [cb.isChecked() for cb in row["checks"]]

        count = self.pitch_row_count()
        if reset_to_defaults or not self.pitch_rows:
            pitches = _default_pitches_for_count(count)
        else:
            pitches = self._resize_pitch_values(self.pitch_values(), count)

        slots = self.yaw_slot_count()
        self._rebuilding_grid = True
        self.pitch_rows = []
        self.yaw_slot_labels = []
        self._clear_grid()

        self.grid_layout.setColumnMinimumWidth(0, 66)
        self.grid_layout.addWidget(self._build_grid_tools(), 0, 0)
        for s in range(slots):
            lab = QLabel(f"S{s}")
            lab.setAlignment(Qt.AlignCenter)
            lab.setFixedWidth(42)
            lab.setWordWrap(False)
            lab.setStyleSheet("font-size: 8pt;")
            self.yaw_slot_labels.append(lab)
            self.grid_layout.addWidget(lab, 0, s + 1)

        for ri, pitch in enumerate(pitches):
            row_index = ri
            grid_row = ri + 1
            pitch_edit = DragDoubleSpinBox(
                minimum=_MIN_PITCH_DEG,
                maximum=_MAX_PITCH_DEG,
                step=1.0,
                decimals=1,
                value=pitch,
                drag_pixels_per_step=6.0,
            )
            pitch_edit.setToolTip(i18n.tip("PITCH_ROWS"))
            pitch_edit.setFixedWidth(66)
            pitch_edit.valueChanged.connect(lambda value, idx=row_index: self._on_pitch_value_changed(idx, value))
            self.grid_layout.addWidget(pitch_edit, grid_row, 0)
            checks = []
            key = _pitch_key(pitch)
            restored = old.get(key)
            for s in range(slots):
                cb = _HoverCheckBox()
                cb.setFixedSize(QSize(22, 22))
                if restored and s < len(restored):
                    cb.setChecked(restored[s])
                else:
                    cb.setChecked(True)
                cb.toggled.connect(self._on_selection_changed)
                cb.hover_changed.connect(lambda hovering, idx=row_index, slot=s: self._on_view_hover(idx, slot, hovering))
                self.grid_layout.addWidget(cb, grid_row, s + 1, alignment=Qt.AlignCenter)
                checks.append(cb)
            self.pitch_rows.append({"pitch": pitch, "pitch_edit": pitch_edit, "checks": checks})

        self._rebuilding_grid = False
        self._update_yaw_labels()
        self._update_selected_label()
        self.views_changed.emit()

    def _resize_pitch_values(self, current: list[float], count: int) -> list[float]:
        values = [_clamp_pitch(p) for p in current[:count]]
        if len(values) < count:
            used = {_pitch_key(p) for p in values}
            for candidate in _default_pitches_for_count(count):
                key = _pitch_key(candidate)
                if key in used:
                    continue
                values.append(candidate)
                used.add(key)
                if len(values) >= count:
                    break
        while len(values) < count:
            values.append(self._nearest_unique_pitch(0.0, values))
        return values

    def _nearest_unique_pitch(self, value: float, existing: list[float]) -> float:
        base = _clamp_pitch(value)
        used = {_pitch_key(v) for v in existing}
        if _pitch_key(base) not in used:
            return base
        for delta in range(1, int(_MAX_PITCH_DEG - _MIN_PITCH_DEG) + 1):
            for candidate in (base + delta, base - delta):
                candidate = _clamp_pitch(candidate)
                if _pitch_key(candidate) not in used:
                    return candidate
        return base

    def _on_pitch_value_changed(self, row_index: int, value: float) -> None:
        if self._rebuilding_grid or self._normalizing_pitch or row_index >= len(self.pitch_rows):
            return
        existing = [
            float(row["pitch"])
            for idx, row in enumerate(self.pitch_rows)
            if idx != row_index
        ]
        adjusted = self._nearest_unique_pitch(value, existing)
        row = self.pitch_rows[row_index]
        if _pitch_key(adjusted) != _pitch_key(value):
            self._normalizing_pitch = True
            row["pitch_edit"].setValue(adjusted)
            self._normalizing_pitch = False
        row["pitch"] = adjusted
        self._update_selected_label()
        self.views_changed.emit()

    def _view_name_for(self, row_index: int, slot: int) -> str | None:
        if row_index < 0 or row_index >= len(self.pitch_rows):
            return None
        pitch = float(self.pitch_rows[row_index]["pitch"])
        return f"pit{_angle_token(pitch)}_s{slot}"

    def _on_view_hover(self, row_index: int, slot: int, hovering: bool) -> None:
        name = self._view_name_for(row_index, slot)
        if hovering:
            if name != self._hovered_view_name:
                self._hovered_view_name = name
                self.hovered_view_changed.emit(name)
            return
        if name is None or self._hovered_view_name == name:
            self._hovered_view_name = None
            self.hovered_view_changed.emit(None)

    def _build_grid_tools(self) -> QWidget:
        tools = QWidget()
        layout = QHBoxLayout(tools)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.all_on_btn = QToolButton()
        self.all_on_btn.setObjectName("iconToolButton")
        self.all_on_btn.setIcon(select_all_icon())
        self.all_on_btn.setToolTip(i18n.t("SELECT_ALL"))
        self.all_on_btn.setAccessibleName(i18n.t("SELECT_ALL"))
        self.all_on_btn.setFixedSize(28, 28)
        self.all_on_btn.clicked.connect(self._all_on)
        layout.addWidget(self.all_on_btn)

        self.all_off_btn = QToolButton()
        self.all_off_btn.setObjectName("iconToolButton")
        self.all_off_btn.setIcon(deselect_all_icon())
        self.all_off_btn.setToolTip(i18n.t("DESELECT_ALL"))
        self.all_off_btn.setAccessibleName(i18n.t("DESELECT_ALL"))
        self.all_off_btn.setFixedSize(28, 28)
        self.all_off_btn.clicked.connect(self._all_off)
        layout.addWidget(self.all_off_btn)
        return tools

    def _update_yaw_labels(self) -> None:
        offset = self.yaw_offset()
        slots = self.yaw_slot_count()
        step = 360.0 / float(slots)
        for i, lab in enumerate(self.yaw_slot_labels):
            yaw = _normalize_angle(offset + i * step)
            lab.setText(f"S{i}\n{yaw:g}")
            lab.setToolTip(f"{yaw:g}°")

    def _update_selected_label(self) -> None:
        try:
            views = self.collect_views(include_disabled=True)
        except Exception:
            text = f"{i18n.t('SELECTED_VIEWS')}: -"
            self.selected_label.setText(text)
            self.summary_changed.emit(text)
            return
        sel = sum(1 for v in views if v["enabled"])
        warn = ""
        if sel > _BLOCK_ENABLED_VIEWS:
            warn = f" [{i18n.t('EXCEED')}]"
        elif sel > _WARN_ENABLED_VIEWS:
            warn = f" [{i18n.t('HIGH')}]"
        text = f"{i18n.t('SELECTED_VIEWS')}: {sel} / {len(views)}{warn}"
        if self._output_count_text:
            text = f"{text}   {self._output_count_text}"
        self.selected_label.setText(text)
        self.summary_changed.emit(text)

    def _all_on(self) -> None:
        self._set_all_checked(True)

    def _all_off(self) -> None:
        self._set_all_checked(False)

    def _set_all_checked(self, checked: bool) -> None:
        changed = False
        self.grid_widget.setUpdatesEnabled(False)
        try:
            for row in self.pitch_rows:
                for cb in row["checks"]:
                    if cb.isChecked() == checked:
                        continue
                    was_blocked = cb.blockSignals(True)
                    try:
                        cb.setChecked(checked)
                    finally:
                        cb.blockSignals(was_blocked)
                    changed = True
        finally:
            self.grid_widget.setUpdatesEnabled(True)

        if changed:
            self._on_selection_changed()

    def _cube6_views(self, yaw_offset: float) -> list[dict]:
        return [
            {"name": "px", "yaw": 90.0 - yaw_offset, "pitch": 0.0, "enabled": True, "slot": 0, "label": "px"},
            {"name": "nx", "yaw": -90.0 - yaw_offset, "pitch": 0.0, "enabled": True, "slot": 1, "label": "nx"},
            {"name": "pz", "yaw": 0.0 - yaw_offset, "pitch": 0.0, "enabled": True, "slot": 2, "label": "pz"},
            {"name": "nz", "yaw": 180.0 - yaw_offset, "pitch": 0.0, "enabled": True, "slot": 3, "label": "nz"},
            {"name": "top", "yaw": 0.0 - yaw_offset, "pitch": 90.0, "enabled": True, "slot": 4, "label": "top"},
            {"name": "bottom", "yaw": 0.0 - yaw_offset, "pitch": -90.0, "enabled": True, "slot": 5, "label": "bottom"},
        ]


class _HoverCheckBox(QCheckBox):
    hover_changed = Signal(bool)

    def enterEvent(self, event) -> None:  # type: ignore[override]
        self.hover_changed.emit(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self.hover_changed.emit(False)
        super().leaveEvent(event)
