"""Pitch/Yaw グリッド、ビュー設定"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui import i18n
from gui.common.drag_spinbox import DragDoubleSpinBox
from gui.common.icons import delete_icon, minus_icon, plus_icon, select_all_icon

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
_CUBE6_YAW_SLOTS = 4
_CUBE6_PITCHES = (-90.0, 0.0, 90.0)
_CUBE6_ENABLED_CELLS = frozenset({(0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 3)})
_CUBE6_VIEW_CELLS = (
    ("px", 1, 0),
    ("nx", 1, 2),
    ("pz", 1, 3),
    ("nz", 1, 1),
    # Use cubemap axis names for generated files. In the converter's pitch
    # convention, -90 samples the upper pole and +90 samples the lower pole.
    ("py", 0, 3),
    ("ny", 2, 3),
)
_CUBE6_CELL_TO_NAME = {(row, slot): name for name, row, slot in _CUBE6_VIEW_CELLS}
_PITCH_DELETE_BUTTON_SIZE = 24
_PITCH_CELL_SPACING = 2
_PITCH_EDIT_MIN_WIDTH = 78
_PITCH_EDIT_WIDTH_SAMPLE = "-999.9"
_YAW_SLOT_COLUMN_MIN_WIDTH = 39

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


class _ViewModeSelector(QWidget):
    """Radio selector with the small combo-like API used by the view grid."""

    currentIndexChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("radioOptionRow")
        self._items: list[tuple[str, object]] = []
        self._buttons: list[QRadioButton] = []
        self._current_index = -1
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addStretch()
        self._layout = layout

    def addItem(self, text: str, data: object) -> None:
        index = len(self._items)
        button = QRadioButton(text)
        button.setObjectName("optionRadio")
        button.setToolTip(self.toolTip())
        button.clicked.connect(lambda _checked=False, idx=index: self.setCurrentIndex(idx))
        self._group.addButton(button)
        self._items.append((text, data))
        self._buttons.append(button)
        self._layout.insertWidget(index, button)
        if self._current_index < 0:
            self.setCurrentIndex(index)

    def currentData(self) -> object | None:
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index][1]
        return None

    def currentText(self) -> str:
        return self.itemText(self._current_index)

    def findData(self, data: object) -> int:
        for index, (_text, item_data) in enumerate(self._items):
            if item_data == data:
                return index
        return -1

    def itemText(self, index: int) -> str:
        if 0 <= index < len(self._items):
            return self._items[index][0]
        return ""

    def setCurrentIndex(self, index: int) -> None:
        if not 0 <= index < len(self._items):
            return
        changed = index != self._current_index
        self._current_index = index
        button = self._buttons[index]
        if not button.isChecked():
            button.setChecked(True)
        if changed:
            self.currentIndexChanged.emit(index)

    def setToolTip(self, tooltip: str) -> None:  # noqa: N802 - Qt API
        super().setToolTip(tooltip)
        for button in self._buttons:
            button.setToolTip(tooltip)


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
        self.pitch_delete_buttons: list[QToolButton] = []
        self._yaw_slot_count = _DEFAULT_YAW_SLOTS
        self._output_count_text = ""
        self._show_settings = show_settings
        self._show_summary = show_summary
        self._rebuilding_grid = False
        self._normalizing_pitch = False
        self._applying_preset = False
        self._hovered_view_name: str | None = None

        self._build_ui()
        self._apply_cube6_preset()

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
        self.view_mode_combo = _ViewModeSelector()
        self.view_mode_combo.setToolTip(i18n.tip("VIEW_MODE"))
        self.view_mode_combo.addItem(i18n.t("CUBE6_LABEL"), VIEW_MODE_CUBE6)
        self.view_mode_combo.addItem(i18n.t("CUSTOM_GRID"), VIEW_MODE_CUSTOM)
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

        # ビュー選択グリッド
        self.grid_section = QWidget()
        self.grid_section.setToolTip(i18n.tip("VIEW_SELECTION_SECTION"))
        grid_section_layout = QVBoxLayout(self.grid_section)
        grid_section_layout.setContentsMargins(0, 0, 0, 0)
        grid_section_layout.setSpacing(4)

        self.grid_controls_widget = QWidget()
        grid_controls = QHBoxLayout(self.grid_controls_widget)
        grid_controls.setContentsMargins(0, 0, 0, 0)
        grid_controls.setSpacing(2)
        self.grid_title_label = QLabel(i18n.t("VIEW_SELECTION_COMPACT_SECTION"))
        self.grid_title_label.setToolTip(i18n.tip("VIEW_SELECTION_SECTION"))
        grid_controls.addWidget(self.grid_title_label)
        self.all_toggle_btn = self._make_grid_control_button(
            select_all_icon(),
            i18n.t("SELECT_ALL"),
            self._toggle_all_views,
        )
        self.all_toggle_btn.setCheckable(True)
        grid_controls.addWidget(self.all_toggle_btn)
        grid_controls.addStretch()

        self.yaw_remove_btn = self._make_grid_control_button(
            minus_icon(),
            i18n.t("YAW_SLOT_REMOVE"),
            lambda: self.set_yaw_slot_count(self.yaw_slot_count() - 1),
        )
        self.yaw_count_label = QLabel()
        self.yaw_count_label.setToolTip(i18n.tip("YAW_SLOTS"))
        self.yaw_add_btn = self._make_grid_control_button(
            plus_icon(),
            i18n.t("YAW_SLOT_ADD"),
            lambda: self.set_yaw_slot_count(self.yaw_slot_count() + 1),
        )
        self.pitch_count_label = QLabel()
        self.pitch_count_label.setToolTip(i18n.tip("PITCH_ROWS"))
        self.pitch_add_btn = self._make_grid_control_button(
            plus_icon(),
            i18n.t("PITCH_ROW_ADD"),
            self.add_pitch_row,
        )
        for widget in (
            self.yaw_remove_btn,
            self.yaw_count_label,
            self.yaw_add_btn,
        ):
            grid_controls.addWidget(widget)
        grid_section_layout.addWidget(self.grid_controls_widget)

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setHorizontalSpacing(2)
        self.grid_layout.setVerticalSpacing(4)
        grid_section_layout.addWidget(self.grid_widget)

        self.pitch_controls_widget = QWidget()
        pitch_controls = QHBoxLayout(self.pitch_controls_widget)
        pitch_controls.setContentsMargins(0, 0, 0, 0)
        pitch_controls.setSpacing(4)
        pitch_controls.addWidget(self.pitch_add_btn)
        pitch_controls.addWidget(self.pitch_count_label)
        pitch_controls.addStretch()
        grid_section_layout.addWidget(self.pitch_controls_widget)
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

    # -- public API --

    def view_mode(self) -> str:
        return self.view_mode_combo.currentData() or VIEW_MODE_CUSTOM

    def yaw_offset(self) -> float:
        return float(self.yaw_offset_edit.value())

    def yaw_slot_count(self) -> int:
        return int(self._yaw_slot_count)

    def pitch_row_count(self) -> int:
        if self.pitch_rows:
            return len(self.pitch_rows)
        return _DEFAULT_PITCH_ROWS

    def set_yaw_slot_count(self, count: int) -> None:
        count = max(_MIN_YAW_SLOTS, min(_MAX_YAW_SLOTS, int(count)))
        if count == self._yaw_slot_count:
            self._update_grid_control_state()
            return
        self._mark_custom_if_user_changed()
        rows = self._row_snapshots()
        self._yaw_slot_count = count
        self._rebuild_grid(rows)

    def set_pitch_row_count(self, count: int) -> None:
        count = max(_MIN_PITCH_ROWS, min(_MAX_PITCH_ROWS, int(count)))
        current = len(self.pitch_rows)
        if count == current:
            self._update_grid_control_state()
            return
        self._mark_custom_if_user_changed()
        rows = self._row_snapshots()
        if count < len(rows):
            rows = rows[:count]
        else:
            while len(rows) < count:
                rows.append({
                    "pitch": self._next_pitch_for_rows(rows),
                    "checks": [True] * self.yaw_slot_count(),
                })
        self._rebuild_grid(rows)

    def add_pitch_row(self) -> None:
        self.set_pitch_row_count(self.pitch_row_count() + 1)

    def remove_pitch_row(self, row_index: int) -> None:
        if self.pitch_row_count() <= _MIN_PITCH_ROWS:
            self._update_grid_control_state()
            return
        if row_index < 0 or row_index >= len(self.pitch_rows):
            return
        self._mark_custom_if_user_changed()
        rows = self._row_snapshots()
        rows.pop(row_index)
        self._rebuild_grid(rows)

    def pitch_values(self) -> list[float]:
        if self.pitch_rows:
            return [float(row["pitch"]) for row in self.pitch_rows]
        return _default_pitches_for_count(self.pitch_row_count())

    def pitch_rows_text(self) -> str:
        return ",".join(f"{pitch:g}" for pitch in self.pitch_values())

    def hovered_view_name(self) -> str | None:
        return self._hovered_view_name

    def collect_views(self, include_disabled: bool = False) -> list[dict]:
        return self._grid_views(include_disabled=include_disabled)

    def apply_settings_snapshot(self, snapshot: dict) -> None:
        """Restore the view grid from a Step 4 settings snapshot."""
        if not isinstance(snapshot, dict):
            return

        mode = str(snapshot.get("mode") or VIEW_MODE_CUBE6)
        if mode not in {VIEW_MODE_CUBE6, VIEW_MODE_CUSTOM}:
            mode = VIEW_MODE_CUBE6

        try:
            yaw_offset = float(snapshot.get("yaw_offset", self.yaw_offset()))
        except (TypeError, ValueError):
            yaw_offset = self.yaw_offset()
        self.yaw_offset_edit.setValue(yaw_offset)

        rows = self._rows_from_settings_snapshot(snapshot)
        mode_blocked = self.view_mode_combo.blockSignals(True)
        self._applying_preset = True
        try:
            idx = self.view_mode_combo.findData(mode)
            if idx >= 0:
                self.view_mode_combo.setCurrentIndex(idx)
            if rows:
                self._rebuild_grid(rows)
            elif mode == VIEW_MODE_CUBE6:
                self._apply_cube6_preset()
            else:
                self._apply_custom_defaults()
        finally:
            self._applying_preset = False
            self.view_mode_combo.blockSignals(mode_blocked)

    def _rows_from_settings_snapshot(self, snapshot: dict) -> list[dict]:
        try:
            yaw_slots = int(snapshot.get("yaw_slots") or self.yaw_slot_count())
        except (TypeError, ValueError):
            yaw_slots = self.yaw_slot_count()
        yaw_slots = max(_MIN_YAW_SLOTS, min(_MAX_YAW_SLOTS, yaw_slots))

        raw_pitches = snapshot.get("pitch_rows")
        pitches: list[float] = []
        if isinstance(raw_pitches, list):
            for value in raw_pitches[:_MAX_PITCH_ROWS]:
                try:
                    pitches.append(_clamp_pitch(float(value)))
                except (TypeError, ValueError):
                    continue

        raw_views = snapshot.get("views")
        views = raw_views if isinstance(raw_views, list) else []
        if not pitches and views:
            for view in views:
                if not isinstance(view, dict):
                    continue
                try:
                    pitch = _clamp_pitch(float(view.get("pitch")))
                except (TypeError, ValueError):
                    continue
                if _pitch_key(pitch) not in {_pitch_key(existing) for existing in pitches}:
                    pitches.append(pitch)
                if len(pitches) >= _MAX_PITCH_ROWS:
                    break

        if not pitches:
            return []

        rows: list[dict] = []
        for row_index, pitch in enumerate(pitches):
            checks: list[bool] = []
            for slot in range(yaw_slots):
                view_index = row_index * yaw_slots + slot
                if view_index < len(views) and isinstance(views[view_index], dict):
                    checks.append(bool(views[view_index].get("enabled", True)))
                else:
                    checks.append(True)
            rows.append({"pitch": pitch, "checks": checks})

        self._yaw_slot_count = yaw_slots
        return rows

    def _grid_views(self, include_disabled: bool = False) -> list[dict]:
        step = 360.0 / float(self.yaw_slot_count())
        views = []
        for row_index, row in enumerate(self.pitch_rows):
            pitch = float(row["pitch"])
            for slot, cb in enumerate(row["checks"]):
                enabled = cb.isChecked()
                if not include_disabled and not enabled:
                    continue
                name = self._view_name_for(row_index, slot)
                if name is None:
                    name = self._generated_view_name(pitch, slot)
                views.append({
                    "name": name,
                    "yaw": float(_normalize_angle(self.yaw_offset() + slot * step)),
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
        if self.view_mode() == VIEW_MODE_CUBE6:
            self._apply_cube6_preset()
        else:
            self._apply_custom_defaults()

    def _on_params_changed(self, *_args) -> None:
        self._update_yaw_labels()
        self._update_selected_label()
        self.views_changed.emit()

    def _on_selection_changed(self, *_args) -> None:
        if self._rebuilding_grid:
            return
        self._mark_custom_if_user_changed()
        self._update_selected_label()
        self._sync_all_toggle_state()
        self.views_changed.emit()

    def _parse_pitches(self) -> list[float]:
        return self.pitch_values()

    def _clear_grid(self) -> None:
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _row_snapshots(self) -> list[dict]:
        return [
            {
                "pitch": float(row["pitch"]),
                "checks": [cb.isChecked() for cb in row["checks"]],
            }
            for row in self.pitch_rows
        ]

    def _next_pitch_for_rows(self, rows: list[dict]) -> float:
        existing = [_clamp_pitch(row["pitch"]) for row in rows]
        used = {_pitch_key(p) for p in existing}
        for candidate in _default_pitches_for_count(min(len(rows) + 1, _MAX_PITCH_ROWS)):
            if _pitch_key(candidate) not in used:
                return candidate
        return self._nearest_unique_pitch(0.0, existing)

    def _rebuild_grid(self, rows: list[dict]) -> None:
        if not rows:
            rows = [{"pitch": 0.0, "checks": [True] * self.yaw_slot_count()}]
        rows = [
            {
                "pitch": _clamp_pitch(row["pitch"]),
                "checks": list(row.get("checks", [])),
            }
            for row in rows[:_MAX_PITCH_ROWS]
        ]
        slots = self.yaw_slot_count()
        self._rebuilding_grid = True
        try:
            self.pitch_rows = []
            self.yaw_slot_labels = []
            self.pitch_delete_buttons = []
            self._clear_grid()

            pitch_edit_width = self._pitch_edit_width()
            pitch_column_width = _PITCH_DELETE_BUTTON_SIZE + _PITCH_CELL_SPACING + pitch_edit_width
            for col in range(_MAX_YAW_SLOTS + 1):
                self.grid_layout.setColumnMinimumWidth(col, 0)
                self.grid_layout.setColumnStretch(col, 0)
            self.grid_layout.setColumnMinimumWidth(0, pitch_column_width)
            for s in range(slots):
                self.grid_layout.setColumnMinimumWidth(s + 1, _YAW_SLOT_COLUMN_MIN_WIDTH)
                self.grid_layout.setColumnStretch(s + 1, 1)
                yaw_cell = QWidget()
                yaw_cell.setMinimumWidth(_YAW_SLOT_COLUMN_MIN_WIDTH)
                yaw_cell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                yaw_layout = QHBoxLayout(yaw_cell)
                yaw_layout.setContentsMargins(0, 0, 0, 0)
                yaw_layout.setSpacing(0)
                lab = QLabel(f"S{s}")
                lab.setAlignment(Qt.AlignCenter)
                lab.setFixedWidth(_YAW_SLOT_COLUMN_MIN_WIDTH)
                lab.setWordWrap(False)
                lab.setStyleSheet("font-size: 8pt;")
                yaw_layout.addStretch(1)
                yaw_layout.addWidget(lab)
                yaw_layout.addStretch(1)
                self.yaw_slot_labels.append(lab)
                self.grid_layout.addWidget(yaw_cell, 0, s + 1)

            for ri, row in enumerate(rows):
                pitch = _clamp_pitch(row["pitch"])
                restored = row["checks"]
                row_index = ri
                grid_row = ri + 1
                pitch_cell = QWidget()
                pitch_layout = QHBoxLayout(pitch_cell)
                pitch_layout.setContentsMargins(0, 0, 0, 0)
                pitch_layout.setSpacing(_PITCH_CELL_SPACING)

                delete_btn = QToolButton()
                delete_btn.setObjectName("iconToolButton")
                delete_btn.setIcon(delete_icon())
                delete_btn.setToolTip(i18n.t("PITCH_ROW_REMOVE"))
                delete_btn.setAccessibleName(i18n.t("PITCH_ROW_REMOVE"))
                delete_btn.setFixedSize(_PITCH_DELETE_BUTTON_SIZE, _PITCH_DELETE_BUTTON_SIZE)
                delete_btn.clicked.connect(lambda _checked=False, idx=row_index: self.remove_pitch_row(idx))
                self.pitch_delete_buttons.append(delete_btn)
                pitch_layout.addWidget(delete_btn)

                pitch_edit = DragDoubleSpinBox(
                    minimum=_MIN_PITCH_DEG,
                    maximum=_MAX_PITCH_DEG,
                    step=1.0,
                    decimals=1,
                    value=pitch,
                    drag_pixels_per_step=6.0,
                )
                pitch_edit.setToolTip(i18n.tip("PITCH_ROWS"))
                pitch_edit.setFixedWidth(pitch_edit_width)
                pitch_edit.valueChanged.connect(lambda value, idx=row_index: self._on_pitch_value_changed(idx, value))
                pitch_layout.addWidget(pitch_edit)
                pitch_layout.addStretch(1)
                pitch_cell.setFixedWidth(pitch_column_width)
                self.grid_layout.addWidget(pitch_cell, grid_row, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)

                checks = []
                for s in range(slots):
                    cb = _HoverCheckBox()
                    cb.setFixedSize(QSize(22, 22))
                    if s < len(restored):
                        cb.setChecked(bool(restored[s]))
                    else:
                        cb.setChecked(True)
                    cb.toggled.connect(self._on_selection_changed)
                    cb.hover_changed.connect(lambda hovering, idx=row_index, slot=s: self._on_view_hover(idx, slot, hovering))
                    self.grid_layout.addWidget(cb, grid_row, s + 1, alignment=Qt.AlignCenter)
                    checks.append(cb)
                self.pitch_rows.append({
                    "pitch": pitch,
                    "pitch_edit": pitch_edit,
                    "checks": checks,
                    "delete_btn": delete_btn,
                })
        finally:
            self._rebuilding_grid = False
        self._update_yaw_labels()
        self._update_selected_label()
        self._update_grid_control_state()
        self.views_changed.emit()

    def _pitch_edit_width(self) -> int:
        metrics = self.fontMetrics()
        return max(_PITCH_EDIT_MIN_WIDTH, metrics.horizontalAdvance(_PITCH_EDIT_WIDTH_SAMPLE) + 22)

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
        self._mark_custom_if_user_changed()
        self._update_selected_label()
        self.views_changed.emit()

    def _generated_view_name(self, pitch: float, slot: int) -> str:
        return f"pit{_angle_token(pitch)}_s{slot}"

    def _view_name_for(self, row_index: int, slot: int) -> str | None:
        if row_index < 0 or row_index >= len(self.pitch_rows):
            return None
        if self.view_mode() == VIEW_MODE_CUBE6 and self._has_cube6_geometry():
            semantic_name = _CUBE6_CELL_TO_NAME.get((row_index, slot))
            if semantic_name:
                return semantic_name
        pitch = float(self.pitch_rows[row_index]["pitch"])
        return self._generated_view_name(pitch, slot)

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

    def _make_grid_control_button(self, icon, label: str, callback) -> QToolButton:
        button = QToolButton()
        button.setObjectName("iconToolButton")
        button.setIcon(icon)
        button.setToolTip(label)
        button.setAccessibleName(label)
        button.setFixedSize(24, 24)
        button.clicked.connect(callback)
        return button

    def _update_yaw_labels(self) -> None:
        offset = self.yaw_offset()
        slots = self.yaw_slot_count()
        step = 360.0 / float(slots)
        for i, lab in enumerate(self.yaw_slot_labels):
            yaw = _normalize_angle(offset + i * step)
            display_yaw = f"{yaw:.1f}"
            lab.setText(f"S{i}\n{display_yaw}")
            lab.setToolTip(f"{display_yaw}°")

    def _update_grid_control_state(self) -> None:
        yaw_count = self.yaw_slot_count()
        pitch_count = self.pitch_row_count()
        self.yaw_count_label.setText(i18n.t("YAW_SLOT_COUNT_FORMAT").format(count=yaw_count))
        self.pitch_count_label.setText(i18n.t("PITCH_ROW_COUNT_FORMAT").format(count=pitch_count))
        self.yaw_remove_btn.setEnabled(yaw_count > _MIN_YAW_SLOTS)
        self.yaw_add_btn.setEnabled(yaw_count < _MAX_YAW_SLOTS)
        self.pitch_add_btn.setEnabled(pitch_count < _MAX_PITCH_ROWS)
        for button in self.pitch_delete_buttons:
            button.setEnabled(pitch_count > _MIN_PITCH_ROWS)
        self._sync_all_toggle_state()

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

    def _toggle_all_views(self, checked: bool) -> None:
        self._set_all_checked(bool(checked))

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
        else:
            self._sync_all_toggle_state()

    def _sync_all_toggle_state(self) -> None:
        views = self.collect_views(include_disabled=True)
        all_checked = bool(views) and all(view["enabled"] for view in views)
        was_blocked = self.all_toggle_btn.blockSignals(True)
        try:
            self.all_toggle_btn.setChecked(all_checked)
        finally:
            self.all_toggle_btn.blockSignals(was_blocked)
        label = i18n.t("DESELECT_ALL") if all_checked else i18n.t("SELECT_ALL")
        self.all_toggle_btn.setToolTip(label)
        self.all_toggle_btn.setAccessibleName(label)

    def _apply_grid_preset(
        self,
        *,
        yaw_slots: int,
        pitches: tuple[float, ...] | list[float],
        enabled_cells: frozenset[tuple[int, int]] | None,
    ) -> None:
        self._applying_preset = True
        try:
            self._yaw_slot_count = max(_MIN_YAW_SLOTS, min(_MAX_YAW_SLOTS, int(yaw_slots)))
            rows = []
            for ri, pitch in enumerate(pitches[:_MAX_PITCH_ROWS]):
                checks = [
                    ((ri, slot) in enabled_cells) if enabled_cells is not None else True
                    for slot in range(self.yaw_slot_count())
                ]
                rows.append({"pitch": pitch, "checks": checks})
            self._rebuild_grid(rows)
        finally:
            self._applying_preset = False

    def _apply_cube6_preset(self) -> None:
        self._apply_grid_preset(
            yaw_slots=_CUBE6_YAW_SLOTS,
            pitches=list(_CUBE6_PITCHES),
            enabled_cells=_CUBE6_ENABLED_CELLS,
        )

    def _apply_custom_defaults(self) -> None:
        self._apply_grid_preset(
            yaw_slots=_DEFAULT_YAW_SLOTS,
            pitches=_default_pitches_for_count(_DEFAULT_PITCH_ROWS),
            enabled_cells=None,
        )

    def _mark_custom_if_user_changed(self) -> None:
        if self._applying_preset or self.view_mode() != VIEW_MODE_CUBE6:
            return
        custom_idx = self.view_mode_combo.findData(VIEW_MODE_CUSTOM)
        if custom_idx < 0:
            return
        was_blocked = self.view_mode_combo.blockSignals(True)
        try:
            self.view_mode_combo.setCurrentIndex(custom_idx)
        finally:
            self.view_mode_combo.blockSignals(was_blocked)

    def _has_cube6_geometry(self) -> bool:
        return (
            self.yaw_slot_count() == _CUBE6_YAW_SLOTS
            and len(self.pitch_rows) == len(_CUBE6_PITCHES)
            and all(
                _pitch_key(float(row["pitch"])) == _pitch_key(expected)
                for row, expected in zip(self.pitch_rows, _CUBE6_PITCHES, strict=True)
            )
        )

    def _cube6_views(self, yaw_offset: float) -> list[dict]:
        step = 360.0 / float(_CUBE6_YAW_SLOTS)
        return [
            {
                "name": name,
                "yaw": _normalize_angle(float(yaw_offset) + slot * step),
                "pitch": _CUBE6_PITCHES[row],
                "enabled": True,
                "slot": slot,
                "label": name,
            }
            for name, row, slot in _CUBE6_VIEW_CELLS
        ]


class _HoverCheckBox(QCheckBox):
    hover_changed = Signal(bool)

    def enterEvent(self, event) -> None:  # type: ignore[override]
        self.hover_changed.emit(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self.hover_changed.emit(False)
        super().leaveEvent(event)
