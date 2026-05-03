"""Pitch/Yaw グリッド、ビュー設定"""
from __future__ import annotations

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui import i18n
from gui.common.collapsible_section import CollapsibleSection

_MIN_YAW_SLOTS = 4
_MAX_YAW_SLOTS = 8
_DEFAULT_YAW_SLOTS = 6
_MAX_PITCH_ROWS = 9
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


class ViewConfigWidget(QWidget):
    """ビュー選択グリッドウィジェット。views_changed シグナルで変更を通知。"""

    views_changed = Signal()

    def __init__(self, parent: QWidget | None = None, *, show_settings: bool = True) -> None:
        super().__init__(parent)
        self.pitch_rows: list[dict] = []
        self.yaw_slot_labels: list[QLabel] = []
        self._output_count_text = ""
        self._show_settings = show_settings

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
        self.yaw_offset_edit = QLineEdit("45.0")
        self.yaw_offset_edit.setToolTip(i18n.tip("YAW_OFFSET"))
        self.yaw_offset_edit.setFixedWidth(72)
        self.yaw_offset_edit.textChanged.connect(self._on_params_changed)
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
        self.yaw_slots_combo.currentTextChanged.connect(lambda _: self._apply_pitch_rows())
        self.yaw_slots_label = QLabel(i18n.t("YAW_SLOTS_LABEL"))
        self.yaw_slots_label.setToolTip(i18n.tip("YAW_SLOTS"))
        custom_row.addWidget(self.yaw_slots_label)
        custom_row.addWidget(self.yaw_slots_combo)

        self.pitch_edit = QLineEdit("-30,0,30")
        self.pitch_edit.setToolTip(i18n.tip("PITCH_ROWS"))
        self.pitch_edit.setFixedWidth(132)
        self.pitch_label = QLabel(i18n.t("PITCH_ROWS_LABEL"))
        self.pitch_label.setToolTip(i18n.tip("PITCH_ROWS"))
        custom_row.addWidget(self.pitch_label)
        custom_row.addWidget(self.pitch_edit)

        self.apply_btn = QPushButton(i18n.t("APPLY"))
        self.apply_btn.setToolTip(i18n.tip("APPLY_BTN"))
        self.apply_btn.clicked.connect(self._apply_pitch_rows)
        self.apply_btn.setMinimumWidth(58)
        custom_row.addWidget(self.apply_btn)
        ctrl.addWidget(self.custom_controls_widget)

        # ビュー選択グリッド
        self.grid_section = CollapsibleSection(i18n.t("VIEW_SELECTION_SECTION"), expanded=False)
        btn_row = QHBoxLayout()
        all_on = QPushButton(i18n.t("SELECT_ALL"))
        all_on.clicked.connect(self._all_on)
        btn_row.addWidget(all_on)
        all_off = QPushButton(i18n.t("DESELECT_ALL"))
        all_off.clicked.connect(self._all_off)
        btn_row.addWidget(all_off)
        btn_row.addStretch()
        self.grid_section.content_layout.addLayout(btn_row)

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setHorizontalSpacing(8)
        self.grid_layout.setVerticalSpacing(4)
        self.grid_section.content_layout.addWidget(self.grid_widget)
        ctrl.addWidget(self.grid_section)

        summary_row = QHBoxLayout()
        self.selected_label = QLabel(f"{i18n.t('SELECTED_VIEWS')}: 0")
        self.selected_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        summary_row.addStretch()
        summary_row.addWidget(self.selected_label)
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
        try:
            return float(self.yaw_offset_edit.text().strip())
        except Exception:
            return 0.0

    def yaw_slot_count(self) -> int:
        try:
            return int(self.yaw_slots_combo.currentText())
        except Exception:
            return _DEFAULT_YAW_SLOTS

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
                views.append({
                    "name": f"pit{_angle_token(pitch)}_s{slot}",
                    "yaw": float(offset + slot * step),
                    "pitch": pitch,
                    "enabled": enabled,
                    "slot": slot,
                    "label": f"p{pitch:g}/s{slot}",
                })
        return views

    def set_output_count_text(self, text: str) -> None:
        self._output_count_text = text
        self._update_selected_label()

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
        self._update_selected_label()
        self.views_changed.emit()

    def _parse_pitches(self) -> list[float]:
        raw = self.pitch_edit.text().strip()
        if not raw:
            raise ValueError("Pitch行が空です")
        tokens = [t for t in re.split(r"[,\s]+", raw) if t]
        pitches, seen = [], set()
        for t in tokens:
            p = float(t)
            if p < -90.0 or p > 90.0:
                raise ValueError(f"Pitch範囲外 [-90, 90]: {p}")
            key = f"{p:.6f}"
            if key not in seen:
                seen.add(key)
                pitches.append(p)
        if not pitches:
            raise ValueError("有効なPitchがありません")
        if len(pitches) > _MAX_PITCH_ROWS:
            raise ValueError(f"Pitch行は最大 {_MAX_PITCH_ROWS} 行です")
        return pitches

    def _clear_grid(self) -> None:
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _apply_pitch_rows(self) -> None:
        old = {}
        for row in self.pitch_rows:
            key = f"{row['pitch']:.6f}"
            old[key] = [cb.isChecked() for cb in row["checks"]]

        try:
            pitches = self._parse_pitches()
        except Exception as e:
            QMessageBox.critical(self, "Pitchエラー", str(e))
            return

        slots = self.yaw_slot_count()
        self.pitch_rows = []
        self.yaw_slot_labels = []
        self._clear_grid()

        self.grid_layout.addWidget(QLabel(i18n.t("PITCH_SLOT_HEADER")), 0, 0)
        for s in range(slots):
            lab = QLabel(f"S{s}")
            lab.setAlignment(Qt.AlignCenter)
            self.yaw_slot_labels.append(lab)
            self.grid_layout.addWidget(lab, 0, s + 1)

        for ri, pitch in enumerate(pitches, start=1):
            self.grid_layout.addWidget(QLabel(f"{pitch:g}°"), ri, 0)
            checks = []
            key = f"{pitch:.6f}"
            restored = old.get(key)
            for s in range(slots):
                cb = QCheckBox()
                if restored and s < len(restored):
                    cb.setChecked(restored[s])
                else:
                    cb.setChecked(abs(pitch) < 1e-6)
                cb.toggled.connect(self._on_selection_changed)
                self.grid_layout.addWidget(cb, ri, s + 1, alignment=Qt.AlignCenter)
                checks.append(cb)
            self.pitch_rows.append({"pitch": pitch, "checks": checks})

        self._update_yaw_labels()
        self._update_selected_label()
        self.views_changed.emit()

    def _update_yaw_labels(self) -> None:
        offset = self.yaw_offset()
        slots = self.yaw_slot_count()
        step = 360.0 / float(slots)
        for i, lab in enumerate(self.yaw_slot_labels):
            yaw = _normalize_angle(offset + i * step)
            lab.setText(f"S{i}\n{yaw:.1f}°")

    def _update_selected_label(self) -> None:
        try:
            views = self.collect_views(include_disabled=True)
        except Exception:
            self.selected_label.setText(f"{i18n.t('SELECTED_VIEWS')}: -")
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

    def _all_on(self) -> None:
        for row in self.pitch_rows:
            for cb in row["checks"]:
                cb.setChecked(True)
        self._on_selection_changed()

    def _all_off(self) -> None:
        for row in self.pitch_rows:
            for cb in row["checks"]:
                cb.setChecked(False)
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
