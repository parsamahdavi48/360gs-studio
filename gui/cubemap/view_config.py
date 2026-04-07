"""ピッチ/ヨーグリッド、ビュー設定"""
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.pitch_rows: list[dict] = []
        self.yaw_slot_labels: list[QLabel] = []

        self._build_ui()
        self._apply_pitch_rows()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        ctrl = QHBoxLayout()

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.setToolTip("カスタムグリッド: ピッチ/ヨーを自由に設定\nCube6: 標準キューブマップ6面 (前後左右+上下)")
        self.view_mode_combo.addItem("カスタムグリッド", VIEW_MODE_CUSTOM)
        self.view_mode_combo.addItem("Cube6 (4面+上下)", VIEW_MODE_CUBE6)
        self.view_mode_combo.currentIndexChanged.connect(self._on_view_mode_changed)
        ctrl.addWidget(QLabel("ビューモード:"))
        ctrl.addWidget(self.view_mode_combo)

        self.yaw_offset_edit = QLineEdit("45.0")
        self.yaw_offset_edit.setToolTip("全ビューのヨー角にオフセットを加算 (度)。スティッチ線を避けるために45度推奨")
        self.yaw_offset_edit.setFixedWidth(60)
        self.yaw_offset_edit.textChanged.connect(self._on_params_changed)
        ctrl.addWidget(QLabel("ヨーオフセット:"))
        ctrl.addWidget(self.yaw_offset_edit)

        self.yaw_slots_combo = QComboBox()
        self.yaw_slots_combo.setToolTip("水平方向の分割数 (4-8)。360度をN等分した角度でビューを配置")
        self.yaw_slots_combo.addItems([str(v) for v in range(_MIN_YAW_SLOTS, _MAX_YAW_SLOTS + 1)])
        self.yaw_slots_combo.setCurrentText(str(_DEFAULT_YAW_SLOTS))
        self.yaw_slots_combo.currentTextChanged.connect(lambda _: self._apply_pitch_rows())
        ctrl.addWidget(QLabel("ヨースロット:"))
        ctrl.addWidget(self.yaw_slots_combo)

        self.pitch_edit = QLineEdit("-30,0,30")
        self.pitch_edit.setToolTip("垂直方向のピッチ角をカンマ区切りで指定 (度)。-90〜90。例: -30,0,30")
        self.pitch_edit.setFixedWidth(120)
        ctrl.addWidget(QLabel("ピッチ行:"))
        ctrl.addWidget(self.pitch_edit)

        apply_btn = QPushButton("適用")
        apply_btn.setToolTip("ピッチ行とヨースロットの変更をグリッドに反映")
        apply_btn.clicked.connect(self._apply_pitch_rows)
        ctrl.addWidget(apply_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Cube6 オプション
        self._cube6_row = QHBoxLayout()
        self.cube6_drop_top = QCheckBox("上面(+90)を除外")
        self.cube6_drop_top.setToolTip("天頂面 (真上) をキューブマップから除外。空しか映らない場合に")
        self.cube6_drop_top.toggled.connect(self._on_selection_changed)
        self._cube6_row.addWidget(self.cube6_drop_top)
        self.cube6_drop_bottom = QCheckBox("底面(-90)を除外")
        self.cube6_drop_bottom.setToolTip("底面 (真下) をキューブマップから除外。三脚/撮影者が映る場合に")
        self.cube6_drop_bottom.toggled.connect(self._on_selection_changed)
        self._cube6_row.addWidget(self.cube6_drop_bottom)
        self._cube6_row.addStretch()
        self._cube6_container = QWidget()
        QHBoxLayout(self._cube6_container).addLayout(self._cube6_row)
        layout.addWidget(self._cube6_container)

        # ビュー選択ボタン
        btn_row = QHBoxLayout()
        all_on = QPushButton("全選択")
        all_on.clicked.connect(self._all_on)
        btn_row.addWidget(all_on)
        all_off = QPushButton("全解除")
        all_off.clicked.connect(self._all_off)
        btn_row.addWidget(all_off)
        btn_row.addStretch()
        self.selected_label = QLabel("選択ビュー: 0")
        self.selected_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        btn_row.addWidget(self.selected_label)
        layout.addLayout(btn_row)

        # グリッド
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        layout.addWidget(self.grid_widget)

        self._on_view_mode_changed(0)

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

    # -- internal --

    def _on_view_mode_changed(self, _index: int) -> None:
        is_custom = self.view_mode() == VIEW_MODE_CUSTOM
        self.yaw_slots_combo.setVisible(is_custom)
        self.pitch_edit.setVisible(is_custom)
        self.grid_widget.setVisible(is_custom)
        self._cube6_container.setVisible(not is_custom)
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
            raise ValueError("ピッチ行が空です")
        tokens = [t for t in re.split(r"[,\s]+", raw) if t]
        pitches, seen = [], set()
        for t in tokens:
            p = float(t)
            if p < -90.0 or p > 90.0:
                raise ValueError(f"ピッチ範囲外 [-90, 90]: {p}")
            key = f"{p:.6f}"
            if key not in seen:
                seen.add(key)
                pitches.append(p)
        if not pitches:
            raise ValueError("有効なピッチがありません")
        if len(pitches) > _MAX_PITCH_ROWS:
            raise ValueError(f"ピッチ行は最大 {_MAX_PITCH_ROWS} 行です")
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
            QMessageBox.critical(self, "ピッチエラー", str(e))
            return

        slots = self.yaw_slot_count()
        self.pitch_rows = []
        self.yaw_slot_labels = []
        self._clear_grid()

        self.grid_layout.addWidget(QLabel("ピッチ / スロット"), 0, 0)
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
            self.selected_label.setText("選択ビュー: -")
            return
        sel = sum(1 for v in views if v["enabled"])
        warn = ""
        if sel > _BLOCK_ENABLED_VIEWS:
            warn = " [超過]"
        elif sel > _WARN_ENABLED_VIEWS:
            warn = " [多い]"
        self.selected_label.setText(f"選択ビュー: {sel} / {len(views)}{warn}")

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
        dt = self.cube6_drop_top.isChecked()
        db = self.cube6_drop_bottom.isChecked()
        return [
            {"name": "px", "yaw": 90.0 - yaw_offset, "pitch": 0.0, "enabled": True, "slot": 0, "label": "px"},
            {"name": "nx", "yaw": -90.0 - yaw_offset, "pitch": 0.0, "enabled": True, "slot": 1, "label": "nx"},
            {"name": "pz", "yaw": 0.0 - yaw_offset, "pitch": 0.0, "enabled": True, "slot": 2, "label": "pz"},
            {"name": "nz", "yaw": 180.0 - yaw_offset, "pitch": 0.0, "enabled": True, "slot": 3, "label": "nz"},
            {"name": "top", "yaw": 0.0 - yaw_offset, "pitch": 90.0, "enabled": not dt, "slot": 4, "label": "top"},
            {"name": "bottom", "yaw": 0.0 - yaw_offset, "pitch": -90.0, "enabled": not db, "slot": 5, "label": "bottom"},
        ]
