#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

try:
    from PySide6.QtCore import QSize, Qt, Signal
    from PySide6.QtGui import QIcon, QKeySequence, QPixmap, QShortcut
    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSlider,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except Exception as e:  # pragma: no cover - environment-dependent import
    QSize = None
    Qt = None
    Signal = None
    QIcon = None
    QKeySequence = None
    QPixmap = None
    QShortcut = None
    QApplication = None
    QHBoxLayout = None
    QLabel = None
    QMainWindow = None
    QMessageBox = None
    QPushButton = None
    QSlider = None
    QToolButton = None
    QVBoxLayout = None
    QWidget = None
    _PYSIDE_IMPORT_ERROR = e
else:
    _PYSIDE_IMPORT_ERROR = None

# i18n は PySide6 に依存しないので無条件 import
from gui import i18n
from apply_frame_decisions import pending_drop_image_paths as find_pending_drop_image_paths
if _PYSIDE_IMPORT_ERROR is None:
    from gui.common.zoomable_image_label import ZoomableImageLabel
else:  # pragma: no cover - PySide6 missing
    ZoomableImageLabel = None


_ICON_DIR = Path(__file__).resolve().parent / "gui" / "assets" / "icons"


def _review_icon(name: str) -> QIcon:
    return QIcon(str(_ICON_DIR / f"{name}.svg"))


if QMainWindow is not None:
    class ReviewWidget(QWidget):
        decisions_changed = Signal()

        def __init__(self, scene_dir: Path, csv_path: Path) -> None:
            super().__init__()
            self.scene_dir = scene_dir
            self.csv_path = csv_path
            self.rows = self._load_rows(csv_path)
            if not self.rows:
                raise RuntimeError(f"No rows found in {csv_path}")
            self._initial_decisions = [row["decision"] for row in self.rows]
            self.problem_indices = self._collect_problem_indices()

            self.index = 0
            self._slider_sync = False
            self.current_pixmap: QPixmap | None = None

            self._build_ui()
            self._bind_shortcuts()
            self._render_current()

        def _load_rows(self, path: Path) -> List[Dict[str, str]]:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            for row in rows:
                decision = row.get("decision", "keep").strip().lower()
                row["decision"] = "drop" if decision == "drop" else "keep"
            return rows

        def _is_problem_row(self, row: Dict[str, str]) -> bool:
            status = row.get("status", "").strip().lower()
            return status not in {"", "ok"}

        def _collect_problem_indices(self) -> List[int]:
            return [i for i, row in enumerate(self.rows) if self._is_problem_row(row)]

        def _build_ui(self) -> None:
            layout = QVBoxLayout(self)

            top_row = QHBoxLayout()
            self.title_label = QLabel()
            self.title_label.setStyleSheet("font-weight: 700;")
            top_row.addWidget(self.title_label)

            top_row.addStretch(1)

            self.decision_label = QLabel()
            self.decision_label.setStyleSheet("font-weight: 700;")
            top_row.addWidget(self.decision_label)

            self.flag_button = QToolButton()
            self.flag_button.setCheckable(True)
            self.flag_button.setText("")
            self.flag_button.setToolTip(i18n.t("REVIEW_FLAG_TIP"))
            self.flag_button.setIconSize(QSize(20, 20))
            self.flag_button.setFixedSize(36, 32)
            self.flag_button.clicked.connect(lambda _checked=False: self.toggle_decision())
            top_row.addWidget(self.flag_button)

            self.reset_decision_button = QToolButton()
            self.reset_decision_button.setText("")
            self.reset_decision_button.setIcon(_review_icon("rotate-ccw"))
            self.reset_decision_button.setToolTip(i18n.t("REVIEW_RESET_DECISION_TIP"))
            self.reset_decision_button.setIconSize(QSize(20, 20))
            self.reset_decision_button.setFixedSize(36, 32)
            self.reset_decision_button.clicked.connect(lambda _checked=False: self.reset_decision())
            top_row.addWidget(self.reset_decision_button)
            layout.addLayout(top_row)

            self.image_view = ZoomableImageLabel()
            self.image_view.setMinimumHeight(260)
            self.image_view.setStyleSheet("border: 1px solid palette(mid);")
            layout.addWidget(self.image_view, stretch=1)

            timeline_row = QHBoxLayout()
            self.frame_slider = QSlider(Qt.Horizontal)
            self.frame_slider.setToolTip(i18n.t("REVIEW_FRAME_SLIDER_TIP"))
            self.frame_slider.setRange(0, max(0, len(self.rows) - 1))
            self.frame_slider.setEnabled(len(self.rows) > 1)
            self.frame_slider.valueChanged.connect(self._on_slider_changed)
            timeline_row.addWidget(self.frame_slider, stretch=1)

            self.frame_position_label = QLabel()
            timeline_row.addWidget(self.frame_position_label)
            layout.addLayout(timeline_row)

            # アドバイザリー（このフレームが要注意な理由を自動表示）
            self.advisory_label = QLabel()
            self.advisory_label.setWordWrap(False)
            self.advisory_label.setStyleSheet(
                "padding: 6px 10px; border-radius: 4px; font-weight: 600; font-size: 10pt;"
            )
            self.advisory_label.setFixedHeight(32)
            layout.addWidget(self.advisory_label)

            self.info_label = QLabel()
            self.info_label.setWordWrap(True)
            layout.addWidget(self.info_label)

            self.problem_summary_label = QLabel()
            self.problem_summary_label.setWordWrap(False)
            self.problem_summary_label.setStyleSheet("color: palette(text); font-weight: 500;")
            layout.addWidget(self.problem_summary_label)

            btn_row = QHBoxLayout()
            self.prev_problem_button = QPushButton(i18n.t("REVIEW_BTN_PREV_PROBLEM"))
            self.prev_problem_button.setToolTip(i18n.t("REVIEW_BTN_PROBLEM_TIP"))
            self.prev_problem_button.clicked.connect(self.prev_problem)
            btn_row.addWidget(self.prev_problem_button)

            self.next_problem_button = QPushButton(i18n.t("REVIEW_BTN_NEXT_PROBLEM"))
            self.next_problem_button.setToolTip(i18n.t("REVIEW_BTN_PROBLEM_TIP"))
            self.next_problem_button.clicked.connect(self.next_problem)
            btn_row.addWidget(self.next_problem_button)

            btn_row.addStretch(1)
            layout.addLayout(btn_row)

        def _bind_shortcuts(self) -> None:
            QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.prev_row)
            QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.next_row)
            QShortcut(QKeySequence("F"), self, activated=self.next_problem)
            QShortcut(QKeySequence("Shift+F"), self, activated=self.prev_problem)
            QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_decision)
            QShortcut(QKeySequence("Q"), self, activated=self._close_review_window)
            QShortcut(QKeySequence("0"), self, activated=self.reset_zoom)

        def _close_review_window(self) -> None:
            window = self.window()
            if isinstance(window, ReviewWindow):
                window.close()

        def _current_row(self) -> Dict[str, str]:
            return self.rows[self.index]

        def _decision_color(self, decision: str) -> str:
            return "#b00020" if decision == "drop" else "#1b7f3b"

        def _decision_text(self, decision: str) -> str:
            if decision == "drop":
                return i18n.t("REVIEW_DECISION_DROP")
            return i18n.t("REVIEW_DECISION_KEEP")

        def _advisory_for_row(self, row: Dict[str, str], idx: int) -> tuple[str, str, str]:
            """status のみに基づく advisory。

            フレーム抽出時に代表フレーム選択は済んでいるので、
            低品質のまま残ったフレームだけを強く確認対象として扱う。
            「ブレ top X%」のような相対順位 advisory は出さない（誤判定の元）。

            Returns (text, fg, bg).
            """
            status = row.get("status", "ok").strip().lower()

            # 橙: 品質条件を満たす代表候補が探索範囲内になかったフレーム
            if "fallback_keep" in status:
                return i18n.t("REVIEW_ADVISORY_FALLBACK"), "#fef3c7", "#7c2d12"

            # 青: 自動間引き
            if "thinned" in status:
                return i18n.t("REVIEW_ADVISORY_THINNED"), "#dbeafe", "#1e3a8a"

            # 青: 自動置換済み
            if "replaced" in status:
                return i18n.t("REVIEW_ADVISORY_REPLACED"), "#dbeafe", "#1e3a8a"

            # 青: 変化補正による追加候補
            if "smart_added" in status:
                return i18n.t("REVIEW_ADVISORY_SMART_ADDED"), "#dbeafe", "#1e3a8a"

            # 緑: 通常品質
            return i18n.t("REVIEW_ADVISORY_NORMAL"), "#a7f3d0", "#064e3b"

        def _format_quality_value(self, value: str | None) -> str:
            if value in (None, ""):
                return "-"
            try:
                return f"{float(value):.2f}"
            except (TypeError, ValueError):
                return "-"

        def _quality_summary(self, row: Dict[str, str]) -> str:
            final_score = self._format_quality_value(row.get("quality_score_final"))
            original_score = self._format_quality_value(row.get("quality_score_original"))
            threshold = self._format_quality_value(row.get("quality_min_score"))

            parts = [final_score]
            detail_parts: list[str] = []
            if original_score != "-" and original_score != final_score:
                detail_parts.append(i18n.t("REVIEW_QUALITY_ORIGINAL_FORMAT").format(score=original_score))
            if threshold != "-":
                detail_parts.append(i18n.t("REVIEW_QUALITY_THRESHOLD_FORMAT").format(score=threshold))
            if detail_parts:
                parts.append(f"({' / '.join(detail_parts)})")
            return " ".join(parts)

        def _render_current(self) -> None:
            row = self._current_row()
            seq = int(row.get("seq", self.index + 1))
            total = len(self.rows)

            image_rel = row.get("output_file", "")
            image_path = self.scene_dir / image_rel

            self.title_label.setText(f"{seq}/{total}  {image_rel}")
            self._slider_sync = True
            self.frame_slider.setValue(self.index)
            self._slider_sync = False
            self.frame_position_label.setText(
                i18n.t("REVIEW_FRAME_POSITION_FORMAT").format(
                    seq=seq,
                    total=total,
                    name=Path(image_rel).name,
                )
            )

            decision = row.get("decision", "keep")
            self.decision_label.setText(f"{i18n.t('REVIEW_DECISION_PREFIX')}{self._decision_text(decision)}")
            self.decision_label.setStyleSheet(f"font-weight: 700; color: {self._decision_color(decision)};")
            self._update_decision_buttons(decision)

            # アドバイザリー (このフレームが要注意な理由)
            adv_text, adv_fg, adv_bg = self._advisory_for_row(row, self.index)
            self.advisory_label.setText(adv_text)
            self.advisory_label.setStyleSheet(
                f"padding: 6px 10px; border-radius: 4px; font-weight: 600; font-size: 10pt; "
                f"color: {adv_fg}; background-color: {adv_bg};"
            )

            smart_added_count = sum(1 for r in self.rows if "smart_added" in r.get("status", "").strip().lower())
            replaced_count = sum(1 for r in self.rows if "replaced" in r.get("status", "").strip().lower())
            fallback_count = sum(1 for r in self.rows if "fallback_keep" in r.get("status", "").strip().lower())
            thinned_count = sum(1 for r in self.rows if "thinned" in r.get("status", "").strip().lower())
            problem_count = len(self.problem_indices)
            current_problem = i18n.t("REVIEW_INFO_YES") if self._is_problem_row(row) else i18n.t("REVIEW_INFO_NO")
            self.problem_summary_label.setText(
                i18n.t("REVIEW_PROBLEMS_FORMAT").format(
                    n=problem_count,
                    a=smart_added_count,
                    r=replaced_count,
                    f=fallback_count,
                    t=thinned_count,
                    cur=current_problem,
                )
            )

            # 動画内位置
            ts_raw = row.get("timestamp_sec", "-")
            try:
                ts_str = f"{float(ts_raw):.2f}s"
            except (ValueError, TypeError):
                ts_str = ts_raw

            info_text = i18n.t("REVIEW_INFO_FORMAT").format(
                ts=ts_str,
                quality=self._quality_summary(row),
            )
            self.info_label.setText(info_text)

            if not image_path.exists():
                self.current_pixmap = None
                self.image_view.setText(i18n.t("REVIEW_IMAGE_NOT_FOUND").format(path=image_path))
                return

            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                self.current_pixmap = None
                self.image_view.setText(i18n.t("REVIEW_IMAGE_LOAD_FAILED").format(path=image_path))
                return

            self.current_pixmap = pixmap
            self.image_view.set_source_pixmap(pixmap)

        def reset_zoom(self) -> None:
            self.image_view.reset_view()

        def _on_slider_changed(self, value: int) -> None:
            if self._slider_sync:
                return
            if 0 <= value < len(self.rows):
                self.index = value
                self._render_current()

        def _update_decision_buttons(self, decision: str) -> None:
            keep = decision != "drop"
            self.flag_button.setChecked(keep)
            self.flag_button.setIcon(_review_icon("flag-keep" if keep else "flag-drop"))
            self.flag_button.setStyleSheet(
                "QToolButton {"
                f"background: {'#14532d' if keep else '#3b1717'};"
                f"border: 1px solid {'#22c55e' if keep else '#991b1b'};"
                "border-radius: 4px;"
                "}"
            )
            reset_enabled = decision != self._initial_decisions[self.index]
            self.reset_decision_button.setEnabled(reset_enabled)
            self.reset_decision_button.setStyleSheet(
                "QToolButton { border-radius: 4px; }"
                "QToolButton:disabled { opacity: 0.45; }"
            )

        def has_decision_changes(self) -> bool:
            return any(
                row.get("decision", "keep") != initial
                for row, initial in zip(self.rows, self._initial_decisions)
            )

        def pending_drop_image_paths(self) -> list[Path]:
            return find_pending_drop_image_paths(self.scene_dir, self.csv_path.name)

        def has_pending_finalize(self) -> bool:
            return self.has_decision_changes() or bool(self.pending_drop_image_paths())

        def _write_rows(self) -> None:
            fieldnames = list(self.rows[0].keys())
            with self.csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)

        def _set_current_decision(self, decision: str) -> None:
            row = self._current_row()
            old_decision = row.get("decision", "keep")
            if old_decision == decision:
                self._render_current()
                return

            row["decision"] = decision
            try:
                self._write_rows()
            except Exception as e:
                row["decision"] = old_decision
                QMessageBox.critical(
                    self,
                    i18n.t("REVIEW_SAVE_FAILED_HEADER"),
                    i18n.t("REVIEW_SAVE_FAILED_BODY").format(error=e),
                )
            self._render_current()
            self.decisions_changed.emit()

        def prev_row(self) -> None:
            if self.index > 0:
                self.index -= 1
                self._render_current()

        def next_row(self) -> None:
            if self.index < len(self.rows) - 1:
                self.index += 1
                self._render_current()

        def next_problem(self) -> None:
            if not self.problem_indices:
                QMessageBox.information(self, i18n.t("REVIEW_INFO_HEADER"), i18n.t("REVIEW_NO_PROBLEMS"))
                return

            for idx in self.problem_indices:
                if idx > self.index:
                    self.index = idx
                    self._render_current()
                    return

            self.index = self.problem_indices[0]
            self._render_current()

        def prev_problem(self) -> None:
            if not self.problem_indices:
                QMessageBox.information(self, i18n.t("REVIEW_INFO_HEADER"), i18n.t("REVIEW_NO_PROBLEMS"))
                return

            for idx in reversed(self.problem_indices):
                if idx < self.index:
                    self.index = idx
                    self._render_current()
                    return

            self.index = self.problem_indices[-1]
            self._render_current()

        def toggle_decision(self) -> None:
            row = self._current_row()
            next_decision = "drop" if row.get("decision", "keep") == "keep" else "keep"
            self._set_current_decision(next_decision)

        def reset_decision(self) -> None:
            self._set_current_decision(self._initial_decisions[self.index])


    class ReviewWindow(QMainWindow):
        def __init__(self, scene_dir: Path, csv_path: Path) -> None:
            super().__init__()
            self.setWindowTitle(i18n.t("REVIEW_TITLE"))
            self.resize(1280, 860)
            self.review_widget = ReviewWidget(scene_dir, csv_path)
            self.setCentralWidget(self.review_widget)

else:
    class ReviewWidget:  # pragma: no cover - placeholder when PySide6 missing
        pass


    class ReviewWindow:  # pragma: no cover - placeholder when PySide6 missing
        pass


def ensure_gui_deps() -> None:
    if QApplication is None:
        raise RuntimeError(f"PySide6 is required to run this GUI: {_PYSIDE_IMPORT_ERROR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review extracted frames and edit keep/drop decisions.")
    parser.add_argument(
        "scene_dir",
        nargs="?",
        default=".",
        help="Scene directory containing selected_frames.csv and images/",
    )
    parser.add_argument(
        "--csv",
        default="selected_frames.csv",
        help="CSV filename under scene_dir (default=selected_frames.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir).resolve()
    csv_path = scene_dir / args.csv

    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)

    try:
        ensure_gui_deps()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    app = QApplication(sys.argv)
    try:
        window = ReviewWindow(scene_dir, csv_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
