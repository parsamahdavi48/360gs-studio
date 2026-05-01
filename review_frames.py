#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence, QPainter, QPixmap, QShortcut
    from PySide6.QtWidgets import (
        QApplication,
        QGraphicsScene,
        QGraphicsView,
        QHBoxLayout,
        QLineEdit,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )
except Exception as e:  # pragma: no cover - environment-dependent import
    Qt = None
    QKeySequence = None
    QPainter = None
    QPixmap = None
    QShortcut = None
    QApplication = None
    QGraphicsScene = None
    QGraphicsView = None
    QHBoxLayout = None
    QLabel = None
    QMainWindow = None
    QMessageBox = None
    QPushButton = None
    QLineEdit = None
    QScrollArea = None
    QSizePolicy = None
    QVBoxLayout = None
    QWidget = None
    _PYSIDE_IMPORT_ERROR = e
else:
    _PYSIDE_IMPORT_ERROR = None

# i18n は PySide6 に依存しないので無条件 import
from gui import i18n
if _PYSIDE_IMPORT_ERROR is None:
    from gui.common.collapsible_section import CollapsibleSection
else:  # pragma: no cover - PySide6 missing
    CollapsibleSection = None


if QGraphicsView is not None:
    class ZoomableImageView(QGraphicsView):
        """ホイールで拡大縮小、ドラッグでパンできる画像ビュー。"""

        ZOOM_STEP = 1.20

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._scene = QGraphicsScene(self)
            self.setScene(self._scene)
            self._pixmap_item = None
            self._fit_pending = True

            self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.setStyleSheet("background-color: #101010; border: 1px solid #333;")
            self.setMinimumHeight(340)
            self.setFocusPolicy(Qt.NoFocus)  # キー入力は親ウィンドウへ

        def show_pixmap(self, pixmap: QPixmap) -> None:
            self._scene.clear()
            self._pixmap_item = self._scene.addPixmap(pixmap)
            self._scene.setSceneRect(self._pixmap_item.boundingRect())
            # 新しい画像が来たら fit-to-window にリセット
            self.resetTransform()
            self._fit_pending = True
            self._fit_to_view()

        def show_message(self, text: str) -> None:
            self._scene.clear()
            self._pixmap_item = None
            item = self._scene.addText(text)
            item.setDefaultTextColor(Qt.gray)
            self._scene.setSceneRect(item.boundingRect())
            self.resetTransform()
            self._fit_pending = False
            self.centerOn(item)

        def _fit_to_view(self) -> None:
            if self._pixmap_item is None:
                return
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

        def reset_zoom(self) -> None:
            """フィットウィンドウに戻す。"""
            self.resetTransform()
            self._fit_to_view()

        def wheelEvent(self, event) -> None:
            if self._pixmap_item is None:
                return
            delta = event.angleDelta().y()
            if delta == 0:
                return
            factor = self.ZOOM_STEP if delta > 0 else 1.0 / self.ZOOM_STEP
            self.scale(factor, factor)
            event.accept()

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            # 初回だけ自動フィット。以降のリサイズはユーザーのズームを尊重。
            if self._fit_pending and self._pixmap_item is not None:
                self._fit_to_view()
                self._fit_pending = False

        def keyPressEvent(self, event) -> None:
            # 矢印キーは親のフレーム送りで使うので、ビュー側では拾わない
            if event.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
                event.ignore()
                return
            super().keyPressEvent(event)
else:
    class ZoomableImageView:  # pragma: no cover - placeholder when PySide6 missing
        pass


if QMainWindow is not None:
    class ReviewWidget(QWidget):
        def __init__(self, scene_dir: Path, csv_path: Path) -> None:
            super().__init__()
            self.scene_dir = scene_dir
            self.csv_path = csv_path
            self.rows = self._load_rows(csv_path)
            if not self.rows:
                raise RuntimeError(f"No rows found in {csv_path}")
            self.problem_indices = self._collect_problem_indices()
            self.blur_median = self._compute_blur_median()

            self.index = 0
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

        def _blur_score(self, row: Dict[str, str]) -> float:
            """blur_score_final を float で返す。取得できなければ inf。"""
            try:
                return float(row.get("blur_score_final", "inf"))
            except (ValueError, TypeError):
                return float("inf")

        def _compute_blur_median(self) -> float:
            """blur_score_final の中央値。情報表示用 (advisory 判定には使わない)。"""
            scores = [self._blur_score(row) for row in self.rows]
            valid = sorted(s for s in scores if s != float("inf"))
            if not valid:
                return 0.0
            n = len(valid)
            if n % 2 == 0:
                return (valid[n // 2 - 1] + valid[n // 2]) / 2.0
            return valid[n // 2]

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
            layout.addLayout(top_row)

            self.image_view = ZoomableImageView()
            self.image_view.setMinimumHeight(260)
            layout.addWidget(self.image_view, stretch=1)

            # アドバイザリー（このフレームが要注意な理由を自動表示）
            self.advisory_label = QLabel()
            self.advisory_label.setWordWrap(True)
            self.advisory_label.setStyleSheet(
                "padding: 6px 10px; border-radius: 4px; font-weight: 600; font-size: 10pt;"
            )
            self.advisory_label.setMinimumHeight(32)
            layout.addWidget(self.advisory_label)

            self.info_label = QLabel()
            self.info_label.setWordWrap(True)
            layout.addWidget(self.info_label)

            self.problem_summary_label = QLabel()
            self.problem_summary_label.setStyleSheet("color: #666;")
            layout.addWidget(self.problem_summary_label)

            btn_row = QHBoxLayout()
            self.prev_button = QPushButton(i18n.t("REVIEW_BTN_PREV"))
            self.prev_button.clicked.connect(self.prev_row)
            btn_row.addWidget(self.prev_button)

            self.next_button = QPushButton(i18n.t("REVIEW_BTN_NEXT"))
            self.next_button.clicked.connect(self.next_row)
            btn_row.addWidget(self.next_button)

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

            action_row = QHBoxLayout()
            self.toggle_button = QPushButton(i18n.t("REVIEW_BTN_TOGGLE"))
            self.toggle_button.clicked.connect(self.toggle_decision)
            action_row.addWidget(self.toggle_button)

            self.jump_edit = QLineEdit()
            self.jump_edit.setPlaceholderText(i18n.t("REVIEW_JUMP_PLACEHOLDER"))
            self.jump_edit.setFixedWidth(90)
            action_row.addWidget(self.jump_edit)

            self.jump_button = QPushButton(i18n.t("REVIEW_BTN_JUMP"))
            self.jump_button.clicked.connect(self.jump_to_seq)
            action_row.addWidget(self.jump_button)

            action_row.addStretch(1)

            self.save_button = QPushButton(i18n.t("REVIEW_BTN_SAVE"))
            self.save_button.clicked.connect(self.save)
            action_row.addWidget(self.save_button)
            layout.addLayout(action_row)

            # ブレスコアによる一括除外行（手動絶対閾値ツール）
            blur_row = QHBoxLayout()
            blur_row.addWidget(QLabel(i18n.t("REVIEW_BLUR_THRESHOLD_LABEL")))
            self.blur_threshold_edit = QLineEdit()
            self.blur_threshold_edit.setPlaceholderText(i18n.t("REVIEW_BLUR_THRESHOLD_PLACEHOLDER"))
            self.blur_threshold_edit.setFixedWidth(100)
            blur_row.addWidget(self.blur_threshold_edit)

            self.blur_drop_button = QPushButton(i18n.t("REVIEW_BLUR_DROP_BTN"))
            self.blur_drop_button.clicked.connect(self.drop_below_blur_threshold)
            blur_row.addWidget(self.blur_drop_button)

            blur_row.addStretch(1)
            layout.addLayout(blur_row)

            # 使い方ヘルプ（必要な時だけ展開）
            self.help_section = CollapsibleSection(i18n.t("REVIEW_HELP_HEADER"), expanded=False)
            self.help_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

            help_body = QLabel(i18n.t("REVIEW_HELP_BODY"))
            help_body.setStyleSheet(
                "color: #888; font-size: 9pt; padding: 4px 8px; "
                "background-color: #1a1a2e; border-radius: 6px;"
            )
            help_body.setWordWrap(True)

            help_scroll = QScrollArea()
            help_scroll.setWidgetResizable(True)
            help_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            help_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            help_scroll.setMinimumHeight(120)
            help_scroll.setMaximumHeight(180)
            help_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            help_scroll.setStyleSheet("QScrollArea { border: 0; background: transparent; }")
            help_scroll.setWidget(help_body)

            self.help_section.content_layout.addWidget(help_scroll)
            layout.addWidget(self.help_section)

        def _bind_shortcuts(self) -> None:
            QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.prev_row)
            QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.next_row)
            QShortcut(QKeySequence("F"), self, activated=self.next_problem)
            QShortcut(QKeySequence("Shift+F"), self, activated=self.prev_problem)
            QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_decision)
            QShortcut(QKeySequence("S"), self, activated=self.save)
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

        def _format_process(self, row: Dict[str, str]) -> str:
            status = row.get("status", "ok").strip().lower()
            orig = row.get("original_index", "?")
            final = row.get("final_index", "?")
            if "fallback_keep" in status:
                return i18n.t("REVIEW_PROCESS_FALLBACK")
            if "thinned" in status:
                return i18n.t("REVIEW_PROCESS_THINNED")
            if "replaced" in status:
                return i18n.t("REVIEW_PROCESS_REPLACED").format(orig=orig, final=final)
            return i18n.t("REVIEW_PROCESS_OK")

        def _advisory_for_row(self, row: Dict[str, str], idx: int) -> tuple[str, str, str]:
            """status のみに基づく advisory。

            フレーム抽出時にブレ判定 + 置換は済んでいるので、
            置換不可のフレームだけを強く確認対象として扱う。
            「ブレ top X%」のような相対順位 advisory は出さない（誤判定の元）。

            Returns (text, fg, bg).
            """
            status = row.get("status", "ok").strip().lower()

            # 橙: ブレを検出したが置換先がなかったフレーム
            if "fallback_keep" in status:
                return i18n.t("REVIEW_ADVISORY_FALLBACK"), "#fef3c7", "#7c2d12"

            # 青: 自動間引き
            if "thinned" in status:
                return i18n.t("REVIEW_ADVISORY_THINNED"), "#dbeafe", "#1e3a8a"

            # 青: 自動置換済み
            if "replaced" in status:
                return i18n.t("REVIEW_ADVISORY_REPLACED"), "#dbeafe", "#1e3a8a"

            # 緑: 通常品質
            return i18n.t("REVIEW_ADVISORY_NORMAL"), "#a7f3d0", "#064e3b"

        def _render_current(self) -> None:
            row = self._current_row()
            seq = int(row.get("seq", self.index + 1))
            total = len(self.rows)

            image_rel = row.get("output_file", "")
            image_path = self.scene_dir / image_rel

            self.title_label.setText(f"{seq}/{total}  {image_rel}")

            decision = row.get("decision", "keep")
            self.decision_label.setText(f"{i18n.t('REVIEW_DECISION_PREFIX')}{self._decision_text(decision)}")
            self.decision_label.setStyleSheet(f"font-weight: 700; color: {self._decision_color(decision)};")

            # アドバイザリー (このフレームが要注意な理由)
            adv_text, adv_fg, adv_bg = self._advisory_for_row(row, self.index)
            self.advisory_label.setText(adv_text)
            self.advisory_label.setStyleSheet(
                f"padding: 6px 10px; border-radius: 4px; font-weight: 600; font-size: 10pt; "
                f"color: {adv_fg}; background-color: {adv_bg};"
            )

            replaced_count = sum(1 for r in self.rows if "replaced" in r.get("status", "").strip().lower())
            fallback_count = sum(1 for r in self.rows if "fallback_keep" in r.get("status", "").strip().lower())
            thinned_count = sum(1 for r in self.rows if "thinned" in r.get("status", "").strip().lower())
            problem_count = len(self.problem_indices)
            current_problem = i18n.t("REVIEW_INFO_YES") if self._is_problem_row(row) else i18n.t("REVIEW_INFO_NO")
            self.problem_summary_label.setText(
                i18n.t("REVIEW_PROBLEMS_FORMAT").format(
                    n=problem_count, r=replaced_count, f=fallback_count, t=thinned_count, cur=current_problem
                )
            )

            # ブレ表示: 中央値比で文脈を与える（情報のみ。判定には使わない）
            blur_final = self._blur_score(row)
            if blur_final != float("inf"):
                if self.blur_median > 0:
                    pct = round(blur_final / self.blur_median * 100)
                    blur_str = i18n.t("REVIEW_BLUR_VALUE_FORMAT").format(
                        score=blur_final, median=self.blur_median, pct=pct
                    )
                else:
                    blur_str = i18n.t("REVIEW_BLUR_VALUE_NO_MEDIAN").format(score=blur_final)
            else:
                blur_str = "-"

            # 撮影時刻
            ts_raw = row.get("timestamp_sec", "-")
            try:
                ts_str = f"{float(ts_raw):.2f}s"
            except (ValueError, TypeError):
                ts_str = ts_raw

            # 変化スコア
            change_raw = row.get("change_score_final", "-")
            try:
                change_str = f"{float(change_raw):.3f}"
            except (ValueError, TypeError):
                change_str = change_raw

            info_text = i18n.t("REVIEW_INFO_FORMAT").format(
                ts=ts_str, blur=blur_str, change=change_str,
                process=self._format_process(row),
            )
            self.info_label.setText(info_text)

            if not image_path.exists():
                self.current_pixmap = None
                self.image_view.show_message(i18n.t("REVIEW_IMAGE_NOT_FOUND").format(path=image_path))
                return

            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                self.current_pixmap = None
                self.image_view.show_message(i18n.t("REVIEW_IMAGE_LOAD_FAILED").format(path=image_path))
                return

            self.current_pixmap = pixmap
            self.image_view.show_pixmap(pixmap)

        def reset_zoom(self) -> None:
            self.image_view.reset_zoom()

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

        def jump_to_seq(self) -> None:
            text = self.jump_edit.text().strip()
            if not text:
                return

            try:
                seq = int(text)
            except ValueError:
                QMessageBox.warning(
                    self, i18n.t("REVIEW_INVALID_INPUT"), i18n.t("REVIEW_SEQ_INTEGER_ERR")
                )
                return

            if seq < 1 or seq > len(self.rows):
                QMessageBox.warning(
                    self, i18n.t("REVIEW_INVALID_INPUT"),
                    i18n.t("REVIEW_OUT_OF_RANGE_ERR").format(max=len(self.rows)),
                )
                return

            self.index = seq - 1
            self._render_current()

        def drop_below_blur_threshold(self) -> None:
            """閾値以下のblur_score_finalを持つフレームを一括で除外扱いにする。"""
            text = self.blur_threshold_edit.text().strip()
            if not text:
                # 未入力なら、現在のフレームのスコアを閾値として提案
                score = self._blur_score(self._current_row())
                if score != float("inf"):
                    self.blur_threshold_edit.setText(f"{score:.1f}")
                QMessageBox.information(
                    self, i18n.t("REVIEW_THRESHOLD_HEADER"),
                    i18n.t("REVIEW_THRESHOLD_NEED_INPUT"),
                )
                return

            try:
                threshold = float(text)
            except ValueError:
                QMessageBox.warning(
                    self, i18n.t("REVIEW_INVALID_INPUT"), i18n.t("REVIEW_THRESHOLD_NUMERIC_ERR")
                )
                return

            count = 0
            for row in self.rows:
                score = self._blur_score(row)
                if score <= threshold and row.get("decision") != "drop":
                    row["decision"] = "drop"
                    count += 1

            self._render_current()
            QMessageBox.information(
                self, i18n.t("REVIEW_BULK_DROP_HEADER"),
                i18n.t("REVIEW_BULK_DROP_RESULT").format(thr=f"{threshold:.1f}", n=count),
            )

        def toggle_decision(self) -> None:
            row = self._current_row()
            row["decision"] = "drop" if row.get("decision", "keep") == "keep" else "keep"
            self._render_current()

        def save(self) -> None:
            if not self.rows:
                return

            fieldnames = list(self.rows[0].keys())
            with self.csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)

            keep_count = sum(1 for r in self.rows if r.get("decision") != "drop")
            drop_count = len(self.rows) - keep_count
            QMessageBox.information(
                self,
                i18n.t("REVIEW_SAVED_HEADER"),
                i18n.t("REVIEW_SAVED_BODY").format(
                    path=str(self.csv_path), k=keep_count, d=drop_count
                ),
            )


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
