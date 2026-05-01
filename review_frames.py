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
    QVBoxLayout = None
    QWidget = None
    _PYSIDE_IMPORT_ERROR = e
else:
    _PYSIDE_IMPORT_ERROR = None

# i18n は PySide6 に依存しないので無条件 import
from gui import i18n


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
            self.setMinimumHeight(560)
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
    class ReviewWindow(QMainWindow):
        def __init__(self, scene_dir: Path, csv_path: Path) -> None:
            super().__init__()
            self.scene_dir = scene_dir
            self.csv_path = csv_path
            self.rows = self._load_rows(csv_path)
            if not self.rows:
                raise RuntimeError(f"No rows found in {csv_path}")
            self.problem_indices = self._collect_problem_indices()
            self.blur_worst_indices = self._build_blur_worst_indices()
            self._blur_worst_cursor = 0

            self.index = 0
            self.current_pixmap: QPixmap | None = None

            self.setWindowTitle(i18n.t("REVIEW_TITLE"))
            self.resize(1280, 860)

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

        def _build_blur_worst_indices(self) -> List[int]:
            """ブラースコア昇順 (ワースト=最小が先) のインデックスリスト。"""
            scored = [(i, self._blur_score(row)) for i, row in enumerate(self.rows)]
            scored.sort(key=lambda x: x[1])
            return [i for i, _ in scored]

        def _build_ui(self) -> None:
            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)

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
            self.prev_problem_button.clicked.connect(self.prev_problem)
            btn_row.addWidget(self.prev_problem_button)

            self.next_problem_button = QPushButton(i18n.t("REVIEW_BTN_NEXT_PROBLEM"))
            self.next_problem_button.clicked.connect(self.next_problem)
            btn_row.addWidget(self.next_problem_button)

            self.toggle_button = QPushButton(i18n.t("REVIEW_BTN_TOGGLE"))
            self.toggle_button.clicked.connect(self.toggle_decision)
            btn_row.addWidget(self.toggle_button)

            self.jump_edit = QLineEdit()
            self.jump_edit.setPlaceholderText(i18n.t("REVIEW_JUMP_PLACEHOLDER"))
            self.jump_edit.setFixedWidth(90)
            btn_row.addWidget(self.jump_edit)

            self.jump_button = QPushButton(i18n.t("REVIEW_BTN_JUMP"))
            self.jump_button.clicked.connect(self.jump_to_seq)
            btn_row.addWidget(self.jump_button)

            btn_row.addStretch(1)

            self.save_button = QPushButton(i18n.t("REVIEW_BTN_SAVE"))
            self.save_button.clicked.connect(self.save)
            btn_row.addWidget(self.save_button)
            layout.addLayout(btn_row)

            # ブラー操作行
            blur_row = QHBoxLayout()
            blur_row.addWidget(QLabel(i18n.t("REVIEW_BLUR_CYCLE_LABEL")))

            self.blur_worst_button = QPushButton(i18n.t("REVIEW_BTN_BLUR_WORST"))
            self.blur_worst_button.setToolTip(i18n.t("REVIEW_BTN_BLUR_WORST_TIP"))
            self.blur_worst_button.clicked.connect(self.next_blur_worst)
            blur_row.addWidget(self.blur_worst_button)

            self.blur_prev_button = QPushButton(i18n.t("REVIEW_BTN_BLUR_PREV"))
            self.blur_prev_button.setToolTip(i18n.t("REVIEW_BTN_BLUR_PREV_TIP"))
            self.blur_prev_button.clicked.connect(self.prev_blur_worst)
            blur_row.addWidget(self.blur_prev_button)

            blur_row.addWidget(QLabel(i18n.t("REVIEW_BLUR_THRESHOLD_LABEL")))
            self.blur_threshold_edit = QLineEdit()
            self.blur_threshold_edit.setPlaceholderText(i18n.t("REVIEW_BLUR_THRESHOLD_PLACEHOLDER"))
            self.blur_threshold_edit.setFixedWidth(100)
            blur_row.addWidget(self.blur_threshold_edit)

            self.blur_drop_button = QPushButton(i18n.t("REVIEW_BLUR_DROP_BTN"))
            self.blur_drop_button.clicked.connect(self.drop_below_blur_threshold)
            blur_row.addWidget(self.blur_drop_button)

            blur_row.addStretch(1)

            self.blur_rank_label = QLabel()
            self.blur_rank_label.setStyleSheet("color: #888;")
            blur_row.addWidget(self.blur_rank_label)
            layout.addLayout(blur_row)

            # 使い方ヘルプ（折りたたみ風: header + body）
            help_header = QLabel(i18n.t("REVIEW_HELP_HEADER"))
            help_header.setStyleSheet(
                "color: #c4b5fd; font-weight: 700; padding-top: 6px;"
            )
            layout.addWidget(help_header)
            help_body = QLabel(i18n.t("REVIEW_HELP_BODY"))
            help_body.setStyleSheet(
                "color: #888; font-size: 9pt; padding: 4px 8px; "
                "background-color: #1a1a2e; border-radius: 6px;"
            )
            help_body.setWordWrap(True)
            layout.addWidget(help_body)

        def _bind_shortcuts(self) -> None:
            QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.prev_row)
            QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.next_row)
            QShortcut(QKeySequence("F"), self, activated=self.next_problem)
            QShortcut(QKeySequence("Shift+F"), self, activated=self.prev_problem)
            QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_decision)
            QShortcut(QKeySequence("B"), self, activated=self.next_blur_worst)
            QShortcut(QKeySequence("Shift+B"), self, activated=self.prev_blur_worst)
            QShortcut(QKeySequence("S"), self, activated=self.save)
            QShortcut(QKeySequence("Q"), self, activated=self.close)
            QShortcut(QKeySequence("0"), self, activated=self.reset_zoom)

        def _current_row(self) -> Dict[str, str]:
            return self.rows[self.index]

        def _decision_color(self, decision: str) -> str:
            return "#b00020" if decision == "drop" else "#1b7f3b"

        def _advisory_for_row(self, row: Dict[str, str], idx: int) -> tuple[str, str, str]:
            """この行の要注意度を判定。 (text, fg, bg) を返す。

            最も重大な問題を 1 つ表示する（複数並べると見にくいため）。
            """
            status = row.get("status", "ok").strip().lower()

            # ブレ順位を計算
            rank: int = -1
            total = len(self.blur_worst_indices)
            try:
                rank = self.blur_worst_indices.index(idx) + 1
            except ValueError:
                pass

            # 優先度順に判定
            # 1. ブレ最悪 top 5% は最強警告（赤）
            if rank > 0 and total > 0 and rank <= max(1, int(total * 0.05)):
                text = i18n.t("REVIEW_ADVISORY_BLUR_TOP5").format(rank=rank, total=total)
                return text, "#fee2e2", "#7f1d1d"  # 明るい赤文字 / 濃い赤背景

            # 2. fallback_keep = 置換できない blurry フレーム（橙）
            if "fallback_keep" in status:
                return i18n.t("REVIEW_ADVISORY_FALLBACK"), "#fef3c7", "#7c2d12"

            # 3. ブレ top 15% （橙）
            if rank > 0 and total > 0 and rank <= max(1, int(total * 0.15)):
                text = i18n.t("REVIEW_ADVISORY_BLUR_TOP15").format(rank=rank, total=total)
                return text, "#fef3c7", "#7c2d12"

            # 4. thinned = 既に間引き済み（青／情報）
            if "thinned" in status:
                return i18n.t("REVIEW_ADVISORY_THINNED"), "#dbeafe", "#1e3a8a"

            # 5. replaced = 自動置換済み（青／情報）
            if "replaced" in status:
                return i18n.t("REVIEW_ADVISORY_REPLACED"), "#dbeafe", "#1e3a8a"

            # 6. 問題なし
            return i18n.t("REVIEW_ADVISORY_NORMAL"), "#a7f3d0", "#064e3b"

        def _render_current(self) -> None:
            row = self._current_row()
            seq = int(row.get("seq", self.index + 1))
            total = len(self.rows)

            image_rel = row.get("output_file", "")
            image_path = self.scene_dir / image_rel

            self.title_label.setText(f"{seq}/{total}  {image_rel}")

            decision = row.get("decision", "keep")
            self.decision_label.setText(f"{i18n.t('REVIEW_DECISION_PREFIX')}{decision.upper()}")
            self.decision_label.setStyleSheet(f"font-weight: 700; color: {self._decision_color(decision)};")

            # アドバイザリー (このフレームが要注意な理由)
            adv_text, adv_fg, adv_bg = self._advisory_for_row(row, self.index)
            self.advisory_label.setText(adv_text)
            self.advisory_label.setStyleSheet(
                f"padding: 6px 10px; border-radius: 4px; font-weight: 600; font-size: 10pt; "
                f"color: {adv_fg}; background-color: {adv_bg};"
            )

            replaced_count = sum(1 for r in self.rows if r.get("status", "").strip().lower() == "replaced")
            fallback_count = sum(1 for r in self.rows if r.get("status", "").strip().lower() == "fallback_keep")
            problem_count = len(self.problem_indices)
            current_problem = i18n.t("REVIEW_INFO_YES") if self._is_problem_row(row) else i18n.t("REVIEW_INFO_NO")
            self.problem_summary_label.setText(
                i18n.t("REVIEW_PROBLEMS_FORMAT").format(
                    n=problem_count, r=replaced_count, f=fallback_count, cur=current_problem
                )
            )

            blur_final = self._blur_score(row)
            blur_str = f"{blur_final:.1f}" if blur_final != float("inf") else "-"
            info_text = (
                f"orig={row.get('original_index', '-')}, final={row.get('final_index', '-')}, "
                f"ts={row.get('timestamp_sec', '-')}, status={row.get('status', '-')}, "
                f"blur(orig/final)={row.get('blur_score_original', '-')}/{row.get('blur_score_final', '-')}, "
                f"change(orig/final)={row.get('change_score_original', '-')}/{row.get('change_score_final', '-')}"
            )
            self.info_label.setText(info_text)

            # ブラーランク表示
            try:
                rank = self.blur_worst_indices.index(self.index) + 1
            except ValueError:
                rank = -1
            total = len(self.rows)
            self.blur_rank_label.setText(
                i18n.t("REVIEW_BLUR_RANK_FORMAT").format(rank=rank, total=total, score=blur_str)
            )

            if not image_path.exists():
                self.current_pixmap = None
                self.image_view.show_message(f"Image not found:\n{image_path}")
                return

            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                self.current_pixmap = None
                self.image_view.show_message(f"Failed to load image:\n{image_path}")
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

        def next_blur_worst(self) -> None:
            """ブラースコアが悪い順に次のフレームに飛ぶ。"""
            if not self.blur_worst_indices:
                return
            # 現在位置のランクを探して次へ
            try:
                cur_rank = self.blur_worst_indices.index(self.index)
                next_rank = cur_rank + 1
            except ValueError:
                next_rank = 0
            if next_rank >= len(self.blur_worst_indices):
                next_rank = 0
            self._blur_worst_cursor = next_rank
            self.index = self.blur_worst_indices[next_rank]
            self._render_current()

        def prev_blur_worst(self) -> None:
            """ブラースコアが悪い順に前のフレームに飛ぶ。"""
            if not self.blur_worst_indices:
                return
            try:
                cur_rank = self.blur_worst_indices.index(self.index)
                prev_rank = cur_rank - 1
            except ValueError:
                prev_rank = 0
            if prev_rank < 0:
                prev_rank = len(self.blur_worst_indices) - 1
            self._blur_worst_cursor = prev_rank
            self.index = self.blur_worst_indices[prev_rank]
            self._render_current()

        def drop_below_blur_threshold(self) -> None:
            """閾値以下のblur_score_finalを持つフレームを一括drop。"""
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

else:
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
