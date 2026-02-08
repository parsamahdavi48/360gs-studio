#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
    from PySide6.QtWidgets import (
        QApplication,
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
    QPixmap = None
    QShortcut = None
    QApplication = None
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

            self.index = 0
            self.current_pixmap: QPixmap | None = None

            self.setWindowTitle("Frame Review")
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

            self.image_label = QLabel()
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setStyleSheet("background-color: #101010; border: 1px solid #333;")
            self.image_label.setMinimumHeight(560)
            layout.addWidget(self.image_label, stretch=1)

            self.info_label = QLabel()
            self.info_label.setWordWrap(True)
            layout.addWidget(self.info_label)

            self.problem_summary_label = QLabel()
            self.problem_summary_label.setStyleSheet("color: #666;")
            layout.addWidget(self.problem_summary_label)

            btn_row = QHBoxLayout()
            self.prev_button = QPushButton("Prev (Left)")
            self.prev_button.clicked.connect(self.prev_row)
            btn_row.addWidget(self.prev_button)

            self.next_button = QPushButton("Next (Right)")
            self.next_button.clicked.connect(self.next_row)
            btn_row.addWidget(self.next_button)

            self.prev_problem_button = QPushButton("Prev Problem (Shift+F)")
            self.prev_problem_button.clicked.connect(self.prev_problem)
            btn_row.addWidget(self.prev_problem_button)

            self.next_problem_button = QPushButton("Next Problem (F)")
            self.next_problem_button.clicked.connect(self.next_problem)
            btn_row.addWidget(self.next_problem_button)

            self.toggle_button = QPushButton("Toggle Keep/Drop (Space)")
            self.toggle_button.clicked.connect(self.toggle_decision)
            btn_row.addWidget(self.toggle_button)

            self.jump_edit = QLineEdit()
            self.jump_edit.setPlaceholderText("seq")
            self.jump_edit.setFixedWidth(90)
            btn_row.addWidget(self.jump_edit)

            self.jump_button = QPushButton("Jump Seq")
            self.jump_button.clicked.connect(self.jump_to_seq)
            btn_row.addWidget(self.jump_button)

            btn_row.addStretch(1)

            self.save_button = QPushButton("Save (S)")
            self.save_button.clicked.connect(self.save)
            btn_row.addWidget(self.save_button)
            layout.addLayout(btn_row)

            hint = QLabel(
                "Keys: Left/Right=move, F/Shift+F=next/prev problem, "
                "Space=toggle keep/drop, S=save, Q=quit"
            )
            hint.setStyleSheet("color: #666;")
            layout.addWidget(hint)

        def _bind_shortcuts(self) -> None:
            QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.prev_row)
            QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.next_row)
            QShortcut(QKeySequence("F"), self, activated=self.next_problem)
            QShortcut(QKeySequence("Shift+F"), self, activated=self.prev_problem)
            QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_decision)
            QShortcut(QKeySequence("S"), self, activated=self.save)
            QShortcut(QKeySequence("Q"), self, activated=self.close)

        def _current_row(self) -> Dict[str, str]:
            return self.rows[self.index]

        def _decision_color(self, decision: str) -> str:
            return "#b00020" if decision == "drop" else "#1b7f3b"

        def _render_current(self) -> None:
            row = self._current_row()
            seq = int(row.get("seq", self.index + 1))
            total = len(self.rows)

            image_rel = row.get("output_file", "")
            image_path = self.scene_dir / image_rel

            self.title_label.setText(f"{seq}/{total}  {image_rel}")

            decision = row.get("decision", "keep")
            self.decision_label.setText(f"Decision: {decision.upper()}")
            self.decision_label.setStyleSheet(f"font-weight: 700; color: {self._decision_color(decision)};")

            replaced_count = sum(1 for r in self.rows if r.get("status", "").strip().lower() == "replaced")
            fallback_count = sum(1 for r in self.rows if r.get("status", "").strip().lower() == "fallback_keep")
            problem_count = len(self.problem_indices)
            current_problem = "YES" if self._is_problem_row(row) else "NO"
            self.problem_summary_label.setText(
                f"Problems: {problem_count} (replaced={replaced_count}, fallback_keep={fallback_count})"
                f" | Current problem: {current_problem}"
            )

            info_text = (
                f"orig={row.get('original_index', '-')}, final={row.get('final_index', '-')}, "
                f"ts={row.get('timestamp_sec', '-')}, status={row.get('status', '-')}, "
                f"blur(orig/final)={row.get('blur_score_original', '-')}/{row.get('blur_score_final', '-')}, "
                f"change(orig/final)={row.get('change_score_original', '-')}/{row.get('change_score_final', '-')}"
            )
            self.info_label.setText(info_text)

            if not image_path.exists():
                self.current_pixmap = None
                self.image_label.setText(f"Image not found:\n{image_path}")
                self.image_label.setPixmap(QPixmap())
                return

            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                self.current_pixmap = None
                self.image_label.setText(f"Failed to load image:\n{image_path}")
                self.image_label.setPixmap(QPixmap())
                return

            self.current_pixmap = pixmap
            self._update_pixmap_view()

        def _update_pixmap_view(self) -> None:
            if self.current_pixmap is None:
                return

            target_size = self.image_label.size()
            if target_size.width() <= 1 or target_size.height() <= 1:
                return

            scaled = self.current_pixmap.scaled(
                target_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.image_label.setText("")
            self.image_label.setPixmap(scaled)

        def resizeEvent(self, event) -> None:  # pragma: no cover - UI event
            super().resizeEvent(event)
            self._update_pixmap_view()

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
                QMessageBox.information(self, "Info", "No problem frames found.")
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
                QMessageBox.information(self, "Info", "No problem frames found.")
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
                QMessageBox.warning(self, "Invalid Input", "Seq must be an integer.")
                return

            if seq < 1 or seq > len(self.rows):
                QMessageBox.warning(self, "Out of Range", f"Seq must be between 1 and {len(self.rows)}.")
                return

            self.index = seq - 1
            self._render_current()

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
                "Saved",
                f"Updated {self.csv_path}\nkeep={keep_count}, drop={drop_count}",
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
