#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

try:
    from PySide6.QtCore import QProcess, Qt
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QCheckBox,
        QComboBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except Exception as e:  # pragma: no cover - environment-dependent import
    QProcess = None
    Qt = None
    QCloseEvent = None
    QApplication = None
    QFileDialog = None
    QCheckBox = None
    QComboBox = None
    QFormLayout = None
    QHBoxLayout = None
    QLabel = None
    QLineEdit = None
    QMainWindow = None
    QMessageBox = None
    QPlainTextEdit = None
    QProgressBar = None
    QPushButton = None
    QVBoxLayout = None
    QWidget = None
    _PYSIDE_IMPORT_ERROR = e
else:
    _PYSIDE_IMPORT_ERROR = None


if QMainWindow is not None:
    _YOLO_LINE_RE = re.compile(r"^Processing:\s+")
    _STITCH_TASK_RE = re.compile(r"^Processing\s+(\d+)\s+images\s+with\s+\d+\s+workers\.\.\.$")
    _STITCH_TQDM_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")

    class MaskToolsWindow(QMainWindow):
        def __init__(self, initial_scene_dir: str | None = None) -> None:
            super().__init__()
            self.base_dir = Path(__file__).resolve().parent
            self.proc: QProcess | None = None
            self.current_phase = ""
            self._process_buffer = ""
            self.pending_steps: list[tuple[str, list[str]]] = []
            self.phase_total_items = 0
            self.phase_done_items = 0
            self._stitch_chunk_total = 0
            self._stitch_chunk_done = 0
            self._stitch_done_before_chunk = 0

            self.setWindowTitle("Mask Tools")
            self.resize(1120, 820)

            self._build_ui(initial_scene_dir or ".")
            self._apply_scene_paths()
            self._refresh_action_buttons()

        def _build_ui(self, initial_scene_dir: str) -> None:
            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)

            form_layout = QFormLayout()

            self.scene_dir_edit = QLineEdit(initial_scene_dir)
            self.scene_dir_edit.textChanged.connect(lambda _: self._refresh_action_buttons())
            browse_scene_btn = QPushButton("Browse")
            browse_scene_btn.clicked.connect(self._browse_scene_dir)
            apply_scene_btn = QPushButton("Apply Scene Paths")
            apply_scene_btn.clicked.connect(self._apply_scene_paths)
            row = QHBoxLayout()
            row.addWidget(self.scene_dir_edit)
            row.addWidget(browse_scene_btn)
            row.addWidget(apply_scene_btn)
            form_layout.addRow("Scene Directory", row)

            self.images_dir_edit = QLineEdit()
            self.images_dir_edit.textChanged.connect(lambda _: self._refresh_action_buttons())
            browse_images_btn = QPushButton("Browse")
            browse_images_btn.clicked.connect(self._browse_images_dir)
            row = QHBoxLayout()
            row.addWidget(self.images_dir_edit)
            row.addWidget(browse_images_btn)
            form_layout.addRow("Images Directory", row)

            self.masks_dir_edit = QLineEdit()
            self.masks_dir_edit.textChanged.connect(lambda _: self._refresh_action_buttons())
            browse_masks_btn = QPushButton("Browse")
            browse_masks_btn.clicked.connect(self._browse_masks_dir)
            row = QHBoxLayout()
            row.addWidget(self.masks_dir_edit)
            row.addWidget(browse_masks_btn)
            form_layout.addRow("Masks Directory", row)

            self.yolo_level_combo = QComboBox()
            self.yolo_level_combo.addItems(["0", "1", "2", "3"])
            self.yolo_level_combo.setCurrentText("1")
            form_layout.addRow("YOLO Level", self.yolo_level_combo)

            self.yolo_expand_edit = QLineEdit("2")
            form_layout.addRow("YOLO Expand (px)", self.yolo_expand_edit)

            self.yolo_add_ext_check = QCheckBox("Add original extension in output name (--add_ext)")
            self.yolo_add_ext_check.setChecked(False)
            form_layout.addRow("YOLO Add Ext", self.yolo_add_ext_check)

            self.stitch_fov_edit = QLineEdit("175.0")
            form_layout.addRow("Stitch FOV (deg)", self.stitch_fov_edit)

            self.stitch_workers_edit = QLineEdit(str(os.cpu_count() or 4))
            form_layout.addRow("Stitch Workers", self.stitch_workers_edit)

            layout.addLayout(form_layout)

            btn_row = QHBoxLayout()
            self.run_yolo_button = QPushButton("1) Run YOLO Mask")
            self.run_yolo_button.clicked.connect(self._run_yolo)
            btn_row.addWidget(self.run_yolo_button)

            self.run_stitch_button = QPushButton("2) Run Stitch Mask")
            self.run_stitch_button.clicked.connect(self._run_stitch)
            btn_row.addWidget(self.run_stitch_button)

            self.run_both_button = QPushButton("3) Run YOLO + Stitch")
            self.run_both_button.clicked.connect(self._run_both)
            btn_row.addWidget(self.run_both_button)

            btn_row.addStretch(1)

            self.status_label = QLabel("Idle")
            self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            btn_row.addWidget(self.status_label)

            layout.addLayout(btn_row)

            progress_row = QHBoxLayout()
            self.progress_bar = QProgressBar()
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%")
            progress_row.addWidget(self.progress_bar, stretch=1)

            self.progress_text_label = QLabel("0 / 0")
            self.progress_text_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            progress_row.addWidget(self.progress_text_label)

            layout.addLayout(progress_row)

            self.log_text = QPlainTextEdit()
            self.log_text.setReadOnly(True)
            layout.addWidget(self.log_text, stretch=1)

        def _is_running(self) -> bool:
            return self.proc is not None and self.proc.state() != QProcess.NotRunning

        def _append_log(self, text: str) -> None:
            self.log_text.appendPlainText(text)
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        def _set_running_state(self, running: bool, status_text: str) -> None:
            self.status_label.setText(status_text)
            if running:
                self.run_yolo_button.setEnabled(False)
                self.run_stitch_button.setEnabled(False)
                self.run_both_button.setEnabled(False)
            else:
                self._refresh_action_buttons()

        def _count_matching_files(self, root: Path, suffixes: tuple[str, ...]) -> int:
            if not root.is_dir():
                return 0
            return sum(
                1
                for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in suffixes
            )

        def _set_phase_progress(self, done: int, total: int) -> None:
            done = max(0, done)
            total = max(0, total)
            self.phase_done_items = done
            self.phase_total_items = total

            if total > 0:
                self.progress_bar.setMinimum(0)
                self.progress_bar.setMaximum(total)
                self.progress_bar.setValue(min(done, total))
                self.progress_bar.setFormat("%v / %m (%p%)")
                self.progress_text_label.setText(f"{min(done, total)} / {total}")
                if self._is_running():
                    self.status_label.setText(
                        f"Running: {self.current_phase} ({min(done, total)}/{total})"
                    )
            else:
                self.progress_bar.setMinimum(0)
                self.progress_bar.setMaximum(100)
                self.progress_bar.setValue(0)
                self.progress_bar.setFormat("%p%")
                self.progress_text_label.setText("0 / 0")
                if self._is_running():
                    self.status_label.setText(f"Running: {self.current_phase}")

        def _init_progress_for_phase(self, phase: str) -> None:
            self._stitch_chunk_total = 0
            self._stitch_chunk_done = 0
            self._stitch_done_before_chunk = 0

            if phase == "yolo":
                total = self._count_matching_files(
                    Path(self.images_dir_edit.text().strip()),
                    (".jpg", ".jpeg", ".png"),
                )
                self._set_phase_progress(0, total)
                return

            if phase == "stitch":
                total = self._count_matching_files(
                    Path(self.masks_dir_edit.text().strip()),
                    (".png",),
                )
                self._set_phase_progress(0, total)
                return

            self._set_phase_progress(0, 0)

        def _track_progress_from_line(self, line: str) -> None:
            if self.current_phase == "yolo":
                if _YOLO_LINE_RE.match(line):
                    next_done = self.phase_done_items + 1
                    self._set_phase_progress(next_done, self.phase_total_items)
                return

            if self.current_phase != "stitch":
                return

            task_match = _STITCH_TASK_RE.match(line)
            if task_match:
                chunk_total = int(task_match.group(1))
                if self._stitch_chunk_total > 0:
                    self._stitch_done_before_chunk += self._stitch_chunk_total
                self._stitch_chunk_total = chunk_total
                self._stitch_chunk_done = 0
                self._set_phase_progress(
                    self._stitch_done_before_chunk,
                    self.phase_total_items,
                )
                return

            tqdm_match = _STITCH_TQDM_RE.search(line)
            if not tqdm_match:
                return

            chunk_done = int(tqdm_match.group(1))
            chunk_total = int(tqdm_match.group(2))
            if chunk_total <= 0:
                return

            if self._stitch_chunk_total == 0:
                self._stitch_chunk_total = chunk_total

            # tqdm output can reset when moving to the next directory.
            if chunk_done < self._stitch_chunk_done:
                self._stitch_done_before_chunk += self._stitch_chunk_total
                self._stitch_chunk_total = chunk_total

            self._stitch_chunk_done = chunk_done
            self._set_phase_progress(
                self._stitch_done_before_chunk + self._stitch_chunk_done,
                self.phase_total_items,
            )

        def _refresh_action_buttons(self) -> None:
            if self._is_running():
                self.run_yolo_button.setEnabled(False)
                self.run_stitch_button.setEnabled(False)
                self.run_both_button.setEnabled(False)
                return

            has_images = Path(self.images_dir_edit.text().strip()).is_dir()
            has_masks = Path(self.masks_dir_edit.text().strip()).is_dir()
            self.run_yolo_button.setEnabled(has_images)
            self.run_stitch_button.setEnabled(has_masks)
            self.run_both_button.setEnabled(has_images)

        def _browse_scene_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select scene directory")
            if path:
                self.scene_dir_edit.setText(path)

        def _browse_images_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select images directory")
            if path:
                self.images_dir_edit.setText(path)

        def _browse_masks_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select masks directory")
            if path:
                self.masks_dir_edit.setText(path)

        def _apply_scene_paths(self) -> None:
            scene_dir = Path(self.scene_dir_edit.text().strip() or ".")
            self.images_dir_edit.setText(str(scene_dir / "images"))
            self.masks_dir_edit.setText(str(scene_dir / "masks"))

        def _require_scripts(self) -> tuple[Path, Path]:
            yolo_script = self.base_dir / "yolo_mask.py"
            stitch_script = self.base_dir / "stitch_mask.py"
            if not yolo_script.exists():
                raise FileNotFoundError(f"yolo_mask.py not found: {yolo_script}")
            if not stitch_script.exists():
                raise FileNotFoundError(f"stitch_mask.py not found: {stitch_script}")
            return yolo_script, stitch_script

        def _build_yolo_cmd(self) -> list[str]:
            yolo_script, _ = self._require_scripts()
            images_dir = Path(self.images_dir_edit.text().strip())
            masks_dir = Path(self.masks_dir_edit.text().strip())

            if not images_dir.is_dir():
                raise ValueError(f"Images directory not found: {images_dir}")
            if not masks_dir.exists():
                masks_dir.mkdir(parents=True, exist_ok=True)

            try:
                expand = int(self.yolo_expand_edit.text().strip())
            except Exception as e:
                raise ValueError(f"Invalid YOLO Expand value: {e}") from e

            level = self.yolo_level_combo.currentText()
            cmd = [
                sys.executable,
                str(yolo_script),
                str(images_dir),
                str(masks_dir),
                "--level",
                level,
                "--expand",
                str(expand),
            ]
            if self.yolo_add_ext_check.isChecked():
                cmd.append("--add_ext")
            return cmd

        def _build_stitch_cmd(self) -> list[str]:
            _, stitch_script = self._require_scripts()
            masks_dir = Path(self.masks_dir_edit.text().strip())
            if not masks_dir.is_dir():
                raise ValueError(f"Masks directory not found: {masks_dir}")

            try:
                fov = float(self.stitch_fov_edit.text().strip())
            except Exception as e:
                raise ValueError(f"Invalid Stitch FOV value: {e}") from e
            try:
                workers = int(self.stitch_workers_edit.text().strip())
            except Exception as e:
                raise ValueError(f"Invalid Stitch Workers value: {e}") from e
            if workers <= 0:
                raise ValueError("Stitch Workers must be > 0")

            cmd = [
                sys.executable,
                str(stitch_script),
                str(masks_dir),
                str(masks_dir),
                "--fov",
                str(fov),
                "--workers",
                str(workers),
            ]
            return cmd

        def _start_step_queue(self, steps: list[tuple[str, list[str]]]) -> None:
            if self._is_running():
                QMessageBox.warning(self, "Busy", "Another process is running.")
                return
            if not steps:
                return
            self.pending_steps = list(steps)
            self._run_next_step()

        def _run_next_step(self) -> None:
            if not self.pending_steps:
                self._set_running_state(False, "Idle")
                return

            phase, cmd = self.pending_steps.pop(0)
            self.current_phase = phase
            self._process_buffer = ""
            self._init_progress_for_phase(phase)
            self._append_log("$ " + " ".join(cmd))

            proc = QProcess(self)
            proc.setProgram(cmd[0])
            proc.setArguments(cmd[1:])
            proc.setProcessChannelMode(QProcess.MergedChannels)
            proc.readyReadStandardOutput.connect(self._on_process_output)
            proc.errorOccurred.connect(self._on_process_error)
            proc.finished.connect(self._on_process_finished)
            self.proc = proc

            self._set_running_state(True, f"Running: {phase}")
            proc.start()

        def _on_process_output(self) -> None:
            if self.proc is None:
                return
            data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
            # tqdm style carriage-return updates are converted to newline for readable logs.
            data = data.replace("\r", "\n")
            self._process_buffer += data
            while "\n" in self._process_buffer:
                line, self._process_buffer = self._process_buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line:
                    self._append_log(line)
                    self._track_progress_from_line(line)

        def _on_process_error(self, _error) -> None:
            if self.proc is None:
                return
            self._append_log(f"[{self.current_phase}] process error occurred")

        def _on_process_finished(self, exit_code: int, _status) -> None:
            if self._process_buffer:
                tail = self._process_buffer.replace("\r", "\n").strip()
                if tail:
                    for line in tail.splitlines():
                        self._append_log(line)
                self._process_buffer = ""

            phase = self.current_phase
            if exit_code == 0:
                if self.phase_total_items > 0:
                    self._set_phase_progress(self.phase_total_items, self.phase_total_items)
                self._append_log(f"[{phase}] completed successfully")
                self.proc = None
                if self.pending_steps:
                    self._run_next_step()
                else:
                    self._set_running_state(False, f"Done: {phase}")
            else:
                self._append_log(f"[{phase}] failed (exit={exit_code})")
                self.pending_steps = []
                self.proc = None
                self._set_running_state(False, f"Failed: {phase}")

        def _run_yolo(self) -> None:
            try:
                cmd = self._build_yolo_cmd()
            except Exception as e:
                QMessageBox.critical(self, "Invalid Input", str(e))
                return
            self._start_step_queue([("yolo", cmd)])

        def _run_stitch(self) -> None:
            try:
                cmd = self._build_stitch_cmd()
            except Exception as e:
                QMessageBox.critical(self, "Invalid Input", str(e))
                return
            self._start_step_queue([("stitch", cmd)])

        def _run_both(self) -> None:
            try:
                yolo_cmd = self._build_yolo_cmd()
                stitch_cmd = self._build_stitch_cmd()
            except Exception as e:
                QMessageBox.critical(self, "Invalid Input", str(e))
                return
            self._start_step_queue([("yolo", yolo_cmd), ("stitch", stitch_cmd)])

        def closeEvent(self, event: QCloseEvent) -> None:  # pragma: no cover - UI event
            if self._is_running() and self.proc is not None:
                self.proc.kill()
                self.proc.waitForFinished(3000)
            super().closeEvent(event)

else:
    class MaskToolsWindow:  # pragma: no cover - placeholder when PySide6 missing
        pass


def ensure_gui_deps() -> None:
    if QApplication is None:
        raise RuntimeError(f"PySide6 is required to run this GUI: {_PYSIDE_IMPORT_ERROR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI wrapper for yolo_mask.py and stitch_mask.py.")
    parser.add_argument(
        "--scene-dir",
        default=".",
        help="Initial scene directory shown in GUI",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        ensure_gui_deps()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = MaskToolsWindow(initial_scene_dir=args.scene_dir)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
