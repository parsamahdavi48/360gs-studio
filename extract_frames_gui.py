#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from PySide6.QtCore import QProcess, Qt
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QRadioButton,
        QVBoxLayout,
        QWidget,
        QComboBox,
    )
except Exception as e:  # pragma: no cover - environment-dependent import
    QProcess = None
    Qt = None
    QCloseEvent = None
    QApplication = None
    QFileDialog = None
    QFormLayout = None
    QHBoxLayout = None
    QLabel = None
    QLineEdit = None
    QMainWindow = None
    QMessageBox = None
    QPlainTextEdit = None
    QPushButton = None
    QRadioButton = None
    QVBoxLayout = None
    QWidget = None
    QComboBox = None
    _PYSIDE_IMPORT_ERROR = e
else:
    _PYSIDE_IMPORT_ERROR = None


if QMainWindow is not None:
    class ExtractFramesWindow(QMainWindow):
        def __init__(self, initial_scene_dir: str | None = None) -> None:
            super().__init__()
            self.base_dir = Path(__file__).resolve().parent
            self.proc: QProcess | None = None
            self.current_phase = ""

            self.setWindowTitle("Frame Extractor")
            self.resize(1160, 860)

            self._build_ui(initial_scene_dir or ".")
            self._update_mode_widgets()
            self._refresh_action_buttons()

        def _build_ui(self, initial_scene_dir: str) -> None:
            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)

            form_layout = QFormLayout()

            self.input_video_edit = QLineEdit()
            browse_video_btn = QPushButton("Browse")
            browse_video_btn.clicked.connect(self._browse_video)
            row = QHBoxLayout()
            row.addWidget(self.input_video_edit)
            row.addWidget(browse_video_btn)
            form_layout.addRow("Input Video", row)

            self.scene_dir_edit = QLineEdit(initial_scene_dir)
            self.scene_dir_edit.textChanged.connect(lambda _: self._refresh_action_buttons())
            browse_scene_btn = QPushButton("Browse")
            browse_scene_btn.clicked.connect(self._browse_scene_dir)
            row = QHBoxLayout()
            row.addWidget(self.scene_dir_edit)
            row.addWidget(browse_scene_btn)
            form_layout.addRow("Scene Directory", row)

            self.change_radio = QRadioButton("Change-Based")
            self.fixed_radio = QRadioButton("Fixed Interval")
            self.change_radio.setChecked(True)
            self.change_radio.toggled.connect(self._update_mode_widgets)
            row = QHBoxLayout()
            row.addWidget(self.change_radio)
            row.addWidget(self.fixed_radio)
            row.addStretch(1)
            form_layout.addRow("Mode", row)

            self.interval_sec_edit = QLineEdit("0.5")
            form_layout.addRow("Fixed Interval (sec)", self.interval_sec_edit)

            self.change_threshold_edit = QLineEdit("0.04")
            form_layout.addRow("Change Threshold", self.change_threshold_edit)

            self.min_gap_edit = QLineEdit("0.25")
            form_layout.addRow("Min Gap (sec)", self.min_gap_edit)

            self.max_gap_edit = QLineEdit("2.0")
            form_layout.addRow("Max Gap (sec)", self.max_gap_edit)

            self.analysis_width_edit = QLineEdit("960")
            form_layout.addRow("Analysis Width", self.analysis_width_edit)

            self.blur_percentile_edit = QLineEdit("25.0")
            form_layout.addRow("Blur Percentile", self.blur_percentile_edit)

            self.blur_window_edit = QLineEdit("0")
            form_layout.addRow("Blur Window (frames)", self.blur_window_edit)

            self.image_ext_combo = QComboBox()
            self.image_ext_combo.addItems(["jpg", "png"])
            form_layout.addRow("Image Ext", self.image_ext_combo)

            self.jpg_quality_edit = QLineEdit("2")
            form_layout.addRow("JPEG Quality (ffmpeg -q:v)", self.jpg_quality_edit)

            self.ffmpeg_edit = QLineEdit("ffmpeg")
            form_layout.addRow("ffmpeg", self.ffmpeg_edit)

            self.ffprobe_edit = QLineEdit("ffprobe")
            form_layout.addRow("ffprobe", self.ffprobe_edit)

            self.metashape_output_edit = QLineEdit("metashape_images")
            form_layout.addRow("Metashape Output Folder", self.metashape_output_edit)

            layout.addLayout(form_layout)

            btn_row = QHBoxLayout()
            self.run_button = QPushButton("1) Run Extraction")
            self.run_button.clicked.connect(self._run_extraction)
            btn_row.addWidget(self.run_button)

            self.review_button = QPushButton("2) Open Review GUI")
            self.review_button.clicked.connect(self._open_review)
            btn_row.addWidget(self.review_button)

            self.export_button = QPushButton("3) Export Keep Frames")
            self.export_button.clicked.connect(self._export_keep)
            btn_row.addWidget(self.export_button)

            btn_row.addStretch(1)

            self.status_label = QLabel("Idle")
            self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            btn_row.addWidget(self.status_label)

            layout.addLayout(btn_row)

            self.log_text = QPlainTextEdit()
            self.log_text.setReadOnly(True)
            layout.addWidget(self.log_text, stretch=1)

        def _is_running(self) -> bool:
            return self.proc is not None and self.proc.state() != QProcess.NotRunning

        def _mode(self) -> str:
            return "fixed" if self.fixed_radio.isChecked() else "change"

        def _update_mode_widgets(self) -> None:
            is_fixed = self._mode() == "fixed"
            self.interval_sec_edit.setEnabled(is_fixed)
            self.change_threshold_edit.setEnabled(not is_fixed)
            self.min_gap_edit.setEnabled(not is_fixed)
            self.max_gap_edit.setEnabled(not is_fixed)

        def _refresh_action_buttons(self) -> None:
            if self._is_running():
                self.run_button.setEnabled(False)
                self.review_button.setEnabled(False)
                self.export_button.setEnabled(False)
                return

            self.run_button.setEnabled(True)
            scene_dir = Path(self.scene_dir_edit.text().strip() or ".")
            csv_path = scene_dir / "selected_frames.csv"
            has_csv = csv_path.exists()
            self.review_button.setEnabled(has_csv)
            self.export_button.setEnabled(has_csv)

        def _append_log(self, text: str) -> None:
            self.log_text.appendPlainText(text)
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.End)
            self.log_text.setTextCursor(cursor)

        def _set_running_state(self, running: bool, status_text: str) -> None:
            self.status_label.setText(status_text)
            if running:
                self.run_button.setEnabled(False)
                self.review_button.setEnabled(False)
                self.export_button.setEnabled(False)
            else:
                self._refresh_action_buttons()

        def _browse_video(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select input video",
                "",
                "Video files (*.mp4 *.mov *.mkv *.avi *.m4v);;All files (*.*)",
            )
            if path:
                self.input_video_edit.setText(path)

        def _browse_scene_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select scene directory")
            if path:
                self.scene_dir_edit.setText(path)

        def _build_extract_cmd(self) -> list[str]:
            input_video = self.input_video_edit.text().strip()
            scene_dir = self.scene_dir_edit.text().strip()

            if not input_video:
                raise ValueError("Input video is required")
            if not Path(input_video).exists():
                raise ValueError(f"Input video not found: {input_video}")
            if not scene_dir:
                raise ValueError("Scene directory is required")

            script = self.base_dir / "extract_frames.py"
            if not script.exists():
                raise FileNotFoundError(f"extract_frames.py not found: {script}")

            cmd = [
                sys.executable,
                str(script),
                input_video,
                scene_dir,
                "--mode",
                self._mode(),
                "--analysis-width",
                self.analysis_width_edit.text().strip(),
                "--blur-percentile",
                self.blur_percentile_edit.text().strip(),
                "--blur-window-frames",
                self.blur_window_edit.text().strip(),
                "--image-ext",
                self.image_ext_combo.currentText(),
                "--jpg-quality",
                self.jpg_quality_edit.text().strip(),
                "--ffmpeg",
                self.ffmpeg_edit.text().strip(),
                "--ffprobe",
                self.ffprobe_edit.text().strip(),
            ]

            if self._mode() == "fixed":
                cmd.extend(["--interval-sec", self.interval_sec_edit.text().strip()])
            else:
                cmd.extend(
                    [
                        "--change-threshold",
                        self.change_threshold_edit.text().strip(),
                        "--min-gap-sec",
                        self.min_gap_edit.text().strip(),
                        "--max-gap-sec",
                        self.max_gap_edit.text().strip(),
                    ]
                )

            return cmd

        def _start_process(self, cmd: list[str], phase: str) -> None:
            if self._is_running():
                QMessageBox.warning(self, "Busy", "Another process is running.")
                return

            self.current_phase = phase
            self._append_log("$ " + " ".join(cmd))

            proc = QProcess(self)
            proc.setProgram(cmd[0])
            proc.setArguments(cmd[1:])
            proc.setProcessChannelMode(QProcess.MergedChannels)
            proc.readyReadStandardOutput.connect(self._on_process_output)
            proc.errorOccurred.connect(self._on_process_error)
            proc.finished.connect(self._on_process_finished)
            self.proc = proc

            self._set_running_state(True, "Running")
            proc.start()

        def _on_process_output(self) -> None:
            if self.proc is None:
                return
            data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
            for line in data.splitlines():
                self._append_log(line)

        def _on_process_error(self, _error) -> None:
            if self.proc is None:
                return
            self._append_log(f"[{self.current_phase}] process error occurred")

        def _on_process_finished(self, exit_code: int, _status) -> None:
            phase = self.current_phase
            if exit_code == 0:
                self._append_log(f"[{phase}] completed successfully")
                self._set_running_state(False, f"Done: {phase}")
            else:
                self._append_log(f"[{phase}] failed (exit={exit_code})")
                self._set_running_state(False, f"Failed: {phase}")

            self.proc = None

        def _run_extraction(self) -> None:
            try:
                cmd = self._build_extract_cmd()
            except Exception as e:
                QMessageBox.critical(self, "Invalid Input", str(e))
                return

            self._start_process(cmd, "extract")

        def _open_review(self) -> None:
            scene_dir = self.scene_dir_edit.text().strip() or "."
            script = self.base_dir / "review_frames.py"
            csv_path = Path(scene_dir) / "selected_frames.csv"

            if not script.exists():
                QMessageBox.critical(self, "Error", f"review_frames.py not found: {script}")
                return
            if not csv_path.exists():
                QMessageBox.critical(self, "Error", f"CSV not found: {csv_path}")
                return

            cmd = [sys.executable, str(script), scene_dir]
            try:
                subprocess.Popen(cmd)
                self._append_log("[review] launched")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

        def _export_keep(self) -> None:
            scene_dir = self.scene_dir_edit.text().strip() or "."
            output_name = self.metashape_output_edit.text().strip() or "metashape_images"

            script = self.base_dir / "apply_frame_decisions.py"
            csv_path = Path(scene_dir) / "selected_frames.csv"
            if not script.exists():
                QMessageBox.critical(self, "Error", f"apply_frame_decisions.py not found: {script}")
                return
            if not csv_path.exists():
                QMessageBox.critical(self, "Error", f"CSV not found: {csv_path}")
                return

            result = QMessageBox.question(
                self,
                "Clean Output",
                f"Clean existing image files in '{output_name}' before export?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            should_clean = result == QMessageBox.Yes

            cmd = [
                sys.executable,
                str(script),
                scene_dir,
                "--output",
                output_name,
            ]
            if should_clean:
                cmd.append("--clean-output")

            self._start_process(cmd, "export")

        def closeEvent(self, event: QCloseEvent) -> None:  # pragma: no cover - UI event
            if self._is_running() and self.proc is not None:
                self.proc.kill()
                self.proc.waitForFinished(3000)
            super().closeEvent(event)

else:
    class ExtractFramesWindow:  # pragma: no cover - placeholder when PySide6 missing
        pass


def ensure_gui_deps() -> None:
    if QApplication is None:
        raise RuntimeError(f"PySide6 is required to run this GUI: {_PYSIDE_IMPORT_ERROR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI wrapper for extract_frames.py and review/export workflow.")
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
    window = ExtractFramesWindow(initial_scene_dir=args.scene_dir)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
