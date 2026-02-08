#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
            self._process_buffer = ""
            self.video_info: dict | None = None
            self.last_estimate_summary: dict | None = None

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
            self.change_radio.toggled.connect(self._mark_estimate_stale)
            row = QHBoxLayout()
            row.addWidget(self.change_radio)
            row.addWidget(self.fixed_radio)
            row.addStretch(1)
            form_layout.addRow("Mode", row)

            self.interval_sec_edit = QLineEdit("0.5")
            self.interval_sec_edit.textChanged.connect(self._mark_estimate_stale)
            form_layout.addRow("Fixed Interval (sec)", self.interval_sec_edit)

            self.change_threshold_edit = QLineEdit("0.04")
            self.change_threshold_edit.textChanged.connect(self._mark_estimate_stale)
            form_layout.addRow("Change Threshold", self.change_threshold_edit)

            self.min_gap_edit = QLineEdit("0.25")
            self.min_gap_edit.textChanged.connect(self._mark_estimate_stale)
            form_layout.addRow("Min Gap (sec)", self.min_gap_edit)

            self.max_gap_edit = QLineEdit("2.0")
            self.max_gap_edit.textChanged.connect(self._mark_estimate_stale)
            form_layout.addRow("Max Gap (sec)", self.max_gap_edit)

            self.analysis_width_edit = QLineEdit("960")
            self.analysis_width_edit.textChanged.connect(self._mark_estimate_stale)
            form_layout.addRow("Analysis Width", self.analysis_width_edit)

            self.blur_percentile_edit = QLineEdit("25.0")
            self.blur_percentile_edit.textChanged.connect(self._mark_estimate_stale)
            form_layout.addRow("Blur Percentile", self.blur_percentile_edit)

            self.blur_window_edit = QLineEdit("0")
            self.blur_window_edit.textChanged.connect(self._mark_estimate_stale)
            form_layout.addRow("Blur Window (frames)", self.blur_window_edit)

            self.image_ext_combo = QComboBox()
            self.image_ext_combo.addItems(["jpg", "png"])
            self.image_ext_combo.currentIndexChanged.connect(self._mark_estimate_stale)
            form_layout.addRow("Image Ext", self.image_ext_combo)

            self.jpg_quality_edit = QLineEdit("2")
            self.jpg_quality_edit.textChanged.connect(self._mark_estimate_stale)
            form_layout.addRow("JPEG Quality (ffmpeg -q:v)", self.jpg_quality_edit)

            self.ffmpeg_edit = QLineEdit("ffmpeg")
            self.ffmpeg_edit.textChanged.connect(self._mark_estimate_stale)
            form_layout.addRow("ffmpeg", self.ffmpeg_edit)

            self.ffprobe_edit = QLineEdit("ffprobe")
            self.ffprobe_edit.textChanged.connect(self._mark_estimate_stale)
            form_layout.addRow("ffprobe", self.ffprobe_edit)

            self.metashape_output_edit = QLineEdit("metashape_images")
            form_layout.addRow("Metashape Output Folder", self.metashape_output_edit)

            meta_row = QHBoxLayout()
            self.load_info_button = QPushButton("Load Video Info")
            self.load_info_button.clicked.connect(lambda: self._load_video_info(show_error_popup=True))
            meta_row.addWidget(self.load_info_button)

            self.estimate_button = QPushButton("Estimate Count")
            self.estimate_button.clicked.connect(self._run_estimate)
            meta_row.addWidget(self.estimate_button)
            meta_row.addStretch(1)
            form_layout.addRow("Analysis Tools", meta_row)

            self.video_info_label = QLabel("Video: -")
            self.video_info_label.setStyleSheet("color: #333;")
            form_layout.addRow("Video Info", self.video_info_label)

            self.estimate_label = QLabel("Estimate: -")
            self.estimate_label.setStyleSheet("color: #333;")
            form_layout.addRow("Estimated Output", self.estimate_label)

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
                self.load_info_button.setEnabled(False)
                self.estimate_button.setEnabled(False)
                return

            has_video = Path(self.input_video_edit.text().strip()).exists()
            self.run_button.setEnabled(has_video)
            scene_dir = Path(self.scene_dir_edit.text().strip() or ".")
            csv_path = scene_dir / "selected_frames.csv"
            has_csv = csv_path.exists()
            self.review_button.setEnabled(has_csv)
            self.export_button.setEnabled(has_csv)
            self.load_info_button.setEnabled(has_video)
            self.estimate_button.setEnabled(has_video)

        def _append_log(self, text: str) -> None:
            self.log_text.appendPlainText(text)
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        def _set_running_state(self, running: bool, status_text: str) -> None:
            self.status_label.setText(status_text)
            if running:
                self.run_button.setEnabled(False)
                self.review_button.setEnabled(False)
                self.export_button.setEnabled(False)
                self.load_info_button.setEnabled(False)
                self.estimate_button.setEnabled(False)
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
                self._load_video_info(show_error_popup=False)
                self._mark_estimate_stale()

        def _browse_scene_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select scene directory")
            if path:
                self.scene_dir_edit.setText(path)

        def _mark_estimate_stale(self, *_args) -> None:
            self.last_estimate_summary = None
            self.estimate_label.setText("Estimate: - (stale)")

        @staticmethod
        def _parse_fraction(value: str) -> float:
            if not value:
                return 0.0
            if "/" in value:
                num, den = value.split("/", 1)
                den_f = float(den)
                if den_f == 0:
                    return 0.0
                return float(num) / den_f
            return float(value)

        @staticmethod
        def _format_duration(sec: float) -> str:
            if sec < 0:
                sec = 0
            whole = int(sec)
            h = whole // 3600
            m = (whole % 3600) // 60
            s = whole % 60
            return f"{h:02d}:{m:02d}:{s:02d}"

        def _probe_video_info(self) -> dict:
            input_video = self.input_video_edit.text().strip()
            ffprobe_bin = self.ffprobe_edit.text().strip() or "ffprobe"

            if not input_video:
                raise ValueError("Input video is required")
            if not Path(input_video).exists():
                raise ValueError(f"Input video not found: {input_video}")

            cmd = [
                ffprobe_bin,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                input_video,
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or "ffprobe failed")

            data = json.loads(proc.stdout)
            streams = data.get("streams", [])
            if not streams:
                raise RuntimeError("No video stream found")

            stream = streams[0]
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))

            fps = self._parse_fraction(stream.get("avg_frame_rate", "0"))
            if fps <= 0:
                fps = self._parse_fraction(stream.get("r_frame_rate", "0"))

            duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
            nb_frames_raw = stream.get("nb_frames")
            total_frames = int(nb_frames_raw) if nb_frames_raw and str(nb_frames_raw).isdigit() else 0

            if fps <= 0 and duration > 0 and total_frames > 0:
                fps = total_frames / duration
            if fps <= 0:
                raise RuntimeError("Could not determine FPS from video")
            if total_frames <= 0 and duration > 0:
                total_frames = max(1, int(round(duration * fps)))

            return {
                "width": width,
                "height": height,
                "fps": fps,
                "duration_sec": duration,
                "total_frames": total_frames,
            }

        def _update_video_info_label(self) -> None:
            if not self.video_info:
                self.video_info_label.setText("Video: -")
                return

            info = self.video_info
            duration_str = self._format_duration(float(info["duration_sec"]))
            self.video_info_label.setText(
                f"{info['width']}x{info['height']}, {info['fps']:.3f} fps, "
                f"{duration_str} ({info['duration_sec']:.2f}s), {info['total_frames']} frames"
            )

        def _load_video_info(self, show_error_popup: bool = True) -> bool:
            try:
                self.video_info = self._probe_video_info()
                self._update_video_info_label()
                self._append_log("[probe] loaded video metadata")
                return True
            except Exception as e:
                self.video_info = None
                self._update_video_info_label()
                if show_error_popup:
                    QMessageBox.critical(self, "Video Info Error", str(e))
                else:
                    self._append_log(f"[probe] failed: {e}")
                return False

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

        def _build_estimate_cmd(self) -> list[str]:
            cmd = self._build_extract_cmd()
            cmd.extend(["--estimate-only", "--print-summary-json"])
            return cmd

        def _start_process(self, cmd: list[str], phase: str) -> None:
            if self._is_running():
                QMessageBox.warning(self, "Busy", "Another process is running.")
                return

            self.current_phase = phase
            self._process_buffer = ""
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
            self._process_buffer += data
            while "\n" in self._process_buffer:
                line, self._process_buffer = self._process_buffer.split("\n", 1)
                self._handle_process_line(line.rstrip("\r"))

        def _handle_process_line(self, line: str) -> None:
            if not line:
                return
            self._append_log(line)

            prefix = "SUMMARY_JSON:"
            if line.startswith(prefix):
                payload = line[len(prefix):]
                try:
                    summary = json.loads(payload)
                except Exception as e:
                    self._append_log(f"[summary] parse failed: {e}")
                    return
                self.last_estimate_summary = summary
                self._apply_summary(summary)

        def _apply_summary(self, summary: dict) -> None:
            video = summary.get("video", {})
            if video:
                self.video_info = {
                    "width": int(video.get("width", 0)),
                    "height": int(video.get("height", 0)),
                    "fps": float(video.get("fps", 0.0)),
                    "duration_sec": float(video.get("duration_sec", 0.0)),
                    "total_frames": int(video.get("total_frames", 0)),
                }
                self._update_video_info_label()

            result = summary.get("result", {})
            selected = int(result.get("selected_count", 0))
            total = int(video.get("total_frames", 0))
            ratio = (selected / total * 100.0) if total > 0 else 0.0
            replaced = int(result.get("replaced_count", 0))
            fallback = int(result.get("fallback_keep_count", 0))
            self.estimate_label.setText(
                f"{selected} frames ({ratio:.1f}% of total, replaced={replaced}, fallback={fallback})"
            )

        def _on_process_error(self, _error) -> None:
            if self.proc is None:
                return
            self._append_log(f"[{self.current_phase}] process error occurred")

        def _on_process_finished(self, exit_code: int, _status) -> None:
            if self._process_buffer:
                self._handle_process_line(self._process_buffer.rstrip("\r"))
                self._process_buffer = ""

            phase = self.current_phase
            if exit_code == 0:
                self._append_log(f"[{phase}] completed successfully")
                self._set_running_state(False, f"Done: {phase}")
            else:
                self._append_log(f"[{phase}] failed (exit={exit_code})")
                self._set_running_state(False, f"Failed: {phase}")

            self.proc = None

        def _run_estimate(self) -> None:
            if not self.video_info:
                if not self._load_video_info(show_error_popup=True):
                    return

            try:
                cmd = self._build_estimate_cmd()
            except Exception as e:
                QMessageBox.critical(self, "Invalid Input", str(e))
                return

            self._start_process(cmd, "estimate")

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
