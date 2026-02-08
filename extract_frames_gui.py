#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

try:
    from PySide6.QtCore import QProcess, Qt, QTimer
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
        QProgressBar,
        QPushButton,
        QRadioButton,
        QVBoxLayout,
        QWidget,
        QComboBox,
    )
except Exception as e:  # pragma: no cover - environment-dependent import
    QProcess = None
    Qt = None
    QTimer = None
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
    QProgressBar = None
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
            self.sample_proc: QProcess | None = None
            self._sample_buffer = ""
            self.video_info: dict | None = None
            self.last_estimate_summary: dict | None = None
            self.instant_estimate_text = "-"
            self.sampled_estimate_text = "-"
            self.sample_segments = 5
            self.sample_segment_sec = 12.0
            self.sample_fps = 8.0
            self.sample_timer = QTimer(self)
            self.sample_timer.setSingleShot(True)
            self.sample_timer.timeout.connect(self._start_sample_estimate)
            self.cancel_requested = False

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

            self.metashape_output_edit = QLineEdit("images")
            form_layout.addRow("Export Target Folder", self.metashape_output_edit)

            self.filename_prefix_edit = QLineEdit("")
            self.filename_prefix_edit.setPlaceholderText("auto (video filename)")
            self.filename_prefix_edit.textChanged.connect(self._mark_estimate_stale)
            form_layout.addRow("Filename Prefix", self.filename_prefix_edit)

            meta_row = QHBoxLayout()
            self.load_info_button = QPushButton("Load Video Info")
            self.load_info_button.clicked.connect(lambda: self._load_video_info(show_error_popup=True))
            meta_row.addWidget(self.load_info_button)

            self.estimate_button = QPushButton("Refresh Sampled")
            self.estimate_button.clicked.connect(self._run_sampled_estimate_now)
            meta_row.addWidget(self.estimate_button)
            meta_row.addStretch(1)
            form_layout.addRow("Analysis Tools", meta_row)

            self.video_info_label = QLabel("Video: -")
            form_layout.addRow("Video Info", self.video_info_label)

            self.estimate_label = QLabel()
            self.estimate_label.setWordWrap(True)
            self._refresh_estimate_label()
            form_layout.addRow("Estimated Output", self.estimate_label)

            layout.addLayout(form_layout)

            btn_row = QHBoxLayout()
            self.run_button = QPushButton("1) Run Extraction")
            self.run_button.clicked.connect(self._run_extraction)
            btn_row.addWidget(self.run_button)

            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self._cancel_running_process)
            self.cancel_button.setEnabled(False)
            btn_row.addWidget(self.cancel_button)

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

            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Idle")
            layout.addWidget(self.progress_bar)

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
            self._refresh_action_buttons()

        def _refresh_action_buttons(self) -> None:
            if self._is_running():
                self.run_button.setEnabled(False)
                self.cancel_button.setEnabled(True)
                self.review_button.setEnabled(False)
                self.export_button.setEnabled(False)
                self.load_info_button.setEnabled(False)
                self.estimate_button.setEnabled(False)
                return

            has_video = Path(self.input_video_edit.text().strip()).exists()
            self.run_button.setEnabled(has_video)
            self.cancel_button.setEnabled(False)
            scene_dir = Path(self.scene_dir_edit.text().strip() or ".")
            csv_path = scene_dir / "selected_frames.csv"
            has_csv = csv_path.exists()
            self.review_button.setEnabled(has_csv)
            self.export_button.setEnabled(has_csv)
            self.load_info_button.setEnabled(has_video)
            self.estimate_button.setEnabled(has_video and self._mode() == "change")

        def _append_log(self, text: str) -> None:
            self.log_text.appendPlainText(text)
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        def _set_running_state(self, running: bool, status_text: str) -> None:
            self.status_label.setText(status_text)
            if running:
                self.run_button.setEnabled(False)
                self.cancel_button.setEnabled(True)
                self.review_button.setEnabled(False)
                self.export_button.setEnabled(False)
                self.load_info_button.setEnabled(False)
                self.estimate_button.setEnabled(False)
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(0)
                self.progress_bar.setFormat("0%")
            else:
                self.cancel_button.setEnabled(False)
                if status_text.startswith("Done"):
                    self.progress_bar.setRange(0, 100)
                    self.progress_bar.setValue(100)
                self.progress_bar.setFormat(status_text)
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

        def _browse_scene_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select scene directory")
            if path:
                self.scene_dir_edit.setText(path)

        def _refresh_estimate_label(self) -> None:
            self.estimate_label.setText(
                f"Instant: {self.instant_estimate_text}\n"
                f"Sampled: {self.sampled_estimate_text}"
            )

        def _cancel_sample_estimate(self, reason: str | None = None, log: bool = False) -> None:
            self.sample_timer.stop()
            proc = self.sample_proc
            self.sample_proc = None
            self._sample_buffer = ""
            if proc is not None and proc.state() != QProcess.NotRunning:
                proc.kill()
                proc.waitForFinished(2000)
                if log and reason:
                    self._append_log(f"[sample] canceled: {reason}")

        def _estimate_change_range(self, duration_sec: float, min_gap_sec: float, max_gap_sec: float) -> tuple[int, int]:
            if min_gap_sec <= 0 or max_gap_sec <= 0:
                raise ValueError("Min/Max gap must be > 0")
            if max_gap_sec < min_gap_sec:
                raise ValueError("Max Gap must be >= Min Gap")
            if duration_sec <= 0:
                return 1, 1
            min_count = max(1, int(math.ceil(duration_sec / max_gap_sec)))
            max_count = max(min_count, int(math.ceil(duration_sec / min_gap_sec)))
            return min_count, max_count

        @staticmethod
        def _threshold_to_weight(threshold: float) -> float:
            low = 0.01
            high = 0.12
            if threshold <= low:
                return 0.0
            if threshold >= high:
                return 1.0
            return (threshold - low) / (high - low)

        def _update_instant_estimate(self) -> None:
            if not self.video_info:
                self.instant_estimate_text = "-"
                self._refresh_estimate_label()
                return

            duration_sec = float(self.video_info.get("duration_sec", 0.0))
            fps = float(self.video_info.get("fps", 0.0))
            total_frames = int(self.video_info.get("total_frames", 0))

            if self._mode() == "fixed":
                try:
                    interval_sec = float(self.interval_sec_edit.text().strip())
                    if interval_sec <= 0:
                        raise ValueError("interval <= 0")
                    if total_frames <= 0 and duration_sec > 0 and fps > 0:
                        total_frames = max(1, int(round(duration_sec * fps)))
                    step = max(1, int(round(interval_sec * fps)))
                    indices = list(range(0, max(total_frames, 1), step))
                    if indices[-1] != max(total_frames - 1, 0):
                        indices.append(max(total_frames - 1, 0))
                    self.instant_estimate_text = f"{len(indices)} frames (fixed exact)"
                except Exception:
                    self.instant_estimate_text = "- (invalid fixed params)"
                self._refresh_estimate_label()
                return

            try:
                threshold = float(self.change_threshold_edit.text().strip())
                min_gap = float(self.min_gap_edit.text().strip())
                max_gap = float(self.max_gap_edit.text().strip())
                min_count, max_count = self._estimate_change_range(duration_sec, min_gap, max_gap)
                weight = self._threshold_to_weight(threshold)
                est_gap = min_gap + weight * (max_gap - min_gap)
                est_count = max(1, int(math.ceil(duration_sec / max(est_gap, 1e-6)))) if duration_sec > 0 else 1
                est_count = max(min_count, min(max_count, est_count))
                self.instant_estimate_text = f"{min_count}-{max_count} frames (thr: {est_count})"
            except Exception:
                self.instant_estimate_text = "- (invalid change params)"
            self._refresh_estimate_label()

        def _build_sampled_estimate_cmd(self) -> list[str]:
            cmd = self._build_extract_cmd()
            cmd.extend(
                [
                    "--estimate-only",
                    "--estimate-mode",
                    "sampled",
                    "--sample-segments",
                    str(self.sample_segments),
                    "--sample-segment-sec",
                    str(self.sample_segment_sec),
                    "--sample-fps",
                    str(self.sample_fps),
                    "--print-summary-json",
                ]
            )
            return cmd

        def _schedule_sample_estimate(self, delay_ms: int = 500) -> None:
            if self._mode() != "change":
                self._cancel_sample_estimate()
                self.sampled_estimate_text = "- (not used in fixed mode)"
                self._refresh_estimate_label()
                return
            if self._is_running():
                return
            if not self.video_info:
                self.sampled_estimate_text = "-"
                self._refresh_estimate_label()
                return

            self._cancel_sample_estimate()
            self.sampled_estimate_text = "calculating..."
            self._refresh_estimate_label()
            self.sample_timer.start(max(0, delay_ms))

        def _start_sample_estimate(self) -> None:
            if self._is_running() or self._mode() != "change":
                return
            if not self.video_info:
                return

            try:
                cmd = self._build_sampled_estimate_cmd()
            except Exception as e:
                self.sampled_estimate_text = f"- ({e})"
                self._refresh_estimate_label()
                return

            self._cancel_sample_estimate()
            self._append_log("$ " + " ".join(cmd))

            proc = QProcess(self)
            proc.setProgram(cmd[0])
            proc.setArguments(cmd[1:])
            proc.setProcessChannelMode(QProcess.MergedChannels)
            proc.readyReadStandardOutput.connect(lambda p=proc: self._on_sample_output(p))
            proc.finished.connect(lambda code, status, p=proc: self._on_sample_finished(p, code, status))
            self.sample_proc = proc
            proc.start()

        def _on_sample_output(self, proc: QProcess) -> None:
            if proc is not self.sample_proc:
                return

            data = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
            self._sample_buffer += data
            while "\n" in self._sample_buffer:
                line, self._sample_buffer = self._sample_buffer.split("\n", 1)
                self._handle_sample_line(line.rstrip("\r"))

        def _handle_sample_line(self, line: str) -> None:
            if not line:
                return
            self._append_log(f"[sample] {line}")

            prefix = "SUMMARY_JSON:"
            if not line.startswith(prefix):
                return

            payload = line[len(prefix):]
            try:
                summary = json.loads(payload)
            except Exception as e:
                self._append_log(f"[sample] summary parse failed: {e}")
                return

            self.last_estimate_summary = summary
            self._apply_summary(summary)

        def _on_sample_finished(self, proc: QProcess, exit_code: int, _status) -> None:
            if proc is not self.sample_proc:
                return

            if self._sample_buffer:
                self._handle_sample_line(self._sample_buffer.rstrip("\r"))
                self._sample_buffer = ""

            if exit_code == 0:
                self._append_log("[sample] completed successfully")
            else:
                self._append_log(f"[sample] failed (exit={exit_code})")
                if self._mode() == "change" and not self._is_running():
                    self.sampled_estimate_text = "- (sampled failed)"
                    self._refresh_estimate_label()

            self.sample_proc = None

        def _mark_estimate_stale(self, *_args) -> None:
            self.last_estimate_summary = None
            self._update_instant_estimate()

            if self._mode() == "fixed":
                self.sampled_estimate_text = "- (not used in fixed mode)"
                self._cancel_sample_estimate()
                self._refresh_estimate_label()
                return

            if not self.video_info:
                self.sampled_estimate_text = "- (load video first)"
                self._cancel_sample_estimate()
                self._refresh_estimate_label()
                return

            self._schedule_sample_estimate()

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
            if duration <= 0 and total_frames > 0:
                duration = total_frames / fps
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
                self._mark_estimate_stale()
                return True
            except Exception as e:
                self.video_info = None
                self._update_video_info_label()
                self._cancel_sample_estimate()
                self.instant_estimate_text = "-"
                self.sampled_estimate_text = "-"
                self._refresh_estimate_label()
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

            prefix_text = self.filename_prefix_edit.text().strip()
            if prefix_text:
                cmd.extend(["--filename-prefix", prefix_text])

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

            self._cancel_sample_estimate(reason=f"{phase} started", log=True)
            self.cancel_requested = False

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

        def _terminate_process_gracefully(self, proc: QProcess, phase: str, timeout_ms: int = 3000) -> None:
            if proc.state() == QProcess.NotRunning:
                return
            self._append_log(f"[{phase}] cancel requested; sending terminate")
            proc.terminate()
            QTimer.singleShot(timeout_ms, lambda p=proc, ph=phase: self._force_kill_if_running(p, ph))

        def _force_kill_if_running(self, proc: QProcess, phase: str) -> None:
            if proc.state() == QProcess.NotRunning:
                return
            self._append_log(f"[{phase}] terminate timeout; killing process")
            proc.kill()

        def _cancel_running_process(self) -> None:
            if not self._is_running() or self.proc is None:
                return
            self.cancel_requested = True
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Canceling...")
            self._terminate_process_gracefully(self.proc, self.current_phase)

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
            progress_prefix = "[progress] "
            if line.startswith(progress_prefix):
                progress_text = line[len(progress_prefix):]
                self.status_label.setText(progress_text)
                self._update_progress_bar(progress_text)

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

        def _update_progress_bar(self, progress_text: str) -> None:
            match = re.search(r"^(\S+)\s+(\d+)/(\d+)\s+\S+\s+\(([\d.]+)%\)$", progress_text)
            if match:
                phase = match.group(1)
                done = int(match.group(2))
                total = int(match.group(3))
                pct = float(match.group(4))
                pct_clamped = max(0.0, min(100.0, pct))
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(int(round(pct_clamped)))
                self.progress_bar.setFormat(f"{phase}: {pct_clamped:.1f}% ({done}/{total})")
                return

            pct_match = re.search(r"([\d.]+)%", progress_text)
            if pct_match:
                pct = float(pct_match.group(1))
                pct_clamped = max(0.0, min(100.0, pct))
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(int(round(pct_clamped)))
                self.progress_bar.setFormat(progress_text)

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
            estimate = summary.get("estimate", {})
            used = estimate.get("sampled_segments_used")
            requested = estimate.get("sampled_segments_requested")
            range_min = estimate.get("range_min_count")
            range_max = estimate.get("range_max_count")
            detail_parts = [f"{selected} frames ({ratio:.1f}% of total)"]
            if used is not None and requested is not None:
                detail_parts.append(f"segments={used}/{requested}")
            if range_min is not None and range_max is not None:
                detail_parts.append(f"range={range_min}-{range_max}")
            self.sampled_estimate_text = ", ".join(detail_parts)
            self._refresh_estimate_label()

        def _on_process_error(self, _error) -> None:
            if self.proc is None:
                return
            self._append_log(f"[{self.current_phase}] process error occurred")

        def _on_process_finished(self, exit_code: int, _status) -> None:
            if self._process_buffer:
                self._handle_process_line(self._process_buffer.rstrip("\r"))
                self._process_buffer = ""

            phase = self.current_phase
            was_canceled = self.cancel_requested
            self.cancel_requested = False
            if exit_code == 0:
                self._append_log(f"[{phase}] completed successfully")
                self._set_running_state(False, f"Done: {phase}")
            else:
                if was_canceled:
                    self._append_log(f"[{phase}] canceled by user")
                    self._set_running_state(False, f"Canceled: {phase}")
                else:
                    self._append_log(f"[{phase}] failed (exit={exit_code})")
                    self._set_running_state(False, f"Failed: {phase}")

            self.proc = None

        def _run_sampled_estimate_now(self) -> None:
            if self._mode() != "change":
                self.sampled_estimate_text = "- (not used in fixed mode)"
                self._refresh_estimate_label()
                return

            if not self.video_info:
                if not self._load_video_info(show_error_popup=True):
                    return

            self._schedule_sample_estimate(delay_ms=0)

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
            output_name = self.metashape_output_edit.text().strip() or "images"

            script = self.base_dir / "apply_frame_decisions.py"
            csv_path = Path(scene_dir) / "selected_frames.csv"
            if not script.exists():
                QMessageBox.critical(self, "Error", f"apply_frame_decisions.py not found: {script}")
                return
            if not csv_path.exists():
                QMessageBox.critical(self, "Error", f"CSV not found: {csv_path}")
                return

            if output_name.lower() == "images":
                result = QMessageBox.question(
                    self,
                    "Finalize Images",
                    "Apply keep/drop in-place?\n\n"
                    "This will delete dropped files in images/, renumber keep files, "
                    "and rewrite selected_frames.csv.\n"
                    "A backup CSV will be created.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if result != QMessageBox.Yes:
                    return

                cmd = [
                    sys.executable,
                    str(script),
                    scene_dir,
                    "--finalize-in-place",
                ]
                prefix_text = self.filename_prefix_edit.text().strip()
                if prefix_text:
                    cmd.extend(["--filename-prefix", prefix_text])
            else:
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
            self._cancel_sample_estimate()
            if self._is_running() and self.proc is not None:
                self.proc.terminate()
                if not self.proc.waitForFinished(2000):
                    self.proc.kill()
                    self.proc.waitForFinished(2000)
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
