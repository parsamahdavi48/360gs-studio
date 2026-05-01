"""Step 1: フレーム抽出"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _detect_binary(name: str) -> str:
    """PATH から実行ファイルを検出。見つからなければ素の名前を返す（PATH 解決に委ねる）。"""
    found = shutil.which(name)
    return found if found else name

from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.collapsible_section import CollapsibleSection
from gui.steps.base_step import SETTINGS_PANE_WIDTH, BaseStepWidget, configure_settings_scroll


class ExtractStep(BaseStepWidget):
    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)

        self.video_info: dict | None = None
        self.last_estimate_summary: dict | None = None
        self.instant_estimate_text = "-"
        self.sampled_estimate_text = "-"
        self.sample_segments = 5
        self.sample_segment_sec = 12.0
        self.sample_fps = 8.0

        self.sample_proc: QProcess | None = None
        self._sample_buffer = ""
        self.sample_timer = QTimer(self)
        self.sample_timer.setSingleShot(True)
        self.sample_timer.timeout.connect(self._start_sample_estimate)

        self._build_ui()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        settings_scroll = QScrollArea()
        configure_settings_scroll(settings_scroll)

        settings = QWidget()
        settings.setObjectName("settingsPane")
        layout = QVBoxLayout(settings)
        layout.setContentsMargins(0, 0, 8, 4)
        layout.setSpacing(8)

        work_pane = QWidget()
        work_pane.setObjectName("workPane")
        work_layout = QVBoxLayout(work_pane)
        work_layout.setContentsMargins(12, 12, 12, 12)
        work_layout.setSpacing(10)

        # ===== 基本設定 =====
        basic = QFormLayout()
        basic.setSpacing(6)

        self.video_browse = BrowseWidget(
            mode="file",
            filter_str=i18n.t("VIDEO_FILE_FILTER"),
            placeholder=i18n.t("INPUT_VIDEO_PLACEHOLDER"),
        )
        self.video_browse.setToolTip(i18n.tip("INPUT_VIDEO"))
        self.video_browse.path_changed.connect(self._on_video_changed)
        basic.addRow(i18n.INPUT_VIDEO, self.video_browse)

        # モード選択
        self.change_radio = QRadioButton(i18n.MODE_CHANGE)
        self.change_radio.setToolTip(i18n.tip("MODE_CHANGE"))
        self.fixed_radio = QRadioButton(i18n.MODE_FIXED)
        self.fixed_radio.setToolTip(i18n.tip("MODE_FIXED"))
        self.fixed_radio.setChecked(True)
        self.change_radio.toggled.connect(self._update_mode_widgets)
        self.change_radio.toggled.connect(self._mark_estimate_stale)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.fixed_radio)
        mode_row.addWidget(self.change_radio)
        mode_row.addStretch()
        basic.addRow(i18n.EXTRACTION_MODE, mode_row)

        self.interval_edit = QLineEdit("0.8")
        self.interval_edit.setToolTip(i18n.tip("INTERVAL"))
        self.interval_edit.setFixedWidth(80)
        self.interval_edit.textChanged.connect(self._mark_estimate_stale)
        basic.addRow(i18n.INTERVAL, self.interval_edit)

        self.threshold_edit = QLineEdit("0.04")
        self.threshold_edit.setToolTip(i18n.tip("CHANGE_THRESHOLD"))
        self.threshold_edit.setFixedWidth(80)
        self.threshold_edit.textChanged.connect(self._mark_estimate_stale)
        basic.addRow(i18n.CHANGE_THRESHOLD, self.threshold_edit)

        self.min_gap_edit = QLineEdit("0.25")
        self.min_gap_edit.setToolTip(i18n.tip("MIN_GAP"))
        self.min_gap_edit.setFixedWidth(80)
        self.min_gap_edit.textChanged.connect(self._mark_estimate_stale)
        basic.addRow(i18n.MIN_GAP, self.min_gap_edit)

        self.max_gap_edit = QLineEdit("2.0")
        self.max_gap_edit.setToolTip(i18n.tip("MAX_GAP"))
        self.max_gap_edit.setFixedWidth(80)
        self.max_gap_edit.textChanged.connect(self._mark_estimate_stale)
        basic.addRow(i18n.MAX_GAP, self.max_gap_edit)

        # 画像形式
        fmt_row = QHBoxLayout()
        self.image_ext_combo = QComboBox()
        self.image_ext_combo.setToolTip(i18n.tip("IMAGE_FORMAT"))
        self.image_ext_combo.addItems(["jpg", "png"])
        self.image_ext_combo.setFixedWidth(80)
        self.image_ext_combo.currentIndexChanged.connect(self._mark_estimate_stale)
        fmt_row.addWidget(self.image_ext_combo)
        fmt_row.addWidget(QLabel(i18n.JPEG_QUALITY + ":"))
        self.jpg_quality_edit = QLineEdit("2")
        self.jpg_quality_edit.setToolTip(i18n.tip("JPEG_QUALITY"))
        self.jpg_quality_edit.setFixedWidth(50)
        fmt_row.addWidget(self.jpg_quality_edit)
        fmt_row.addStretch()
        basic.addRow(i18n.IMAGE_FORMAT, fmt_row)

        layout.addLayout(basic)

        # ===== 動画情報 (コンパクト) =====
        info_box = QHBoxLayout()
        info_box.setSpacing(12)

        self.load_info_btn = QPushButton(i18n.t("VIDEO_INFO_LOAD"))
        self.load_info_btn.setToolTip(i18n.tip("VIDEO_INFO_BTN"))
        self.load_info_btn.setMinimumWidth(150)
        self.load_info_btn.clicked.connect(lambda: self._load_video_info(show_error=True))
        info_box.addWidget(self.load_info_btn)
        info_box.addStretch()
        layout.addLayout(info_box)

        work_layout.addWidget(QLabel(i18n.VIDEO_INFO))
        self.video_info_label = QLabel(i18n.t("VIDEO_LABEL_DEFAULT"))
        self.video_info_label.setStyleSheet("color: #8888aa;")
        self.video_info_label.setWordWrap(True)
        work_layout.addWidget(self.video_info_label)

        work_layout.addWidget(QLabel(i18n.FRAME_ESTIMATE))
        self.estimate_label = QLabel()
        self.estimate_label.setWordWrap(True)
        self.estimate_label.setStyleSheet("color: #8888aa; font-size: 9pt;")
        self._refresh_estimate_label()
        work_layout.addWidget(self.estimate_label)
        work_layout.addStretch()

        # ===== 詳細設定 (折りたたみ) =====
        advanced = CollapsibleSection(i18n.t("ADVANCED_SETTINGS"), expanded=False)
        adv_form = QFormLayout()
        adv_form.setSpacing(6)

        self.analysis_width_edit = QLineEdit("1920")
        self.analysis_width_edit.setToolTip(i18n.tip("ANALYSIS_WIDTH"))
        self.analysis_width_edit.setFixedWidth(80)
        self.analysis_width_edit.textChanged.connect(self._mark_estimate_stale)
        adv_form.addRow(i18n.ANALYSIS_WIDTH, self.analysis_width_edit)

        self.blur_percentile_edit = QLineEdit("25.0")
        self.blur_percentile_edit.setToolTip(i18n.tip("BLUR_PERCENTILE"))
        self.blur_percentile_edit.setFixedWidth(80)
        self.blur_percentile_edit.textChanged.connect(self._mark_estimate_stale)
        adv_form.addRow(i18n.BLUR_PERCENTILE, self.blur_percentile_edit)

        self.blur_window_edit = QLineEdit("0")
        self.blur_window_edit.setToolTip(i18n.tip("BLUR_WINDOW"))
        self.blur_window_edit.setFixedWidth(80)
        self.blur_window_edit.textChanged.connect(self._mark_estimate_stale)
        adv_form.addRow(i18n.BLUR_WINDOW, self.blur_window_edit)

        # 立ち止まり間引き
        self.thin_motion_edit = QLineEdit("0.6")
        self.thin_motion_edit.setToolTip(i18n.t("THIN_MOTION_HINT"))
        self.thin_motion_edit.setFixedWidth(80)
        self.thin_motion_edit.textChanged.connect(self._mark_estimate_stale)
        adv_form.addRow(i18n.t("THIN_MOTION_THRESHOLD"), self.thin_motion_edit)

        # キャッシュ無効化チェックボックス
        from PySide6.QtWidgets import QCheckBox  # 局所 import: 既存 import 行への影響回避
        self.no_cache_cb = QCheckBox(i18n.t("NO_CACHE"))
        self.no_cache_cb.setToolTip(i18n.t("NO_CACHE_HINT"))
        adv_form.addRow("", self.no_cache_cb)

        # ffmpeg / ffprobe: PATH から自動検出して初期値にセット。参照ボタンで上書き可能
        ffmpeg_filter = "Executable (*.exe);;すべて (*.*)" if sys.platform == "win32" else "すべて (*.*)"
        self.ffmpeg_browse = BrowseWidget(mode="file", filter_str=ffmpeg_filter,
                                          placeholder="ffmpeg (PATH から自動検出)")
        self.ffmpeg_browse.set_text(_detect_binary("ffmpeg"))
        self.ffmpeg_browse.setToolTip(i18n.tip("FFMPEG_PATH"))
        adv_form.addRow(i18n.FFMPEG_PATH, self.ffmpeg_browse)

        self.ffprobe_browse = BrowseWidget(mode="file", filter_str=ffmpeg_filter,
                                           placeholder="ffprobe (PATH から自動検出)")
        self.ffprobe_browse.set_text(_detect_binary("ffprobe"))
        self.ffprobe_browse.setToolTip(i18n.tip("FFPROBE_PATH"))
        adv_form.addRow(i18n.FFPROBE_PATH, self.ffprobe_browse)

        self.prefix_edit = QLineEdit("")
        self.prefix_edit.setToolTip(i18n.tip("FILENAME_PREFIX"))
        self.prefix_edit.setPlaceholderText(i18n.t("AUTO_PREFIX_HINT"))
        adv_form.addRow(i18n.FILENAME_PREFIX, self.prefix_edit)

        self.refresh_sample_btn = QPushButton(i18n.t("SAMPLE_REFRESH"))
        self.refresh_sample_btn.setToolTip(i18n.tip("SAMPLE_BTN"))
        self.refresh_sample_btn.clicked.connect(self._run_sampled_estimate_now)
        adv_form.addRow("", self.refresh_sample_btn)

        advanced.content_layout.addLayout(adv_form)
        layout.addWidget(advanced)

        layout.addStretch()
        settings_scroll.setWidget(settings)
        splitter.addWidget(settings_scroll)
        splitter.addWidget(work_pane)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([SETTINGS_PANE_WIDTH, 760])
        root_layout.addWidget(splitter)
        self._update_mode_widgets()

    # -- シーンディレクトリ --

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)

    def primary_action_text(self) -> str:
        return i18n.EXTRACT_FRAMES

    # -- モード --

    def _mode(self) -> str:
        return "fixed" if self.fixed_radio.isChecked() else "change"

    def _update_mode_widgets(self) -> None:
        is_fixed = self._mode() == "fixed"
        self.interval_edit.setEnabled(is_fixed)
        self.threshold_edit.setEnabled(not is_fixed)
        self.min_gap_edit.setEnabled(not is_fixed)
        self.max_gap_edit.setEnabled(not is_fixed)

    # -- コマンド構築 --

    def build_commands(self) -> list[tuple[str, list[str]]]:
        return [("extract", self._build_extract_cmd())]

    def _build_extract_cmd(self) -> list[str]:
        video = self.video_browse.text()
        if not video:
            raise ValueError("入力動画が指定されていません")
        if not Path(video).exists():
            raise ValueError(f"入力動画が見つかりません: {video}")
        if not self.scene_dir:
            raise ValueError("シーンフォルダが指定されていません")

        script = self.base_dir / "extract_frames.py"
        if not script.exists():
            raise FileNotFoundError(f"extract_frames.py が見つかりません: {script}")

        cmd = [
            sys.executable, "-u", str(script),
            video, self.scene_dir,
            "--mode", self._mode(),
            "--analysis-width", self.analysis_width_edit.text().strip(),
            "--blur-percentile", self.blur_percentile_edit.text().strip(),
            "--blur-window-frames", self.blur_window_edit.text().strip(),
            "--image-ext", self.image_ext_combo.currentText(),
            "--jpg-quality", self.jpg_quality_edit.text().strip(),
            "--ffmpeg", self.ffmpeg_browse.text() or "ffmpeg",
            "--ffprobe", self.ffprobe_browse.text() or "ffprobe",
        ]
        prefix = self.prefix_edit.text().strip()
        if prefix:
            cmd.extend(["--filename-prefix", prefix])

        if self._mode() == "fixed":
            cmd.extend(["--interval-sec", self.interval_edit.text().strip()])
        else:
            cmd.extend([
                "--change-threshold", self.threshold_edit.text().strip(),
                "--min-gap-sec", self.min_gap_edit.text().strip(),
                "--max-gap-sec", self.max_gap_edit.text().strip(),
            ])

        thin_text = self.thin_motion_edit.text().strip()
        if thin_text:
            try:
                thin_val = float(thin_text)
            except ValueError:
                raise ValueError("立ち止まり間引き閾値は数値で指定してください")
            if thin_val < 0:
                raise ValueError("立ち止まり間引き閾値は 0 以上で指定してください")
            # 0.0 でも明示的に渡す（GUI 値が CLI デフォルトに上書きされないように）
            cmd.extend(["--thin-motion-threshold", f"{thin_val:g}"])

        if self.no_cache_cb.isChecked():
            cmd.append("--no-cache")
        return cmd

    # -- プログレス解析 --

    def on_line(self, line: str) -> tuple[int, int] | None:
        progress_prefix = "[progress] "
        if line.startswith(progress_prefix):
            text = line[len(progress_prefix):]
            match = re.search(r"(\d+)/(\d+)", text)
            if match:
                return int(match.group(1)), int(match.group(2))

        if line.startswith("SUMMARY_JSON:"):
            payload = line[len("SUMMARY_JSON:"):]
            try:
                summary = json.loads(payload)
                self.last_estimate_summary = summary
                self._apply_summary(summary)
            except Exception:
                pass
        return None

    # -- 動画情報 --

    def _on_video_changed(self, path: str) -> None:
        if path and Path(path).exists():
            self._load_video_info(show_error=False)

    @staticmethod
    def _parse_fraction(value: str) -> float:
        if not value:
            return 0.0
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f != 0 else 0.0
        return float(value)

    @staticmethod
    def _format_duration(sec: float) -> str:
        whole = int(max(0, sec))
        h, m, s = whole // 3600, (whole % 3600) // 60, whole % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _load_video_info(self, show_error: bool = True) -> bool:
        try:
            self.video_info = self._probe_video_info()
            self._update_video_info_label()
            self._mark_estimate_stale()
            return True
        except Exception as e:
            self.video_info = None
            self._update_video_info_label()
            self._cancel_sample_estimate()
            self.instant_estimate_text = "-"
            self.sampled_estimate_text = "-"
            self._refresh_estimate_label()
            if show_error:
                QMessageBox.critical(self, i18n.INVALID_INPUT, str(e))
            return False

    def _probe_video_info(self) -> dict:
        video = self.video_browse.text()
        ffprobe = self.ffprobe_browse.text() or "ffprobe"
        if not video:
            raise ValueError("入力動画が指定されていません")
        if not Path(video).exists():
            raise ValueError(f"入力動画が見つかりません: {video}")

        cmd = [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
            "-show_entries", "format=duration", "-of", "json", video,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "ffprobe 失敗")

        data = json.loads(proc.stdout)
        streams = data.get("streams", [])
        if not streams:
            raise RuntimeError("動画ストリームが見つかりません")

        s = streams[0]
        w, h = int(s.get("width", 0)), int(s.get("height", 0))
        fps = self._parse_fraction(s.get("avg_frame_rate", "0"))
        if fps <= 0:
            fps = self._parse_fraction(s.get("r_frame_rate", "0"))
        dur = float(s.get("duration") or data.get("format", {}).get("duration") or 0.0)
        nb = int(s["nb_frames"]) if s.get("nb_frames", "").isdigit() else 0

        if fps <= 0 and dur > 0 and nb > 0:
            fps = nb / dur
        if fps <= 0:
            raise RuntimeError("FPSを取得できません")
        if dur <= 0 and nb > 0:
            dur = nb / fps
        if nb <= 0 and dur > 0:
            nb = max(1, int(round(dur * fps)))

        return {"width": w, "height": h, "fps": fps, "duration_sec": dur, "total_frames": nb}

    def _update_video_info_label(self) -> None:
        if not self.video_info:
            self.video_info_label.setText("動画: -")
            return
        i = self.video_info
        d = self._format_duration(float(i["duration_sec"]))
        self.video_info_label.setText(
            f"{i['width']}x{i['height']}  |  {i['fps']:.2f} fps  |  {d}  |  {i['total_frames']} フレーム"
        )

    # -- フレーム数推定 --

    def _refresh_estimate_label(self) -> None:
        self.estimate_label.setText(
            f"{i18n.INSTANT_ESTIMATE}: {self.instant_estimate_text}    "
            f"{i18n.SAMPLED_ESTIMATE}: {self.sampled_estimate_text}"
        )

    def _mark_estimate_stale(self, *_args) -> None:
        self.last_estimate_summary = None
        self._update_instant_estimate()
        if self._mode() == "fixed":
            self.sampled_estimate_text = "-"
            self._cancel_sample_estimate()
            self._refresh_estimate_label()
            return
        if not self.video_info:
            self.sampled_estimate_text = "-"
            self._cancel_sample_estimate()
            self._refresh_estimate_label()
            return
        self._schedule_sample_estimate()

    def _update_instant_estimate(self) -> None:
        if not self.video_info:
            self.instant_estimate_text = "-"
            self._refresh_estimate_label()
            return
        dur = float(self.video_info.get("duration_sec", 0))
        fps = float(self.video_info.get("fps", 0))
        total = int(self.video_info.get("total_frames", 0))

        if self._mode() == "fixed":
            try:
                iv = float(self.interval_edit.text().strip())
                if iv <= 0:
                    raise ValueError
                if total <= 0 and dur > 0 and fps > 0:
                    total = max(1, int(round(dur * fps)))
                step = max(1, int(round(iv * fps)))
                indices = list(range(0, max(total, 1), step))
                if indices[-1] != max(total - 1, 0):
                    indices.append(max(total - 1, 0))
                self.instant_estimate_text = f"{len(indices)} フレーム"
            except Exception:
                self.instant_estimate_text = "-"
            self._refresh_estimate_label()
            return

        try:
            thr = float(self.threshold_edit.text().strip())
            min_g = float(self.min_gap_edit.text().strip())
            max_g = float(self.max_gap_edit.text().strip())
            if min_g <= 0 or max_g <= 0 or max_g < min_g:
                raise ValueError
            min_c = max(1, int(math.ceil(dur / max_g)))
            max_c = max(min_c, int(math.ceil(dur / min_g)))
            low, high = 0.01, 0.12
            w = max(0.0, min(1.0, (thr - low) / (high - low)))
            est_gap = min_g + w * (max_g - min_g)
            est = max(1, int(math.ceil(dur / max(est_gap, 1e-6)))) if dur > 0 else 1
            est = max(min_c, min(max_c, est))
            self.instant_estimate_text = f"{min_c}-{max_c} フレーム (推定: {est})"
        except Exception:
            self.instant_estimate_text = "-"
        self._refresh_estimate_label()

    # -- サンプル推定 --

    def _cancel_sample_estimate(self) -> None:
        self.sample_timer.stop()
        proc = self.sample_proc
        self.sample_proc = None
        self._sample_buffer = ""
        if proc is not None and proc.state() != QProcess.NotRunning:
            proc.kill()
            proc.waitForFinished(2000)

    def _schedule_sample_estimate(self, delay_ms: int = 500) -> None:
        if self._mode() != "change" or not self.video_info:
            return
        self._cancel_sample_estimate()
        self.sampled_estimate_text = "計算中..."
        self._refresh_estimate_label()
        self.sample_timer.start(max(0, delay_ms))

    def _start_sample_estimate(self) -> None:
        if self._mode() != "change" or not self.video_info:
            return
        try:
            cmd = self._build_extract_cmd()
        except Exception as e:
            self.sampled_estimate_text = f"- ({e})"
            self._refresh_estimate_label()
            return
        cmd.extend([
            "--estimate-only", "--estimate-mode", "sampled",
            "--sample-segments", str(self.sample_segments),
            "--sample-segment-sec", str(self.sample_segment_sec),
            "--sample-fps", str(self.sample_fps),
            "--print-summary-json",
        ])
        self._cancel_sample_estimate()

        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda p=proc: self._on_sample_output(p))
        proc.finished.connect(lambda code, status, p=proc: self._on_sample_finished(p, code))
        self.sample_proc = proc
        proc.start()

    def _on_sample_output(self, proc: QProcess) -> None:
        if proc is not self.sample_proc:
            return
        data = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._sample_buffer += data
        while "\n" in self._sample_buffer:
            line, self._sample_buffer = self._sample_buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line and line.startswith("SUMMARY_JSON:"):
                try:
                    summary = json.loads(line[len("SUMMARY_JSON:"):])
                    self.last_estimate_summary = summary
                    self._apply_summary(summary)
                except Exception:
                    pass

    def _on_sample_finished(self, proc: QProcess, exit_code: int) -> None:
        if proc is not self.sample_proc:
            return
        if self._sample_buffer:
            line = self._sample_buffer.rstrip("\r")
            if line.startswith("SUMMARY_JSON:"):
                try:
                    summary = json.loads(line[len("SUMMARY_JSON:"):])
                    self._apply_summary(summary)
                except Exception:
                    pass
            self._sample_buffer = ""
        if exit_code != 0 and self._mode() == "change":
            self.sampled_estimate_text = "- (失敗)"
            self._refresh_estimate_label()
        self.sample_proc = None

    def _run_sampled_estimate_now(self) -> None:
        if self._mode() != "change":
            self.sampled_estimate_text = "-"
            self._refresh_estimate_label()
            return
        if not self.video_info:
            if not self._load_video_info(show_error=True):
                return
        self._schedule_sample_estimate(delay_ms=0)

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
        total_f = int(video.get("total_frames", 0))
        ratio = (selected / total_f * 100.0) if total_f > 0 else 0.0
        est = summary.get("estimate", {})
        parts = [f"{selected} フレーム ({ratio:.1f}%)"]
        if est.get("sampled_segments_used") is not None:
            parts.append(f"seg={est['sampled_segments_used']}/{est.get('sampled_segments_requested', '?')}")
        if est.get("range_min_count") is not None:
            parts.append(f"range={est['range_min_count']}-{est.get('range_max_count', '?')}")
        self.sampled_estimate_text = ", ".join(parts)
        self._refresh_estimate_label()
