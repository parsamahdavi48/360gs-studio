"""3DGS Studio メインウィンドウ"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.log_panel import LogPanel
from gui.common.process_runner import ProcessRunner
from gui.common.progress_widget import ProgressWidget
from gui.steps.step1_extract import ExtractStep
from gui.steps.step2_review import ReviewStep
from gui.steps.step3_mask import MaskStep
from gui.steps.step4_cubemap import CubemapStep
from gui.theme import apply_theme


class MainWindow(QWidget):
    def __init__(self, initial_scene_dir: str = ".") -> None:
        super().__init__()
        self.base_dir = Path(__file__).resolve().parent.parent
        self.setWindowTitle(f"{i18n.APP_TITLE}  v0.1")
        self.resize(1280, 920)

        self.runner = ProcessRunner(self)
        self._current_step: int = 0

        self._build_ui(initial_scene_dir)
        self._connect_signals()

    def _build_ui(self, initial_scene_dir: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # --- ヘッダー: シーンディレクトリ ---
        header = QHBoxLayout()
        header.setSpacing(8)
        scene_label = QLabel(f"  {i18n.SCENE_DIR}")
        scene_label.setStyleSheet("font-weight: 600; font-size: 11pt;")
        header.addWidget(scene_label)
        self.scene_browse = BrowseWidget(mode="dir", placeholder="シーンフォルダを選択...")
        self.scene_browse.set_text(initial_scene_dir)
        header.addWidget(self.scene_browse, stretch=1)
        root.addLayout(header)

        # --- タブ ---
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.step1 = ExtractStep(self.base_dir)
        self.step2 = ReviewStep(self.base_dir)
        self.step3 = MaskStep(self.base_dir)
        self.step4 = CubemapStep(self.base_dir)

        self.tabs.addTab(self.step1, i18n.STEP1_TITLE)
        self.tabs.addTab(self.step2, i18n.STEP2_TITLE)
        self.tabs.addTab(self.step3, i18n.STEP3_TITLE)
        self.tabs.addTab(self.step4, i18n.STEP4_TITLE)
        self.steps = [self.step1, self.step2, self.step3, self.step4]

        # --- メイン分割: タブ (上) / コントロール+ログ (下) ---
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.tabs)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 8, 0, 0)
        bottom_layout.setSpacing(6)

        # プログレス
        self.progress = ProgressWidget()
        bottom_layout.addWidget(self.progress)

        # ログ
        self.log_panel = LogPanel()
        self.log_panel.setMinimumHeight(100)
        bottom_layout.addWidget(self.log_panel)

        # 実行 / キャンセル (中央揃え)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.addStretch()

        self.run_btn = QPushButton(f"  {i18n.RUN}")
        self.run_btn.setObjectName("primary")
        self.run_btn.setFixedWidth(160)
        self.run_btn.setFixedHeight(36)
        btn_row.addWidget(self.run_btn)

        self.cancel_btn = QPushButton(f"  {i18n.CANCEL}")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.setFixedWidth(160)
        self.cancel_btn.setFixedHeight(36)
        self.cancel_btn.setEnabled(False)
        btn_row.addWidget(self.cancel_btn)

        btn_row.addStretch()
        bottom_layout.addLayout(btn_row)

        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    def _connect_signals(self) -> None:
        self.scene_browse.path_changed.connect(self._on_scene_changed)
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.runner.line_received.connect(self._on_line)
        self.runner.phase_started.connect(self._on_phase_started)
        self.runner.phase_finished.connect(self._on_phase_finished)
        self.runner.queue_finished.connect(self._on_queue_finished)

        self.step2.run_requested.connect(self._on_run)
        self.step3.run_requested.connect(self._on_run)

        self._on_scene_changed(self.scene_browse.text())

    def _current_step_widget(self):
        idx = self.tabs.currentIndex()
        if 0 <= idx < len(self.steps):
            return self.steps[idx]
        return None

    def _on_scene_changed(self, path: str) -> None:
        for step in self.steps:
            step.set_scene_dir(path)

    def _on_tab_changed(self, index: int) -> None:
        self._update_run_button()

    def _update_run_button(self) -> None:
        running = self.runner.is_running()
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)

    def _on_run(self) -> None:
        step = self._current_step_widget()
        if step is None:
            return
        if self.runner.is_running():
            QMessageBox.warning(self, i18n.APP_TITLE, i18n.BUSY_MSG)
            return
        try:
            commands = step.build_commands()
        except (ValueError, FileNotFoundError) as e:
            QMessageBox.critical(self, i18n.INVALID_INPUT, str(e))
            return
        if not commands:
            return
        self._current_step = self.tabs.currentIndex()
        self.progress.reset()
        self.runner.start_queue(commands)
        self._update_run_button()

    def _on_cancel(self) -> None:
        self.runner.cancel()
        self.progress.set_status(i18n.STATUS_CANCELED)
        self._update_run_button()

    def _on_line(self, line: str) -> None:
        self.log_panel.append_log(line)
        step = self.steps[self._current_step] if 0 <= self._current_step < len(self.steps) else None
        if step:
            result = step.on_line(line)
            if result is not None:
                done, total = result
                self.progress.set_progress(done, total)

    def _on_phase_started(self, phase: str) -> None:
        self.progress.set_status(f"{i18n.STATUS_RUNNING}: {phase}")
        self._update_run_button()

    def _on_phase_finished(self, phase: str, exit_code: int, canceled: bool) -> None:
        step = self.steps[self._current_step] if 0 <= self._current_step < len(self.steps) else None
        if step:
            step.on_phase_finished(phase, exit_code, canceled)

    def _on_queue_finished(self, success: bool) -> None:
        step = self.steps[self._current_step] if 0 <= self._current_step < len(self.steps) else None
        if step:
            step.on_queue_finished(success)
        if success:
            self.progress.set_status(i18n.STATUS_DONE)
        else:
            status = self.progress.status_label.text()
            if i18n.STATUS_CANCELED not in status:
                self.progress.set_status(i18n.STATUS_FAILED)
        self._update_run_button()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.runner.is_running():
            self.runner.cancel()
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description="3DGS Studio")
    parser.add_argument("--scene", default=".", help="Initial scene directory")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    apply_theme(app)

    window = MainWindow(initial_scene_dir=args.scene)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
