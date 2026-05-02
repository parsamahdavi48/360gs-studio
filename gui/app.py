"""STechDrive 3DGS Utils メインウィンドウ"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
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
from gui.version import app_version_label


class MainWindow(QWidget):
    def __init__(self, initial_scene_dir: str = "") -> None:
        super().__init__()
        self.base_dir = Path(__file__).resolve().parent.parent
        self.setWindowTitle(f"{i18n.APP_TITLE}  {app_version_label()}")
        self.resize(1280, 920)

        self.runner = ProcessRunner(self)
        self._current_step: int = 0

        self._build_ui(initial_scene_dir)
        self._connect_signals()

    def _build_ui(self, initial_scene_dir: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- ヘッダー ---
        header = QHBoxLayout()
        header.setContentsMargins(14, 10, 14, 10)
        header.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel(i18n.APP_TITLE)
        title.setObjectName("appTitle")
        subtitle = QLabel(i18n.t("APP_SUBTITLE"))
        subtitle.setObjectName("appSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)

        header.addWidget(QLabel(i18n.SCENE_DIR))
        self.scene_browse = BrowseWidget(mode="dir", placeholder=i18n.t("SCENE_DIR_PLACEHOLDER"))
        self.scene_browse.setToolTip(i18n.tip("SCENE_DIR"))
        if initial_scene_dir:
            self.scene_browse.set_text(initial_scene_dir)
        header.addWidget(self.scene_browse, stretch=1)
        header_widget = QWidget()
        header_widget.setObjectName("appHeader")
        header_widget.setLayout(header)
        root.addWidget(header_widget)

        self.step1 = ExtractStep(self.base_dir)
        self.step2 = ReviewStep(self.base_dir)
        self.step3 = MaskStep(self.base_dir)
        self.step4 = CubemapStep(self.base_dir)
        self.steps = [self.step1, self.step2, self.step3, self.step4]
        self.step_titles = [i18n.STEP1_TITLE, i18n.STEP2_TITLE, i18n.STEP3_TITLE, i18n.STEP4_TITLE]
        self.step_nav_titles = [
            i18n.t("STEP1_NAV"),
            i18n.t("STEP2_NAV"),
            i18n.t("STEP3_NAV"),
            i18n.t("STEP4_NAV"),
        ]
        self.step_descriptions = [
            i18n.t("STEP1_DESC"),
            i18n.t("STEP2_DESC"),
            i18n.t("STEP3_DESC"),
            i18n.t("STEP4_DESC"),
        ]

        # --- メイン分割: 作業領域 / 実行状態 ---
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        workspace = QWidget()
        workspace_layout = QHBoxLayout(workspace)
        workspace_layout.setContentsMargins(12, 12, 12, 8)
        workspace_layout.setSpacing(8)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(84)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(6, 8, 6, 8)
        sidebar_layout.setSpacing(6)
        self.step_buttons: list[QPushButton] = []
        for index, title_text in enumerate(self.step_nav_titles):
            btn = QPushButton(title_text)
            btn.setObjectName("navStep")
            btn.setCheckable(True)
            btn.setFixedSize(70, 64)
            btn.setToolTip(self.step_titles[index])
            btn.clicked.connect(lambda _checked=False, i=index: self._set_current_step(i))
            sidebar_layout.addWidget(btn)
            self.step_buttons.append(btn)
        sidebar_layout.addStretch()
        workspace_layout.addWidget(sidebar)

        content_panel = QWidget()
        content_panel.setObjectName("contentPanel")
        content_layout = QVBoxLayout(content_panel)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(8)

        self.step_header = QLabel("")
        self.step_header.setObjectName("stepHeader")
        content_layout.addWidget(self.step_header)
        self.step_subheader = QLabel("")
        self.step_subheader.setObjectName("stepSubheader")
        self.step_subheader.setWordWrap(True)
        content_layout.addWidget(self.step_subheader)

        self.stack = QStackedWidget()
        for step in self.steps:
            self.stack.addWidget(step)
        content_layout.addWidget(self.stack, stretch=1)
        workspace_layout.addWidget(content_panel, stretch=1)
        splitter.addWidget(workspace)

        job_panel = QWidget()
        job_panel.setObjectName("jobPanel")
        bottom_layout = QVBoxLayout(job_panel)
        bottom_layout.setContentsMargins(12, 8, 12, 12)
        bottom_layout.setSpacing(6)

        self.progress = ProgressWidget()
        bottom_layout.addWidget(self.progress)

        self.log_panel = LogPanel()
        self.log_panel.setMinimumHeight(100)
        bottom_layout.addWidget(self.log_panel)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.addStretch()

        self.run_btn = QPushButton(f"  {i18n.RUN}")
        self.run_btn.setObjectName("primary")
        self.run_btn.setToolTip(i18n.tip("RUN"))
        self.run_btn.setMinimumWidth(220)
        self.run_btn.setFixedHeight(36)
        btn_row.addWidget(self.run_btn)

        self.cancel_btn = QPushButton(f"  {i18n.CANCEL}")
        self.cancel_btn.setToolTip(i18n.tip("CANCEL"))
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.setFixedWidth(160)
        self.cancel_btn.setFixedHeight(36)
        self.cancel_btn.setEnabled(False)
        btn_row.addWidget(self.cancel_btn)

        btn_row.addStretch()
        bottom_layout.addLayout(btn_row)

        splitter.addWidget(job_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([660, 220])
        root.addWidget(splitter)
        self._set_current_step(0)

    def _connect_signals(self) -> None:
        self.scene_browse.path_changed.connect(self._on_scene_changed)
        self.step1.scene_dir_suggested.connect(self._on_scene_suggested)
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn.clicked.connect(self._on_cancel)

        self.runner.line_received.connect(self._on_line)
        self.runner.phase_started.connect(self._on_phase_started)
        self.runner.phase_finished.connect(self._on_phase_finished)
        self.runner.queue_finished.connect(self._on_queue_finished)
        for step in self.steps:
            step.primary_action_state_changed.connect(self._update_run_button)

        self._on_scene_changed(self.scene_browse.text())

    def _current_step_widget(self):
        idx = self.stack.currentIndex()
        if 0 <= idx < len(self.steps):
            return self.steps[idx]
        return None

    def _on_scene_changed(self, path: str) -> None:
        for step in self.steps:
            step.set_scene_dir(path)
        step = self._current_step_widget()
        if step is not None:
            step.on_activated()

    def _on_scene_suggested(self, path: str) -> None:
        if self.scene_browse.text():
            return
        candidate = Path(path)
        if not candidate.is_dir():
            return
        self.scene_browse.set_text(str(candidate))

    def _set_current_step(self, index: int) -> None:
        if not 0 <= index < len(self.steps):
            return
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.step_buttons):
            btn.setChecked(i == index)
        self.step_header.setText(self.step_titles[index])
        self.step_subheader.setText(self.step_descriptions[index])
        step = self._current_step_widget()
        if step is not None:
            step.on_activated()
        self._update_run_button()

    def _update_run_button(self) -> None:
        running = self.runner.is_running()
        step = self._current_step_widget()
        scene_selected = bool(self.scene_browse.text())
        if step is not None:
            self.run_btn.setText(f"  {step.primary_action_text()}")
            if scene_selected:
                self.run_btn.setToolTip(step.primary_action_tooltip())
            else:
                self.run_btn.setToolTip(i18n.t("SCENE_REQUIRED_ACTION_HINT"))

        self.run_btn.setVisible(True)
        action_enabled = step.primary_action_enabled() if step is not None else True
        self.run_btn.setEnabled(not running and scene_selected and action_enabled)

        self.cancel_btn.setVisible(True)
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
        self._current_step = self.stack.currentIndex()
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
    parser = argparse.ArgumentParser(description=i18n.APP_TITLE)
    parser.add_argument("--scene", default="", help="Initial scene directory")
    parser.add_argument(
        "--version",
        action="version",
        version=f"{i18n.APP_TITLE} {app_version_label()}",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    apply_theme(app)

    window = MainWindow(initial_scene_dir=args.scene)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
