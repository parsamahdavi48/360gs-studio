"""STechDrive 3DGS Utils メインウィンドウ"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, QSignalBlocker, QSize, Qt, QUrl
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.app_job import frame_app_job
from core.frame_job_spec import import_scene_job
from core.path_safety import PathSafetyIssue, check_path_safety, normalized_path_text
from gui import i18n
from gui.common import dialogs
from gui.common.browse_widget import BrowseWidget
from gui.common.icons import back_icon, folder_icon, help_icon, import_scene_icon, menu_icon, reset_icon
from gui.common.log_panel import LogPanel
from gui.common.process_runner import ProcessRunner
from gui.common.progress_widget import ProgressWidget
from gui.steps.base_step import BaseStepWidget
from gui.steps.dataset_step import DatasetStep
from gui.steps.perspective_export import PerspectiveExportStep
from gui.steps.sfm_step import SfmStep
from gui.steps.step1_extract import ExtractStep
from gui.steps.step2_review import ReviewStep
from gui.steps.step3_mask import MaskStep
from gui.steps.step4_cubemap import CubemapStep
from gui.steps.step5_training import TrainingStep
from gui.theme import apply_theme
from gui.version import app_version_label

_GITHUB_DOC_BASE_URL = "https://github.com/stechdrive/360gs-studio/blob/main/doc"
_STEP_HELP_DOC_STEMS = (
    "extract_frames_gui",
    "review_frames_gui",
    "mask_tools_gui",
    "cubemap_tools_gui",
    "cubemap_tools_gui",
    "training_gui",
    "perspective_export",
)
def app_icon() -> QIcon:
    icon_path = Path(__file__).resolve().parent / "assets" / "app_icon.ico"
    if not icon_path.exists():
        icon_path = Path(__file__).resolve().parent / "assets" / "app_icon.svg"
    return QIcon(str(icon_path))


def step_help_url(index: int, *, lang: str | None = None) -> str:
    if not 0 <= index < len(_STEP_HELP_DOC_STEMS):
        raise IndexError(f"step index out of range: {index}")
    suffix = ".ja.md" if (lang or i18n.LANG).lower().startswith("ja") else ".md"
    return f"{_GITHUB_DOC_BASE_URL}/{_STEP_HELP_DOC_STEMS[index]}{suffix}"


class MainWindow(QMainWindow):
    def __init__(self, initial_scene_dir: str = "") -> None:
        super().__init__()
        self.base_dir = Path(__file__).resolve().parent.parent
        self.setWindowTitle(f"{i18n.APP_TITLE}  {app_version_label()}")
        self.setWindowIcon(app_icon())
        self.resize(1280, 920)

        self.runner = ProcessRunner(self)
        self._current_step: int = 0
        self._auto_scene_from_input: str | None = None
        self._applying_scene_suggestion = False
        self._shutdown = False
        self._scene_import_running = False
        self._scene_import_cancel_requested = False
        self._scene_import_canceled = False
        self._scene_import_scene = ""
        self._scene_import_summary: dict[str, object] | None = None
        self._deferred_scene_sync_path: str | None = None
        self._deferred_scene_sync_step_ids: set[int] = set()
        self._settings = QSettings("360GS Studio", "360GS Studio")

        self._build_ui(initial_scene_dir)
        self._connect_signals()
        self._restore_layout()

    def _build_ui(self, initial_scene_dir: str) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
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
        title_box.addWidget(title)
        header.addLayout(title_box)

        header.addWidget(QLabel(i18n.SCENE_DIR))
        self.scene_browse = BrowseWidget(mode="dir", placeholder=i18n.t("SCENE_DIR_PLACEHOLDER"))
        self.scene_browse.setToolTip(i18n.tip("SCENE_DIR"))
        if initial_scene_dir:
            self.scene_browse.set_text(initial_scene_dir)
        self.clear_scene_btn = self.scene_browse.add_icon_button(
            reset_icon(),
            i18n.t("CLEAR_SCENE_DIR_HINT"),
            self._clear_scene_dir,
            accessible_name=i18n.t("CLEAR_SCENE_DIR"),
        )
        self.clear_scene_btn.setEnabled(bool(initial_scene_dir))
        self.scene_menu_btn = self.scene_browse.add_icon_button(
            menu_icon(),
            i18n.t("SCENE_ACTIONS_MENU_HINT"),
            None,
            accessible_name=i18n.t("SCENE_ACTIONS_MENU"),
        )
        self.scene_menu_btn.setProperty("hideMenuIndicator", "true")
        self.scene_actions_menu = QMenu(self.scene_menu_btn)
        self.scene_actions_menu.setToolTipsVisible(True)
        self.open_scene_action = QAction(
            folder_icon(),
            i18n.t("OPEN_SCENE_DIR"),
            self.scene_actions_menu,
        )
        self.open_scene_action.setToolTip(i18n.t("OPEN_SCENE_DIR_HINT"))
        self.open_scene_action.setStatusTip(i18n.t("OPEN_SCENE_DIR_HINT"))
        self.open_scene_action.triggered.connect(lambda _checked=False: self.scene_browse.browse())
        self.clear_scene_action = QAction(
            reset_icon(),
            i18n.t("CLEAR_SCENE_DIR_MENU_ITEM"),
            self.scene_actions_menu,
        )
        self.clear_scene_action.setToolTip(i18n.t("CLEAR_SCENE_DIR_HINT"))
        self.clear_scene_action.setStatusTip(i18n.t("CLEAR_SCENE_DIR_HINT"))
        self.clear_scene_action.triggered.connect(lambda _checked=False: self._clear_scene_dir())
        self.clear_scene_action.setEnabled(bool(initial_scene_dir))
        self.import_scene_action = QAction(
            import_scene_icon(),
            i18n.t("IMPORT_SCENE_MENU_ITEM"),
            self.scene_actions_menu,
        )
        self.import_scene_action.setToolTip(i18n.tip("IMPORT_SCENE"))
        self.import_scene_action.setStatusTip(i18n.tip("IMPORT_SCENE"))
        self.import_scene_action.triggered.connect(lambda _checked=False: self._import_scene_from_folder())
        self.scene_actions_menu.addAction(self.open_scene_action)
        self.scene_actions_menu.addAction(self.clear_scene_action)
        self.scene_actions_menu.addSeparator()
        self.scene_actions_menu.addAction(self.import_scene_action)
        self.scene_menu_btn.setMenu(self.scene_actions_menu)
        self.scene_menu_btn.setPopupMode(QToolButton.InstantPopup)
        header.addWidget(self.scene_browse, stretch=1)
        ffmpeg_ready = bool(shutil.which("ffmpeg.exe") or shutil.which("ffmpeg"))
        gpu_ready = bool(shutil.which("nvidia-smi.exe") or shutil.which("nvidia-smi"))
        self.component_status = QLabel("FFmpeg ✓" if ffmpeg_ready else "FFmpeg —")
        self.component_status.setObjectName("statusPill")
        self.component_status.setToolTip("Run 360gs-studio doctor for codec and encoder capabilities.")
        header.addWidget(self.component_status)
        self.gpu_status = QLabel("NVIDIA ✓" if gpu_ready else "CPU mode")
        self.gpu_status.setObjectName("statusPill")
        self.gpu_status.setToolTip("GPU features are enabled only after their individual capability preflight passes.")
        header.addWidget(self.gpu_status)
        header_widget = QWidget()
        header_widget.setObjectName("appHeader")
        header_widget.setLayout(header)
        root.addWidget(header_widget)

        self.step1 = ExtractStep(self.base_dir)
        self.step2 = ReviewStep(self.base_dir)
        self.step3 = MaskStep(self.base_dir)
        self.step4 = CubemapStep(self.base_dir)
        self.sfm_step = SfmStep(self.base_dir, self.step4)
        self.dataset_step = DatasetStep(self.base_dir, self.step4)
        self.step5 = TrainingStep(self.base_dir, self.step4)
        self.perspective_step = PerspectiveExportStep(self.base_dir)
        self.step4.enable_user_preferences()
        self.steps = [
            self.step1,
            self.step2,
            self.step3,
            self.sfm_step,
            self.dataset_step,
            self.step5,
            self.perspective_step,
        ]
        self._sfm_step_index = 3
        self._dataset_step_index = 4
        self._training_step_index = 5
        self.step_titles = [
            i18n.STEP1_TITLE,
            i18n.STEP2_TITLE,
            i18n.STEP3_TITLE,
            i18n.STEP4_TITLE,
            i18n.STEP5_TITLE,
            i18n.t("STEP6_TITLE"),
            i18n.t("STEP7_TITLE"),
        ]
        self.step_nav_titles = [
            i18n.t("STEP1_NAV"),
            i18n.t("STEP2_NAV"),
            i18n.t("STEP3_NAV"),
            i18n.t("STEP4_NAV"),
            i18n.t("STEP5_NAV"),
            i18n.t("STEP6_NAV"),
            i18n.t("STEP7_NAV"),
        ]
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

        step_header_row = QHBoxLayout()
        step_header_row.setContentsMargins(0, 0, 0, 0)
        step_header_row.setSpacing(8)
        self.step_back_btn = QToolButton()
        self.step_back_btn.setObjectName("iconToolButton")
        self.step_back_btn.setIcon(back_icon())
        self.step_back_btn.setIconSize(QSize(18, 18))
        self.step_back_btn.setFixedSize(28, 28)
        self.step_back_btn.setVisible(False)
        self.step_back_btn.clicked.connect(self._on_step_header_back)
        step_header_row.addWidget(self.step_back_btn)
        self.step_header = QLabel("")
        self.step_header.setObjectName("stepHeader")
        step_header_row.addWidget(self.step_header)
        self.step_subheader = QLabel("")
        self.step_subheader.setObjectName("stepSubheader")
        self.step_subheader.setWordWrap(False)
        self.step_subheader.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.step_subheader.setVisible(False)
        step_header_row.addWidget(self.step_subheader)
        step_header_row.addStretch(1)
        self.step_help_btn = QToolButton()
        self.step_help_btn.setObjectName("iconToolButton")
        self.step_help_btn.setIcon(help_icon())
        self.step_help_btn.setIconSize(QSize(18, 18))
        self.step_help_btn.setToolTip(i18n.tip("STEP_HELP_BUTTON"))
        self.step_help_btn.setAccessibleName(i18n.t("STEP_HELP_BUTTON"))
        self.step_help_btn.setFixedSize(28, 28)
        self.step_help_btn.clicked.connect(self._open_step_help)
        step_header_row.addWidget(self.step_help_btn)
        content_layout.addLayout(step_header_row)

        self.stack = QStackedWidget()
        for step in self.steps:
            self.stack.addWidget(step)
        content_layout.addWidget(self.stack, stretch=1)
        workspace_layout.addWidget(content_panel, stretch=1)
        root.addWidget(workspace, stretch=1)
        self.setCentralWidget(central)

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

        self.jobs_dock = QDockWidget("Jobs, progress and logs", self)
        self.jobs_dock.setObjectName("jobsDock")
        self.jobs_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.jobs_dock.setWidget(job_panel)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.jobs_dock)
        self._set_current_step(0)

    def _restore_layout(self) -> None:
        geometry = self._settings.value("window/geometry")
        state = self._settings.value("window/state")
        if geometry:
            self.restoreGeometry(geometry)
        if state:
            self.restoreState(state)

    def _save_layout(self) -> None:
        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.setValue("window/state", self.saveState())

    def _connect_signals(self) -> None:
        self.scene_browse.path_changed.connect(self._on_scene_changed)
        self.step1.scene_dir_suggested.connect(self._on_scene_suggested)
        self.step1.input_videos_cleared.connect(self._on_input_videos_cleared)
        self.sfm_step.route_requested.connect(self._open_dataset_route)
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn.clicked.connect(self._on_cancel)

        self.runner.line_received.connect(self._on_line)
        self.runner.phase_started.connect(self._on_phase_started)
        self.runner.phase_log_started.connect(self._on_phase_log_started)
        self.runner.phase_finished.connect(self._on_phase_finished)
        self.runner.queue_finished.connect(self._on_queue_finished)
        for step in self.steps:
            step.primary_action_state_changed.connect(self._update_run_button)
            step.background_task_started.connect(self._on_background_task_started)
            step.background_line_received.connect(self._on_background_line)
            step.background_progress_changed.connect(self._on_background_progress)
            step.background_status_changed.connect(self._on_background_status)
            step.background_task_finished.connect(self._on_background_task_finished)

        self._apply_scene_dir(self.scene_browse.text(), activate_current=True)

    def _current_step_widget(self):
        idx = self.stack.currentIndex()
        if 0 <= idx < len(self.steps):
            return self.steps[idx]
        return None

    def _on_scene_changed(self, path: str) -> None:
        self._apply_scene_dir(path, activate_current=True, defer_step_sync=bool(path))

    def _apply_scene_dir(self, path: str, *, activate_current: bool, defer_step_sync: bool = False) -> None:
        if not self._applying_scene_suggestion and path != self._auto_scene_from_input:
            self._auto_scene_from_input = None
        if defer_step_sync and path:
            self._deferred_scene_sync_path = path
            self._deferred_scene_sync_step_ids = {id(step) for step in self.steps}
        else:
            self._clear_deferred_scene_sync()
            for step in self.steps:
                step.set_scene_dir(path)
        self.clear_scene_btn.setEnabled(bool(path))
        self.clear_scene_action.setEnabled(bool(path))
        if activate_current:
            step = self._current_step_widget()
            if step is not None:
                was_deferred = self._step_scene_sync_deferred(step)
                self._sync_step_scene_if_deferred(step)
                step.on_activated()
                if was_deferred:
                    self._update_run_button()
        self._update_step_header()

    def _set_scene_browse_text_silently(self, path: str) -> None:
        blocker = QSignalBlocker(self.scene_browse.line_edit)
        try:
            self.scene_browse.set_text(path)
        finally:
            del blocker

    def _clear_deferred_scene_sync(self) -> None:
        self._deferred_scene_sync_path = None
        self._deferred_scene_sync_step_ids.clear()

    def _step_scene_sync_deferred(self, step: BaseStepWidget | None) -> bool:
        if step is None:
            return False
        if not self._deferred_scene_sync_path:
            return False
        if self.scene_browse.text() != self._deferred_scene_sync_path:
            return False
        return id(step) in self._deferred_scene_sync_step_ids

    def _sync_step_scene_if_deferred(self, step: BaseStepWidget) -> None:
        path = self._deferred_scene_sync_path
        if not path or self.scene_browse.text() != path or id(step) not in self._deferred_scene_sync_step_ids:
            return
        if step is self.step5 and id(self.dataset_step) in self._deferred_scene_sync_step_ids:
            self.dataset_step.set_scene_dir(path)
            self._deferred_scene_sync_step_ids.discard(id(self.dataset_step))
        step.set_scene_dir(path)
        self._deferred_scene_sync_step_ids.discard(id(step))
        if not self._deferred_scene_sync_step_ids:
            self._deferred_scene_sync_path = None

    def _on_scene_suggested(self, path: str) -> None:
        if self.scene_browse.text():
            return
        candidate = Path(path)
        if not candidate.is_dir():
            return
        scene = str(candidate)
        self._auto_scene_from_input = scene
        self._applying_scene_suggestion = True
        try:
            self.scene_browse.set_text(scene)
        finally:
            self._applying_scene_suggestion = False

    def _on_input_videos_cleared(self) -> None:
        if self._auto_scene_from_input and self.scene_browse.text() == self._auto_scene_from_input:
            self._clear_scene_dir()

    def _clear_scene_dir(self) -> None:
        self._auto_scene_from_input = None
        if self.scene_browse.text():
            self.scene_browse.set_text("")

    def _import_scene_from_folder(self) -> None:
        if self._workflow_busy():
            self.log_panel.append_log(i18n.BUSY_MSG)
            return
        start_dir = self.scene_browse.text().strip() or str(Path.cwd())
        scene = dialogs.get_existing_directory(self, i18n.t("IMPORT_SCENE_SELECT_FOLDER"), start_dir)
        if not scene:
            return

        self._start_scene_import(scene)

    def _start_scene_import(self, scene: str) -> None:
        if self._workflow_busy():
            self.log_panel.append_log(i18n.BUSY_MSG)
            return

        self._scene_import_running = True
        self._scene_import_cancel_requested = False
        self._scene_import_canceled = False
        self._scene_import_scene = scene
        self._scene_import_summary = None
        self.progress.reset()
        self.progress.start_phase()
        self.progress.set_status(i18n.t("IMPORT_SCENE_RUNNING"))
        self.log_panel.append_log(i18n.t("IMPORT_SCENE_STARTED").format(scene=scene))
        self._update_run_button()

        self.runner.start_queue(
            [("scene_import", frame_app_job(import_scene_job(scene_dir=scene)))],
        )

    def _capture_scene_import_summary(self, line: str) -> bool:
        prefix = "SUMMARY_JSON:"
        if not line.startswith(prefix):
            return False
        try:
            payload = json.loads(line[len(prefix) :])
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, dict) or payload.get("kind") != "scene_import":
            return False
        self._scene_import_summary = payload
        return True

    def _finish_scene_import_queue(self, success: bool) -> None:
        summary = self._scene_import_summary or {}
        canceled = self._scene_import_cancel_requested or self._scene_import_canceled
        scene = str(summary.get("scene_dir") or self._scene_import_scene).strip()
        errors = summary.get("errors")
        error_count = len(errors) if isinstance(errors, list) else 0
        self._scene_import_running = False
        self._scene_import_cancel_requested = False
        self._scene_import_canceled = False
        self._scene_import_scene = ""
        self._scene_import_summary = None

        if canceled:
            self.progress.finish_phase(complete=False)
            self.log_panel.append_log(i18n.t("IMPORT_SCENE_CANCELED"))
            self.progress.set_status(i18n.STATUS_CANCELED)
            self._update_run_button()
            return

        if not success and not summary:
            self.progress.finish_phase(complete=False)
            self.log_panel.append_log(i18n.t("IMPORT_SCENE_FAILED"))
            self.progress.set_status(i18n.STATUS_FAILED)
            self._update_run_button()
            return

        if scene:
            self._auto_scene_from_input = None
            self._set_scene_browse_text_silently(scene)
            self._apply_scene_dir(scene, activate_current=False, defer_step_sync=True)
            self.log_panel.append_log(i18n.t("IMPORT_SCENE_DEFERRED_REFRESH"))
        self.progress.finish_phase(complete=success and error_count == 0)
        if not success or error_count:
            self.progress.set_status(i18n.t("IMPORT_SCENE_FAILED"))
        else:
            self.progress.set_status(
                i18n.t("IMPORT_SCENE_DONE").format(
                    images=int(summary.get("image_count") or 0),
                    masks=int(summary.get("mask_count") or 0),
                    output_images=int(summary.get("output_image_count") or 0),
                )
            )
        self._update_run_button()

    def _set_current_step(self, index: int) -> None:
        if not 0 <= index < len(self.steps):
            return
        if index == self.stack.currentIndex() and not self._step_scene_sync_deferred(self.steps[index]):
            return
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.step_buttons):
            btn.setChecked(i == index)
        self._update_step_header()
        self.step_help_btn.setAccessibleName(f"{self.step_titles[index]} {i18n.t('STEP_HELP_BUTTON')}")
        step = self._current_step_widget()
        if step is not None:
            self._sync_step_scene_if_deferred(step)
            step.on_activated()
        self._update_run_button()

    def _open_dataset_route(self, route_id: str) -> None:
        if self._workflow_busy():
            return
        self._set_current_step(self._dataset_step_index)
        if route_id == "dataset_menu":
            self.dataset_step.show_tool("menu")
        else:
            self.dataset_step.show_cubemap_route(route_id)
        self._update_run_button()

    def _update_step_header(self) -> None:
        index = self.stack.currentIndex()
        if not 0 <= index < len(self.step_titles):
            return
        step = self._current_step_widget()
        title = step.header_title() if step is not None else ""
        self.step_header.setText(title or self.step_titles[index])
        back_visible = bool(step and step.header_back_enabled())
        self.step_back_btn.setVisible(back_visible)
        self.step_back_btn.setEnabled(back_visible and not self._workflow_busy())
        tooltip = step.header_back_tooltip() if step is not None else ""
        self.step_back_btn.setToolTip(tooltip)
        self.step_back_btn.setAccessibleName(tooltip)
        self.step_subheader.setText("")
        self.step_subheader.setToolTip("")
        self.step_subheader.setVisible(False)

    def _on_step_header_back(self) -> None:
        if self._workflow_busy():
            return
        step = self._current_step_widget()
        if step is None or not step.header_back_enabled():
            return
        step.header_back()
        self._update_step_header()
        self._update_run_button()


    def _open_step_help(self) -> None:
        url = step_help_url(self.stack.currentIndex())
        if QDesktopServices.openUrl(QUrl(url)):
            return
        QMessageBox.warning(
            self,
            i18n.t("STEP_HELP_OPEN_FAILED_TITLE"),
            i18n.t("STEP_HELP_OPEN_FAILED_BODY").format(url=url),
        )

    def _update_run_button(self) -> None:
        runner_running = self.runner.is_running()
        busy = self._workflow_busy()
        step = self._current_step_widget()
        scene_selected = bool(self.scene_browse.text())
        self._set_workflow_locked(busy)
        if step is not None:
            self.run_btn.setText(f"  {step.primary_action_text()}")
            deferred_scene_sync = self._step_scene_sync_deferred(step)
            if scene_selected and deferred_scene_sync:
                self.run_btn.setToolTip(i18n.t("IMPORT_SCENE_DEFERRED_ACTION_HINT"))
            elif scene_selected:
                self.run_btn.setToolTip(step.primary_action_tooltip())
            else:
                self.run_btn.setToolTip(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        self._update_step_header()

        self.run_btn.setVisible(True)
        if self._step_scene_sync_deferred(step):
            action_enabled = False
        elif step is not None:
            action_enabled = step.primary_action_enabled()
        else:
            action_enabled = True
        self.run_btn.setEnabled(not busy and scene_selected and action_enabled)

        self.cancel_btn.setVisible(True)
        self.cancel_btn.setEnabled(
            runner_running or self._scene_import_running or self._background_task_step() is not None
        )

    def _workflow_busy(self) -> bool:
        return self.runner.is_running() or self._scene_import_running or self._background_task_step() is not None

    def _background_task_step(self) -> BaseStepWidget | None:
        for step in self.steps:
            if step.has_background_task():
                return step
        return None

    def _set_workflow_locked(self, locked: bool) -> None:
        unlocked = not locked
        self.scene_browse.setEnabled(unlocked)
        self.clear_scene_btn.setEnabled(unlocked and bool(self.scene_browse.text()))
        self.open_scene_action.setEnabled(unlocked)
        self.clear_scene_action.setEnabled(unlocked and bool(self.scene_browse.text()))
        self.scene_menu_btn.setEnabled(unlocked)
        self.import_scene_action.setEnabled(unlocked)
        self.stack.setEnabled(unlocked)
        for btn in self.step_buttons:
            btn.setEnabled(unlocked)

    def _on_run(self) -> None:
        step = self._current_step_widget()
        if step is None:
            return
        if self.runner.is_running():
            QMessageBox.warning(self, i18n.APP_TITLE, i18n.BUSY_MSG)
            return
        if not self._confirm_scene_path_is_safe():
            return
        if step.run_primary_action():
            self._update_run_button()
            return
        try:
            commands = step.build_commands()
        except (ValueError, FileNotFoundError) as e:
            QMessageBox.critical(self, i18n.INVALID_INPUT, str(e))
            return
        if not commands:
            return
        if not step.confirm_commands(commands):
            return
        self._current_step = self.stack.currentIndex()
        self.progress.reset()
        self.runner.start_queue(commands, log_dir=step.process_log_dir())
        self._update_run_button()

    def _confirm_scene_path_is_safe(self) -> bool:
        path = self.scene_browse.text().strip()
        if not path:
            return True
        issues = check_path_safety(path)
        if not issues:
            return True

        reasons = "\n".join(f"- {self._path_safety_reason(issue)}" for issue in issues)
        QMessageBox.critical(
            self,
            i18n.t("UNSAFE_SCENE_PATH_TITLE"),
            i18n.t("UNSAFE_SCENE_PATH_BODY").format(
                reasons=reasons,
                path=normalized_path_text(path),
            ),
        )
        return False

    def _path_safety_reason(self, issue: PathSafetyIssue) -> str:
        if issue.code == "non_ascii":
            return i18n.t("UNSAFE_PATH_REASON_NON_ASCII")
        if issue.code == "too_long":
            return i18n.t("UNSAFE_PATH_REASON_TOO_LONG").format(
                length=issue.length,
                limit=issue.limit,
            )
        if issue.code == "control_chars":
            return i18n.t("UNSAFE_PATH_REASON_CONTROL_CHARS").format(value=issue.value)
        if issue.code == "quote":
            return i18n.t("UNSAFE_PATH_REASON_QUOTE")
        return i18n.t("UNSAFE_PATH_REASON_UNKNOWN")

    def _on_cancel(self) -> None:
        if self._scene_import_running:
            if not self._scene_import_cancel_requested:
                self._scene_import_cancel_requested = True
                self.runner.cancel()
                self.log_panel.append_log(i18n.t("IMPORT_SCENE_CANCELING"))
                self.progress.set_status(i18n.t("IMPORT_SCENE_CANCELING"))
            self._update_run_button()
            return
        background_step = self._background_task_step()
        if background_step is not None:
            background_step.cancel_background_task()
            self._update_run_button()
            return
        self.runner.cancel()
        self._update_run_button()

    def _on_line(self, line: str) -> None:
        if self._scene_import_running and self._capture_scene_import_summary(line):
            return
        self.log_panel.append_log(line)
        if self._scene_import_running:
            return
        step = self.steps[self._current_step] if 0 <= self._current_step < len(self.steps) else None
        if step:
            result = step.on_line(line)
            if result is not None:
                done, total = result
                self.progress.set_progress(done, total)

    def _on_phase_started(self, phase: str) -> None:
        if self._scene_import_running:
            self.progress.start_phase()
            self.progress.set_status(i18n.t("IMPORT_SCENE_RUNNING"))
            self._update_run_button()
            return
        step = self.steps[self._current_step] if 0 <= self._current_step < len(self.steps) else None
        self.progress.start_phase()
        if step:
            self.progress.set_status(step.phase_status_text(phase, self.runner.phase_index, self.runner.queue_total))
        else:
            self.progress.set_status(f"{i18n.STATUS_RUNNING}: {phase}")
        if step:
            result = step.on_phase_started(phase)
            if result is not None:
                done, total = result
                self.progress.set_progress(done, total)
        self._update_run_button()

    def _on_phase_log_started(self, phase: str, path: str) -> None:
        if self._scene_import_running:
            return
        step = self.steps[self._current_step] if 0 <= self._current_step < len(self.steps) else None
        if step:
            step.on_phase_log_started(phase, path)

    def _on_phase_finished(self, phase: str, exit_code: int, canceled: bool) -> None:
        if self._scene_import_running:
            self._scene_import_canceled = canceled
            self.progress.finish_phase(complete=exit_code == 0 and not canceled)
            if canceled:
                self.progress.set_status(i18n.STATUS_CANCELED)
            return
        self.progress.finish_phase(complete=exit_code == 0 and not canceled)
        if canceled:
            self.progress.set_status(i18n.STATUS_CANCELED)
        step = self.steps[self._current_step] if 0 <= self._current_step < len(self.steps) else None
        if step:
            step.on_phase_finished(phase, exit_code, canceled)

    def _on_queue_finished(self, success: bool) -> None:
        if self._scene_import_running:
            self._finish_scene_import_queue(success)
            return
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

    def _on_background_task_started(self, status: str) -> None:
        self.progress.reset()
        self.progress.start_phase()
        self.progress.set_status(status or i18n.STATUS_RUNNING)
        self._update_run_button()

    def _on_background_line(self, line: str) -> None:
        self.log_panel.append_log(line)

    def _on_background_progress(self, done: int, total: int) -> None:
        self.progress.set_progress(done, total)

    def _on_background_status(self, status: str) -> None:
        self.progress.set_status(status)

    def _on_background_task_finished(self, success: bool, canceled: bool) -> None:
        self.progress.finish_phase(complete=success and not canceled)
        if canceled:
            self.progress.set_status(i18n.STATUS_CANCELED)
        elif success:
            self.progress.set_status(i18n.STATUS_DONE)
        else:
            self.progress.set_status(i18n.STATUS_FAILED)
        self._update_run_button()

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self.runner.is_running():
            self.runner.cancel()
        for step in self.steps:
            step.shutdown()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_layout()
        self.shutdown()
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description=i18n.APP_TITLE)
    parser.add_argument("--scene", default="", help="Initial scene directory")
    parser.add_argument(
        "--enable-apriltag",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{i18n.APP_TITLE} {app_version_label()}",
    )
    args = parser.parse_args()
    app = QApplication(sys.argv)
    app.setWindowIcon(app_icon())
    apply_theme(app)

    initial_scene = args.scene or os.environ.get("GS360_INITIAL_PROJECT", "")
    window = MainWindow(initial_scene_dir=initial_scene)
    app.aboutToQuit.connect(window.shutdown)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
