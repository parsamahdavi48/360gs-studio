"""STechDrive 3DGS Utils メインウィンドウ"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QPoint, QSignalBlocker, QSize, Qt, QThread, QTimer, QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.path_safety import PathSafetyIssue, check_path_safety, normalized_path_text
from core.scene_import import SceneImportResult, import_scene
from core.scene_layout import scene_images_dir, scene_masks_dir, scene_output_dir
from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.icons import help_icon, import_scene_icon, reset_icon
from gui.common.log_panel import LogPanel
from gui.common.process_runner import ProcessRunner
from gui.common.progress_widget import ProgressWidget
from gui.common.scene_import_worker import SceneImportWorker
from gui.steps.base_step import BaseStepWidget
from gui.steps.step1_extract import ExtractStep
from gui.steps.step2_review import ReviewStep
from gui.steps.step3_mask import MaskStep
from gui.steps.step4_cubemap import CubemapStep
from gui.steps.step5_training import TrainingStep
from gui.theme import apply_theme
from gui.version import app_version_label

_GITHUB_DOC_BASE_URL = "https://github.com/stechdrive/stechdrive-3dgs-utils/blob/main/doc"
_STEP_HELP_DOC_STEMS = (
    "extract_frames_gui",
    "review_frames_gui",
    "mask_tools_gui",
    "cubemap_tools_gui",
    "training_gui",
)
_STEP4_PIPELINE_NOTICE_MS = 1800


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


class MainWindow(QWidget):
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
        self._scene_import_thread: QThread | None = None
        self._scene_import_worker: SceneImportWorker | None = None
        self._deferred_scene_sync_path: str | None = None
        self._deferred_scene_sync_step_ids: set[int] = set()
        self._app_event_filter_installed = False

        self._build_ui(initial_scene_dir)
        self._connect_signals()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._app_event_filter_installed = True

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
        self.import_scene_btn = self.scene_browse.add_icon_button(
            import_scene_icon(),
            i18n.tip("IMPORT_SCENE"),
            self._import_scene_from_folder,
            accessible_name=i18n.t("IMPORT_SCENE"),
        )
        header.addWidget(self.scene_browse, stretch=1)
        header_widget = QWidget()
        header_widget.setObjectName("appHeader")
        header_widget.setLayout(header)
        root.addWidget(header_widget)

        self.step1 = ExtractStep(self.base_dir)
        self.step2 = ReviewStep(self.base_dir)
        self.step3 = MaskStep(self.base_dir)
        self.step4 = CubemapStep(self.base_dir)
        self.step5 = TrainingStep(self.base_dir, self.step4)
        self.step4.enable_user_preferences()
        self.steps = [self.step1, self.step2, self.step3, self.step4, self.step5]
        self.step_titles = [
            i18n.STEP1_TITLE,
            i18n.STEP2_TITLE,
            i18n.STEP3_TITLE,
            i18n.STEP4_TITLE,
            i18n.STEP5_TITLE,
        ]
        self.step_nav_titles = [
            i18n.t("STEP1_NAV"),
            i18n.t("STEP2_NAV"),
            i18n.t("STEP3_NAV"),
            i18n.t("STEP4_NAV"),
            i18n.t("STEP5_NAV"),
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
        self.step4_sub_buttons: dict[str, QWidget] = {}
        self.step4_sub_intent_buttons: dict[str, QToolButton] = {}
        self.step4_sub_status_labels: dict[str, QLabel] = {}
        self.step4_sub_text_labels: dict[str, QLabel] = {}
        self.step4_subnav_rail: QWidget | None = None
        self.step4_subnotice_label = QLabel("", self)
        self.step4_subnotice_label.setObjectName("navSubNotice")
        self.step4_subnotice_label.setWordWrap(False)
        self.step4_subnotice_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.step4_subnotice_label.hide()
        self.step4_subnotice_timer = QTimer(self)
        self.step4_subnotice_timer.setSingleShot(True)
        self.step4_subnotice_timer.timeout.connect(self._hide_step4_pipeline_notice)
        for index, title_text in enumerate(self.step_nav_titles):
            btn = QPushButton(title_text)
            btn.setObjectName("navStep")
            btn.setCheckable(True)
            btn.setFixedSize(70, 64)
            btn.setToolTip(self.step_titles[index])
            btn.clicked.connect(lambda _checked=False, i=index: self._set_current_step(i))
            sidebar_layout.addWidget(btn)
            self.step_buttons.append(btn)
            if index == 3:
                subnav = QWidget()
                subnav.setObjectName("navSubSteps")
                subnav_layout = QHBoxLayout(subnav)
                subnav_layout.setContentsMargins(4, 0, 0, 2)
                subnav_layout.setSpacing(3)
                rail = QWidget()
                rail.setObjectName("navSubRail")
                rail.setFixedWidth(2)
                subnav_layout.addWidget(rail)
                self.step4_subnav_rail = rail
                subnav_rows = QWidget()
                subnav_rows.setObjectName("navSubRows")
                subnav_rows_layout = QVBoxLayout(subnav_rows)
                subnav_rows_layout.setContentsMargins(0, 0, 0, 0)
                subnav_rows_layout.setSpacing(2)
                for stage in ("sfm", "conversion"):
                    sub_btn = QWidget()
                    sub_btn.setObjectName("navSubStep")
                    sub_btn.setFixedSize(63, 22)
                    sub_btn.setProperty("stage", stage)
                    sub_btn.installEventFilter(self)
                    sub_btn_layout = QHBoxLayout(sub_btn)
                    sub_btn_layout.setContentsMargins(2, 0, 1, 0)
                    sub_btn_layout.setSpacing(1)
                    intent_btn = QToolButton()
                    intent_btn.setObjectName("navSubStepIntent")
                    intent_btn.setCheckable(True)
                    intent_btn.setAutoRaise(True)
                    intent_btn.setFixedSize(13, 18)
                    intent_btn.clicked.connect(
                        lambda _checked=False, s=stage: self._toggle_step4_pipeline_stage_intent(s)
                    )
                    status_label = QLabel("")
                    status_label.setObjectName("navSubStepStatus")
                    status_label.setFixedSize(13, 18)
                    status_label.setAlignment(Qt.AlignCenter)
                    status_label.setProperty("stage", stage)
                    status_label.installEventFilter(self)
                    text_label = QLabel("")
                    text_label.setObjectName("navSubStepText")
                    text_label.setWordWrap(False)
                    text_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
                    sub_btn_layout.addWidget(intent_btn)
                    sub_btn_layout.addWidget(text_label, stretch=1)
                    sub_btn_layout.addWidget(status_label)
                    subnav_rows_layout.addWidget(sub_btn)
                    self.step4_sub_buttons[stage] = sub_btn
                    self.step4_sub_intent_buttons[stage] = intent_btn
                    self.step4_sub_status_labels[stage] = status_label
                    self.step4_sub_text_labels[stage] = text_label
                subnav_layout.addWidget(subnav_rows)
                sidebar_layout.addWidget(subnav)
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
        self.step_header = QLabel("")
        self.step_header.setObjectName("stepHeader")
        step_header_row.addWidget(self.step_header)
        self.step_subheader = QLabel("")
        self.step_subheader.setObjectName("stepSubheader")
        self.step_subheader.setWordWrap(False)
        self.step_subheader.setTextInteractionFlags(Qt.TextSelectableByMouse)
        step_header_row.addWidget(self.step_subheader, stretch=1)
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
        self.step1.input_videos_cleared.connect(self._on_input_videos_cleared)
        self.step3.scene_dir_suggested.connect(self._on_scene_suggested)
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn.clicked.connect(self._on_cancel)

        self.runner.line_received.connect(self._on_line)
        self.runner.phase_started.connect(self._on_phase_started)
        self.runner.phase_log_started.connect(self._on_phase_log_started)
        self.runner.phase_finished.connect(self._on_phase_finished)
        self.runner.queue_finished.connect(self._on_queue_finished)
        for step in self.steps:
            step.primary_action_state_changed.connect(self._update_run_button)
        self.step4.primary_action_state_changed.connect(self._refresh_step4_subnav)

        self._on_scene_changed(self.scene_browse.text())

    def _current_step_widget(self):
        idx = self.stack.currentIndex()
        if 0 <= idx < len(self.steps):
            return self.steps[idx]
        return None

    def _on_scene_changed(self, path: str) -> None:
        self._apply_scene_dir(path, activate_current=True)

    def _apply_scene_dir(self, path: str, *, activate_current: bool, defer_step_sync: bool = False) -> None:
        if not self._applying_scene_suggestion and path != self._auto_scene_from_input:
            self._auto_scene_from_input = None
        if defer_step_sync:
            for step in self.steps:
                BaseStepWidget.set_scene_dir(step, path)
            self._deferred_scene_sync_path = path
            self._deferred_scene_sync_step_ids = {id(step) for step in self.steps}
        else:
            self._clear_deferred_scene_sync()
            for step in self.steps:
                step.set_scene_dir(path)
        self.clear_scene_btn.setEnabled(bool(path))
        if activate_current:
            step = self._current_step_widget()
            if step is not None:
                self._sync_step_scene_if_deferred(step)
                step.on_activated()
        self._update_step_header()
        if not self._step_scene_sync_deferred(self.step4):
            self._refresh_step4_subnav()

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
        if step is self.step5 and id(self.step4) in self._deferred_scene_sync_step_ids:
            self.step4.set_scene_dir(path)
            self._deferred_scene_sync_step_ids.discard(id(self.step4))
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
        scene = QFileDialog.getExistingDirectory(self, i18n.t("IMPORT_SCENE_SELECT_FOLDER"), start_dir)
        if not scene:
            return

        self._start_scene_import(scene)

    def _start_scene_import(self, scene: str) -> None:
        if self._workflow_busy():
            self.log_panel.append_log(i18n.BUSY_MSG)
            return

        self._scene_import_running = True
        self._scene_import_cancel_requested = False
        self.progress.reset()
        self.progress.start_phase()
        self.progress.set_status(i18n.t("IMPORT_SCENE_RUNNING"))
        self.log_panel.append_log(i18n.t("IMPORT_SCENE_STARTED").format(scene=scene))
        self._update_run_button()

        thread = QThread(self)
        worker = SceneImportWorker(Path(scene), importer=import_scene)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.log_panel.append_log)
        worker.finished.connect(self._on_scene_import_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_scene_import_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._scene_import_thread = thread
        self._scene_import_worker = worker
        thread.start()

    def _on_scene_import_finished(self, result: object, error: str, canceled: bool) -> None:
        self._scene_import_running = False
        self._scene_import_cancel_requested = False
        if canceled:
            self.progress.finish_phase(complete=False)
            self.log_panel.append_log(i18n.t("IMPORT_SCENE_CANCELED"))
            self.progress.set_status(i18n.STATUS_CANCELED)
            self._update_run_button()
            return
        if error:
            self.progress.finish_phase(complete=False)
            self.log_panel.append_log(f"{i18n.t('IMPORT_SCENE_FAILED')}: {error}")
            self.progress.set_status(i18n.STATUS_FAILED)
            self._update_run_button()
            return

        if not isinstance(result, SceneImportResult):
            self.progress.finish_phase(complete=False)
            self.log_panel.append_log(f"{i18n.t('IMPORT_SCENE_FAILED')}: invalid worker result")
            self.progress.set_status(i18n.STATUS_FAILED)
            self._update_run_button()
            return

        scene = str(result.scene_dir)
        for line in result.summary_lines():
            self.log_panel.append_log(line)
        self._auto_scene_from_input = None
        self._set_scene_browse_text_silently(scene)
        self._apply_scene_dir(scene, activate_current=False, defer_step_sync=True)
        self.log_panel.append_log(i18n.t("IMPORT_SCENE_DEFERRED_REFRESH"))
        self.progress.finish_phase(complete=not bool(result.errors))
        if result.errors:
            self.progress.set_status(i18n.t("IMPORT_SCENE_FAILED"))
        else:
            self.progress.set_status(
                i18n.t("IMPORT_SCENE_DONE").format(
                    images=result.image_count,
                    masks=result.mask_count,
                    output_images=result.output_image_count,
                )
            )
        self._update_run_button()

    def _on_scene_import_thread_finished(self) -> None:
        if self.sender() is self._scene_import_thread:
            self._scene_import_thread = None
            self._scene_import_worker = None

    def _set_current_step(self, index: int) -> None:
        if not 0 <= index < len(self.steps):
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
        if index != 3:
            self._hide_step4_pipeline_notice()
        self._update_run_button()
        self._refresh_step4_subnav()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.MouseButtonPress and self.step4_subnotice_label.isVisible():
            self._hide_step4_pipeline_notice()
        if event.type() == QEvent.Type.MouseButtonRelease:
            for stage, widget in self.step4_sub_buttons.items():
                if watched is widget:
                    if widget.isEnabled() and self.step4.pipeline_stage_intent_toggle_enabled(stage):
                        self._toggle_step4_pipeline_stage_intent(stage)
                        return True
                    return False
            for stage, widget in self.step4_sub_status_labels.items():
                if watched is widget and widget.property("status") == "warning":
                    self._open_step4_pipeline_stage(stage)
                    return True
        return super().eventFilter(watched, event)

    def _update_step_header(self) -> None:
        index = self.stack.currentIndex()
        if not 0 <= index < len(self.step_titles):
            return
        self.step_header.setText(self.step_titles[index])
        path = self._step_header_path(index)
        self.step_subheader.setText(path)
        self.step_subheader.setToolTip(path)

    def _step_header_path(self, index: int) -> str:
        if not self.scene_browse.text():
            return "-"
        scene = Path(self.scene_browse.text())
        if index == 0:
            return str(scene_images_dir(scene))
        if index == 1:
            return str(scene_images_dir(scene))
        if index == 2:
            return str(scene_masks_dir(scene))
        if index == 3:
            return str(scene_output_dir(scene))
        return str(scene_output_dir(scene))

    def _toggle_step4_pipeline_stage_intent(self, stage: str) -> None:
        if not self.step4.pipeline_stage_intent_toggle_enabled(stage):
            self._set_current_step(3)
            self._refresh_step4_subnav()
            self._update_run_button()
            return
        self._set_current_step(3)
        self.step4.toggle_pipeline_stage_intent(stage)
        self._show_step4_pipeline_notice(self.step4.take_pipeline_notice())
        self._refresh_step4_subnav()
        self._update_run_button()

    def _open_step4_pipeline_stage(self, stage: str) -> None:
        if self._workflow_busy():
            return
        self._set_current_step(3)
        self.step4.activate_pipeline_stage(stage)
        self._update_run_button()
        self._refresh_step4_subnav()

    def _show_step4_pipeline_notice(self, text: str) -> None:
        if not text:
            return
        self.step4_subnotice_label.setText(text)
        self.step4_subnotice_label.adjustSize()
        self._position_step4_pipeline_notice()
        self.step4_subnotice_label.show()
        self.step4_subnotice_label.raise_()
        self.step4_subnotice_timer.start(_STEP4_PIPELINE_NOTICE_MS)

    def _hide_step4_pipeline_notice(self) -> None:
        self.step4_subnotice_timer.stop()
        self.step4_subnotice_label.hide()
        self.step4_subnotice_label.setText("")

    def _position_step4_pipeline_notice(self) -> None:
        if not self.step4_subnotice_label.isVisible() and not self.step4_subnotice_label.text():
            return
        anchor = self.step4_sub_buttons.get("sfm")
        if anchor is None:
            return
        top_right = anchor.mapTo(self, QPoint(anchor.width() + 8, 0))
        self.step4_subnotice_label.move(top_right)

    def _refresh_step4_subnav(self) -> None:
        if not self.step4_sub_buttons:
            return
        if self._step_scene_sync_deferred(self.step4):
            return
        if self.step4_subnav_rail is not None:
            self.step4_subnav_rail.setProperty("active", "false")
            self.step4_subnav_rail.style().unpolish(self.step4_subnav_rail)
            self.step4_subnav_rail.style().polish(self.step4_subnav_rail)
        for item in self.step4.pipeline_nav_items():
            button = self.step4_sub_buttons.get(item["stage"])
            if button is None:
                continue
            status_label = self.step4_sub_status_labels.get(item["stage"])
            text_label = self.step4_sub_text_labels.get(item["stage"])
            intent_btn = self.step4_sub_intent_buttons.get(item["stage"])
            if intent_btn is not None:
                intent_btn.setText(str(item["intent_symbol"]))
                intent_btn.setChecked(bool(item["intent_checked"]))
                intent_btn.setEnabled(bool(item["intent_enabled"]))
                intent_btn.setToolTip(str(item["intent_tooltip"]))
            if status_label is not None:
                status_label.setText(str(item["status_symbol"]))
                status_label.setProperty("status", item["status"])
                status_label.setToolTip(str(item["status_tooltip"]))
                status_label.setCursor(
                    Qt.CursorShape.PointingHandCursor
                    if item["status"] == "warning"
                    else Qt.CursorShape.ArrowCursor
                )
            if text_label is not None:
                text_label.setText(str(item["label"]))
                text_label.setProperty("active", "false")
                text_label.setToolTip("")
            button.setToolTip(str(item["row_tooltip"]))
            button.setCursor(
                Qt.CursorShape.PointingHandCursor
                if item["intent_toggle_enabled"]
                else Qt.CursorShape.ArrowCursor
            )
            button.setProperty("active", "false")
            button.setProperty("status", item["status"])
            for widget in (button, intent_btn, status_label, text_label):
                if widget is not None:
                    widget.style().unpolish(widget)
                    widget.style().polish(widget)
        if self.step4_subnotice_label.isVisible():
            self._position_step4_pipeline_notice()

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
        self.cancel_btn.setEnabled(runner_running or self._scene_import_running)

    def _workflow_busy(self) -> bool:
        return self.runner.is_running() or self._scene_import_running

    def _set_workflow_locked(self, locked: bool) -> None:
        unlocked = not locked
        self.scene_browse.setEnabled(unlocked)
        self.clear_scene_btn.setEnabled(unlocked and bool(self.scene_browse.text()))
        self.import_scene_btn.setEnabled(unlocked)
        self.stack.setEnabled(unlocked)
        for btn in self.step_buttons:
            btn.setEnabled(unlocked)
        for btn in self.step4_sub_buttons.values():
            btn.setEnabled(unlocked)
        for btn in self.step4_sub_intent_buttons.values():
            btn.setEnabled(unlocked)
        for btn in self.step4_sub_status_labels.values():
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
                if self._scene_import_worker is not None:
                    self._scene_import_worker.cancel()
                self.log_panel.append_log(i18n.t("IMPORT_SCENE_CANCELING"))
                self.progress.set_status(i18n.t("IMPORT_SCENE_CANCELING"))
            self._update_run_button()
            return
        self.runner.cancel()
        self.progress.finish_phase(complete=False)
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
        step = self.steps[self._current_step] if 0 <= self._current_step < len(self.steps) else None
        label = step.phase_display_name(phase) if step else phase
        self.progress.start_phase()
        self.progress.set_status(f"{i18n.STATUS_RUNNING}: {label}")
        if step:
            result = step.on_phase_started(phase)
            if result is not None:
                done, total = result
                self.progress.set_progress(done, total)
        self._update_run_button()

    def _on_phase_log_started(self, phase: str, path: str) -> None:
        step = self.steps[self._current_step] if 0 <= self._current_step < len(self.steps) else None
        if step:
            step.on_phase_log_started(phase, path)

    def _on_phase_finished(self, phase: str, exit_code: int, canceled: bool) -> None:
        self.progress.finish_phase(complete=exit_code == 0 and not canceled)
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

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        app = QApplication.instance()
        if app is not None and self._app_event_filter_installed:
            app.removeEventFilter(self)
            self._app_event_filter_installed = False
        if self.runner.is_running():
            self.runner.cancel()
        if self._scene_import_thread is not None and self._scene_import_thread.isRunning():
            if self._scene_import_worker is not None:
                self._scene_import_worker.cancel()
            self._scene_import_thread.quit()
            self._scene_import_thread.wait(3000)
        for step in self.steps:
            step.shutdown()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.shutdown()
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
    app.setWindowIcon(app_icon())
    apply_theme(app)

    window = MainWindow(initial_scene_dir=args.scene)
    app.aboutToQuit.connect(window.shutdown)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
