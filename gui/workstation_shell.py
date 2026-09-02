"""Dockable workstation panels shared by the 360GS Studio shell."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.artifact_registry import ArtifactRecord, load_artifacts
from gs360studio.platform.job_store import list_jobs
from gs360studio.platform.project_store import load_project
from gui import i18n


class ProjectArtifactBrowser(QWidget):
    """Compact, read-only view of the active project and its produced assets."""

    path_open_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("projectBrowser")
        self._scene_dir = Path()

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        heading_row = QHBoxLayout()
        heading = QLabel(i18n.t("SHELL_PROJECT").upper())
        heading.setObjectName("dockSectionTitle")
        heading_row.addWidget(heading)
        heading_row.addStretch(1)
        self.refresh_button = QPushButton(i18n.t("SHELL_REFRESH"))
        self.refresh_button.setObjectName("compactButton")
        self.refresh_button.clicked.connect(self.refresh)
        heading_row.addWidget(self.refresh_button)
        root.addLayout(heading_row)

        self.project_name = QLabel(i18n.t("SHELL_NO_PROJECT"))
        self.project_name.setObjectName("projectName")
        self.project_name.setWordWrap(True)
        root.addWidget(self.project_name)

        self.project_path = QLabel(i18n.t("SHELL_CHOOSE_PROJECT"))
        self.project_path.setObjectName("mutedText")
        self.project_path.setWordWrap(True)
        self.project_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.project_path)

        self.tree = QTreeWidget()
        self.tree.setObjectName("artifactTree")
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.itemDoubleClicked.connect(self._open_item)
        root.addWidget(self.tree, stretch=1)

        self.open_button = QPushButton(i18n.t("SHELL_OPEN_SELECTED"))
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_selected)
        self.tree.currentItemChanged.connect(
            lambda item, _previous: self.open_button.setEnabled(bool(item and item.data(0, Qt.UserRole)))
        )
        root.addWidget(self.open_button)

    @property
    def scene_dir(self) -> Path:
        return self._scene_dir

    def set_scene_dir(self, scene_dir: str) -> None:
        self._scene_dir = Path(scene_dir) if scene_dir else Path()
        self.refresh()

    def refresh(self) -> None:
        self.tree.clear()
        if not self._scene_dir:
            self.project_name.setText(i18n.t("SHELL_NO_PROJECT"))
            self.project_path.setText(i18n.t("SHELL_CHOOSE_PROJECT"))
            self.open_button.setEnabled(False)
            return

        scene = self._scene_dir
        self.project_name.setText(scene.name or str(scene))
        self.project_path.setText(str(scene))

        manifest_name = ""
        manifest_version = ""
        try:
            manifest, _report = load_project(scene, migrate=False)
            manifest_name = manifest.name
            manifest_version = f"Schema v{manifest.SCHEMA_VERSION}"
        except (FileNotFoundError, OSError, ValueError):
            pass
        if manifest_name:
            self.project_name.setText(manifest_name)

        project_item = QTreeWidgetItem([manifest_version or i18n.t("SHELL_PROJECT_FILES")])
        project_item.setData(0, Qt.UserRole, str(scene))
        self.tree.addTopLevelItem(project_item)
        for label, relative in (
            (i18n.t("SHELL_IMAGES"), "images"),
            (i18n.t("SHELL_MASKS"), "masks"),
            (i18n.t("SHELL_OUTPUT"), "output"),
        ):
            path = scene / relative
            empty = "" if path.exists() else f" ({i18n.t('SHELL_EMPTY')})"
            child = QTreeWidgetItem([f"{label}{empty}"])
            child.setData(0, Qt.UserRole, str(path))
            project_item.addChild(child)

        self._append_artifact_group(scene, i18n.t("SHELL_SFM_ARTIFACTS"), "sfm")
        self._append_artifact_group(scene, i18n.t("SHELL_DATASET_ARTIFACTS"), "dataset")

        jobs = list_jobs(scene)
        jobs_item = QTreeWidgetItem([f"{i18n.t('SHELL_JOBS')} ({len(jobs)})"])
        self.tree.addTopLevelItem(jobs_item)
        for job in sorted(jobs, key=lambda item: item.created_at, reverse=True)[:25]:
            label = f"{job.job_type} · {job.status}"
            job_item = QTreeWidgetItem([label])
            job_item.setToolTip(0, job.job_id)
            jobs_item.addChild(job_item)

        project_item.setExpanded(True)
        self.open_button.setEnabled(False)

    def _append_artifact_group(self, scene: Path, title: str, group: str) -> None:
        try:
            records = load_artifacts(scene, group)  # type: ignore[arg-type]
        except (OSError, ValueError):
            records = []
        parent = QTreeWidgetItem([f"{title} ({len(records)})"])
        self.tree.addTopLevelItem(parent)
        for record in reversed(records[-25:]):
            parent.addChild(self._artifact_item(scene, record))

    @staticmethod
    def _artifact_item(scene: Path, record: ArtifactRecord) -> QTreeWidgetItem:
        item = QTreeWidgetItem([f"{record.kind} · {record.status}"])
        root = Path(record.root)
        resolved = root if root.is_absolute() else scene / root
        item.setData(0, Qt.UserRole, str(resolved))
        item.setToolTip(0, f"{record.id}\n{resolved}")
        return item

    def _open_selected(self) -> None:
        self._open_item(self.tree.currentItem())

    def _open_item(self, item: QTreeWidgetItem | None, _column: int = 0) -> None:
        if item is None:
            return
        value = str(item.data(0, Qt.UserRole) or "")
        if value:
            self.path_open_requested.emit(value)


class ContextInspector(QWidget):
    """Small always-visible summary of the selected workspace and action."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("contextInspector")

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        heading = QLabel(i18n.t("SHELL_INSPECTOR").upper())
        heading.setObjectName("dockSectionTitle")
        root.addWidget(heading)

        self.workspace_label = QLabel(i18n.STEP1_TITLE)
        self.workspace_label.setObjectName("inspectorTitle")
        self.workspace_label.setWordWrap(True)
        root.addWidget(self.workspace_label)

        self.workspace_description = QLabel("")
        self.workspace_description.setObjectName("mutedText")
        self.workspace_description.setWordWrap(True)
        root.addWidget(self.workspace_description)

        divider = QWidget()
        divider.setObjectName("inspectorDivider")
        divider.setFixedHeight(1)
        root.addWidget(divider)

        self.scene_caption = QLabel(i18n.t("SHELL_ACTIVE_PROJECT").upper())
        self.scene_caption.setObjectName("dockSectionTitle")
        root.addWidget(self.scene_caption)
        self.scene_label = QLabel("None")
        self.scene_label.setObjectName("inspectorValue")
        self.scene_label.setWordWrap(True)
        self.scene_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.scene_label)

        self.action_caption = QLabel(i18n.t("SHELL_PRIMARY_ACTION").upper())
        self.action_caption.setObjectName("dockSectionTitle")
        root.addWidget(self.action_caption)
        self.action_label = QLabel(i18n.t("SHELL_SELECT_PROJECT"))
        self.action_label.setObjectName("inspectorValue")
        self.action_label.setWordWrap(True)
        root.addWidget(self.action_label)

        root.addStretch(1)
        hint = QLabel(i18n.t("SHELL_LAYOUT_HINT"))
        hint.setObjectName("mutedText")
        hint.setWordWrap(True)
        root.addWidget(hint)

    def set_context(self, *, workspace: str, description: str, scene: str, action: str) -> None:
        self.workspace_label.setText(workspace)
        self.workspace_description.setText(description)
        self.scene_label.setText(scene or "None")
        self.action_label.setText(action or i18n.t("SHELL_NO_ACTION"))
