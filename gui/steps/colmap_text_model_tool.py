"""Metashape XML/PLY to portable COLMAP text dataset tool."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.app_job import dataset_app_job
from core.dataset_job_spec import metashape_colmap_job, write_dataset_job
from core.dataset_mask_policy import (
    DATASET_MASK_CONVERT_SFM,
    DATASET_MASK_GENERATE_TRAINING,
    DATASET_MASK_NONE,
    DATASET_MASK_REUSE_EXISTING,
)
from core.metashape_preview_targets import (
    build_metashape_preview_targets,
    metashape_output_count_for_actions,
)
from core.orientation_correction import FINAL_ORIENTATION_NONE
from core.projection_contract import PROJECTION_EQUIRECTANGULAR
from core.scene_inventory import build_scene_inventory
from core.scene_layout import (
    jobs_dir,
    scene_images_dir,
    scene_masks_dir,
    scene_output_dir,
    step4_meta_dir,
)
from core.workflow_artifacts import (
    DATASET_KIND_COLMAP_DATASET,
    SFM_KIND_METASHAPE_XML_PLY,
    register_dataset_artifact,
    register_sfm_artifact,
)
from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.drag_spinbox import DragDoubleSpinBox
from gui.common.form_rows import add_tooltip_row
from gui.common.runner_types import StepCommandQueue
from gui.cubemap.preview_renderer import PreviewWidget
from gui.cubemap.view_config import _BLOCK_ENABLED_VIEWS, _WARN_ENABLED_VIEWS, ViewConfigWidget
from gui.steps.base_step import SETTINGS_PANE_MARGINS, SETTINGS_PANE_WIDTH, BaseStepWidget
from gui.steps.dataset_mask_step import DatasetMaskStep
from gui.steps.output_reset import clear_path, path_has_contents
from gui.steps.step4_contracts import (
    _AXIS_NONE,
    _NORMAL_OUTPUT_SCALE,
)
from gui.steps.step4_widgets import make_output_image_controls

_PROGRESS_RE = re.compile(r"^\[progress\]\s+(\d+)\s*/\s*(\d+)")


class ColmapTextModelTool(BaseStepWidget):
    """Create a portable COLMAP dataset from Metashape XML and PLY exports."""

    def __init__(self, base_dir: Path, parent=None) -> None:
        super().__init__(base_dir, parent)
        self._syncing_paths = False
        self._images_user_edited = False
        self._masks_user_edited = False
        self._xml_user_edited = False
        self._ply_user_edited = False
        self._metashape_preview_action_counts: dict[str, int] | None = None
        self._dataset_mask_step: DatasetMaskStep | None = None
        self._dataset_mask_tab_index: int | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left_pane = QWidget()
        left_pane.setFixedWidth(SETTINGS_PANE_WIDTH)
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(*SETTINGS_PANE_MARGINS)
        left_layout.setSpacing(8)

        self.settings_tabs = QTabWidget()
        self.settings_tabs.setObjectName("step4SettingsTabs")
        self.settings_tabs.tabBar().setUsesScrollButtons(False)
        self.settings_tabs.tabBar().setExpanding(False)
        self.settings_tabs.addTab(self._build_input_tab(), i18n.t("STEP4_TAB_INPUT"))
        self.settings_tabs.addTab(self._build_output_tab(), i18n.t("STEP4_TAB_OUTPUT"))
        self.settings_tabs.addTab(self._build_details_tab(), i18n.t("STEP4_TAB_DETAILS"))
        self._dataset_mask_step = DatasetMaskStep(
            self.base_dir,
            dataset_root_provider=self._output_dir,
            source_images_dir_provider=self._images_dir,
            source_masks_dir_provider=self._masks_dir,
            generated_source_masks_dir_provider=self._dataset_training_source_masks_dir,
            link_mask_paths=False,
            mode_tip_keys={
                DATASET_MASK_CONVERT_SFM: "COLMAP_TEXT_MASK_MODE_CONVERT_SFM",
                DATASET_MASK_GENERATE_TRAINING: "COLMAP_TEXT_MASK_MODE_GENERATE_TRAINING",
                DATASET_MASK_REUSE_EXISTING: "COLMAP_TEXT_MASK_MODE_REUSE_EXISTING",
                DATASET_MASK_NONE: "COLMAP_TEXT_MASK_MODE_NONE",
            },
            parent=self,
        )
        self._dataset_mask_step.primary_action_state_changed.connect(self.primary_action_state_changed)
        self._dataset_mask_step.settings_scroll.setObjectName("step4TabScroll")
        self._dataset_mask_tab_index = self.settings_tabs.addTab(
            self._dataset_mask_step.settings_scroll,
            i18n.t("STEP4_TAB_MASK_SETTINGS"),
        )
        self._dataset_mask_step.hide()
        left_layout.addWidget(self.settings_tabs, stretch=1)
        self.settings_tabs.currentChanged.connect(lambda _index: self._on_settings_tab_changed())

        preview_pane = QWidget()
        preview_pane.setObjectName("workPane")
        preview_layout = QVBoxLayout(preview_pane)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(8)

        preview_header = QHBoxLayout()
        preview_header.setContentsMargins(0, 0, 0, 0)
        preview_header.setSpacing(8)
        preview_title = QLabel(i18n.t("CUBEMAP_PREVIEW_SECTION"))
        preview_title.setObjectName("paneTitle")
        preview_header.addWidget(preview_title)
        preview_header.addStretch()
        self.preview = PreviewWidget()
        preview_header.addWidget(self.preview.projection_toggle_btn)
        preview_layout.addLayout(preview_header)
        preview_layout.addWidget(self.preview, stretch=1)

        self.preview_pane = preview_pane
        self.work_stack = QStackedWidget()
        self.work_stack.addWidget(preview_pane)
        self.work_stack.addWidget(self._dataset_mask_step.preview_pane)

        splitter.addWidget(left_pane)
        splitter.addWidget(self.work_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([SETTINGS_PANE_WIDTH, 760])
        layout.addWidget(splitter)

        self._connect_signals()
        self._sync_mask_settings_context()
        self._update_output_count()

    def _build_input_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        form = QFormLayout()
        form.setSpacing(6)

        self.images_browse = BrowseWidget(mode="dir", placeholder=i18n.t("COLMAP_TEXT_IMAGES_PLACEHOLDER"))
        self.images_browse.setToolTip(i18n.tip("COLMAP_TEXT_IMAGES"))
        self.images_browse.line_edit.setToolTip(i18n.tip("COLMAP_TEXT_IMAGES"))
        add_tooltip_row(form, i18n.t("COLMAP_TEXT_IMAGES"), self.images_browse, i18n.tip("COLMAP_TEXT_IMAGES"))

        self.masks_browse = BrowseWidget(mode="dir", placeholder=i18n.t("COLMAP_TEXT_MASKS_PLACEHOLDER"))
        self.masks_browse.setToolTip(i18n.tip("COLMAP_TEXT_MASKS"))
        self.masks_browse.line_edit.setToolTip(i18n.tip("COLMAP_TEXT_MASKS"))
        add_tooltip_row(form, i18n.t("COLMAP_TEXT_MASKS"), self.masks_browse, i18n.tip("COLMAP_TEXT_MASKS"))

        self.xml_browse = BrowseWidget(
            mode="file",
            filter_str="XML (*.xml);;すべて (*.*)",
            placeholder=i18n.t("MS_XML_PLACEHOLDER"),
        )
        self.xml_browse.setToolTip(i18n.tip("COLMAP_TEXT_XML"))
        self.xml_browse.line_edit.setToolTip(i18n.tip("COLMAP_TEXT_XML"))
        add_tooltip_row(form, i18n.METASHAPE_XML, self.xml_browse, i18n.tip("COLMAP_TEXT_XML"))

        self.ply_browse = BrowseWidget(
            mode="file",
            filter_str="PLY (*.ply);;すべて (*.*)",
            placeholder=i18n.t("MS_PLY_PLACEHOLDER"),
        )
        self.ply_browse.setToolTip(i18n.tip("COLMAP_TEXT_PLY"))
        self.ply_browse.line_edit.setToolTip(i18n.tip("COLMAP_TEXT_PLY"))
        add_tooltip_row(form, i18n.METASHAPE_PLY, self.ply_browse, i18n.tip("COLMAP_TEXT_PLY"))

        layout.addLayout(form)
        layout.addStretch()
        return tab

    def _build_output_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.view_config = ViewConfigWidget(show_settings=True, show_summary=True)

        self.yaw_per_frame_edit = DragDoubleSpinBox(
            minimum=-180.0,
            maximum=180.0,
            step=1.0,
            decimals=1,
            value=30.0,
            drag_pixels_per_step=6.0,
        )
        self.yaw_per_frame_edit.setFixedWidth(76)
        self.yaw_per_frame_edit.setToolTip(i18n.t("YAW_OFFSET_PER_FRAME_HINT"))
        self.yaw_per_frame_label = QLabel(i18n.t("YAW_OFFSET_PER_FRAME"))
        self.yaw_per_frame_label.setToolTip(i18n.t("YAW_OFFSET_PER_FRAME_HINT"))
        self.view_config.angle_row.addWidget(self.yaw_per_frame_label)
        self.view_config.angle_row.addWidget(self.yaw_per_frame_edit)
        self.view_config.angle_row.addStretch()

        self.scale_combo = QComboBox()
        self.scale_combo.setToolTip(i18n.tip("OUTPUT_SCALE"))
        self.scale_combo.addItem("Full", 1.0)
        self.scale_combo.addItem("Normal", _NORMAL_OUTPUT_SCALE)
        self.scale_combo.addItem("Half", 0.5)
        normal_scale_index = self.scale_combo.findData(_NORMAL_OUTPUT_SCALE)
        if normal_scale_index >= 0:
            self.scale_combo.setCurrentIndex(normal_scale_index)
        self.scale_combo.setFixedWidth(90)

        scale_row = QWidget()
        scale_layout = QHBoxLayout(scale_row)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.setSpacing(8)
        scale_label = QLabel(i18n.OUTPUT_SCALE + ":")
        scale_label.setToolTip(i18n.tip("OUTPUT_SCALE"))
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.scale_combo)
        scale_layout.addStretch()
        self.view_config.extra_controls_layout.addWidget(scale_row)

        layout.addWidget(self.view_config.settings_widget)
        layout.addStretch()
        return tab

    def _build_details_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.output_image_controls = make_output_image_controls(tab)
        self.output_format_combo = self.output_image_controls.output_format_combo
        self.output_bit_depth_combo = self.output_image_controls.output_bit_depth_combo
        self.invert_masks_cb = self.output_image_controls.invert_masks_cb
        self.jpg_quality_label = self.output_image_controls.jpg_quality_label
        self.jpg_quality_edit = self.output_image_controls.jpg_quality_edit
        layout.addWidget(self.output_image_controls.widget)
        layout.addStretch()
        return tab

    def _connect_signals(self) -> None:
        self.images_browse.path_changed.connect(lambda _path: self._on_path_changed("images"))
        self.masks_browse.path_changed.connect(lambda _path: self._on_path_changed("masks"))
        self.xml_browse.path_changed.connect(lambda _path: self._on_path_changed("xml"))
        self.ply_browse.path_changed.connect(lambda _path: self._on_path_changed("ply"))

        self.view_config.views_changed.connect(self._on_views_changed)
        self.view_config.hovered_view_changed.connect(lambda _name: self._render_preview())
        self.scale_combo.currentIndexChanged.connect(self._on_output_option_changed)
        self.yaw_per_frame_edit.valueChanged.connect(self._on_output_option_changed)
        self.output_format_combo.currentIndexChanged.connect(self._on_output_option_changed)
        self.output_bit_depth_combo.currentIndexChanged.connect(self._on_output_option_changed)
        self.invert_masks_cb.toggled.connect(self._on_output_option_changed)
        self.jpg_quality_edit.textEdited.connect(lambda _text: self._on_output_option_changed())
        self.preview.mask_overlay_changed.connect(self._render_preview)
        self.preview.current_image_changed.connect(self._render_preview)

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        self._images_user_edited = False
        self._masks_user_edited = False
        self._xml_user_edited = False
        self._ply_user_edited = False
        self.preview.set_scene_dir(path, refresh=False)
        self._refresh_default_paths()
        self._sync_preview_inputs()
        self._sync_mask_settings_context()

    def on_activated(self) -> None:
        self._refresh_default_paths()
        self._sync_preview_inputs()
        self._sync_mask_settings_context()
        self.primary_action_state_changed.emit()

    def focus_output_tab(self) -> None:
        self.settings_tabs.setCurrentIndex(1)

    def _mask_tab_selected(self) -> bool:
        return self._dataset_mask_tab_index is not None and self.settings_tabs.currentIndex() == self._dataset_mask_tab_index

    def _on_settings_tab_changed(self) -> None:
        self._sync_mask_settings_context()
        if self._mask_tab_selected() and self._dataset_mask_step is not None:
            self.work_stack.setCurrentWidget(self._dataset_mask_step.preview_pane)
            self._dataset_mask_step.on_activated()
        else:
            self.work_stack.setCurrentWidget(self.preview_pane)
        self.primary_action_state_changed.emit()

    def _sync_mask_settings_context(self) -> None:
        if self._dataset_mask_step is None:
            return
        if self.scene_dir and self._dataset_mask_step.scene_dir != self.scene_dir:
            self._dataset_mask_step.set_scene_dir(self.scene_dir)
        self._dataset_mask_step.set_dataset_projection("normal")

    def primary_action_text(self) -> str:
        return i18n.t("DATASET_RUN_COLMAP_TEXT")

    def primary_action_tooltip(self) -> str:
        return i18n.tip("DATASET_RUN_COLMAP_TEXT")

    def primary_action_enabled(self) -> bool:
        return (
            bool(self.scene_dir)
            and self._images_dir().is_dir()
            and self._xml_path().is_file()
            and self._ply_path().is_file()
        )

    def build_commands(self) -> StepCommandQueue:
        images = self._images_dir()
        masks = self._masks_dir()
        xml = self._xml_path()
        ply = self._ply_path()
        output = self._output_dir()
        write_source_masks = self._writes_source_masks()
        self._validate_inputs(
            images,
            masks,
            xml,
            ply,
            output,
            use_source_masks=self._dataset_mask_mode() == DATASET_MASK_CONVERT_SFM,
        )
        self._validate_output_options()

        if not self._prepare_output_dir(output, preserve_masks=self._preserves_output_masks()):
            return []

        views = self.view_config.collect_views(include_disabled=True)
        mask_dir = self._dataset_mask_source_dir_for_job() if write_source_masks else None
        job_path = jobs_dir(Path(self.scene_dir)) / "metashape_colmap_job.json"
        payload = metashape_colmap_job(
            scene_dir=Path(self.scene_dir),
            images_dir=images,
            masks_dir=mask_dir,
            xml_path=xml,
            ply_path=ply,
            output_dir=output,
            views=views,
            output_scale=float(self.scale_combo.currentData()),
            output_format=str(self.output_format_combo.currentData() or "jpg"),
            output_bit_depth=str(self.output_bit_depth_combo.currentData() or "8"),
            jpg_quality=self._jpg_quality(),
            undistort_alpha=1.0,
            axis_transform=self._axis_transform_mode(),
            final_orientation=self._final_orientation(),
        )
        write_dataset_job(job_path, payload)
        commands: StepCommandQueue = []
        if self._dataset_mask_step is not None:
            self._sync_mask_settings_context()
            commands.extend(self._dataset_mask_step.build_source_mask_commands())
        commands.append(("metashape_colmap", dataset_app_job(payload, job_path)))
        return commands

    def _validate_inputs(
        self,
        images: Path,
        masks: Path,
        xml: Path,
        ply: Path,
        output: Path,
        *,
        use_source_masks: bool,
    ) -> None:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        if not images.is_dir():
            raise ValueError(i18n.t("COLMAP_TEXT_IMAGES_NOT_FOUND").format(path=images))
        if use_source_masks and self.masks_browse.text().strip() and not masks.is_dir():
            raise ValueError(i18n.t("COLMAP_TEXT_MASKS_NOT_FOUND").format(path=masks))
        if not xml.is_file():
            raise ValueError(i18n.t("COLMAP_TEXT_XML_NOT_FOUND").format(path=xml))
        if not ply.is_file():
            raise ValueError(i18n.t("COLMAP_TEXT_PLY_NOT_FOUND").format(path=ply))

        root = scene_output_dir(Path(self.scene_dir)).resolve()
        try:
            resolved_output = output.resolve()
        except OSError:
            resolved_output = output.absolute()
        if resolved_output == root:
            raise ValueError(i18n.t("COLMAP_TEXT_OUTPUT_ROOT_ERROR").format(path=output))
        try:
            resolved_output.relative_to(root)
        except ValueError as exc:
            raise ValueError(i18n.t("COLMAP_TEXT_OUTPUT_OUTSIDE_ERROR").format(path=output, root=root)) from exc

    def _validate_output_options(self) -> None:
        views = self.view_config.collect_views(include_disabled=True)
        enabled = sum(1 for view in views if view["enabled"])
        if enabled <= 0:
            raise ValueError("少なくとも1つのビューを有効にしてください")
        if enabled > _BLOCK_ENABLED_VIEWS:
            raise ValueError(f"ビュー数が多すぎます ({enabled})。{_BLOCK_ENABLED_VIEWS} 以下にしてください。")
        self._jpg_quality()

    def _prepare_output_dir(self, output: Path, *, preserve_masks: bool = False) -> bool:
        if self.scene_dir is None:
            return False
        scene = Path(self.scene_dir)
        output_root = scene_output_dir(scene)
        preserved_masks: Path | None = None
        if path_has_contents(output):
            result = QMessageBox.question(
                self,
                i18n.t("OUTPUT_RESET_TITLE"),
                i18n.t("OUTPUT_RESET_MESSAGE").format(path=str(output)),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                return False
            source_masks = output / "masks"
            if preserve_masks and source_masks.is_dir():
                preserved_masks = output_root / f".{output.name}_preserved_masks"
                if preserved_masks.exists():
                    clear_path(preserved_masks, allowed_roots=[output_root])
                shutil.move(str(source_masks), str(preserved_masks))
            clear_path(output, allowed_roots=[output_root])
        output.mkdir(parents=True, exist_ok=True)
        if preserved_masks is not None and preserved_masks.is_dir():
            shutil.move(str(preserved_masks), str(output / "masks"))
        return True

    def _dataset_mask_mode(self) -> str:
        if self._dataset_mask_step is None:
            return DATASET_MASK_CONVERT_SFM
        return self._dataset_mask_step.mask_mode()

    def _writes_source_masks(self) -> bool:
        return self._dataset_mask_mode() in {DATASET_MASK_CONVERT_SFM, DATASET_MASK_GENERATE_TRAINING}

    def _preserves_output_masks(self) -> bool:
        return self._dataset_mask_mode() == DATASET_MASK_REUSE_EXISTING

    def _dataset_mask_source_dir_for_job(self) -> Path | None:
        if self._dataset_mask_step is None:
            masks = self._masks_dir()
            return masks if masks.is_dir() else None
        self._sync_mask_settings_context()
        return self._dataset_mask_step.source_mask_dir_for_dataset(
            require_existing=self._dataset_mask_mode() != DATASET_MASK_GENERATE_TRAINING
        )

    def _dataset_training_source_masks_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return step4_meta_dir(Path(self.scene_dir)) / "dataset_masks" / "metashape_colmap_training_source_masks"

    def phase_display_name(self, phase: str) -> str:
        if phase == "metashape_colmap":
            return i18n.t("PHASE_COLMAP_TEXT_MODEL")
        if self._dataset_mask_step is not None:
            return self._dataset_mask_step.phase_display_name(phase)
        return super().phase_display_name(phase)

    def on_line(self, line: str) -> tuple[int, int] | None:
        progress = _PROGRESS_RE.match(line)
        if progress:
            return int(progress.group(1)), int(progress.group(2))
        return None

    def on_queue_finished(self, success: bool) -> None:
        if not success or not self.scene_dir:
            return
        scene = Path(self.scene_dir)
        settings = {
            "profile": "colmap",
            "view_mode": self.view_config.view_mode(),
            "output_scale": float(self.scale_combo.currentData()),
        }
        sfm_record = register_sfm_artifact(
            scene,
            artifact_id=_artifact_id("metashape_sfm"),
            kind=SFM_KIND_METASHAPE_XML_PLY,
            root=scene,
            files={
                "images_dir": self._images_dir(),
                "masks_dir": self._masks_dir(),
                "xml": self._xml_path(),
                "ply": self._ply_path(),
            },
            source_inputs=[self._images_dir(), self._masks_dir(), self._xml_path(), self._ply_path()],
            settings=settings,
        )
        register_dataset_artifact(
            scene,
            artifact_id=_artifact_id("metashape_colmap"),
            root=self._output_dir(),
            kind=DATASET_KIND_COLMAP_DATASET,
            source_artifact_id=sfm_record.id,
            source_inputs=[self._output_dir()],
            settings=settings,
            metadata={"source_kind": "metashape_xml_ply"},
        )

    def _images_dir(self) -> Path:
        text = self.images_browse.text()
        if text:
            return Path(text)
        if not self.scene_dir:
            return Path()
        return scene_images_dir(Path(self.scene_dir))

    def _masks_dir(self) -> Path:
        text = self.masks_browse.text()
        if text:
            return Path(text)
        if not self.scene_dir:
            return Path()
        return scene_masks_dir(Path(self.scene_dir))

    def _xml_path(self) -> Path:
        text = self.xml_browse.text()
        return Path(text) if text else Path()

    def _ply_path(self) -> Path:
        text = self.ply_browse.text()
        return Path(text) if text else Path()

    def _output_dir(self) -> Path:
        return self._default_output_dir()

    def _default_output_dir(self) -> Path:
        if not self.scene_dir:
            return Path()
        return scene_output_dir(Path(self.scene_dir)) / "metashape_colmap"

    def _axis_transform_mode(self) -> str:
        return _AXIS_NONE

    def _final_orientation(self) -> str:
        return FINAL_ORIENTATION_NONE

    def _jpg_quality(self) -> int:
        try:
            quality = int(self.jpg_quality_edit.text().strip())
        except ValueError as exc:
            raise ValueError("JPG/WebP 品質は整数で指定してください") from exc
        if not 1 <= quality <= 100:
            raise ValueError("JPG/WebP 品質は 1-100 の範囲で指定してください")
        return quality

    def _refresh_default_paths(self) -> None:
        if not self.scene_dir:
            self._syncing_paths = True
            try:
                self.images_browse.set_text("")
                self.masks_browse.set_text("")
                self.xml_browse.set_text("")
                self.ply_browse.set_text("")
            finally:
                self._syncing_paths = False
            return

        scene = Path(self.scene_dir)
        self._syncing_paths = True
        try:
            if not self._images_user_edited:
                self.images_browse.set_text(str(scene_images_dir(scene)))
            if not self._masks_user_edited:
                masks = scene_masks_dir(scene)
                self.masks_browse.set_text(str(masks) if masks.is_dir() else "")
            if not self._xml_user_edited:
                xml = _guess_single_file(scene, "*.xml", exclude_names=set())
                self.xml_browse.set_text(str(xml) if xml else "")
            if not self._ply_user_edited:
                ply = _guess_single_file(scene, "*.ply", exclude_names={"pointcloud.ply"})
                self.ply_browse.set_text(str(ply) if ply else "")
        finally:
            self._syncing_paths = False

    def _on_path_changed(self, field: str) -> None:
        if self._syncing_paths:
            return
        if field == "images":
            self._images_user_edited = True
        elif field == "masks":
            self._masks_user_edited = True
        elif field == "xml":
            self._xml_user_edited = True
        elif field == "ply":
            self._ply_user_edited = True
        if field in {"images", "masks", "xml"}:
            self._sync_preview_inputs()
        self.primary_action_state_changed.emit()

    def _on_output_option_changed(self, *_args) -> None:
        self._update_output_count()
        self._render_preview()
        self.primary_action_state_changed.emit()

    def _on_views_changed(self) -> None:
        self._update_output_count()
        self._render_preview()
        self.primary_action_state_changed.emit()

    def _sync_preview_inputs(self) -> None:
        if not self.scene_dir:
            self._metashape_preview_action_counts = None
            self.preview.set_scene_dir("", refresh=False)
            self.preview.set_perspective_supported_paths(())
            self.preview.set_image_paths(None, refresh=False)
            self.preview.set_image_dir("", refresh=True)
            self._update_output_count()
            self._render_preview()
            return

        self.preview.set_scene_dir(str(self.scene_dir), refresh=False)
        images = self._images_dir()
        self.preview.set_image_dir(str(images) if images.is_dir() else "", refresh=False)
        if not self._sync_metashape_preview_targets(images):
            self._metashape_preview_action_counts = None
            self.preview.set_image_paths(None, refresh=False)
            self._sync_preview_perspective_paths(images)
        self.preview.refresh_image_list(prefer_current=True)
        self._update_output_count()
        self._render_preview()

    def _sync_metashape_preview_targets(self, images: Path) -> bool:
        if not self.scene_dir or not images.is_dir():
            return False
        xml = self._xml_path()
        if not xml.is_file():
            return False
        masks = self._masks_dir()
        try:
            targets = build_metashape_preview_targets(
                scene_dir=self.scene_dir,
                images_dir=images,
                masks_dir=masks if masks.is_dir() else None,
                xml_path=xml,
            )
        except Exception:
            return False
        self._metashape_preview_action_counts = targets.action_counts
        self.preview.set_perspective_supported_paths(targets.equirect_paths)
        self.preview.set_image_paths(targets.image_paths, refresh=False)
        return True

    def _sync_preview_perspective_paths(self, images: Path) -> None:
        if not self.scene_dir or not images.is_dir():
            self.preview.set_perspective_supported_paths(())
            return
        try:
            inventory = build_scene_inventory(self.scene_dir, images_dir=images)
        except Exception:
            self.preview.set_perspective_supported_paths(())
            return
        self.preview.set_perspective_supported_paths(
            image.path for image in inventory.images if image.projection == PROJECTION_EQUIRECTANGULAR
        )

    def _render_preview(self) -> None:
        try:
            views = self.view_config.collect_views(include_disabled=True)
        except Exception:
            views = []
        masks = self._masks_dir()
        self.preview.render(views, str(masks) if masks.is_dir() else "")

    def _update_output_count(self) -> None:
        if not hasattr(self, "view_config"):
            return
        try:
            views = self.view_config.collect_views(include_disabled=True)
        except Exception:
            self.view_config.set_output_count_text(f"{i18n.t('OUTPUT_IMAGE_COUNT_LABEL')}: -")
            return
        enabled = sum(1 for view in views if view["enabled"])
        if self._metashape_preview_action_counts is not None:
            total = metashape_output_count_for_actions(
                self._metashape_preview_action_counts,
                enabled_view_count=enabled,
            )
        else:
            total = len(getattr(self.preview, "preview_images", []) or []) * enabled
        warn = ""
        if enabled > _BLOCK_ENABLED_VIEWS:
            warn = f" [{i18n.t('EXCEED')}]"
        elif enabled > _WARN_ENABLED_VIEWS:
            warn = f" [{i18n.t('HIGH')}]"
        count_text = i18n.t("OUTPUT_IMAGE_COUNT_FORMAT").format(count=total)
        self.view_config.set_output_count_text(f"{i18n.t('OUTPUT_IMAGE_COUNT_LABEL')}: {count_text}{warn}")


def _guess_single_file(scene: Path, pattern: str, *, exclude_names: set[str]) -> Path | None:
    candidates = tuple(
        sorted(
            (
                path
                for path in scene.glob(pattern)
                if path.is_file() and path.name.lower() not in {name.lower() for name in exclude_names}
            ),
            key=lambda path: path.name.lower(),
        )
    )
    return candidates[0] if len(candidates) == 1 else None


def _artifact_id(prefix: str) -> str:
    from core.scene_project import utc_now_iso

    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '')}"
