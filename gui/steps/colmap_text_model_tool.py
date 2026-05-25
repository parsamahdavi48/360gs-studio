"""Metashape XML/PLY to portable COLMAP text dataset tool."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.app_job import dataset_app_job
from core.dataset_job_spec import metashape_colmap_job, write_dataset_job
from core.orientation_correction import FINAL_ORIENTATION_LICHTFELD, FINAL_ORIENTATION_NONE
from core.projection_contract import PROJECTION_EQUIRECTANGULAR
from core.scene_inventory import build_scene_inventory
from core.scene_layout import (
    jobs_dir,
    scene_images_dir,
    scene_masks_dir,
    scene_output_dir,
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
from gui.steps.output_reset import clear_path, path_has_contents
from gui.steps.step4_contracts import (
    _AXIS_BRUSH,
    _AXIS_NONE,
    _AXIS_POSTSHOT,
    _NORMAL_OUTPUT_SCALE,
    _PROFILE_BRUSH,
    _PROFILE_LICHTFELD,
    _PROFILE_POSTSHOT,
)


class ColmapTextModelTool(BaseStepWidget):
    """Create a portable COLMAP dataset from Metashape XML and PLY exports."""

    def __init__(self, base_dir: Path, parent=None) -> None:
        super().__init__(base_dir, parent)
        self._syncing_paths = False
        self._images_user_edited = False
        self._masks_user_edited = False
        self._xml_user_edited = False
        self._ply_user_edited = False
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
        left_layout.addWidget(self.settings_tabs, stretch=1)

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

        splitter.addWidget(left_pane)
        splitter.addWidget(preview_pane)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([SETTINGS_PANE_WIDTH, 760])
        layout.addWidget(splitter)

        self._connect_signals()
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
        form = QFormLayout()
        form.setSpacing(6)

        self.profile_combo = QComboBox()
        self.profile_combo.setToolTip(i18n.tip("TARGET_PROFILE"))
        self.profile_combo.addItem(i18n.PROFILE_POSTSHOT, _PROFILE_POSTSHOT)
        self.profile_combo.addItem(i18n.PROFILE_BRUSH, _PROFILE_BRUSH)
        self.profile_combo.addItem(i18n.PROFILE_LICHTFELD, _PROFILE_LICHTFELD)
        self.profile_combo.setCurrentIndex(self.profile_combo.findData(_PROFILE_LICHTFELD))
        add_tooltip_row(form, i18n.TARGET_PROFILE, self.profile_combo, i18n.tip("TARGET_PROFILE"))
        layout.addLayout(form)

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

        format_row = QWidget()
        format_layout = QHBoxLayout(format_row)
        format_layout.setContentsMargins(0, 0, 0, 0)
        format_layout.setSpacing(8)

        self.output_format_combo = QComboBox()
        self.output_format_combo.setToolTip(i18n.tip("OUTPUT_FORMAT"))
        self.output_format_combo.addItem(i18n.t("OUTPUT_FORMAT_AUTO"), "auto")
        for fmt in ("jpg", "png", "tiff", "webp"):
            self.output_format_combo.addItem(fmt, fmt)
        self.output_format_combo.setFixedWidth(96)

        self.output_bit_depth_combo = QComboBox()
        self.output_bit_depth_combo.setToolTip(i18n.tip("OUTPUT_BIT_DEPTH"))
        self.output_bit_depth_combo.addItem(i18n.t("OUTPUT_BIT_DEPTH_8"), "8")
        self.output_bit_depth_combo.addItem(i18n.t("OUTPUT_BIT_DEPTH_SOURCE"), "source")
        self.output_bit_depth_combo.setFixedWidth(86)

        output_format_label = QLabel(i18n.t("OUTPUT_FORMAT_COMPACT"))
        output_format_label.setToolTip(i18n.tip("OUTPUT_FORMAT"))
        format_layout.addWidget(output_format_label)
        format_layout.addWidget(self.output_format_combo)
        output_bit_depth_label = QLabel(i18n.t("OUTPUT_BIT_DEPTH_COMPACT"))
        output_bit_depth_label.setToolTip(i18n.tip("OUTPUT_BIT_DEPTH"))
        format_layout.addWidget(output_bit_depth_label)
        format_layout.addWidget(self.output_bit_depth_combo)
        format_layout.addStretch()
        layout.addWidget(format_row)

        quality_row = QWidget()
        quality_layout = QHBoxLayout(quality_row)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        quality_layout.setSpacing(8)
        self.invert_masks_cb = QCheckBox(i18n.INVERT_MASKS)
        self.invert_masks_cb.setToolTip(i18n.tip("INVERT_MASKS"))
        quality_layout.addWidget(self.invert_masks_cb)
        quality_layout.addSpacing(8)
        self.jpg_quality_label = QLabel(i18n.t("JPG_QUALITY_COMPACT"))
        self.jpg_quality_label.setToolTip(i18n.tip("JPG_QUALITY"))
        quality_layout.addWidget(self.jpg_quality_label)
        self.jpg_quality_edit = QLineEdit("95")
        self.jpg_quality_edit.setToolTip(i18n.tip("JPG_QUALITY"))
        self.jpg_quality_edit.setFixedWidth(64)
        quality_layout.addWidget(self.jpg_quality_edit)
        quality_layout.addStretch()
        layout.addWidget(quality_row)
        layout.addStretch()
        return tab

    def _connect_signals(self) -> None:
        self.images_browse.path_changed.connect(lambda _path: self._on_path_changed("images"))
        self.masks_browse.path_changed.connect(lambda _path: self._on_path_changed("masks"))
        self.xml_browse.path_changed.connect(lambda _path: self._on_path_changed("xml"))
        self.ply_browse.path_changed.connect(lambda _path: self._on_path_changed("ply"))

        self.profile_combo.currentIndexChanged.connect(self._on_output_option_changed)
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

    def on_activated(self) -> None:
        self._refresh_default_paths()
        self._sync_preview_inputs()
        self.primary_action_state_changed.emit()

    def focus_output_tab(self) -> None:
        self.settings_tabs.setCurrentIndex(1)

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
        self._validate_inputs(images, masks, xml, ply, output)
        self._validate_output_options()

        if not self._prepare_output_dir(output):
            return []

        views = self.view_config.collect_views(include_disabled=True)
        mask_dir = masks if masks.is_dir() else None
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
        return [("metashape_colmap", dataset_app_job(payload, job_path))]

    def _validate_inputs(self, images: Path, masks: Path, xml: Path, ply: Path, output: Path) -> None:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        if not images.is_dir():
            raise ValueError(i18n.t("COLMAP_TEXT_IMAGES_NOT_FOUND").format(path=images))
        if self.masks_browse.text().strip() and not masks.is_dir():
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

    def _prepare_output_dir(self, output: Path) -> bool:
        if self.scene_dir is None:
            return False
        scene = Path(self.scene_dir)
        output_root = scene_output_dir(scene)
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
            clear_path(output, allowed_roots=[output_root])
        output.mkdir(parents=True, exist_ok=True)
        return True

    def phase_display_name(self, phase: str) -> str:
        if phase == "metashape_colmap":
            return i18n.t("PHASE_COLMAP_TEXT_MODEL")
        return super().phase_display_name(phase)

    def on_queue_finished(self, success: bool) -> None:
        if not success or not self.scene_dir:
            return
        scene = Path(self.scene_dir)
        settings = {
            "profile": self._profile_id(),
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

    def _profile_id(self) -> str:
        data = self.profile_combo.currentData()
        return data if data in {_PROFILE_POSTSHOT, _PROFILE_BRUSH, _PROFILE_LICHTFELD} else _PROFILE_LICHTFELD

    def _axis_transform_mode(self) -> str:
        profile = self._profile_id()
        if profile == _PROFILE_POSTSHOT:
            return _AXIS_POSTSHOT
        if profile == _PROFILE_BRUSH:
            return _AXIS_BRUSH
        return _AXIS_NONE

    def _final_orientation(self) -> str:
        return FINAL_ORIENTATION_LICHTFELD if self._profile_id() == _PROFILE_LICHTFELD else FINAL_ORIENTATION_NONE

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
        if not self._syncing_paths:
            if field == "images":
                self._images_user_edited = True
            elif field == "masks":
                self._masks_user_edited = True
            elif field == "xml":
                self._xml_user_edited = True
            elif field == "ply":
                self._ply_user_edited = True
        if field in {"images", "masks"}:
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
            self.preview.set_scene_dir("", refresh=False)
            self.preview.set_perspective_supported_paths(())
            self.preview.set_image_dir("", refresh=True)
            self._update_output_count()
            self._render_preview()
            return

        self.preview.set_scene_dir(str(self.scene_dir), refresh=False)
        images = self._images_dir()
        self._sync_preview_perspective_paths(images)
        self.preview.set_image_dir(str(images) if images.is_dir() else "", refresh=True)
        self._update_output_count()
        self._render_preview()

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
