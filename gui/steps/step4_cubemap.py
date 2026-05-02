"""Step 4: キューブマップ変換 (Postshot / LichtFeld 出力)"""
from __future__ import annotations

import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QSize, Qt

from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.collapsible_section import CollapsibleSection
from gui.common.form_rows import add_tooltip_row
from gui.cubemap.view_config import ViewConfigWidget, _BLOCK_ENABLED_VIEWS, _WARN_ENABLED_VIEWS
from gui.cubemap.preview_renderer import PreviewWidget
from gui.version import APP_VERSION
from gui.steps.base_step import (
    SETTINGS_PANE_MARGINS,
    SETTINGS_PANE_WIDTH,
    BaseStepWidget,
    configure_settings_scroll,
)

_CONVERT_RE = re.compile(r"^Converting\s+(\d+)\s+(?:images|files)\.\.\.$")
_PROGRESS_RE = re.compile(r"^\[progress\]\s+(\d+)\s*/\s*(\d+)")
_PROFILE_POSTSHOT = "postshot"
_PROFILE_BRUSH = "brush"
_PROFILE_LICHTFELD = "lichtfeld"
_PROFILE_CUSTOM = "custom"
_AXIS_POSTSHOT = "postshot"
_AXIS_BRUSH = "brush"
_AXIS_NONE = "none"
_NORMAL_OUTPUT_SCALE = 2.0 / math.pi
_EXPORT_SETTINGS_NAME = "stechdrive_export_settings.json"


class ElidedPathLabel(QLabel):
    """Keep long paths on one line while preserving the full path internally."""

    def __init__(self, text: str = "-", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = text
        self.setWordWrap(False)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self._full_text = text or "-"
        self._apply_elide()

    def full_text(self) -> str:
        return self._full_text

    def sizeHint(self) -> QSize:
        base = super().sizeHint()
        return QSize(0, base.height())

    def minimumSizeHint(self) -> QSize:
        base = super().minimumSizeHint()
        return QSize(0, base.height())

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_elide()

    def _apply_elide(self) -> None:
        width = max(0, self.contentsRect().width())
        text = self._full_text
        if width > 0:
            text = QFontMetrics(self.font()).elidedText(text, Qt.ElideMiddle, width)
        QLabel.setText(self, text)


class CubemapStep(BaseStepWidget):
    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._converted_total = 0
        self._processed = 0
        self._explicit_progress = False
        self._syncing_profile_controls = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # 左パネル: 設定 (スクロール可能)
        top_scroll = QScrollArea()
        configure_settings_scroll(top_scroll)
        top = QWidget()
        top.setObjectName("settingsPane")
        top.setMinimumWidth(0)
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(*SETTINGS_PANE_MARGINS)
        top_layout.setSpacing(8)
        left_layout = top_layout  # 既存コードとの互換用エイリアス

        form = QFormLayout()
        form.setSpacing(6)

        output_dir_label = QLabel(i18n.OUTPUT_DIR)
        output_dir_label.setToolTip(i18n.tip("OUTPUT_DIR_CUBEMAP"))
        self.output_path_label = ElidedPathLabel("-")
        self.output_path_label.setToolTip(i18n.tip("OUTPUT_DIR_CUBEMAP"))
        left_layout.addWidget(output_dir_label)
        left_layout.addWidget(self.output_path_label)

        profile_row = QHBoxLayout()
        profile_row.setSpacing(8)
        self.profile_combo = QComboBox()
        self.profile_combo.setToolTip(i18n.tip("TARGET_PROFILE"))
        self.profile_combo.addItem(i18n.PROFILE_POSTSHOT, _PROFILE_POSTSHOT)
        self.profile_combo.addItem(i18n.PROFILE_BRUSH, _PROFILE_BRUSH)
        self.profile_combo.addItem(i18n.PROFILE_LICHTFELD, _PROFILE_LICHTFELD)
        self.profile_combo.addItem(i18n.PROFILE_CUSTOM, _PROFILE_CUSTOM)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        profile_row.addWidget(self.profile_combo)
        output_scale_label = QLabel(i18n.OUTPUT_SCALE + ":")
        output_scale_label.setToolTip(i18n.tip("OUTPUT_SCALE"))
        profile_row.addWidget(output_scale_label)
        self.scale_combo = QComboBox()
        self.scale_combo.setToolTip(i18n.tip("OUTPUT_SCALE"))
        self.scale_combo.addItem("Full (Quality)", 1.0)
        self.scale_combo.addItem("Normal", _NORMAL_OUTPUT_SCALE)
        self.scale_combo.addItem("Half (Light)", 0.5)
        full_scale_index = self.scale_combo.findData(1.0)
        if full_scale_index >= 0:
            self.scale_combo.setCurrentIndex(full_scale_index)
        self.scale_combo.setFixedWidth(136)
        profile_row.addWidget(self.scale_combo)
        profile_row.addStretch()
        add_tooltip_row(form, i18n.TARGET_PROFILE, profile_row, i18n.tip("TARGET_PROFILE"))

        self.profile_hint = QLabel("")
        self.profile_hint.setStyleSheet("color: #8888aa; font-size: 9pt;")
        self.profile_hint.setVisible(False)
        form.addRow("", self.profile_hint)

        left_layout.addLayout(form)

        # Metashapeインポート設定（折りたたみ）
        preprocess = CollapsibleSection(i18n.METASHAPE_PREPROCESS, expanded=False)
        pp_form = QFormLayout()

        self.preprocess_cb = QCheckBox(i18n.t("PREPROCESS_RUN_LABEL"))
        self.preprocess_cb.setToolTip(i18n.tip("PREPROCESS_CB"))
        self.preprocess_cb.setChecked(True)
        self.preprocess_cb.toggled.connect(self._on_preprocess_toggle)
        pp_form.addRow("", self.preprocess_cb)

        self.ms_images_path_label = ElidedPathLabel("-")
        self.ms_images_path_label.setToolTip(i18n.tip("MS_IMAGES"))
        add_tooltip_row(pp_form, i18n.t("MS_IMAGES_LABEL"), self.ms_images_path_label, i18n.tip("MS_IMAGES"))

        self.ms_xml_browse = BrowseWidget(mode="file", filter_str="XML (*.xml);;すべて (*.*)")
        self.ms_xml_browse.setToolTip(i18n.tip("MS_XML"))
        add_tooltip_row(pp_form, i18n.METASHAPE_XML, self.ms_xml_browse, i18n.tip("MS_XML"))

        self.ms_ply_browse = BrowseWidget(mode="file", filter_str="PLY (*.ply);;すべて (*.*)")
        self.ms_ply_browse.setToolTip(i18n.tip("MS_PLY"))
        add_tooltip_row(pp_form, i18n.METASHAPE_PLY, self.ms_ply_browse, i18n.tip("MS_PLY"))

        import_advanced = CollapsibleSection(i18n.t("ADVANCED_SETTINGS"), expanded=False)
        import_adv_form = QFormLayout()
        import_adv_form.setSpacing(6)

        self.metashape_import_options_row = QWidget()
        import_option_row = QHBoxLayout(self.metashape_import_options_row)
        import_option_row.setContentsMargins(0, 0, 0, 0)
        import_option_row.setSpacing(8)
        self.ms_scale_label = QLabel(i18n.t("SCALE_FACTOR_COMPACT"))
        self.ms_scale_label.setToolTip(i18n.tip("SCALE_FACTOR"))
        import_option_row.addWidget(self.ms_scale_label)

        self.ms_scale_edit = QLineEdit("1.0")
        self.ms_scale_edit.setFixedWidth(72)
        self.ms_scale_edit.setToolTip(i18n.tip("SCALE_FACTOR"))
        self.ms_scale_edit.textEdited.connect(self._on_profile_option_changed)
        import_option_row.addWidget(self.ms_scale_edit)

        self.ms_use_ply_cb = QCheckBox(i18n.t("MS_USE_PLY"))
        self.ms_use_ply_cb.setToolTip(i18n.tip("MS_USE_PLY"))
        self.ms_use_ply_cb.toggled.connect(self._on_profile_option_changed)
        import_option_row.addWidget(self.ms_use_ply_cb)

        self.ms_no_fix_rot_cb = QCheckBox(i18n.NO_FIX_ROTATION)
        self.ms_no_fix_rot_cb.setToolTip(i18n.tip("NO_FIX_ROTATION"))
        self.ms_no_fix_rot_cb.toggled.connect(self._on_profile_option_changed)
        import_option_row.addWidget(self.ms_no_fix_rot_cb)
        import_option_row.addStretch()
        import_adv_form.addRow(self.metashape_import_options_row)
        import_advanced.content_layout.addLayout(import_adv_form)

        preprocess.content_layout.addLayout(pp_form)
        preprocess.content_layout.addWidget(import_advanced)
        left_layout.addWidget(preprocess)
        self._pp_widgets = [
            self.ms_images_path_label, self.ms_xml_browse, self.ms_ply_browse,
            self.ms_scale_edit, self.ms_use_ply_cb, self.ms_no_fix_rot_cb,
        ]

        # 高度な出力設定（折りたたみ）
        adv_output = CollapsibleSection(i18n.t("ADVANCED_OUTPUT_SECTION"), expanded=False)
        adv_form = QFormLayout()
        adv_form.setSpacing(6)

        self.axis_transform_combo = QComboBox()
        self.axis_transform_combo.setToolTip(i18n.tip("AXIS_TRANSFORM"))
        self.axis_transform_combo.addItem(i18n.t("AXIS_TRANSFORM_POSTSHOT"), _AXIS_POSTSHOT)
        self.axis_transform_combo.addItem(i18n.t("AXIS_TRANSFORM_BRUSH"), _AXIS_BRUSH)
        self.axis_transform_combo.addItem(i18n.t("AXIS_TRANSFORM_NONE"), _AXIS_NONE)
        self.axis_transform_combo.setFixedWidth(180)
        self.axis_transform_combo.currentIndexChanged.connect(self._on_profile_option_changed)
        add_tooltip_row(adv_form, i18n.t("AXIS_TRANSFORM"), self.axis_transform_combo, i18n.tip("AXIS_TRANSFORM"))

        self.yaw_per_frame_edit = QLineEdit("30.0")
        self.yaw_per_frame_edit.setFixedWidth(80)
        self.yaw_per_frame_edit.setToolTip(i18n.t("YAW_OFFSET_PER_FRAME_HINT"))
        add_tooltip_row(
            adv_form,
            i18n.t("YAW_OFFSET_PER_FRAME"),
            self.yaw_per_frame_edit,
            i18n.t("YAW_OFFSET_PER_FRAME_HINT"),
        )

        self.output_format_combo = QComboBox()
        self.output_format_combo.addItem(i18n.t("OUTPUT_FORMAT_AUTO"), "auto")
        for fmt in ("jpg", "png", "tiff", "webp"):
            self.output_format_combo.addItem(fmt, fmt)
        self.output_format_combo.setFixedWidth(180)
        adv_form.addRow(i18n.t("OUTPUT_FORMAT"), self.output_format_combo)

        self.output_bit_depth_combo = QComboBox()
        self.output_bit_depth_combo.addItem(i18n.t("OUTPUT_BIT_DEPTH_8"), "8")
        self.output_bit_depth_combo.addItem(i18n.t("OUTPUT_BIT_DEPTH_SOURCE"), "source")
        self.output_bit_depth_combo.setFixedWidth(180)
        adv_form.addRow(i18n.t("OUTPUT_BIT_DEPTH"), self.output_bit_depth_combo)

        self.invert_masks_cb = QCheckBox(i18n.INVERT_MASKS)
        self.invert_masks_cb.setToolTip(i18n.tip("INVERT_MASKS"))
        adv_form.addRow("", self.invert_masks_cb)

        self.no_image_cb = QCheckBox(i18n.NO_IMAGE)
        self.no_image_cb.setToolTip(i18n.tip("NO_IMAGE"))
        adv_form.addRow("", self.no_image_cb)

        self.jpg_quality_edit = QLineEdit("95")
        self.jpg_quality_edit.setFixedWidth(80)
        adv_form.addRow(i18n.t("JPG_QUALITY"), self.jpg_quality_edit)

        self.export_colmap_cb = QCheckBox(i18n.t("EXPORT_COLMAP"))
        self.export_colmap_cb.setToolTip(i18n.t("EXPORT_COLMAP_HINT"))
        adv_form.addRow("", self.export_colmap_cb)

        adv_output.content_layout.addLayout(adv_form)
        left_layout.addWidget(adv_output)

        # ビュー設定
        self.view_config = ViewConfigWidget()
        self.view_config.views_changed.connect(self._on_views_changed)
        left_layout.addWidget(self.view_config)

        left_layout.addStretch()

        # 右パネル: プレビュー
        preview_pane = QWidget()
        preview_pane.setObjectName("workPane")
        preview_layout = QVBoxLayout(preview_pane)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(8)
        preview_title = QLabel(i18n.t("CUBEMAP_PREVIEW_SECTION"))
        preview_title.setObjectName("paneTitle")
        preview_layout.addWidget(preview_title)
        self.preview = PreviewWidget()
        self.preview.mask_slider.valueChanged.connect(lambda _: self._render_preview())
        self.preview.current_image_changed.connect(lambda: self._render_preview())
        preview_layout.addWidget(self.preview, stretch=1)

        top_scroll.setWidget(top)
        splitter.addWidget(top_scroll)
        splitter.addWidget(preview_pane)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([SETTINGS_PANE_WIDTH, 760])
        layout.addWidget(splitter)

        lichtfeld_index = self.profile_combo.findData(_PROFILE_LICHTFELD)
        if lichtfeld_index >= 0:
            self.profile_combo.setCurrentIndex(lichtfeld_index)
        self._on_profile_changed(self.profile_combo.currentIndex())

    # -- シーンディレクトリ --

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        if not path:
            self.output_path_label.setToolTip(i18n.tip("OUTPUT_DIR_CUBEMAP"))
            self.output_path_label.set_full_text("-")
            self.ms_images_path_label.setToolTip(i18n.tip("MS_IMAGES"))
            self.ms_images_path_label.set_full_text("-")
            return
        p = Path(path)
        output = str(self._output_dir())
        self.output_path_label.setToolTip(f"{i18n.tip('OUTPUT_DIR_CUBEMAP')}\n{output}")
        self.output_path_label.set_full_text(output)
        images_dir = str(self._metashape_images_dir())
        self.ms_images_path_label.setToolTip(f"{i18n.tip('MS_IMAGES')}\n{images_dir}")
        self.ms_images_path_label.set_full_text(images_dir)
        self.ms_xml_browse.set_text(str(self._guess_xml(p)))
        self.ms_ply_browse.set_text(self._guess_ply(p))
        self.preview.set_scene_dir(path)
        self._update_output_count()
        self._render_preview()

    def primary_action_text(self) -> str:
        return i18n.t("EXPORT")

    def primary_action_tooltip(self) -> str:
        return i18n.tip("RUN_CUBEMAP")

    # -- プロファイル --

    def _profile_id(self) -> str:
        return self.profile_combo.currentData() or _PROFILE_CUSTOM

    def _effective_profile(self) -> str:
        mode = self._axis_transform_mode()
        if mode == _AXIS_NONE:
            return _PROFILE_LICHTFELD
        if mode == _AXIS_BRUSH:
            return _PROFILE_BRUSH
        return _PROFILE_POSTSHOT

    @staticmethod
    def _profile_axis_default(profile: str) -> str:
        if profile == _PROFILE_LICHTFELD:
            return _AXIS_NONE
        if profile == _PROFILE_BRUSH:
            return _AXIS_BRUSH
        return _AXIS_POSTSHOT

    @staticmethod
    def _profile_use_ply_default(profile: str) -> bool:
        return profile == _PROFILE_LICHTFELD

    @staticmethod
    def _profile_scale_default(_profile: str) -> float:
        return 1.0

    @staticmethod
    def _profile_no_fix_rotation_default(_profile: str) -> bool:
        return False

    def _set_combo_data(self, combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _sync_profile_defaults(self, profile: str) -> None:
        if profile == _PROFILE_CUSTOM:
            return
        self._syncing_profile_controls = True
        try:
            self._set_combo_data(self.axis_transform_combo, self._profile_axis_default(profile))
            self.ms_use_ply_cb.setChecked(self._profile_use_ply_default(profile))
            self.ms_scale_edit.setText("1.0")
            self.ms_no_fix_rot_cb.setChecked(self._profile_no_fix_rotation_default(profile))
        finally:
            self._syncing_profile_controls = False

    def _axis_transform_mode(self) -> str:
        data = self.axis_transform_combo.currentData()
        return data if data in {_AXIS_POSTSHOT, _AXIS_BRUSH, _AXIS_NONE} else _AXIS_POSTSHOT

    def _on_profile_changed(self, _index: int) -> None:
        p = self._profile_id()
        self._sync_profile_defaults(p)
        self.profile_hint.setText(i18n.t("PROFILE_CUSTOM_HINT") if p == _PROFILE_CUSTOM else "")
        self.profile_hint.setVisible(p == _PROFILE_CUSTOM)
        self._sync_ply_browse_enabled()

    def _on_profile_option_changed(self, *_args) -> None:
        if self._syncing_profile_controls:
            return
        current = self._profile_id()
        if current != _PROFILE_CUSTOM:
            axis_changed = self._axis_transform_mode() != self._profile_axis_default(current)
            ply_changed = self.ms_use_ply_cb.isChecked() != self._profile_use_ply_default(current)
            no_fix_changed = (
                self.ms_no_fix_rot_cb.isChecked()
                != self._profile_no_fix_rotation_default(current)
            )
            try:
                scale_changed = not math.isclose(
                    float(self.ms_scale_edit.text().strip()),
                    self._profile_scale_default(current),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            except ValueError:
                scale_changed = self.ms_scale_edit.text().strip() != "1.0"
            if axis_changed or ply_changed or no_fix_changed or scale_changed:
                custom_idx = self.profile_combo.findData(_PROFILE_CUSTOM)
                if custom_idx >= 0:
                    self.profile_combo.setCurrentIndex(custom_idx)
        self._sync_ply_browse_enabled()

    def _on_preprocess_toggle(self, enabled: bool) -> None:
        for w in self._pp_widgets:
            w.setEnabled(enabled)
        self._sync_ply_browse_enabled()

    def _preprocess_uses_ply(self) -> bool:
        return self.ms_use_ply_cb.isChecked()

    def _sync_ply_browse_enabled(self) -> None:
        self.ms_ply_browse.setEnabled(self.preprocess_cb.isChecked() and self._preprocess_uses_ply())

    # -- ビュー --

    def _on_views_changed(self) -> None:
        self._update_output_count()
        self._render_preview()

    def _render_preview(self) -> None:
        try:
            views = self.view_config.collect_views(include_disabled=True)
        except Exception:
            views = []
        self.preview.render(views, str(self._mask_dir()))

    def _count_input_images(self) -> int:
        scene = Path(self.scene_dir) if self.scene_dir else Path(".")
        images = scene / "images"
        roots = [images] if images.is_dir() else [scene]
        exts = {".jpg", ".jpeg", ".png"}
        seen: set[str] = set()
        count = 0
        for root in roots:
            if not root.is_dir():
                continue
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    key = str(p.resolve()).lower()
                    if key not in seen:
                        seen.add(key)
                        count += 1
        return count

    def _update_output_count(self) -> None:
        label = i18n.t("OUTPUT_IMAGE_COUNT_LABEL")
        try:
            views = self.view_config.collect_views(include_disabled=True)
        except Exception:
            self.view_config.set_output_count_text(f"{label}: -")
            return
        enabled = sum(1 for v in views if v["enabled"])
        sources = self._count_input_images()
        total = sources * enabled
        warn = ""
        if enabled > _BLOCK_ENABLED_VIEWS:
            warn = " [超過]"
        elif enabled > _WARN_ENABLED_VIEWS:
            warn = " [多い]"
        count_text = i18n.t("OUTPUT_IMAGE_COUNT_FORMAT").format(count=total)
        self.view_config.set_output_count_text(f"{label}: {count_text}{warn}")

    # -- コマンド構築 --

    def build_commands(self) -> list[tuple[str, list[str]]]:
        self._validate_bundle()

        preprocess_cmd: list[str] | None = None
        if self.preprocess_cb.isChecked():
            preprocess_cmd = self._build_preprocess_cmd()

        if not self._prepare_output_dir():
            return []

        steps = []
        if preprocess_cmd is not None:
            steps.append(("metashape", preprocess_cmd))
        steps.append(("cubemap", self._build_cubemap_cmd()))
        if self.export_colmap_cb.isChecked():
            steps.append(("colmap", self._build_colmap_cmd()))
        return steps

    def _build_preprocess_cmd(self) -> list[str]:
        script = self.base_dir / "vendor" / "metashape_360_lfs" / "metashape_360_lfs.py"
        if not script.exists():
            raise FileNotFoundError(f"metashape_360_lfs.py が見つかりません: {script}")
        scene = Path(self.scene_dir)
        if not scene.is_dir():
            raise ValueError(f"シーンフォルダが見つかりません: {scene}")

        images = str(self._metashape_images_dir())
        xml = self.ms_xml_browse.text()
        if not images or not Path(images).is_dir():
            raise ValueError(f"Metashape画像フォルダが見つかりません: {images}")
        if not xml or not Path(xml).is_file():
            raise ValueError(f"Metashape XMLが見つかりません: {xml}")

        scale = float(self.ms_scale_edit.text().strip())
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("スケール係数は正の有限値である必要があります")

        cmd = [
            sys.executable, "-u", str(script),
            "--images", images,
            "--xml", xml,
            "--output", str(scene),
            "--scale", f"{scale:g}",
        ]
        if self._preprocess_uses_ply():
            ply = self.ms_ply_browse.text()
            if not ply or not Path(ply).is_file():
                raise ValueError(f"PLYファイルが見つかりません: {ply}")
            cmd.extend(["--ply", ply])
        if self.ms_no_fix_rot_cb.isChecked():
            cmd.append("--no-fix-rotation")
        return cmd

    def _build_cubemap_cmd(self) -> list[str]:
        script = self.base_dir / "cubemap_transforms_json.py"
        if not script.exists():
            raise FileNotFoundError(f"cubemap_transforms_json.py が見つかりません: {script}")

        scene = Path(self.scene_dir)
        if not scene.is_dir():
            raise ValueError(f"シーンフォルダが見つかりません: {scene}")

        output = self._output_dir()

        views = self.view_config.collect_views(include_disabled=True)
        enabled = sum(1 for v in views if v["enabled"])
        if enabled <= 0:
            raise ValueError("少なくとも1つのビューを有効にしてください")
        if enabled > _BLOCK_ENABLED_VIEWS:
            raise ValueError(f"ビュー数が多すぎます ({enabled})。{_BLOCK_ENABLED_VIEWS} 以下にしてください。")

        views_json = self._write_views_config(output, views)

        scale = float(self.scale_combo.currentData())
        cmd = [
            sys.executable, "-u", str(script),
            str(scene), str(output),
            "--fov", "90",
            "--output_scale", f"{scale:g}",
            "--views-json", str(views_json),
        ]

        axis_mode = self._axis_transform_mode()
        if axis_mode == _AXIS_NONE:
            cmd.append("--no_transform")
        if axis_mode == _AXIS_BRUSH:
            cmd.append("--brush")
        if self.invert_masks_cb.isChecked():
            cmd.append("--invert_masks")
        if self.no_image_cb.isChecked():
            cmd.append("--no_image")

        # 高度な出力設定
        try:
            yaw_step = float(self.yaw_per_frame_edit.text().strip())
        except ValueError:
            raise ValueError("フレーム別ヨー回転は数値で指定してください")
        cmd.extend(["--yaw-offset-per-frame", f"{yaw_step:g}"])

        out_fmt = self.output_format_combo.currentData() or "auto"
        cmd.extend(["--output-format", out_fmt])

        out_depth = self.output_bit_depth_combo.currentData() or "8"
        cmd.extend(["--output-bit-depth", out_depth])

        try:
            jpgq = int(self.jpg_quality_edit.text().strip())
        except ValueError:
            raise ValueError("JPG/WebP 品質は整数で指定してください")
        if not 1 <= jpgq <= 100:
            raise ValueError("JPG/WebP 品質は 1-100 の範囲で指定してください")
        cmd.extend(["--jpg-quality", str(jpgq)])
        return cmd

    def _build_colmap_cmd(self) -> list[str]:
        script = self.base_dir / "transforms_to_colmap.py"
        if not script.exists():
            raise FileNotFoundError(f"transforms_to_colmap.py が見つかりません: {script}")

        scene = Path(self.scene_dir)
        output = self._output_dir()
        colmap_dir = output / "colmap"

        cmd = [
            sys.executable, "-u", str(script),
            str(output), str(colmap_dir),
        ]
        ply = output / "pointcloud.ply"
        if ply.is_file():
            cmd.extend(["--ply", str(ply)])
        else:
            # cubemap 出力ディレクトリ内の任意 .ply をフォールバック
            plys = sorted([p for p in output.glob("*.ply") if p.is_file()])
            if plys:
                cmd.extend(["--ply", str(plys[0])])
        return cmd

    def _write_views_config(self, output_dir: Path, views: list[dict]) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "views_config.json"
        payload = self._views_config_payload(views)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _views_config_payload(views: list[dict]) -> dict:
        return {
            "fov": 90.0,
            "views": [
                {"name": v["name"], "yaw": float(v["yaw"]), "pitch": float(v["pitch"]), "enabled": bool(v["enabled"])}
                for v in views
            ],
        }

    def _export_settings_path(self) -> Path:
        return self._output_dir() / _EXPORT_SETTINGS_NAME

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _collect_export_settings(self) -> dict:
        views = self.view_config.collect_views(include_disabled=True)
        scale = float(self.scale_combo.currentData())
        yaw_step = float(self.yaw_per_frame_edit.text().strip())
        jpg_quality = int(self.jpg_quality_edit.text().strip())
        scene = Path(self.scene_dir) if self.scene_dir else Path(".")
        output = self._output_dir()
        profile = self._profile_id()

        return {
            "app": "stechdrive-3dgs-utils",
            "app_version": APP_VERSION,
            "settings_version": 1,
            "created_at": self._utc_now_iso(),
            "scene_dir": str(scene),
            "output_dir": str(output),
            "target_profile": profile,
            "effective_profile": self._effective_profile(),
            "axis_transform": self._axis_transform_mode(),
            "fov": 90.0,
            "image_size": {
                "label": self.scale_combo.currentText(),
                "scale": scale,
            },
            "view_config": {
                "mode": self.view_config.view_mode(),
                "yaw_offset": self.view_config.yaw_offset(),
                "yaw_slots": self.view_config.yaw_slot_count(),
                "pitch_rows_text": self.view_config.pitch_edit.text().strip(),
                "cube6_drop_top": self.view_config.cube6_drop_top.isChecked(),
                "cube6_drop_bottom": self.view_config.cube6_drop_bottom.isChecked(),
                "views": [
                    {
                        "name": v["name"],
                        "yaw": float(v["yaw"]),
                        "pitch": float(v["pitch"]),
                        "enabled": bool(v["enabled"]),
                    }
                    for v in views
                ],
            },
            "views_config_path": "views_config.json",
            "views_config_snapshot": self._views_config_payload(views),
            "conversion": {
                "yaw_offset_per_frame": yaw_step,
                "output_format": self.output_format_combo.currentData() or "auto",
                "output_bit_depth": self.output_bit_depth_combo.currentData() or "8",
                "jpg_quality": jpg_quality,
                "invert_masks": self.invert_masks_cb.isChecked(),
                "no_image": self.no_image_cb.isChecked(),
                "export_colmap": self.export_colmap_cb.isChecked(),
            },
            "metashape_import": {
                "enabled": self.preprocess_cb.isChecked(),
                "use_ply": self._preprocess_uses_ply(),
                "images_dir": str(self._metashape_images_dir()),
                "xml": self.ms_xml_browse.text(),
                "ply": self.ms_ply_browse.text()
                if self.preprocess_cb.isChecked() and self._preprocess_uses_ply()
                else "",
                "scale": float(self.ms_scale_edit.text().strip()),
                "no_fix_rotation": self.ms_no_fix_rot_cb.isChecked(),
            },
            "inputs": {
                "transforms_json": str(scene / "transforms.json"),
                "masks_dir": str(self._mask_dir()),
                "ply_source": str(self._resolve_ply_source() or ""),
            },
            "output_files": {
                "settings": _EXPORT_SETTINGS_NAME,
                "views_config": "views_config.json",
                "transforms_json": "transforms.json",
                "images_dir": "images",
                "masks_dir": "masks",
            },
        }

    def _write_export_settings(self) -> None:
        path = self._export_settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._collect_export_settings()
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _output_dir(self) -> Path:
        if not self.scene_dir:
            return Path("output")
        return Path(self.scene_dir) / "output"

    def _mask_dir(self) -> Path:
        if not self.scene_dir:
            return Path("masks")
        return Path(self.scene_dir) / "masks"

    def _metashape_images_dir(self) -> Path:
        if not self.scene_dir:
            return Path("images")
        return Path(self.scene_dir) / "images"

    def _prepare_output_dir(self) -> bool:
        output = self._output_dir()
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))

        scene = Path(self.scene_dir).resolve()
        try:
            resolved_output = output.resolve()
        except OSError:
            resolved_output = output.absolute()
        if resolved_output.parent != scene:
            raise ValueError(f"出力フォルダがシーンフォルダ外です: {output}")

        if self.no_image_cb.isChecked():
            output.mkdir(parents=True, exist_ok=True)
            return True

        if output.exists() and any(output.iterdir()):
            result = QMessageBox.question(
                self,
                i18n.t("OUTPUT_RESET_TITLE"),
                i18n.t("OUTPUT_RESET_MESSAGE").format(path=str(output)),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                return False
            self._clear_output_dir(output)

        output.mkdir(parents=True, exist_ok=True)
        return True

    @staticmethod
    def _clear_output_dir(output: Path) -> None:
        for child in output.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    # -- バンドル検証 --

    def _validate_bundle(self) -> None:
        profile = self._effective_profile()
        source = self._resolve_ply_source()
        if source is not None:
            return
        if profile == _PROFILE_LICHTFELD and self.preprocess_cb.isChecked() and self._preprocess_uses_ply():
            return
        if profile == _PROFILE_LICHTFELD:
            raise ValueError("LichtFeldプロファイルにはpointcloud.plyが必要です。前処理でPLYを有効にしてください。")
        raise ValueError(
            "Postshot/BrushプロファイルにはMetashapeからエクスポートしたRAW PLYが必要です。"
            "LichtFeld用のpointcloud.plyは使用できません。"
        )

    def _resolve_ply_source(self) -> Path | None:
        scene = Path(self.scene_dir) if self.scene_dir else Path(".")
        if self._axis_transform_mode() == _AXIS_NONE:
            candidates = [scene / "pointcloud.ply"]
            for c in candidates:
                if c.is_file():
                    return c
            return None
        else:
            candidates = [scene / "metashape.ply", scene / "sparse.ply"]

        for c in candidates:
            if c.is_file():
                return c
        plys = sorted(
            [p for p in scene.glob("*.ply") if p.is_file() and p.name.lower() != "pointcloud.ply"],
            key=lambda x: x.name.lower(),
        )
        if plys:
            return plys[0]
        return None

    # -- バンドル後処理 --

    def on_queue_finished(self, success: bool) -> None:
        if success:
            try:
                self._finalize_bundle()
            except Exception:
                pass

    def _finalize_bundle(self) -> None:
        output = self._output_dir()
        output.mkdir(parents=True, exist_ok=True)

        source = self._resolve_ply_source()
        if source is not None:
            dest = output / source.name
            if source.resolve() != dest.resolve():
                shutil.copy2(source, dest)

            transforms = output / "transforms.json"
            if transforms.is_file():
                data = json.loads(transforms.read_text(encoding="utf-8"))
                data["ply_file_path"] = dest.name
                transforms.write_text(json.dumps(data, indent=2), encoding="utf-8")

        if not self.no_image_cb.isChecked():
            self._write_export_settings()

    # -- プログレス --

    def on_line(self, line: str) -> tuple[int, int] | None:
        progress = _PROGRESS_RE.match(line)
        if progress:
            self._processed = int(progress.group(1))
            self._converted_total = int(progress.group(2))
            self._explicit_progress = True
            return self._processed, self._converted_total

        m = _CONVERT_RE.match(line)
        if m:
            self._converted_total = int(m.group(1))
            self._processed = 0
            self._explicit_progress = False
            return 0, self._converted_total

        if line.startswith("Processing:") and self._converted_total > 0 and not self._explicit_progress:
            self._processed += 1
            return self._processed, self._converted_total

        return None

    # -- ヘルパー --

    @staticmethod
    def _guess_xml(scene_dir: Path) -> Path:
        for name in ["metashape.xml", "cameras.xml"]:
            c = scene_dir / name
            if c.is_file():
                return c
        xmls = sorted([p for p in scene_dir.glob("*.xml") if p.is_file()], key=lambda x: x.name.lower())
        return xmls[0] if xmls else scene_dir / "metashape.xml"

    @staticmethod
    def _guess_ply(scene_dir: Path) -> str:
        for name in ["metashape.ply", "sparse.ply"]:
            c = scene_dir / name
            if c.is_file():
                return str(c)
        plys = sorted(
            [p for p in scene_dir.glob("*.ply") if p.is_file() and p.name.lower() != "pointcloud.ply"],
            key=lambda x: x.name.lower(),
        )
        return str(plys[0]) if plys else ""
