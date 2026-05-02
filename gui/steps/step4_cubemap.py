"""Step 4: キューブマップ変換 (Postshot / LichtFeld 出力)"""
from __future__ import annotations

import json
import math
import re
import shutil
import sys
from pathlib import Path

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
from PySide6.QtCore import Qt

from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.collapsible_section import CollapsibleSection
from gui.common.form_rows import add_tooltip_row
from gui.cubemap.view_config import ViewConfigWidget, _BLOCK_ENABLED_VIEWS, _WARN_ENABLED_VIEWS
from gui.cubemap.preview_renderer import PreviewWidget
from gui.steps.base_step import (
    SETTINGS_PANE_MARGINS,
    SETTINGS_PANE_WIDTH,
    BaseStepWidget,
    configure_settings_scroll,
)

_CONVERT_RE = re.compile(r"^Converting\s+(\d+)\s+images\.\.\.$")
_PROFILE_POSTSHOT = "postshot"
_PROFILE_BRUSH = "brush"
_PROFILE_LICHTFELD = "lichtfeld"
_PROFILE_CUSTOM = "custom"


class CubemapStep(BaseStepWidget):
    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._converted_total = 0
        self._processed = 0
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
        self.output_path_label = QLabel("-")
        self.output_path_label.setToolTip(i18n.tip("OUTPUT_DIR_CUBEMAP"))
        self.output_path_label.setWordWrap(True)
        self.output_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
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
        self.scale_combo.addItem("Half (0.5x)", 0.5)
        self.scale_combo.addItem("Full (1.0x)", 1.0)
        full_scale_index = self.scale_combo.findData(1.0)
        if full_scale_index >= 0:
            self.scale_combo.setCurrentIndex(full_scale_index)
        self.scale_combo.setFixedWidth(120)
        profile_row.addWidget(self.scale_combo)
        profile_row.addStretch()
        add_tooltip_row(form, i18n.TARGET_PROFILE, profile_row, i18n.tip("TARGET_PROFILE"))

        self.profile_hint = QLabel("")
        self.profile_hint.setStyleSheet("color: #8888aa; font-size: 9pt;")
        self.profile_hint.setVisible(False)
        form.addRow("", self.profile_hint)

        self.json_name_edit = QLineEdit("transforms.json")
        self.json_name_edit.setToolTip(i18n.tip("JSON_NAME"))
        self.json_name_edit.setFixedWidth(160)
        add_tooltip_row(form, i18n.JSON_NAME, self.json_name_edit, i18n.tip("JSON_NAME"))
        self.mask_browse = BrowseWidget(mode="dir")
        self.mask_browse.setToolTip(i18n.tip("MASK_DIR_CUBEMAP"))

        # オプション (GroupBox)
        opt_group = CollapsibleSection(i18n.t("CONVERSION_OPTIONS"), expanded=False)
        opt_inner = QVBoxLayout()
        opt_row1 = QHBoxLayout()
        self.mask_from_alpha_cb = QCheckBox(i18n.MASK_FROM_ALPHA)
        self.mask_from_alpha_cb.setToolTip(i18n.tip("MASK_FROM_ALPHA"))
        opt_row1.addWidget(self.mask_from_alpha_cb)
        self.no_image_cb = QCheckBox(i18n.NO_IMAGE)
        self.no_image_cb.setToolTip(i18n.tip("NO_IMAGE"))
        opt_row1.addWidget(self.no_image_cb)
        self.no_transform_cb = QCheckBox(i18n.NO_TRANSFORM)
        self.no_transform_cb.setToolTip(i18n.tip("NO_TRANSFORM"))
        opt_row1.addWidget(self.no_transform_cb)
        opt_row1.addStretch()
        opt_inner.addLayout(opt_row1)
        opt_row2 = QHBoxLayout()
        self.duplicate_cb = QCheckBox(i18n.DUPLICATE)
        self.duplicate_cb.setToolTip(i18n.tip("DUPLICATE"))
        opt_row2.addWidget(self.duplicate_cb)
        self.invert_masks_cb = QCheckBox(i18n.INVERT_MASKS)
        self.invert_masks_cb.setToolTip(i18n.tip("INVERT_MASKS"))
        opt_row2.addWidget(self.invert_masks_cb)
        opt_row2.addStretch()
        opt_inner.addLayout(opt_row2)
        opt_w = QWidget()
        opt_w.setLayout(opt_inner)
        opt_group.content_layout.addWidget(opt_w)

        left_layout.addLayout(form)
        mask_dir_label = QLabel(i18n.MASK_DIR)
        mask_dir_label.setToolTip(i18n.tip("MASK_DIR_CUBEMAP"))
        left_layout.addWidget(mask_dir_label)
        left_layout.addWidget(self.mask_browse)
        left_layout.addWidget(opt_group)

        # Metashape前処理（折りたたみ）
        preprocess = CollapsibleSection(i18n.METASHAPE_PREPROCESS, expanded=False)
        pp_form = QFormLayout()

        self.preprocess_cb = QCheckBox(i18n.t("PREPROCESS_RUN_LABEL"))
        self.preprocess_cb.setToolTip(i18n.tip("PREPROCESS_CB"))
        self.preprocess_cb.setChecked(True)
        self.preprocess_cb.toggled.connect(self._on_preprocess_toggle)
        pp_form.addRow("", self.preprocess_cb)

        self.ms_images_browse = BrowseWidget(mode="dir")
        self.ms_images_browse.setToolTip(i18n.tip("MS_IMAGES"))
        add_tooltip_row(pp_form, i18n.t("MS_IMAGES_LABEL"), self.ms_images_browse, i18n.tip("MS_IMAGES"))

        self.ms_xml_browse = BrowseWidget(mode="file", filter_str="XML (*.xml);;すべて (*.*)")
        self.ms_xml_browse.setToolTip(i18n.tip("MS_XML"))
        add_tooltip_row(pp_form, i18n.METASHAPE_XML, self.ms_xml_browse, i18n.tip("MS_XML"))

        self.ms_ply_browse = BrowseWidget(mode="file", filter_str="PLY (*.ply);;すべて (*.*)")
        self.ms_ply_browse.setToolTip(i18n.tip("MS_PLY"))
        add_tooltip_row(pp_form, i18n.METASHAPE_PLY, self.ms_ply_browse, i18n.tip("MS_PLY"))

        self.ms_use_ply_cb = QCheckBox(i18n.USE_PLY)
        self.ms_use_ply_cb.setToolTip(i18n.tip("USE_PLY"))
        self.ms_use_ply_cb.toggled.connect(self._on_ms_ply_toggle)
        pp_form.addRow("", self.ms_use_ply_cb)

        self.ms_scale_edit = QLineEdit("1.0")
        self.ms_scale_edit.setToolTip(i18n.tip("SCALE_FACTOR"))
        add_tooltip_row(pp_form, i18n.SCALE_FACTOR, self.ms_scale_edit, i18n.tip("SCALE_FACTOR"))

        self.ms_no_fix_rot_cb = QCheckBox(i18n.NO_FIX_ROTATION)
        self.ms_no_fix_rot_cb.setToolTip(i18n.tip("NO_FIX_ROTATION"))
        pp_form.addRow("", self.ms_no_fix_rot_cb)

        preprocess.content_layout.addLayout(pp_form)
        left_layout.addWidget(preprocess)
        self._pp_widgets = [
            self.ms_images_browse, self.ms_xml_browse, self.ms_ply_browse,
            self.ms_use_ply_cb, self.ms_scale_edit, self.ms_no_fix_rot_cb,
        ]

        # 高度な出力設定（折りたたみ）
        adv_output = CollapsibleSection(i18n.t("ADVANCED_OUTPUT_SECTION"), expanded=False)
        adv_form = QFormLayout()
        adv_form.setSpacing(6)

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
            self.output_path_label.setText("-")
            return
        p = Path(path)
        self.output_path_label.setText(str(self._output_dir()))
        self.mask_browse.set_text(str(p / "masks"))
        self.ms_images_browse.set_text(str(p / "images"))
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
        pid = self._profile_id()
        if pid == _PROFILE_CUSTOM:
            if self.no_transform_cb.isChecked() and self.ms_use_ply_cb.isChecked():
                return _PROFILE_LICHTFELD
            return _PROFILE_POSTSHOT
        return pid

    def _on_profile_changed(self, _index: int) -> None:
        p = self._profile_id()
        if p == _PROFILE_LICHTFELD:
            self.no_transform_cb.setChecked(True)
            self.ms_use_ply_cb.setChecked(True)
            self.no_transform_cb.setEnabled(False)
            self.ms_use_ply_cb.setEnabled(False)
            self.profile_hint.setText("")
            self.profile_hint.setVisible(False)
        elif p == _PROFILE_BRUSH:
            self.no_transform_cb.setChecked(False)
            self.ms_use_ply_cb.setChecked(False)
            self.no_transform_cb.setEnabled(False)
            self.ms_use_ply_cb.setEnabled(False)
            self.profile_hint.setText("")
            self.profile_hint.setVisible(False)
        elif p == _PROFILE_POSTSHOT:
            self.no_transform_cb.setChecked(False)
            self.ms_use_ply_cb.setChecked(False)
            self.no_transform_cb.setEnabled(False)
            self.ms_use_ply_cb.setEnabled(False)
            self.profile_hint.setText("")
            self.profile_hint.setVisible(False)
        else:
            self.no_transform_cb.setEnabled(True)
            self.ms_use_ply_cb.setEnabled(True)
            self.profile_hint.setText("カスタム: 手動設定")
            self.profile_hint.setVisible(True)
        self._on_ms_ply_toggle()

    def _on_preprocess_toggle(self, enabled: bool) -> None:
        for w in self._pp_widgets:
            w.setEnabled(enabled)
        if not enabled:
            self.ms_use_ply_cb.setEnabled(False)
        elif self._profile_id() in (_PROFILE_POSTSHOT, _PROFILE_BRUSH, _PROFILE_LICHTFELD):
            self.ms_use_ply_cb.setEnabled(False)
        self._on_ms_ply_toggle()

    def _on_ms_ply_toggle(self, *_args) -> None:
        enabled = self.preprocess_cb.isChecked() and self.ms_use_ply_cb.isChecked()
        self.ms_ply_browse.setEnabled(enabled)

    # -- ビュー --

    def _on_views_changed(self) -> None:
        self._update_output_count()
        self._render_preview()

    def _render_preview(self) -> None:
        try:
            views = self.view_config.collect_views(include_disabled=True)
        except Exception:
            views = []
        self.preview.render(views, self.mask_browse.text())

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

        images = self.ms_images_browse.text()
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
        if self.ms_use_ply_cb.isChecked():
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
        json_name = self.json_name_edit.text().strip() or "transforms.json"

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
            "--json", json_name,
            "--fov", "90",
            "--output_scale", f"{scale:g}",
            "--views-json", str(views_json),
        ]

        mask_dir = self.mask_browse.text()
        if mask_dir and Path(mask_dir).is_dir():
            cmd.extend(["--mask_dir", mask_dir])
        if self.mask_from_alpha_cb.isChecked():
            cmd.append("--mask_from_alpha")
        if self.no_image_cb.isChecked():
            cmd.append("--no_image")
        if self.no_transform_cb.isChecked():
            cmd.append("--no_transform")
        if self._profile_id() == _PROFILE_BRUSH:
            cmd.append("--brush")
        if self.duplicate_cb.isChecked():
            cmd.append("--duplicate")
        if self.invert_masks_cb.isChecked():
            cmd.append("--invert_masks")

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
        json_name = self.json_name_edit.text().strip() or "transforms.json"
        colmap_dir = output / "colmap"

        cmd = [
            sys.executable, "-u", str(script),
            str(output), str(colmap_dir),
            "--json", json_name,
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
        payload = {
            "fov": 90.0,
            "views": [
                {"name": v["name"], "yaw": float(v["yaw"]), "pitch": float(v["pitch"]), "enabled": bool(v["enabled"])}
                for v in views
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _output_dir(self) -> Path:
        if not self.scene_dir:
            return Path("output")
        return Path(self.scene_dir) / "output"

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
        if profile == _PROFILE_LICHTFELD and self.preprocess_cb.isChecked() and self.ms_use_ply_cb.isChecked():
            return
        if profile == _PROFILE_LICHTFELD:
            raise ValueError("LichtFeldプロファイルにはpointcloud.plyが必要です。前処理でPLYを有効にしてください。")
        raise ValueError("PostshotプロファイルにはMetashape PLY（metashape.plyまたはsparse.ply）が必要です。")

    def _resolve_ply_source(self) -> Path | None:
        scene = Path(self.scene_dir) if self.scene_dir else Path(".")
        profile = self._effective_profile()
        if profile == _PROFILE_LICHTFELD:
            candidates = [scene / "pointcloud.ply"]
        else:
            candidates = [scene / "metashape.ply", scene / "sparse.ply"]

        for c in candidates:
            if c.is_file():
                return c
        plys = sorted([p for p in scene.glob("*.ply") if p.is_file()], key=lambda x: x.name.lower())
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
        scene = Path(self.scene_dir)
        output = self._output_dir()
        output.mkdir(parents=True, exist_ok=True)

        source = self._resolve_ply_source()
        if source is None:
            return
        dest = output / source.name
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)

        transforms = output / "transforms.json"
        if transforms.is_file():
            data = json.loads(transforms.read_text(encoding="utf-8"))
            data["ply_file_path"] = dest.name
            transforms.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # -- プログレス --

    def on_line(self, line: str) -> tuple[int, int] | None:
        m = _CONVERT_RE.match(line)
        if m:
            self._converted_total = int(m.group(1))
            self._processed = 0
            return 0, self._converted_total

        if line.startswith("Processing:") and self._converted_total > 0:
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
        for name in ["metashape.ply", "sparse.ply", "pointcloud.ply"]:
            c = scene_dir / name
            if c.is_file():
                return str(c)
        plys = sorted([p for p in scene_dir.glob("*.ply") if p.is_file()], key=lambda x: x.name.lower())
        return str(plys[0]) if plys else ""
