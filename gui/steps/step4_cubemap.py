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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.collapsible_section import CollapsibleSection
from gui.cubemap.view_config import ViewConfigWidget, _BLOCK_ENABLED_VIEWS, _WARN_ENABLED_VIEWS
from gui.cubemap.preview_renderer import PreviewWidget
from gui.steps.base_step import BaseStepWidget

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
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # 左パネル: 設定 (スクロール可能)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(8)

        form = QFormLayout()

        self.output_browse = BrowseWidget(mode="dir")
        form.addRow(i18n.OUTPUT_DIR, self.output_browse)

        self.json_name_edit = QLineEdit("transforms.json")
        form.addRow(i18n.JSON_NAME, self.json_name_edit)

        self.profile_combo = QComboBox()
        self.profile_combo.addItem(i18n.PROFILE_POSTSHOT, _PROFILE_POSTSHOT)
        self.profile_combo.addItem(i18n.PROFILE_BRUSH, _PROFILE_BRUSH)
        self.profile_combo.addItem(i18n.PROFILE_LICHTFELD, _PROFILE_LICHTFELD)
        self.profile_combo.addItem(i18n.PROFILE_CUSTOM, _PROFILE_CUSTOM)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        form.addRow(i18n.TARGET_PROFILE, self.profile_combo)

        self.profile_hint = QLabel("")
        self.profile_hint.setWordWrap(True)
        form.addRow("", self.profile_hint)

        self.scale_combo = QComboBox()
        self.scale_combo.addItem("Half (0.5x)", 0.5)
        self.scale_combo.addItem("Full (1.0x)", 1.0)
        form.addRow(i18n.OUTPUT_SCALE, self.scale_combo)

        self.mask_browse = BrowseWidget(mode="dir")
        form.addRow(i18n.MASK_DIR, self.mask_browse)

        # オプション (GroupBox)
        opt_group = CollapsibleSection("変換オプション", expanded=False)
        opt_inner = QVBoxLayout()
        opt_row1 = QHBoxLayout()
        self.mask_from_alpha_cb = QCheckBox(i18n.MASK_FROM_ALPHA)
        opt_row1.addWidget(self.mask_from_alpha_cb)
        self.no_image_cb = QCheckBox(i18n.NO_IMAGE)
        opt_row1.addWidget(self.no_image_cb)
        self.no_transform_cb = QCheckBox(i18n.NO_TRANSFORM)
        opt_row1.addWidget(self.no_transform_cb)
        opt_row1.addStretch()
        opt_inner.addLayout(opt_row1)
        opt_row2 = QHBoxLayout()
        self.duplicate_cb = QCheckBox(i18n.DUPLICATE)
        opt_row2.addWidget(self.duplicate_cb)
        self.invert_masks_cb = QCheckBox(i18n.INVERT_MASKS)
        opt_row2.addWidget(self.invert_masks_cb)
        opt_row2.addStretch()
        opt_inner.addLayout(opt_row2)
        opt_w = QWidget()
        opt_w.setLayout(opt_inner)
        opt_group.content_layout.addWidget(opt_w)

        left_layout.addLayout(form)
        left_layout.addWidget(opt_group)

        # Metashape前処理（折りたたみ）
        preprocess = CollapsibleSection(i18n.METASHAPE_PREPROCESS, expanded=False)
        pp_form = QFormLayout()

        self.preprocess_cb = QCheckBox("Metashape前処理を実行")
        self.preprocess_cb.setChecked(True)
        self.preprocess_cb.toggled.connect(self._on_preprocess_toggle)
        pp_form.addRow("", self.preprocess_cb)

        self.ms_images_browse = BrowseWidget(mode="dir")
        pp_form.addRow("画像フォルダ", self.ms_images_browse)

        self.ms_xml_browse = BrowseWidget(mode="file", filter_str="XML (*.xml);;すべて (*.*)")
        pp_form.addRow(i18n.METASHAPE_XML, self.ms_xml_browse)

        self.ms_ply_browse = BrowseWidget(mode="file", filter_str="PLY (*.ply);;すべて (*.*)")
        pp_form.addRow(i18n.METASHAPE_PLY, self.ms_ply_browse)

        self.ms_use_ply_cb = QCheckBox(i18n.USE_PLY)
        self.ms_use_ply_cb.toggled.connect(self._on_ms_ply_toggle)
        pp_form.addRow("", self.ms_use_ply_cb)

        self.ms_scale_edit = QLineEdit("1.0")
        pp_form.addRow(i18n.SCALE_FACTOR, self.ms_scale_edit)

        self.ms_no_fix_rot_cb = QCheckBox(i18n.NO_FIX_ROTATION)
        pp_form.addRow("", self.ms_no_fix_rot_cb)

        preprocess.content_layout.addLayout(pp_form)
        left_layout.addWidget(preprocess)
        self._pp_widgets = [
            self.ms_images_browse, self.ms_xml_browse, self.ms_ply_browse,
            self.ms_use_ply_cb, self.ms_scale_edit, self.ms_no_fix_rot_cb,
        ]

        # ビュー設定
        self.view_config = ViewConfigWidget()
        self.view_config.views_changed.connect(self._on_views_changed)
        left_layout.addWidget(self.view_config)

        # 出力見積もり
        self.estimate_label = QLabel("出力画像数推定: -")
        self.estimate_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        left_layout.addWidget(self.estimate_label)
        left_layout.addStretch()

        # 右パネル: プレビュー
        self.preview = PreviewWidget()
        self.preview.mask_slider.valueChanged.connect(lambda _: self._render_preview())
        self.preview.mask_edit.textChanged.connect(lambda _: self._render_preview())
        self.preview.sample_edit.textChanged.connect(lambda _: self._render_preview())

        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([560, 800])
        layout.addWidget(splitter)

        self._on_profile_changed(0)

    # -- シーンディレクトリ --

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        if not path:
            return
        p = Path(path)
        self.output_browse.set_text(str(p / "cubic"))
        self.mask_browse.set_text(str(p / "masks"))
        self.ms_images_browse.set_text(str(p / "images"))
        self.ms_xml_browse.set_text(str(self._guess_xml(p)))
        self.ms_ply_browse.set_text(self._guess_ply(p))
        self.preview.set_scene_dir(path)
        self._update_estimate()
        self._render_preview()

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
            self.profile_hint.setText("LichtFeld: --no_transform ON, PLY ON")
        elif p == _PROFILE_BRUSH:
            self.no_transform_cb.setChecked(False)
            self.ms_use_ply_cb.setChecked(False)
            self.no_transform_cb.setEnabled(False)
            self.ms_use_ply_cb.setEnabled(False)
            self.profile_hint.setText("Brush: --brush ON, PLY OFF")
        elif p == _PROFILE_POSTSHOT:
            self.no_transform_cb.setChecked(False)
            self.ms_use_ply_cb.setChecked(False)
            self.no_transform_cb.setEnabled(False)
            self.ms_use_ply_cb.setEnabled(False)
            self.profile_hint.setText("Postshot: --no_transform OFF, PLY OFF")
        else:
            self.no_transform_cb.setEnabled(True)
            self.ms_use_ply_cb.setEnabled(True)
            self.profile_hint.setText("カスタム: 手動設定")
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
        self._update_estimate()
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

    def _update_estimate(self) -> None:
        try:
            views = self.view_config.collect_views(include_disabled=True)
        except Exception:
            self.estimate_label.setText("出力画像数推定: -")
            return
        enabled = sum(1 for v in views if v["enabled"])
        sources = self._count_input_images()
        total = sources * enabled
        warn = ""
        if enabled > _BLOCK_ENABLED_VIEWS:
            warn = " [超過]"
        elif enabled > _WARN_ENABLED_VIEWS:
            warn = " [多い]"
        self.estimate_label.setText(f"出力画像数推定: {total} ({sources} x {enabled}){warn}")

    # -- コマンド構築 --

    def build_commands(self) -> list[tuple[str, list[str]]]:
        self._validate_bundle()

        steps = []
        if self.preprocess_cb.isChecked():
            steps.append(("metashape", self._build_preprocess_cmd()))
        steps.append(("cubemap", self._build_cubemap_cmd()))
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

        output = Path(self.output_browse.text() or str(scene / "cubic"))
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
        output = Path(self.output_browse.text() or str(scene / "cubic"))
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
