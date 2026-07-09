"""Step 4 activation, primary-action, and scene preview helpers."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer

from core.metashape_preview_targets import build_metashape_preview_targets
from core.projection_contract import PROJECTION_EQUIRECTANGULAR, PROJECTION_UNKNOWN
from core.scene_inventory import build_fast_scene_inventory, build_scene_inventory
from core.scene_layout import scene_asset_metadata_path
from gui import i18n
from gui.steps.step4_contracts import (
    _PIPELINE_STAGE_CONVERSION,
    _PIPELINE_STAGE_SFM,
    _PIPELINE_STATUS_WARNING,
)


class Step4ActivationMixin:
    def open_scene_preview(self) -> None:
        self._open_scene_preview()

    def _open_scene_preview(self) -> None:
        from gui.scene_preview.window import ScenePreviewWindow

        scene = Path(self.scene_dir) if self.scene_dir else None
        was_visible = bool(self._scene_preview_window is not None and self._scene_preview_window.isVisible())
        if self._scene_preview_window is None:
            self._scene_preview_window = ScenePreviewWindow(scene_dir=None, parent=self)
        else:
            was_visible = self._scene_preview_window.isVisible()
        self._scene_preview_window.set_scene_dir(scene, refresh=False)
        self._scene_preview_window.show()
        self._scene_preview_window.raise_()
        if was_visible:
            self._scene_preview_window.activateWindow()
        self._defer_scene_preview_window_refresh()

    def _defer_scene_preview_window_refresh(self) -> None:
        self._scene_preview_window_refresh_token = getattr(self, "_scene_preview_window_refresh_token", 0) + 1
        token = self._scene_preview_window_refresh_token
        QTimer.singleShot(0, lambda: self._refresh_scene_preview_window_if_current(token))

    def _refresh_scene_preview_window_if_current(self, token: int) -> None:
        if token != getattr(self, "_scene_preview_window_refresh_token", 0):
            return
        window = self._scene_preview_window
        if window is None or not window.isVisible():
            return
        window.refresh()

    def primary_action_text(self) -> str:
        return i18n.t("RUN")

    def primary_action_tooltip(self) -> str:
        if self._apriltag_tab_selected():
            return i18n.tip("APRILTAG_TAB_PRIMARY_ACTION")
        return i18n.tip("RUN_CUBEMAP")

    def primary_action_enabled(self) -> bool:
        if self._apriltag_tab_selected():
            return False
        selected = False
        status_by_stage = {
            _PIPELINE_STAGE_SFM: self._pipeline_sfm_status(),
            _PIPELINE_STAGE_CONVERSION: self._pipeline_conversion_status(),
        }
        for stage, (status, _symbol, _detail) in status_by_stage.items():
            if not self._pipeline_stage_runs_in_app(stage):
                continue
            selected = True
            if status == _PIPELINE_STATUS_WARNING:
                return False
        return selected

    def on_activated(self) -> None:
        if self._dataset_mask_step is not None:
            self._dataset_mask_step.set_dataset_projection(self._dataset_output_projection())
        self._refresh_metashape_auto_inputs_if_empty()
        self._sync_sfm_input_paths()
        self._sync_preview_perspective_paths()
        self.preview.refresh_image_list(prefer_current=True)
        self._refresh_input_image_count()
        self._update_path_labels()
        self._update_output_count()
        self._render_preview()
        if self._dataset_mask_tab_selected() and self._dataset_mask_step is not None:
            self._dataset_mask_step.on_activated()

    def _sync_preview_perspective_paths(self) -> None:
        if not self.scene_dir:
            self._metashape_preview_action_counts = None
            self.preview.set_image_paths(None, refresh=False)
            self.preview.set_perspective_supported_paths(())
            return
        if self._is_metashape_method() and self._sync_metashape_preview_targets():
            return
        self._metashape_preview_action_counts = None
        self.preview.set_image_paths(None, refresh=False)
        try:
            inventory = build_fast_scene_inventory(Path(self.scene_dir))
            if any(
                image.projection == PROJECTION_UNKNOWN or image.size is None
                for image in inventory.images
            ):
                inventory = build_scene_inventory(Path(self.scene_dir))
        except Exception:
            self.preview.set_perspective_supported_paths(())
            return
        self.preview.set_perspective_supported_paths(
            image.path for image in inventory.images if image.projection == PROJECTION_EQUIRECTANGULAR
        )

    def _sync_metashape_preview_targets(self) -> bool:
        if not self.scene_dir:
            return False
        xml_text = self.ms_xml_browse.text().strip()
        if not xml_text:
            return False
        xml = Path(xml_text)
        if not xml.is_file():
            return False
        try:
            scene = Path(self.scene_dir)
            images = self._metashape_images_dir()
            masks = self._mask_dir()
            masks_arg = masks if masks.is_dir() else None
            cache_key = self._metashape_preview_targets_key(scene, images, masks_arg, xml)
            targets = None
            if cache_key == getattr(self, "_metashape_preview_targets_cache_key", None):
                targets = getattr(self, "_metashape_preview_targets_cache", None)
            if targets is None:
                targets = build_metashape_preview_targets(
                    scene_dir=scene,
                    images_dir=images,
                    masks_dir=masks_arg,
                    xml_path=xml,
                )
                self._metashape_preview_targets_cache_key = cache_key
                self._metashape_preview_targets_cache = targets
        except Exception:
            return False
        self._metashape_preview_action_counts = targets.action_counts
        self.preview.set_perspective_supported_paths(targets.equirect_paths)
        self.preview.set_image_paths(targets.image_paths, refresh=False)
        return True

    @staticmethod
    def _metashape_preview_targets_key(scene: Path, images: Path, masks: Path | None, xml: Path) -> tuple:
        return (
            Step4ActivationMixin._path_key(scene),
            Step4ActivationMixin._path_key(images),
            Step4ActivationMixin._path_key(masks) if masks is not None else "",
            Step4ActivationMixin._file_token(images),
            Step4ActivationMixin._file_token(masks) if masks is not None else ("", None, None),
            Step4ActivationMixin._file_token(xml),
            Step4ActivationMixin._file_token(scene_asset_metadata_path(scene)),
        )

    @staticmethod
    def _path_key(path: Path) -> str:
        try:
            return str(path.resolve(strict=False)).replace("\\", "/").casefold()
        except OSError:
            return str(path).replace("\\", "/").casefold()

    @staticmethod
    def _file_token(path: Path) -> tuple[str, int | None, int | None]:
        key = Step4ActivationMixin._path_key(path)
        try:
            stat = path.stat()
        except OSError:
            return key, None, None
        return key, int(stat.st_size), int(stat.st_mtime_ns)
