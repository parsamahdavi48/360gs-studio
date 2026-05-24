"""Step 4 activation, primary-action, and scene preview helpers."""

from __future__ import annotations

from pathlib import Path

from core.projection_contract import PROJECTION_EQUIRECTANGULAR
from core.scene_inventory import build_scene_inventory
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
        if self._scene_preview_window is None:
            self._scene_preview_window = ScenePreviewWindow(scene_dir=scene, parent=self)
        else:
            self._scene_preview_window.set_scene_dir(scene)
        self._scene_preview_window.show()
        self._scene_preview_window.raise_()
        self._scene_preview_window.activateWindow()

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
        self._refresh_metashape_auto_inputs_if_empty()
        self._sync_sfm_input_paths()
        self._sync_preview_perspective_paths()
        self.preview.refresh_image_list(prefer_current=True)
        self._refresh_input_image_count()
        self._update_path_labels()
        self._update_output_count()
        self._render_preview()

    def _sync_preview_perspective_paths(self) -> None:
        if not self.scene_dir:
            self.preview.set_perspective_supported_paths(())
            return
        try:
            inventory = build_scene_inventory(Path(self.scene_dir))
        except Exception:
            self.preview.set_perspective_supported_paths(())
            return
        self.preview.set_perspective_supported_paths(
            image.path for image in inventory.images if image.projection == PROJECTION_EQUIRECTANGULAR
        )

