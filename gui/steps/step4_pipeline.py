"""Step 4 pipeline navigation and readiness state."""

from __future__ import annotations

from pathlib import Path

from gui import i18n
from gui.steps.step4_contracts import (
    _PIPELINE_STAGE_CONVERSION,
    _PIPELINE_STAGE_SFM,
    _PIPELINE_STATUS_OFF,
    _PIPELINE_STATUS_READY,
    _PIPELINE_STATUS_WARNING,
)


class Step4PipelineMixin:
    def pipeline_stage(self) -> str:
        current = self.settings_tabs.currentIndex()
        if current == self.output_tab_index:
            return _PIPELINE_STAGE_CONVERSION
        if current == self.details_tab_index:
            return ""
        return _PIPELINE_STAGE_SFM

    def activate_pipeline_stage(self, stage: str) -> None:
        target = self.output_tab_index if stage == _PIPELINE_STAGE_CONVERSION else self.input_tab_index
        if self._settings_tab_available(target):
            self.settings_tabs.setCurrentIndex(target)

    def pipeline_stage_intent(self, stage: str) -> bool:
        if stage == _PIPELINE_STAGE_SFM:
            return self._sfm_route_backend().sfm_intent(self)
        if stage == _PIPELINE_STAGE_CONVERSION:
            return self._sfm_route_backend().conversion_intent(self)
        return False

    def pipeline_stage_intent_enabled(self, stage: str) -> bool:
        return stage in {_PIPELINE_STAGE_SFM, _PIPELINE_STAGE_CONVERSION}

    def pipeline_stage_intent_toggle_enabled(self, stage: str) -> bool:
        if stage == _PIPELINE_STAGE_SFM:
            return self._sfm_route_backend().sfm_intent_toggle_enabled(self)
        return self.pipeline_stage_intent_enabled(stage)

    def _pipeline_stage_runs_in_app(self, stage: str) -> bool:
        if stage == _PIPELINE_STAGE_SFM:
            return self._sfm_route_backend().sfm_runs_in_app(self)
        return self.pipeline_stage_intent(stage)

    def toggle_pipeline_stage_intent(self, stage: str) -> None:
        self.set_pipeline_stage_intent(stage, not self.pipeline_stage_intent(stage))

    def pipeline_stage_toggle_blocked_notice(self, stage: str) -> str:
        if stage == _PIPELINE_STAGE_SFM and self._is_metashape_method():
            key = (
                "STEP4_PIPELINE_NOTICE_METASHAPE_INPUT_LOCKED_ON"
                if self.pipeline_stage_intent(stage)
                else "STEP4_PIPELINE_NOTICE_METASHAPE_INPUT_LOCKED_OFF"
            )
            return i18n.t(key)
        return ""

    def set_pipeline_stage_intent(self, stage: str, enabled: bool) -> None:
        enabled = bool(enabled)
        if stage == _PIPELINE_STAGE_SFM:
            self._sfm_route_backend().set_sfm_intent(self, enabled)
        elif stage == _PIPELINE_STAGE_CONVERSION:
            self._sfm_route_backend().set_conversion_intent(self, enabled)
        self._sync_settings_tabs()
        self._update_path_labels()
        self._update_training_paths()
        self._update_output_count()
        self.primary_action_state_changed.emit()

    def take_pipeline_notice(self) -> str:
        text = self._pipeline_notice_text
        self._pipeline_notice_text = ""
        return text

    def _set_pipeline_notice(self, key: str) -> None:
        if key and not self._syncing_project_settings:
            self._pipeline_notice_text = i18n.t(key)

    def _set_colmap_stage_intents(self, *, run_sfm: bool, run_conversion: bool, notice_key: str = "") -> None:
        self._colmap_sfm_intent = bool(run_sfm)
        self._conversion_intent = bool(run_conversion)
        self._set_pipeline_notice(notice_key)
        self._sync_colmap_sfm_controls()

    def _set_spheresfm_stage_intents(self, *, run_sfm: bool, run_conversion: bool) -> None:
        self._spheresfm_sfm_intent = bool(run_sfm)
        self._spheresfm_conversion_intent = bool(run_conversion)

    def pipeline_nav_items(self) -> list[dict[str, object]]:
        items: list[tuple[str, str, tuple[str, str, str]]] = [
            (_PIPELINE_STAGE_SFM, i18n.t("STEP4_PIPELINE_SFM"), self._pipeline_sfm_status()),
            (
                _PIPELINE_STAGE_CONVERSION,
                i18n.t("STEP4_PIPELINE_CONVERSION"),
                self._pipeline_conversion_status(),
            ),
        ]
        result: list[dict[str, object]] = []
        for stage, label, (status, symbol, detail) in items:
            status_text = i18n.t(f"STEP4_PIPELINE_STATUS_{status.upper()}")
            status_tooltip = f"{label}: {status_text}\n{detail}"
            if status == _PIPELINE_STATUS_WARNING:
                status_tooltip = self._append_tooltip_note(
                    status_tooltip,
                    i18n.t("STEP4_PIPELINE_STATUS_WARNING_CLICK"),
                )
            intent = self.pipeline_stage_intent(stage)
            intent_enabled = self.pipeline_stage_intent_enabled(stage)
            intent_toggle_enabled = self.pipeline_stage_intent_toggle_enabled(stage)
            blocked_notice = self.pipeline_stage_toggle_blocked_notice(stage)
            intent_key = "STEP4_PIPELINE_INTENT_ON" if intent else "STEP4_PIPELINE_INTENT_OFF"
            if stage == _PIPELINE_STAGE_SFM and self._is_metashape_method():
                intent_tooltip = i18n.t("STEP4_PIPELINE_INTENT_METASHAPE_INPUT")
                intent_symbol = "●" if intent else "○"
                intent_checked = intent
            elif intent and not intent_enabled:
                intent_tooltip = i18n.t("STEP4_PIPELINE_INTENT_LOCKED_ON").format(stage=label)
                intent_symbol = "●" if intent else "○"
                intent_checked = intent
            elif not intent_enabled:
                intent_tooltip = i18n.t("STEP4_PIPELINE_INTENT_DISABLED").format(stage=label)
                intent_symbol = "●" if intent else "○"
                intent_checked = intent
            else:
                intent_tooltip = i18n.t(intent_key).format(stage=label)
                if self._is_metashape_method() and stage == _PIPELINE_STAGE_CONVERSION:
                    intent_tooltip = self._append_tooltip_note(
                        intent_tooltip,
                        i18n.t(
                            "STEP4_PIPELINE_METASHAPE_CUBE_DISABLES_INPUT"
                            if intent
                            else "STEP4_PIPELINE_METASHAPE_CUBE_ENABLES_INPUT"
                        ),
                    )
                elif self._is_colmap_method() and stage == _PIPELINE_STAGE_SFM and not intent:
                    intent_tooltip = self._append_tooltip_note(
                        intent_tooltip,
                        i18n.t("STEP4_PIPELINE_COLMAP_SFM_ENABLES_CUBE"),
                    )
                elif (
                    self._is_colmap_method()
                    and stage == _PIPELINE_STAGE_CONVERSION
                    and intent
                    and self.pipeline_stage_intent(_PIPELINE_STAGE_SFM)
                ):
                    intent_tooltip = self._append_tooltip_note(
                        intent_tooltip,
                        i18n.t("STEP4_PIPELINE_COLMAP_CUBE_DISABLES_SFM"),
                    )
                intent_symbol = "●" if intent else "○"
                intent_checked = intent
            result.append(
                {
                    "stage": stage,
                    "label": label,
                    "status": status,
                    "status_symbol": symbol,
                    "status_tooltip": status_tooltip,
                    "intent_checked": intent_checked,
                    "intent_enabled": intent_enabled,
                    "intent_toggle_enabled": intent_toggle_enabled,
                    "toggle_blocked_notice": blocked_notice,
                    "intent_symbol": intent_symbol,
                    "intent_tooltip": intent_tooltip,
                    "row_tooltip": i18n.t(
                        "STEP4_PIPELINE_ROW_TOGGLE_TOOLTIP"
                        if intent_toggle_enabled
                        else "STEP4_PIPELINE_ROW_TOOLTIP"
                    ).format(
                        stage=label,
                        status=status_text,
                        detail=detail,
                    ),
                    "current_tab_tooltip": i18n.t("STEP4_PIPELINE_CURRENT_TAB").format(stage=label),
                }
            )
        return result

    def _pipeline_sfm_status(self) -> tuple[str, str, str]:
        if not self.pipeline_stage_intent(_PIPELINE_STAGE_SFM):
            if self._is_metashape_method():
                return (_PIPELINE_STATUS_OFF, "-", i18n.t("STEP4_PIPELINE_DETAIL_METASHAPE_OFF"))
            if self._is_colmap_method():
                return (_PIPELINE_STATUS_OFF, "-", i18n.t("STEP4_PIPELINE_DETAIL_COLMAP_OFF"))
            return (_PIPELINE_STATUS_OFF, "-", i18n.t("STEP4_PIPELINE_DETAIL_SPHERESFM_OFF"))
        if not self.scene_dir:
            return (_PIPELINE_STATUS_WARNING, "!", i18n.t("STEP4_PIPELINE_DETAIL_SCENE_REQUIRED"))
        if self._is_metashape_method():
            missing = self._metashape_input_missing_detail()
            if missing is not None:
                return (_PIPELINE_STATUS_WARNING, "!", missing)
            return (_PIPELINE_STATUS_READY, "✓", i18n.t("STEP4_PIPELINE_DETAIL_METASHAPE_READY"))
        if self._is_colmap_method():
            return (_PIPELINE_STATUS_READY, "✓", i18n.t("STEP4_PIPELINE_DETAIL_COLMAP_RUNS"))
        return (_PIPELINE_STATUS_READY, "✓", i18n.t("STEP4_PIPELINE_DETAIL_SPHERESFM_RUNS"))

    def _pipeline_conversion_status(self) -> tuple[str, str, str]:
        if not self.pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION):
            return (_PIPELINE_STATUS_OFF, "-", i18n.t("STEP4_PIPELINE_DETAIL_CONVERSION_OFF"))
        if not self.scene_dir:
            return (_PIPELINE_STATUS_WARNING, "!", i18n.t("STEP4_PIPELINE_DETAIL_SCENE_REQUIRED"))
        if self._is_spheresfm_method() and self._spheresfm_runs_sfm():
            return (_PIPELINE_STATUS_READY, "✓", i18n.t("STEP4_PIPELINE_DETAIL_CONVERSION_AFTER_SFM"))
        if self._is_spheresfm_method() and self._find_spheresfm_sparse_model() is None:
            return (_PIPELINE_STATUS_WARNING, "!", i18n.t("STEP4_PIPELINE_DETAIL_SPHERESFM_NEEDS_SPARSE"))
        if self._is_metashape_method():
            missing = self._metashape_input_missing_detail()
            if missing is not None:
                return (_PIPELINE_STATUS_WARNING, "!", missing)
        return (_PIPELINE_STATUS_READY, "✓", i18n.t("STEP4_PIPELINE_DETAIL_CONVERSION_RUNS"))

    def _metashape_input_missing_detail(self) -> str | None:
        if not self.scene_dir:
            return i18n.t("STEP4_PIPELINE_DETAIL_SCENE_REQUIRED")
        if not (getattr(self, "_syncing_scene_dir", False) or getattr(self, "_syncing_project_settings", False)):
            self._refresh_metashape_auto_inputs_if_empty()
        self._update_metashape_input_hint()
        xml_text = self.ms_xml_browse.text().strip()
        if not xml_text:
            return i18n.t("STEP4_PIPELINE_DETAIL_METASHAPE_NEEDS_XML")
        xml = Path(xml_text)
        if self._metashape_input_output_path_issue(xml):
            return i18n.t("STEP4_PIPELINE_DETAIL_METASHAPE_INPUT_IN_OUTPUT")
        if not xml.is_file():
            return i18n.t("STEP4_PIPELINE_DETAIL_METASHAPE_NEEDS_XML")
        if self._preprocess_uses_ply():
            ply_text = self.ms_ply_browse.text().strip()
            if not ply_text:
                return i18n.t("STEP4_PIPELINE_DETAIL_METASHAPE_NEEDS_PLY")
            ply = Path(ply_text)
            if self._metashape_input_output_path_issue(ply):
                return i18n.t("STEP4_PIPELINE_DETAIL_METASHAPE_INPUT_IN_OUTPUT")
            if not ply.is_file():
                return i18n.t("STEP4_PIPELINE_DETAIL_METASHAPE_NEEDS_PLY")
        return None
