"""Step 4 SfM route state, path summary, and input-path sync helpers."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtWidgets import QLabel, QWidget

from gui import i18n
from gui.steps.sfm_route_backends import get_sfm_route_backend
from gui.steps.sfm_route_specs import normalize_sfm_route
from gui.steps.step4_contracts import (
    _COLMAP_MAPPER_GLOMAP,
    _GENERATED_POINTCLOUD_NAME,
    _METHOD_COLMAP,
    _METHOD_METASHAPE,
    _METHOD_SPHERESFM,
    _PIPELINE_STAGE_CONVERSION,
    _PIPELINE_STAGE_SFM,
    _SPHERESFM_RUN_CONVERT_ONLY,
    _SPHERESFM_RUN_FULL,
    _SPHERESFM_RUN_SFM_ONLY,
)
from gui.steps.step4_widgets import ElidedPathLabel


class Step4RouteStateMixin:
    # -- 書き出し方式 --

    def _export_method(self) -> str:
        return self._export_method_value

    def _sfm_route_backend(self):
        return get_sfm_route_backend(self._export_method())

    def _is_metashape_method(self) -> bool:
        return self._export_method() == _METHOD_METASHAPE

    def _is_colmap_method(self) -> bool:
        return self._export_method() == _METHOD_COLMAP

    def _is_spheresfm_method(self) -> bool:
        return self._export_method() == _METHOD_SPHERESFM

    def _spheresfm_runs_conversion(self) -> bool:
        if not self._is_spheresfm_method():
            return False
        return self._spheresfm_conversion_intent

    def _spheresfm_runs_sfm(self) -> bool:
        if not self._is_spheresfm_method():
            return False
        return self._spheresfm_sfm_intent

    @staticmethod
    def _normalize_spheresfm_run_scope(value: str) -> str:
        if value in {_SPHERESFM_RUN_FULL, _SPHERESFM_RUN_SFM_ONLY, _SPHERESFM_RUN_CONVERT_ONLY}:
            return value
        return _SPHERESFM_RUN_FULL

    def _set_export_method(self, method: str) -> None:
        method = normalize_sfm_route(method)
        self._export_method_value = method
        if hasattr(self, "export_method_selector"):
            self.export_method_selector.set_route(method)
        self._on_export_method_changed()

    def _on_export_method_changed(self) -> None:
        metashape = self._is_metashape_method()
        self._sync_settings_tabs(prefer_route_tab=True)
        if not metashape:
            self.export_colmap_cb.setChecked(False)
        self._sync_output_shape_controls()
        self._sync_yaw_per_frame_control()
        self._sync_colmap_sfm_controls()
        self._sync_sfm_input_paths()
        self._update_path_labels()
        self._update_training_paths()
        self._update_output_count()
        self.primary_action_state_changed.emit()

    def _sync_yaw_per_frame_control(self) -> None:
        if not hasattr(self, "yaw_per_frame_edit"):
            return
        if self._is_colmap_method() or self._is_realityscan_profile():
            if self.yaw_per_frame_edit.isEnabled():
                self._yaw_per_frame_non_colmap_value = float(self.yaw_per_frame_edit.value())
            self.yaw_per_frame_edit.setValue(0.0)
            self.yaw_per_frame_edit.setEnabled(False)
            tip_key = (
                "YAW_OFFSET_PER_FRAME_REALITYSCAN_HINT"
                if self._is_realityscan_profile()
                else "YAW_OFFSET_PER_FRAME_COLMAP_HINT"
            )
            self.yaw_per_frame_edit.setToolTip(i18n.t(tip_key))
            self.yaw_per_frame_label.setToolTip(i18n.t(tip_key))
            return

        if not self.yaw_per_frame_edit.isEnabled():
            self.yaw_per_frame_edit.setEnabled(True)
            self.yaw_per_frame_edit.setValue(self._yaw_per_frame_non_colmap_value)
        self.yaw_per_frame_edit.setToolTip(i18n.t("YAW_OFFSET_PER_FRAME_HINT"))
        self.yaw_per_frame_label.setToolTip(i18n.t("YAW_OFFSET_PER_FRAME_HINT"))

    def _sync_settings_tabs(self, *, prefer_route_tab: bool = False) -> None:
        current = self.settings_tabs.currentIndex()
        self.metashape_section.setVisible(False)
        self.sfm_input_section.setVisible(True)
        self.metashape_sfm_input_widget.setVisible(self._is_metashape_method())
        self.colmap_sfm_input_widget.setVisible(self._is_colmap_method())
        self.spheresfm_sfm_input_widget.setVisible(self._is_spheresfm_method())
        self.metashape_output_section.setVisible(self._is_metashape_method())
        self.colmap_section.setVisible(self._is_colmap_method() and self.pipeline_stage_intent(_PIPELINE_STAGE_SFM))
        self.spheresfm_section.setVisible(self._is_spheresfm_method() and self._spheresfm_runs_sfm())
        self.spheresfm_convert_section.setVisible(self._is_spheresfm_method() and self._spheresfm_runs_conversion())
        spheresfm_conversion_off = self._is_spheresfm_method() and not self._spheresfm_runs_conversion()
        conversion_enabled = not spheresfm_conversion_off
        self.settings_tabs.setTabEnabled(self.output_tab_index, conversion_enabled)
        route_index = self.input_tab_index
        if spheresfm_conversion_off and current == self.output_tab_index:
            self.settings_tabs.setCurrentIndex(route_index)
        elif prefer_route_tab:
            self.settings_tabs.setCurrentIndex(route_index)
        elif not self._settings_tab_available(self.settings_tabs.currentIndex()):
            self.settings_tabs.setCurrentIndex(route_index)

    def _settings_tab_available(self, index: int) -> bool:
        return (
            0 <= index < self.settings_tabs.count()
            and self.settings_tabs.isTabVisible(index)
            and self.settings_tabs.isTabEnabled(index)
        )

    def _update_path_labels(self) -> None:
        if not self.scene_dir:
            self._set_single_path_summary(
                self.sfm_path_summary_row,
                self.sfm_path_summary_kind,
                self.sfm_path_summary_value,
                i18n.t("STEP4_SUMMARY_INPUT"),
                "-",
            )
            self._set_single_path_summary(
                self.cubemap_path_summary_row,
                self.cubemap_path_summary_kind,
                self.cubemap_path_summary_value,
                i18n.t("STEP4_SUMMARY_OUTPUT"),
                "-",
            )
            self._set_training_path_summary("-", "-")
            return
        self._update_sfm_path_summary()
        self._update_cubemap_path_summary()
        self._update_training_path_summary()

    def _set_single_path_summary(
        self,
        row: QWidget,
        kind_label: QLabel,
        value_label: ElidedPathLabel,
        kind: str,
        value: str,
        *,
        tooltip: str = "",
    ) -> None:
        kind_label.setText(kind)
        value_label.set_full_text(value or "-")
        summary_tooltip = tooltip or f"{kind}: {value or '-'}"
        row.setToolTip(summary_tooltip)
        kind_label.setToolTip(summary_tooltip)
        value_label.setToolTip(summary_tooltip)

    def _set_training_path_summary(self, dataset: str, output: str, *, tooltip: str = "") -> None:
        self.training_dataset_summary_kind.setText(i18n.t("STEP4_SUMMARY_INPUT"))
        self.training_output_summary_kind.setText(i18n.t("STEP4_SUMMARY_OUTPUT"))
        self.training_dataset_summary_value.set_full_text(dataset or "-")
        self.training_output_summary_value.set_full_text(output or "-")
        summary_tooltip = tooltip or (
            f"{i18n.t('STEP4_SUMMARY_INPUT')}: {dataset or '-'}\n{i18n.t('STEP4_SUMMARY_OUTPUT')}: {output or '-'}"
        )
        for widget in (
            self.training_path_summary_row,
            self.training_dataset_summary_kind,
            self.training_dataset_summary_value,
            self.training_output_summary_kind,
            self.training_output_summary_value,
        ):
            widget.setToolTip(summary_tooltip)

    def _summary_path_text(self, path: Path, *, directory: bool = True) -> str:
        separator = os.sep
        if self.scene_dir:
            try:
                text = path.relative_to(Path(self.scene_dir)).as_posix()
                separator = "/"
            except ValueError:
                text = str(path)
        else:
            text = str(path)
        if directory and text != "-":
            text = text.rstrip("/\\") + separator
        return text

    def _summary_full_path(self, path: Path, *, directory: bool = True) -> str:
        text = str(path)
        if directory:
            text = text.rstrip("/\\") + os.sep
        return text

    def _update_sfm_path_summary(self) -> None:
        if self._is_metashape_method():
            parts: list[str] = []
            tooltip_parts: list[str] = []
            xml_text = self.ms_xml_browse.text().strip()
            if xml_text:
                xml = Path(xml_text)
                parts.append(f"XML {self._summary_path_text(xml, directory=False)}")
                tooltip_parts.append(f"{i18n.METASHAPE_XML}: {xml}")
            else:
                parts.append("XML")
                tooltip_parts.append(f"{i18n.METASHAPE_XML}: -")
            if self._preprocess_uses_ply():
                ply_text = self.ms_ply_browse.text().strip()
                if ply_text:
                    ply = Path(ply_text)
                    parts.append(f"PLY {self._summary_path_text(ply, directory=False)}")
                    tooltip_parts.append(f"{i18n.METASHAPE_PLY}: {ply}")
                else:
                    parts.append("PLY")
                    tooltip_parts.append(f"{i18n.METASHAPE_PLY}: -")
            self._set_single_path_summary(
                self.sfm_path_summary_row,
                self.sfm_path_summary_kind,
                self.sfm_path_summary_value,
                i18n.t("STEP4_SUMMARY_INPUT"),
                " / ".join(parts),
                tooltip="\n".join(tooltip_parts),
            )
            return

        if self._is_colmap_method():
            if self.pipeline_stage_intent(_PIPELINE_STAGE_SFM):
                target = self._colmap_rig_dir()
                kind = i18n.t("STEP4_SUMMARY_OUTPUT")
                tip_key = "OUTPUT_DIR_COLMAP_PROJECT"
            else:
                target = self._find_colmap_sparse_model() or self._colmap_sparse_dir()
                kind = i18n.t("STEP4_SUMMARY_INPUT")
                tip_key = "COLMAP_SPARSE_MODEL"
            self._set_single_path_summary(
                self.sfm_path_summary_row,
                self.sfm_path_summary_kind,
                self.sfm_path_summary_value,
                kind,
                self._summary_path_text(target),
                tooltip=f"{i18n.tip(tip_key)}\n{self._summary_full_path(target)}",
            )
            return

        if self._spheresfm_runs_sfm():
            target = self._spheresfm_project_dir()
            kind = i18n.t("STEP4_SUMMARY_WORK")
            tip_key = "OUTPUT_DIR_SPHERESFM_PROJECT"
        else:
            target = self._find_spheresfm_sparse_model() or self._spheresfm_sparse_dir()
            kind = i18n.t("STEP4_SUMMARY_INPUT")
            tip_key = "SPHERESFM_SPARSE_MODEL"
        self._set_single_path_summary(
            self.sfm_path_summary_row,
            self.sfm_path_summary_kind,
            self.sfm_path_summary_value,
            kind,
            self._summary_path_text(target),
            tooltip=f"{i18n.tip(tip_key)}\n{self._summary_full_path(target)}",
        )

    def _update_cubemap_path_summary(self) -> None:
        target = self._colmap_rig_dir() if self._is_colmap_method() else self._display_output_dir()
        if self._is_colmap_method():
            tip_key = "OUTPUT_DIR_COLMAP_PROJECT"
        elif self._uses_direct_equirect_output() or (
            self._spheresfm_runs_conversion() and self._uses_spheresfm_3dgut_output()
        ):
            tip_key = "OUTPUT_DIR_LICHTFELD_DIRECT"
        else:
            tip_key = "OUTPUT_DIR_CUBEMAP"
        value = self._summary_path_text(target)
        if not self.pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION):
            value = i18n.t("STEP4_SUMMARY_DISABLED_PATH").format(path=value)
        self._set_single_path_summary(
            self.cubemap_path_summary_row,
            self.cubemap_path_summary_kind,
            self.cubemap_path_summary_value,
            i18n.t("STEP4_SUMMARY_OUTPUT"),
            value,
            tooltip=f"{i18n.tip(tip_key)}\n{self._summary_full_path(target)}",
        )

    def _update_training_path_summary(self) -> None:
        dataset = self._training_dataset_dir()
        output = self._training_output_dir()
        self._set_training_path_summary(
            self._summary_path_text(dataset),
            self._summary_path_text(output),
            tooltip=(
                f"{i18n.t('STEP4_SUMMARY_INPUT')}: {self._summary_full_path(dataset)}\n"
                f"{i18n.t('STEP4_SUMMARY_OUTPUT')}: {self._summary_full_path(output)}"
            ),
        )

    def _on_metashape_input_path_changed(self, *_args) -> None:
        if (
            self._syncing_metashape_auto_inputs
            or getattr(self, "_syncing_scene_dir", False)
            or getattr(self, "_syncing_project_settings", False)
        ):
            return
        if self.sender() is self.ms_ply_browse and not self._syncing_metashape_auto_inputs:
            self._set_metashape_ply_approved(bool(self.ms_ply_browse.text().strip()))
        self._update_metashape_input_hint()
        self._update_path_labels()
        self._sync_preview_perspective_paths()
        self.preview.refresh_image_list(prefer_current=True)
        self._refresh_input_image_count()
        self._update_output_count()
        self._render_preview()
        self.primary_action_state_changed.emit()

    def _on_metashape_ply_text_edited(self, _text: str) -> None:
        if self._syncing_metashape_auto_inputs:
            return
        self._set_metashape_ply_approved(bool(self.ms_ply_browse.text().strip()))

    def _approve_metashape_ply(self) -> None:
        self._set_metashape_ply_approved(bool(self.ms_ply_browse.text().strip()))
        self._update_metashape_input_hint()
        self.primary_action_state_changed.emit()

    def _set_metashape_ply_approved(self, approved: bool, *, auto_candidate: bool = False) -> None:
        self._metashape_ply_approved = bool(approved)
        self._metashape_ply_auto_candidate = bool(auto_candidate and not approved)
        if not hasattr(self, "ms_ply_approve_btn"):
            return
        has_path = bool(self.ms_ply_browse.text().strip())
        self.ms_ply_approve_btn.setEnabled(has_path and self._preprocess_uses_ply())
        self.ms_ply_approve_btn.setChecked(self._metashape_ply_approved)
        if self._metashape_ply_approved:
            tooltip = i18n.t("MS_PLY_APPROVED")
        elif self._metashape_ply_auto_candidate:
            tooltip = i18n.t("MS_PLY_APPROVE_CANDIDATE")
        else:
            tooltip = i18n.t("MS_PLY_APPROVE")
        self.ms_ply_approve_btn.setToolTip(tooltip)

    def _on_colmap_sparse_path_changed(self, _path: str) -> None:
        if not self._syncing_sfm_input_paths:
            self._colmap_sparse_user_edited = True
        self._update_sfm_input_tooltips()
        self._update_training_paths()
        self._update_path_labels()
        self.primary_action_state_changed.emit()

    def _on_spheresfm_sparse_path_changed(self, _path: str) -> None:
        if not self._syncing_sfm_input_paths:
            self._spheresfm_sparse_user_edited = True
        self._update_sfm_input_tooltips()
        self._update_training_paths()
        self._update_path_labels()
        self.primary_action_state_changed.emit()

    def _sync_sfm_input_paths(self, *, force: bool = False) -> None:
        if not hasattr(self, "colmap_sparse_browse"):
            return
        self._syncing_sfm_input_paths = True
        try:
            if not self.scene_dir:
                self.colmap_sparse_browse.set_text("")
                self.spheresfm_sparse_browse.set_text("")
                return
            if force or (not self._colmap_sparse_user_edited and not self.colmap_sparse_browse.text().strip()):
                model = self._auto_find_colmap_sparse_model()
                self.colmap_sparse_browse.set_text(str(model) if model else "")
            if force or (not self._spheresfm_sparse_user_edited and not self.spheresfm_sparse_browse.text().strip()):
                model = self._auto_find_spheresfm_sparse_model()
                self.spheresfm_sparse_browse.set_text(str(model) if model else "")
        finally:
            self._syncing_sfm_input_paths = False
        self._update_sfm_input_tooltips()
        self._update_path_labels()

    def _update_sfm_input_tooltips(self) -> None:
        if not hasattr(self, "colmap_sparse_browse"):
            return
        colmap_note = self._sparse_model_input_note(self.colmap_sparse_browse.text())
        colmap_tip = self._append_tooltip_note(i18n.tip("COLMAP_SPARSE_MODEL"), colmap_note)
        self.colmap_sparse_browse.setToolTip(colmap_tip)
        self.colmap_sparse_browse.line_edit.setToolTip(colmap_tip)

        spheresfm_note = self._sparse_model_input_note(self.spheresfm_sparse_browse.text())
        spheresfm_tip = self._append_tooltip_note(i18n.tip("SPHERESFM_SPARSE_MODEL"), spheresfm_note)
        self.spheresfm_sparse_browse.setToolTip(spheresfm_tip)
        self.spheresfm_sparse_browse.line_edit.setToolTip(spheresfm_tip)

    def _sparse_model_input_note(self, text: str) -> str:
        raw = text.strip()
        if not raw:
            return i18n.t("SFM_SPARSE_MODEL_AUTO_HINT")
        path = Path(raw)
        if self._has_colmap_sparse_model(path):
            return i18n.t("SFM_SPARSE_MODEL_READY_HINT")
        return i18n.t("SFM_SPARSE_MODEL_INVALID_HINT").format(path=raw)

    def _apply_metashape_auto_inputs(self, scene: Path) -> None:
        xml = self._guess_xml(scene)
        ply = self._guess_ply(scene)
        self._syncing_metashape_auto_inputs = True
        try:
            self.ms_xml_browse.set_text(str(xml) if xml else "")
            self.ms_ply_browse.set_text(str(ply) if ply else "")
        finally:
            self._syncing_metashape_auto_inputs = False
        self._set_metashape_ply_approved(ply is not None, auto_candidate=False)

    def _refresh_metashape_auto_inputs_if_empty(self) -> None:
        if not self.scene_dir:
            return
        scene = Path(self.scene_dir)
        old_xml_candidates = self._metashape_auto_xml_candidates
        old_ply_candidates = self._metashape_auto_ply_candidates
        changed = False
        if not self.ms_xml_browse.text().strip():
            xml = self._guess_xml(scene)
            if xml:
                self._syncing_metashape_auto_inputs = True
                try:
                    self.ms_xml_browse.set_text(str(xml))
                finally:
                    self._syncing_metashape_auto_inputs = False
                changed = True
        if not self.ms_ply_browse.text().strip():
            ply = self._guess_ply(scene)
            if ply:
                self._syncing_metashape_auto_inputs = True
                try:
                    self.ms_ply_browse.set_text(str(ply))
                finally:
                    self._syncing_metashape_auto_inputs = False
                self._set_metashape_ply_approved(True, auto_candidate=False)
                changed = True
        if (
            changed
            or old_xml_candidates != self._metashape_auto_xml_candidates
            or old_ply_candidates != self._metashape_auto_ply_candidates
        ):
            self._update_metashape_input_hint()

    def _update_metashape_input_hint(self) -> None:
        if not hasattr(self, "ms_xml_browse") or not hasattr(self, "ms_ply_browse"):
            return
        if not self.scene_dir:
            self._set_metashape_input_tooltips("", "")
            return

        scene = Path(self.scene_dir)
        xml_note = ""
        ply_note = ""
        xml_text = self.ms_xml_browse.text().strip()
        if not xml_text:
            if self._metashape_auto_xml_candidates:
                xml_note = i18n.t("MS_XML_MANUAL_SELECTION_HINT").format(
                    names=self._format_candidate_names(self._metashape_auto_xml_candidates)
                )
            else:
                xml_note = i18n.t("MS_XML_MISSING_HINT")
        else:
            xml = Path(xml_text)
            issue = self._metashape_input_output_path_issue(xml)
            if issue:
                xml_note = issue
            elif not xml.is_file():
                xml_note = i18n.t("MS_XML_SELECTED_MISSING_HINT").format(path=xml_text)

        if self._preprocess_uses_ply():
            ply_text = self.ms_ply_browse.text().strip()
            if not ply_text:
                if self._metashape_auto_ply_candidates:
                    ply_note = i18n.t("MS_PLY_MANUAL_SELECTION_HINT").format(
                        names=self._format_candidate_names(self._metashape_auto_ply_candidates)
                    )
                elif (scene / _GENERATED_POINTCLOUD_NAME).is_file():
                    ply_note = i18n.t("MS_PLY_ONLY_POINTCLOUD_HINT")
                else:
                    ply_note = i18n.t("MS_PLY_MISSING_HINT")
            else:
                ply = Path(ply_text)
                issue = self._metashape_input_output_path_issue(ply)
                if issue:
                    ply_note = issue
                elif not ply.is_file():
                    ply_note = i18n.t("MS_PLY_SELECTED_MISSING_HINT").format(path=ply_text)

        self._set_metashape_input_tooltips(xml_note, ply_note)
        self._set_metashape_ply_approved(
            self._metashape_ply_approved,
            auto_candidate=self._metashape_ply_auto_candidate,
        )

    def _set_metashape_input_tooltips(self, xml_note: str, ply_note: str) -> None:
        xml_tip = self._append_tooltip_note(i18n.tip("MS_XML"), xml_note)
        ply_tip = self._append_tooltip_note(i18n.tip("MS_PLY"), ply_note)
        self.ms_xml_browse.setToolTip(xml_tip)
        self.ms_xml_browse.line_edit.setToolTip(xml_tip)
        self.ms_ply_browse.setToolTip(ply_tip)
        self.ms_ply_browse.line_edit.setToolTip(ply_tip)

    @staticmethod
    def _append_tooltip_note(base: str, note: str) -> str:
        return f"{base}\n{note}" if note else base

    def _sync_colmap_sfm_controls(self) -> None:
        enabled = self._is_colmap_method() and self._colmap_sfm_intent
        self.colmap_exec_browse.setEnabled(enabled)
        self.colmap_pipeline_row.setEnabled(enabled)
        self._on_colmap_mapper_changed()

    def _on_colmap_mapper_changed(self, *_args) -> None:
        needs_glomap = (
            self._is_colmap_method()
            and self._colmap_sfm_intent
            and self.colmap_mapper_combo.currentData() == _COLMAP_MAPPER_GLOMAP
        )
        self.glomap_exec_row_label.setVisible(needs_glomap)
        self.glomap_exec_browse.setVisible(needs_glomap)
        self.glomap_exec_browse.setEnabled(needs_glomap)
