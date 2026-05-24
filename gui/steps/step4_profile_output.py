"""Step 4 target profile and output-shape helpers."""

from __future__ import annotations

import math

from PySide6.QtWidgets import QComboBox

from gui import i18n
from gui.steps.step4_contracts import (
    _AXIS_BRUSH,
    _AXIS_NONE,
    _AXIS_POSTSHOT,
    _OUTPUT_SHAPE_EQUIRECT_3DGUT,
    _OUTPUT_SHAPE_PROJECTED,
    _PROFILE_BRUSH,
    _PROFILE_CUSTOM,
    _PROFILE_LICHTFELD,
    _PROFILE_POSTSHOT,
    _PROFILE_REALITYSCAN,
)
from gui.steps.step4_output_shape_selector import OutputShapeSelector


class Step4ProfileOutputMixin:
    # -- プロファイル --

    def _profile_id(self) -> str:
        return self.profile_combo.currentData() or _PROFILE_CUSTOM

    def _effective_profile(self) -> str:
        if self._profile_id() == _PROFILE_REALITYSCAN:
            return _PROFILE_REALITYSCAN
        mode = self._axis_transform_mode()
        if mode == _AXIS_NONE:
            return _PROFILE_LICHTFELD
        if mode == _AXIS_BRUSH:
            return _PROFILE_BRUSH
        return _PROFILE_POSTSHOT

    @staticmethod
    def _profile_axis_default(profile: str) -> str:
        if profile == _PROFILE_REALITYSCAN:
            return _AXIS_BRUSH
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

    def _set_combo_data(self, combo: QComboBox, value: object) -> None:
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
            if profile == _PROFILE_REALITYSCAN:
                self.export_colmap_cb.setChecked(False)
                self._set_combo_data(self.realityscan_pose_prior_combo, "exact")
                self._set_combo_data(self.realityscan_calibration_prior_combo, "exact")
                self.realityscan_include_rig_cb.setChecked(False)
        finally:
            self._syncing_profile_controls = False

    def _is_realityscan_profile(self) -> bool:
        return self._is_metashape_method() and self._profile_id() == _PROFILE_REALITYSCAN

    def _axis_transform_mode(self) -> str:
        data = self.axis_transform_combo.currentData()
        return data if data in {_AXIS_POSTSHOT, _AXIS_BRUSH, _AXIS_NONE} else _AXIS_POSTSHOT

    def _sync_realityscan_axis_display(self) -> None:
        if self._axis_brush_index < 0:
            return
        realityscan = self._profile_id() == _PROFILE_REALITYSCAN
        self.axis_transform_combo.setItemText(
            self._axis_brush_index,
            i18n.t("AXIS_TRANSFORM_REALITYSCAN_AUTO") if realityscan else i18n.t("AXIS_TRANSFORM_BRUSH"),
        )
        self.axis_transform_combo.setToolTip(
            i18n.tip("AXIS_TRANSFORM_REALITYSCAN_AUTO") if realityscan else i18n.tip("AXIS_TRANSFORM")
        )

    def _sync_realityscan_options_visibility(self) -> None:
        visible = self._is_realityscan_profile()
        self.realityscan_options_row.setVisible(visible)
        if self.realityscan_options_label is not None:
            self.realityscan_options_label.setVisible(visible)

    def _sync_profile_tooltip(self) -> None:
        tooltip_key = "TARGET_PROFILE_REALITYSCAN" if self._profile_id() == _PROFILE_REALITYSCAN else "TARGET_PROFILE"
        tooltip = i18n.tip(tooltip_key)
        self.profile_combo.setToolTip(tooltip)
        if self.profile_label is not None:
            self.profile_label.setToolTip(tooltip)

    def _spheresfm_profile_id(self) -> str:
        return self.spheresfm_profile_combo.currentData() or _PROFILE_LICHTFELD

    def _spheresfm_axis_transform_mode(self) -> str:
        data = self.spheresfm_axis_transform_combo.currentData()
        return data if data in {_AXIS_POSTSHOT, _AXIS_BRUSH, _AXIS_NONE} else _AXIS_NONE

    def _spheresfm_effective_profile(self) -> str:
        if self._uses_spheresfm_3dgut_output():
            return _PROFILE_LICHTFELD
        mode = self._spheresfm_axis_transform_mode()
        if mode == _AXIS_NONE:
            return _PROFILE_LICHTFELD
        if mode == _AXIS_BRUSH:
            return _PROFILE_BRUSH
        return _PROFILE_POSTSHOT

    def _uses_lichtfeld_final_correction(self) -> bool:
        return (
            self._is_metashape_method()
            and self._effective_profile() == _PROFILE_LICHTFELD
            and not self._is_realityscan_profile()
        )

    def _uses_spheresfm_lichtfeld_final_correction(self) -> bool:
        return self._is_spheresfm_method() and self._spheresfm_effective_profile() == _PROFILE_LICHTFELD

    def _on_profile_changed(self, _index: int) -> None:
        p = self._profile_id()
        self._sync_profile_defaults(p)
        self.profile_hint.setText(i18n.t("PROFILE_CUSTOM_HINT") if p == _PROFILE_CUSTOM else "")
        self.profile_hint.setVisible(p == _PROFILE_CUSTOM)
        if (
            self.output_shape_combo.currentData() == _OUTPUT_SHAPE_EQUIRECT_3DGUT
            and not self._metashape_profile_allows_equirect_output()
        ):
            self._set_combo_data(self.output_shape_combo, _OUTPUT_SHAPE_PROJECTED)
        elif self.output_shape_combo.currentData() == _OUTPUT_SHAPE_EQUIRECT_3DGUT and (
            self._axis_transform_mode() != _AXIS_NONE or not self._preprocess_uses_ply()
        ):
            self._set_combo_data(self.output_shape_combo, _OUTPUT_SHAPE_PROJECTED)
        self._sync_profile_tooltip()
        self._sync_realityscan_axis_display()
        self._sync_realityscan_options_visibility()
        self._sync_ply_browse_enabled()
        self._sync_output_shape_controls()
        self._sync_yaw_per_frame_control()
        self._update_path_labels()
        self._update_output_count()

    def _on_profile_option_changed(self, *_args) -> None:
        if self._syncing_profile_controls:
            return
        current = self._profile_id()
        if current != _PROFILE_CUSTOM:
            axis_changed = self._axis_transform_mode() != self._profile_axis_default(current)
            ply_changed = self.ms_use_ply_cb.isChecked() != self._profile_use_ply_default(current)
            no_fix_changed = self.ms_no_fix_rot_cb.isChecked() != self._profile_no_fix_rotation_default(current)
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
        self._sync_output_shape_controls()

    def _on_spheresfm_profile_changed(self, _index: int) -> None:
        p = self._spheresfm_profile_id()
        if p != _PROFILE_CUSTOM:
            self._syncing_profile_controls = True
            try:
                self._set_combo_data(self.spheresfm_axis_transform_combo, self._profile_axis_default(p))
            finally:
                self._syncing_profile_controls = False
        self.spheresfm_profile_hint.setText(i18n.t("PROFILE_CUSTOM_HINT") if p == _PROFILE_CUSTOM else "")
        self.spheresfm_profile_hint.setVisible(p == _PROFILE_CUSTOM)
        if (
            self.spheresfm_output_shape_combo.currentData() == _OUTPUT_SHAPE_EQUIRECT_3DGUT
            and not self._spheresfm_profile_allows_equirect_output()
        ):
            self._set_combo_data(self.spheresfm_output_shape_combo, _OUTPUT_SHAPE_PROJECTED)
        self._sync_output_shape_controls()
        self._update_path_labels()
        self._update_output_count()

    def _on_spheresfm_profile_option_changed(self, *_args) -> None:
        if self._syncing_profile_controls:
            return
        current = self._spheresfm_profile_id()
        if current != _PROFILE_CUSTOM:
            axis_changed = self._spheresfm_axis_transform_mode() != self._profile_axis_default(current)
            if axis_changed:
                custom_idx = self.spheresfm_profile_combo.findData(_PROFILE_CUSTOM)
                if custom_idx >= 0:
                    self.spheresfm_profile_combo.setCurrentIndex(custom_idx)
        self._sync_output_shape_controls()

    # -- 出力形状 --

    def _output_shape(self) -> str:
        combo = self.spheresfm_output_shape_combo if self._is_spheresfm_method() else self.output_shape_combo
        data = combo.currentData()
        return data if data in {_OUTPUT_SHAPE_PROJECTED, _OUTPUT_SHAPE_EQUIRECT_3DGUT} else _OUTPUT_SHAPE_PROJECTED

    def _uses_direct_equirect_output(self) -> bool:
        return (
            self._is_metashape_method()
            and self._output_shape() == _OUTPUT_SHAPE_EQUIRECT_3DGUT
            and self._metashape_profile_allows_equirect_output()
            and not self._is_realityscan_profile()
        )

    def _uses_spheresfm_3dgut_output(self) -> bool:
        return (
            self._is_spheresfm_method()
            and self._output_shape() == _OUTPUT_SHAPE_EQUIRECT_3DGUT
            and self._spheresfm_profile_allows_equirect_output()
        )

    def _uses_spheresfm_projected_output(self) -> bool:
        return self._is_spheresfm_method() and self._output_shape() == _OUTPUT_SHAPE_PROJECTED

    def _on_output_shape_changed(self, *_args) -> None:
        if self._syncing_output_shape_controls:
            return
        if (
            self._is_metashape_method()
            and self._output_shape() == _OUTPUT_SHAPE_EQUIRECT_3DGUT
            and not self._metashape_profile_allows_equirect_output()
        ):
            self._set_combo_data(self.output_shape_combo, _OUTPUT_SHAPE_PROJECTED)
            return
        if (
            self._is_spheresfm_method()
            and self._output_shape() == _OUTPUT_SHAPE_EQUIRECT_3DGUT
            and not self._spheresfm_profile_allows_equirect_output()
        ):
            self._set_combo_data(self.spheresfm_output_shape_combo, _OUTPUT_SHAPE_PROJECTED)
            return
        if self._uses_direct_equirect_output():
            self._ensure_direct_equirect_defaults()
        if self._uses_spheresfm_3dgut_output():
            self._ensure_spheresfm_3dgut_defaults()
        self._sync_output_shape_controls()
        self._sync_settings_tabs()
        self._update_path_labels()
        self._update_training_paths()
        self._update_output_count()
        self._schedule_render_preview()
        self.primary_action_state_changed.emit()

    def _ensure_direct_equirect_defaults(self) -> None:
        self._syncing_profile_controls = True
        try:
            self._set_combo_data(self.axis_transform_combo, _AXIS_NONE)
            self.ms_use_ply_cb.setChecked(True)
        finally:
            self._syncing_profile_controls = False

    def _ensure_spheresfm_3dgut_defaults(self) -> None:
        self._syncing_profile_controls = True
        try:
            self._set_combo_data(self.spheresfm_axis_transform_combo, _AXIS_NONE)
        finally:
            self._syncing_profile_controls = False

    def _metashape_profile_allows_equirect_output(self) -> bool:
        return self._profile_id() == _PROFILE_LICHTFELD

    def _spheresfm_profile_allows_equirect_output(self) -> bool:
        return self._spheresfm_profile_id() == _PROFILE_LICHTFELD

    def _sync_output_shape_option_state(
        self,
        selector: OutputShapeSelector,
        *,
        equirect_enabled: bool,
    ) -> None:
        projected_idx = selector.findData(_OUTPUT_SHAPE_PROJECTED)
        if projected_idx >= 0:
            selector.setItemEnabled(projected_idx, True)
            selector.setItemToolTip(projected_idx, i18n.tip("OUTPUT_SHAPE_PROJECTED"))

        equirect_idx = selector.findData(_OUTPUT_SHAPE_EQUIRECT_3DGUT)
        if equirect_idx >= 0:
            selector.setItemEnabled(equirect_idx, equirect_enabled)
            tooltip_key = (
                "OUTPUT_SHAPE_EQUIRECT_3DGUT"
                if equirect_enabled
                else "OUTPUT_SHAPE_EQUIRECT_3DGUT_DISABLED"
            )
            selector.setItemToolTip(equirect_idx, i18n.tip(tooltip_key))

    def _sync_output_shape_controls(self) -> None:
        self._sync_output_shape_option_state(
            self.output_shape_combo,
            equirect_enabled=self._metashape_profile_allows_equirect_output(),
        )
        self._sync_output_shape_option_state(
            self.spheresfm_output_shape_combo,
            equirect_enabled=self._spheresfm_profile_allows_equirect_output(),
        )
        direct = self._uses_direct_equirect_output()
        spheresfm = self._is_spheresfm_method()
        spheresfm_runs_conversion = self._spheresfm_runs_conversion()
        spheresfm_runs_sfm = self._spheresfm_runs_sfm()
        spheresfm_3dgut = spheresfm_runs_conversion and self._uses_spheresfm_3dgut_output()
        spheresfm_projected = spheresfm_runs_conversion and self._uses_spheresfm_projected_output()
        metashape_dataset_writer = self._uses_metashape_nerf_dataset_writer()
        if direct or spheresfm_3dgut or metashape_dataset_writer:
            if self._saved_projected_export_targets is None:
                self._saved_projected_export_targets = (
                    self.export_images_cb.isChecked(),
                    self.export_masks_cb.isChecked(),
                )
            self.export_images_cb.setChecked(True)
            self.export_masks_cb.setChecked(True)
            self.export_colmap_cb.setChecked(False)
        elif self._saved_projected_export_targets is not None:
            images, masks = self._saved_projected_export_targets
            self.export_images_cb.setChecked(images)
            self.export_masks_cb.setChecked(masks)
            self._saved_projected_export_targets = None

        if spheresfm:
            self.export_colmap_cb.setChecked(False)
        if self._is_realityscan_profile():
            self.export_colmap_cb.setChecked(False)

        route_uses_view_export = not direct and (not spheresfm or spheresfm_projected)
        self.export_targets_row.setEnabled(route_uses_view_export and not metashape_dataset_writer)
        self.view_config.settings_widget.setEnabled(route_uses_view_export)
        self.output_details_section.setEnabled(route_uses_view_export)
        self.output_shape_combo.setEnabled(self._is_metashape_method() and not self._is_realityscan_profile())
        self.spheresfm_output_shape_combo.setEnabled(spheresfm_runs_conversion)
        self.spheresfm_exec_browse.setEnabled(spheresfm_runs_sfm)
        self.spheresfm_use_masks_cb.setEnabled(spheresfm_runs_sfm)
        self.spheresfm_pipeline_row.setEnabled(spheresfm_runs_sfm)
        self.spheresfm_pose_browse.setEnabled(spheresfm_runs_sfm)
        self.axis_transform_combo.setEnabled(
            self._is_metashape_method() and not direct and not self._is_realityscan_profile()
        )
        self.axis_transform_combo.setVisible(False)
        if self.axis_transform_label is not None:
            self.axis_transform_label.setVisible(False)
        self.spheresfm_profile_combo.setEnabled(spheresfm_projected)
        self.spheresfm_axis_transform_combo.setEnabled(spheresfm_projected)
        self.spheresfm_axis_transform_combo.setVisible(False)
        if self.spheresfm_axis_transform_label is not None:
            self.spheresfm_axis_transform_label.setVisible(False)
        self._sync_realityscan_options_visibility()
        self.ms_use_ply_cb.setEnabled(self._is_metashape_method() and not direct)
        self.export_colmap_cb.setEnabled(
            self._is_metashape_method() and not direct and not self._is_realityscan_profile()
        )
        self.export_colmap_cb.setVisible(False)
        self.settings_tabs.setTabEnabled(self.output_tab_index, (not spheresfm) or spheresfm_runs_conversion)

    def _preprocess_uses_ply(self) -> bool:
        return self.ms_use_ply_cb.isChecked()

    def _sync_ply_browse_enabled(self) -> None:
        self.ms_ply_browse.setEnabled(self._preprocess_uses_ply())
        self._set_metashape_ply_approved(
            self._metashape_ply_approved,
            auto_candidate=self._metashape_ply_auto_candidate,
        )
        self._update_metashape_input_hint()

    def _writes_images(self) -> bool:
        return self.export_images_cb.isChecked()

    def _writes_masks(self) -> bool:
        return self.export_masks_cb.isChecked()

    def _writes_any_view_assets(self) -> bool:
        return self._writes_images() or self._writes_masks()

