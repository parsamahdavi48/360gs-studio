"""Step 3 mask model-control state helpers."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt

from core.mask_view_recipes import QUALITY_CHOICES
from gui import i18n
from gui.common import dialogs
from gui.steps import mask_commands as mask_command_defs
from gui.steps.mask_commands import MaskCommandContext
from gui.steps.step3_mask_settings import (
    Step3MaskSettingsState,
    normalize_sam31_merge_mode,
    split_sam_prompt_text,
)

_PERSON_BACKENDS = (
    mask_command_defs.PERSON_BACKEND_YOLO_SAM,
    mask_command_defs.PERSON_BACKEND_MASK2FORMER,
    mask_command_defs.PERSON_BACKEND_SAM31,
)
_PERSON_SAM31_PROMPT = "person"
_SKY_INFERENCE_SIZES = ("512", "768", "1008", "1024")
_SKY_INFERENCE_SIZE_DEFAULT_INDEX = 1
_SKY_SAM31_INFERENCE_SIZE = "1008"
_SKY_BACKENDS = ("mask2former", "sam31")
_SKY_SAM31_CHECKPOINT = Path("models") / "sam3.1" / "sam3.1_multiplex.pt"


class Step3MaskControlsMixin:
    def _selected_classes(self) -> list[int]:
        return [i for i, cb in enumerate(self.class_cbs) if cb.isChecked()]

    def _selected_ade_labels(self) -> list[str]:
        labels = [
            name.strip() for name, cb in zip(self.ade_class_names, self.ade_class_cbs, strict=True) if cb.isChecked()
        ]
        return [label for label in labels if label] or ["person", "sky"]

    _split_sam_prompt_text = staticmethod(split_sam_prompt_text)

    def _selected_sam_prompts(self) -> list[str]:
        prompts = [prompt for prompt, cb in self.sam_prompt_cbs if cb.isChecked()]
        prompts.extend(split_sam_prompt_text(self.sam_custom_prompt_edit.text()))
        return list(dict.fromkeys(prompts)) or [_PERSON_SAM31_PROMPT]

    def _selected_sam_subtract_prompts(self) -> list[str]:
        return list(dict.fromkeys(split_sam_prompt_text(self.sam_subtract_prompt_edit.text())))

    def _sam31_merge_mode_arg(self) -> str:
        return normalize_sam31_merge_mode(self.sam_apply_mode_combo.currentData())

    def _person_backend_arg(self) -> str:
        idx = self.person_backend_combo.currentIndex()
        return _PERSON_BACKENDS[max(0, min(idx, len(_PERSON_BACKENDS) - 1))]

    def _person_uses_sam31(self) -> bool:
        return self._person_backend_arg() == "sam31"

    def _person_uses_mask2former(self) -> bool:
        return self._person_backend_arg() == "mask2former"

    def _on_person_backend_changed(self) -> None:
        if self._person_uses_mask2former():
            self.sky_backend_combo.setCurrentIndex(0)
        elif self._person_uses_sam31():
            self.sky_backend_combo.setCurrentIndex(1)
            if self.sky_inference_size_combo.currentText() != _SKY_SAM31_INFERENCE_SIZE:
                idx = self.sky_inference_size_combo.findText(_SKY_SAM31_INFERENCE_SIZE)
                if idx >= 0:
                    self.sky_inference_size_combo.setCurrentIndex(idx)
        else:
            self.sky_backend_combo.setCurrentIndex(0)
        self._update_task_controls()

    def _yolo_expand_arg(self) -> str:
        return str(self.yolo_expand_edit.value())

    def _quality_arg(self) -> str:
        idx = max(0, min(self.yolo_level_combo.currentIndex(), len(QUALITY_CHOICES) - 1))
        return QUALITY_CHOICES[idx]

    def _bottom_enhance_args(self) -> list[str]:
        return []

    def _sky_backend_arg(self) -> str:
        idx = self.sky_backend_combo.currentIndex()
        return _SKY_BACKENDS[max(0, min(idx, len(_SKY_BACKENDS) - 1))]

    def _sam31_checkpoint_path(self) -> Path:
        return self.base_dir / _SKY_SAM31_CHECKPOINT

    def _sam31_available(self) -> bool:
        return self._sam31_checkpoint_path().is_file()

    def _on_sky_backend_changed(self) -> None:
        if (
            self._sky_backend_arg() == "sam31"
            and self.sky_inference_size_combo.currentText() != _SKY_SAM31_INFERENCE_SIZE
        ):
            idx = self.sky_inference_size_combo.findText(_SKY_SAM31_INFERENCE_SIZE)
            if idx >= 0:
                self.sky_inference_size_combo.setCurrentIndex(idx)
        self._schedule_render_mask_preview()

    def _on_sky_inference_size_changed(self) -> None:
        if (
            self._sky_backend_arg() == "sam31"
            and self.sky_inference_size_combo.currentText() != _SKY_SAM31_INFERENCE_SIZE
        ):
            idx = self.sky_inference_size_combo.findText(_SKY_SAM31_INFERENCE_SIZE)
            if idx >= 0:
                self.sky_inference_size_combo.blockSignals(True)
                try:
                    self.sky_inference_size_combo.setCurrentIndex(idx)
                finally:
                    self.sky_inference_size_combo.blockSignals(False)
        self._schedule_render_mask_preview()

    def _sky_mode_arg(self) -> str:
        idx = self.sky_mode_combo.currentIndex()
        modes = ("full", "direct", "top", "bottom", "hybrid")
        return modes[max(0, min(idx, len(modes) - 1))]

    def _sky_inference_size_arg(self) -> str:
        text = self.sky_inference_size_combo.currentText().strip()
        return text if text else str(_SKY_INFERENCE_SIZES[_SKY_INFERENCE_SIZE_DEFAULT_INDEX])

    def _sky_min_area_ratio_arg(self) -> str:
        return f"{float(self.sky_min_area_edit.value()) / 100.0:g}"

    def _sky_postprocess_args(self) -> list[str]:
        args = ["--min-area-ratio", self._sky_min_area_ratio_arg()]
        if self.sky_top_connected_cb.isChecked():
            args.append("--top-connected")
        return args

    def _update_task_controls(self) -> None:
        equirect = self._has_equirect_images()
        if not equirect and self.run_stitch_cb.isChecked():
            self.run_stitch_cb.setChecked(False)
            return
        stitch_enabled = equirect and self.run_stitch_cb.isChecked()
        overexp_enabled = self.run_overexp_cb.isChecked()
        custom_enabled = self.run_custom_cb.isChecked()

        self._update_person_backend_availability()
        model = self._person_backend_arg()
        person_sam31 = self._person_uses_sam31()
        person_mask2former = self._person_uses_mask2former()
        yolo_sam_enabled = model == "yolo_sam"
        semantic_enabled = model in {"mask2former", "sam31"}
        self.yolo_section.setEnabled(True)
        self.sky_section.setVisible(semantic_enabled)
        self.ade_class_list_section.setVisible(person_mask2former)
        self.sam_prompt_section.setVisible(person_sam31)
        self.yolo_class_list_section.setVisible(yolo_sam_enabled)
        self.yolo_level_label.setVisible(True)
        self.yolo_level_combo.setVisible(True)
        self.yolo_level_label.setEnabled(True)
        self.yolo_level_combo.setEnabled(True)
        self.yolo_expand_label.setEnabled(True)
        self.yolo_expand_edit.setEnabled(True)
        self.sam_apply_mode_label.setVisible(person_sam31)
        self.sam_apply_mode_combo.setVisible(person_sam31)
        self.yolo_bottom_settings_row.setVisible(False)
        self.yolo_bottom_settings_row.setEnabled(False)
        self.yolo_bottom_enhance_label.setEnabled(False)
        self.yolo_bottom_enhance_combo.setEnabled(False)
        self.yolo_class_list_section.setEnabled(yolo_sam_enabled)
        self.sky_inference_size_combo.setEnabled(person_mask2former)
        self.sky_inference_size_label.setEnabled(person_mask2former)
        self.sky_model_details_section.setVisible(person_mask2former)
        self.sky_postprocess_section.setVisible(semantic_enabled)
        self.sky_min_score_edit.setVisible(person_mask2former)
        if self.sky_min_score_label is not None:
            self.sky_min_score_label.setVisible(person_mask2former)
        self.run_stitch_cb.setEnabled(equirect)
        self.run_stitch_cb.setToolTip(
            i18n.tip("MASK_TASK_STITCH") if equirect else i18n.tip("MASK_TASK_STITCH_DISABLED_NORMAL")
        )
        self.stitch_boundary_width_edit.setEnabled(stitch_enabled)
        self.stitch_workers_edit.setEnabled(stitch_enabled or overexp_enabled)
        self.overexp_threshold_edit.setEnabled(overexp_enabled)
        self.overexp_dilate_edit.setEnabled(overexp_enabled)
        self.custom_mask_clear_btn.setEnabled(bool(self._custom_mask_path_text()))
        self.custom_mask_path_label.setEnabled(custom_enabled or bool(self._custom_mask_path_text()))
        self._update_sky_backend_availability()
        self._render_mask_preview()
        self._update_ready_status()

    def _update_person_backend_availability(self) -> None:
        sam_available = self._sam31_available()
        model = self.person_backend_combo.model()
        item = model.item(2) if hasattr(model, "item") else None
        if item is not None:
            item.setEnabled(True)
            item.setToolTip(i18n.tip("PERSON_MODEL_SAM31") if sam_available else i18n.tip("SAM31_CHECKPOINT_DOWNLOAD"))
            self.person_backend_combo.setItemData(
                2,
                i18n.tip("PERSON_MODEL_SAM31") if sam_available else i18n.tip("SAM31_CHECKPOINT_DOWNLOAD"),
                Qt.ToolTipRole,
            )

    def _update_sky_backend_availability(self) -> None:
        sam_available = self._sam31_available()
        model = self.sky_backend_combo.model()
        item = model.item(1) if hasattr(model, "item") else None
        if item is not None:
            item.setEnabled(True)
            item.setToolTip(i18n.tip("SKY_MODEL_SAM31") if sam_available else i18n.tip("SAM31_CHECKPOINT_DOWNLOAD"))
            self.sky_backend_combo.setItemData(
                1,
                i18n.tip("SKY_MODEL_SAM31") if sam_available else i18n.tip("SAM31_CHECKPOINT_DOWNLOAD"),
                Qt.ToolTipRole,
            )

    def _custom_mask_path_text(self) -> str:
        return self._custom_mask_path.strip()

    def _set_custom_mask_path(self, path: str | Path, *, activate: bool = True) -> None:
        self._custom_mask_path = str(path).strip()
        self.custom_mask_path_label.setText(self._custom_mask_path or i18n.t("CUSTOM_MASK_NOT_SELECTED"))
        self.custom_mask_path_label.setToolTip(self._custom_mask_path or i18n.tip("CUSTOM_MASK_FILE"))
        if activate and self._custom_mask_path and not self.run_custom_cb.isChecked():
            self.run_custom_cb.blockSignals(True)
            try:
                self.run_custom_cb.setChecked(True)
            finally:
                self.run_custom_cb.blockSignals(False)
        self._update_task_controls()

    def _clear_custom_mask_path(self) -> None:
        self._custom_mask_path = ""
        self.custom_mask_path_label.setText(i18n.t("CUSTOM_MASK_NOT_SELECTED"))
        self.custom_mask_path_label.setToolTip(i18n.tip("CUSTOM_MASK_FILE"))
        if self.run_custom_cb.isChecked():
            self.run_custom_cb.blockSignals(True)
            try:
                self.run_custom_cb.setChecked(False)
            finally:
                self.run_custom_cb.blockSignals(False)
        self._update_task_controls()

    def _browse_custom_mask(self, *, activate: bool = True) -> bool:
        start_dir = self.scene_dir or str(Path.home())
        selected, _filter = dialogs.get_open_file_name(
            self,
            i18n.t("CUSTOM_MASK_SELECT_FILE"),
            start_dir,
            i18n.t("CUSTOM_MASK_FILE_FILTER"),
        )
        if not selected:
            return False
        self._set_custom_mask_path(selected, activate=activate)
        return True

    def _on_custom_mask_toggled(self, checked: bool) -> None:
        if checked and not self._custom_mask_path_text():
            if not self._browse_custom_mask(activate=True):
                self.run_custom_cb.blockSignals(True)
                try:
                    self.run_custom_cb.setChecked(False)
                finally:
                    self.run_custom_cb.blockSignals(False)
        self._update_task_controls()

    def _on_images_dir_changed(self, path: str) -> None:
        self.mask_preview.set_images_dir(path)
        self._refresh_image_projection_map()
        self._update_preview_projection_enabled()
        self._render_mask_preview()

    def _mask_settings_state(self) -> Step3MaskSettingsState:
        return Step3MaskSettingsState(
            projection=self._projection(),
            projection_mixed=self._projection_mixed,
            tasks=tuple(self._selected_mask_tasks()),
            primary_backend=self._person_backend_arg(),
            quality=self._quality_arg(),
            yolo_level_index=self.yolo_level_combo.currentIndex(),
            yolo_level_label=self.yolo_level_combo.currentText(),
            yolo_expand=self._yolo_expand_arg(),
            yolo_classes=tuple(self._selected_classes()),
            yolo_extra_args=tuple(self._bottom_enhance_args()),
            ade_labels=tuple(self._selected_ade_labels()),
            sam_prompts=tuple(self._selected_sam_prompts()),
            sam_subtract_prompts=tuple(self._selected_sam_subtract_prompts()),
            sam31_merge_mode=self._sam31_merge_mode_arg(),
            sky_backend=self._sky_backend_arg(),
            sky_inference_size=self._sky_inference_size_arg(),
            sky_min_score=f"{float(self.sky_min_score_edit.value()):g}",
            sky_min_area_ratio=self._sky_min_area_ratio_arg(),
            sky_top_connected=self.sky_top_connected_cb.isChecked(),
            stitch_enabled=self._has_equirect_images() and self.run_stitch_cb.isChecked(),
            stitch_boundary_width=self._stitch_boundary_width(),
            stitch_workers=str(self.stitch_workers_edit.value()),
            overexposure_enabled=self.run_overexp_cb.isChecked(),
            overexposure_threshold=str(self.overexp_threshold_edit.value()),
            overexposure_dilate=str(self.overexp_dilate_edit.value()),
            custom_enabled=self.run_custom_cb.isChecked(),
            custom_mask=self._custom_mask_path_text(),
            images_dir=self._images_dir_text(),
            masks_dir=self._masks_dir_text(),
        )

    def _mask_command_context(self, *, projection: str | None = None) -> MaskCommandContext:
        return self._mask_settings_state().command_context(
            python_executable=sys.executable,
            base_dir=self.base_dir,
            projection=projection,
        )

    def _mask_settings_snapshot(self) -> dict:
        return self._mask_settings_state().snapshot()
