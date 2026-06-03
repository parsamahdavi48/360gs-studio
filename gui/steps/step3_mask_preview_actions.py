"""Step 3 temporary mask preview orchestration."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import QProcess

from gui import i18n
from gui.mask.mask_preview import MaskPreviewConfig
from gui.steps.mask_commands import MASK_APPLY_REPLACE

_PROJECTION_EQUIRECT = "equirect"


class Step3MaskPreviewActionsMixin:
    def _on_preview_current_image_changed(self) -> None:
        self._update_preview_projection_enabled()
        self._schedule_render_mask_preview()

    def _schedule_render_mask_preview(self) -> None:
        self._mask_preview_render_pending = True
        self._mask_preview_render_timer.start()

    def _flush_scheduled_mask_preview(self) -> None:
        if not self._mask_preview_render_pending:
            return
        self._mask_preview_render_pending = False
        self._render_mask_preview()

    def _render_mask_preview(self) -> None:
        self.mask_preview.render(self._mask_preview_config_from_controls())

    def _mask_preview_config_from_controls(self) -> MaskPreviewConfig:
        try:
            width = self._stitch_boundary_width()
        except ValueError:
            width = None
        current_projection = self._projection_for_image(self.mask_preview.current_image_path())
        return MaskPreviewConfig(
            use_yolo=True,
            use_stitch=current_projection == _PROJECTION_EQUIRECT and self.run_stitch_cb.isChecked(),
            use_overexposure=self.run_overexp_cb.isChecked(),
            use_sky=False,
            stitch_boundary_width_deg=width,
            overexposure_threshold=int(self.overexp_threshold_edit.value()),
            overexposure_dilate=int(self.overexp_dilate_edit.value()),
            masks_dir=self._masks_dir_text(),
            use_custom=self.run_custom_cb.isChecked(),
            custom_mask_path=self._custom_mask_path_text(),
            settings_key=self._mask_generation_settings_key(),
        )

    def _mask_generation_settings_key(self) -> tuple:
        current_image = self.mask_preview.current_image_path()
        return (
            self._projection(),
            self._projection_for_image(current_image),
            self._person_backend_arg(),
            self.yolo_level_combo.currentIndex(),
            int(self.yolo_expand_edit.value()),
            tuple(self._selected_classes()),
            tuple(self._selected_semantic_labels()),
            tuple(self._selected_sam_prompts()),
            tuple(self._selected_sam_subtract_prompts()),
            self._mask_merge_mode_arg(),
            self._sky_backend_arg(),
            self._sky_inference_size_arg(),
            int(self.sky_expand_edit.value()),
            float(self.sky_min_score_edit.value()),
            float(self.sky_min_area_edit.value()),
            bool(self.sky_top_connected_cb.isChecked()),
        )

    def _seed_merge_preview_base_mask(self, image_path: Path, output_path: Path) -> None:
        if self._mask_merge_mode_arg() == MASK_APPLY_REPLACE:
            return
        existing_path = self._mask_output_path_for_image(image_path)
        if not existing_path.is_file():
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(existing_path, output_path)

    def _run_mask_preview(self) -> None:
        if self._mask_preview_proc is not None and self._mask_preview_proc.state() != QProcess.NotRunning:
            return
        if self._current_reprocess_active:
            return

        image_path = self.mask_preview.current_image_path()
        if image_path is None:
            self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_NO_IMAGE"))
            return
        config = self._mask_preview_config_from_controls()
        if not self._selected_mask_tasks():
            self.mask_preview.set_status_text(i18n.t("MASK_TASK_REQUIRED"))
            return

        if self.mask_preview.preview_mode() == "thumbnails":
            self.mask_preview.set_preview_mode("single")

        if self._person_backend_arg() == "yolo_sam":
            if not self._confirm_yolo_sam_license_notice():
                self.mask_preview.set_status_text(i18n.t("YOLO_SAM_LICENSE_NOTICE_CANCELED"))
                return
        elif not self._confirm_sky_license_notice():
            self.mask_preview.set_status_text(i18n.t("SKY_LICENSE_NOTICE_CANCELED"))
            return
        elif self._uses_sam31_for_primary_mask() and not self._ensure_sam31_checkpoint_available():
            self.mask_preview.set_status_text(i18n.t("SAM31_DOWNLOAD_CANCELED"))
            return

        self._cleanup_mask_preview_temp()
        self._mask_preview_temp = tempfile.TemporaryDirectory(prefix="stechdrive_mask_preview_")
        masks_root = Path(self._mask_preview_temp.name)
        output_path = self._mask_output_path_for_image(image_path, masks_root=masks_root)
        self._seed_merge_preview_base_mask(image_path, output_path)

        try:
            commands = self._build_image_external_commands(image_path, masks_root=masks_root)
        except (ValueError, FileNotFoundError) as e:
            self.mask_preview.set_status_text(str(e))
            self._cleanup_mask_preview_temp()
            return

        self._mask_preview_image = image_path
        self._mask_preview_output = output_path
        self._mask_preview_config = config
        self._mask_preview_commands = commands
        self.mask_preview.set_mask_preview_running(True)
        self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_RUNNING"))
        self._start_next_mask_preview_command()

    def _start_next_mask_preview_command(self) -> None:
        if not self._mask_preview_commands:
            image_path = self._mask_preview_image
            output_path = self._mask_preview_output
            config = self._mask_preview_config
            ok = image_path is not None and output_path is not None and config is not None
            if ok and image_path is not None and output_path is not None:
                try:
                    self._apply_current_image_postprocess(
                        image_path,
                        output_path,
                        replace=not output_path.is_file(),
                    )
                except Exception as e:
                    ok = False
                    self.mask_preview.set_status_text(str(e))
            if ok and image_path is not None and output_path is not None and config is not None:
                ok = output_path.is_file() and self.mask_preview.set_temporary_preview_mask(
                    image_path, output_path, config
                )
            self.mask_preview.set_mask_preview_running(False)
            if ok:
                self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_TEMP"))
            else:
                self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_FAILED"))
            self._cleanup_mask_preview_temp()
            self._mask_preview_proc = None
            self._mask_preview_image = None
            self._mask_preview_output = None
            self._mask_preview_config = None
            self._render_mask_preview()
            return

        _phase, cmd = self._mask_preview_commands.pop(0)
        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._drain_mask_preview_output)
        proc.errorOccurred.connect(self._on_mask_preview_error)
        proc.finished.connect(self._on_mask_preview_finished)
        self._mask_preview_proc = proc
        proc.start()

    def _drain_mask_preview_output(self) -> None:
        if self._mask_preview_proc is not None:
            self._mask_preview_proc.readAllStandardOutput()

    def _on_mask_preview_error(self, _error: QProcess.ProcessError) -> None:
        self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_FAILED"))

    def _on_mask_preview_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._mask_preview_proc = None
        if exit_code == 0:
            self._start_next_mask_preview_command()
            return
        self.mask_preview.set_mask_preview_running(False)
        self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_FAILED"))
        self._cleanup_mask_preview_temp()
        self._mask_preview_image = None
        self._mask_preview_output = None
        self._mask_preview_config = None
        self._mask_preview_commands = []
        self._render_mask_preview()

    def _cleanup_mask_preview_temp(self) -> None:
        if self._mask_preview_temp is None:
            return
        try:
            self._mask_preview_temp.cleanup()
        except Exception:
            pass
        self._mask_preview_temp = None
