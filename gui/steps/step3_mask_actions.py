"""Step 3 mask preview and selected-image reprocessing actions."""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QProcess, QTimer

from gui import i18n
from gui.common.runner_types import ExternalCommandQueue
from gui.steps.mask_commands import (
    build_custom_cmd,
    build_init_masks_cmd,
    build_mask2former_cmd,
    build_overexposure_cmd,
    build_primary_mask_cmd,
    build_sam31_prompt_cmd,
    build_stitch_cmd,
)
from gui.steps.mask_postprocess import MaskPostprocessOptions, apply_mask_postprocess
from gui.steps.step3_mask_preview_actions import Step3MaskPreviewActionsMixin

_PROJECTION_EQUIRECT = "equirect"
_STITCH_BOUNDARY_MIN = 0.0
_STITCH_BOUNDARY_MAX = 30.0
_STITCH_BOUNDARY_DEFAULT = 5.0


class Step3MaskActionsMixin(Step3MaskPreviewActionsMixin):
    def _stitch_boundary_width(self) -> float:
        value = self._clamp_stitch_boundary_width(float(self.stitch_boundary_width_edit.value()))
        if value != self.stitch_boundary_width_edit.value():
            self.stitch_boundary_width_edit.setValue(value)
        return value

    @staticmethod
    def _clamp_stitch_boundary_width(value: float) -> float:
        if not math.isfinite(value):
            return _STITCH_BOUNDARY_DEFAULT
        return max(_STITCH_BOUNDARY_MIN, min(_STITCH_BOUNDARY_MAX, value))

    # -- コマンド構築 --

    def _build_yolo_cmd(
        self,
        *,
        projection: str | None = None,
        image_list: str | Path | None = None,
    ) -> list[str]:
        images = self._images_dir_text()
        masks = self._masks_dir_text()
        return build_primary_mask_cmd(
            self._mask_command_context(projection=projection),
            images,
            masks,
            backend=self._person_backend_arg(),
            image_list=image_list,
        )

    def _build_yolo_preview_cmd(
        self,
        image_path: Path,
        output_dir: Path,
        *,
        projection: str | None = None,
    ) -> list[str]:
        return build_primary_mask_cmd(
            self._mask_command_context(projection=projection),
            str(image_path),
            str(output_dir),
            backend=self._person_backend_arg(),
        )

    def _mask_output_dir_for_image(self, image_path: Path, masks_root: Path | None = None) -> Path:
        masks_root = masks_root or Path(self._masks_dir_text())
        try:
            rel_parent = image_path.resolve().relative_to(Path(self._images_dir_text()).resolve()).parent
        except Exception:
            rel_parent = Path()
        return masks_root / rel_parent

    def _mask_output_path_for_image(self, image_path: Path, masks_root: Path | None = None) -> Path:
        return self._mask_output_dir_for_image(image_path, masks_root=masks_root) / f"{image_path.stem}.png"

    def _build_yolo_current_cmd(self, image_path: Path, masks_root: Path | None = None) -> list[str]:
        output_dir = self._mask_output_dir_for_image(image_path, masks_root=masks_root)
        return self._build_yolo_preview_cmd(
            image_path,
            output_dir,
            projection=self._projection_for_image(image_path),
        )

    def _build_sam31_prompt_cmd(
        self,
        images: str | Path,
        masks: str | Path,
        *,
        prompts: list[str],
        subtract_prompts: list[str] | None = None,
        merge_mode: str | None = None,
        replace: bool = False,
        projection: str | None = None,
        image_list: str | Path | None = None,
    ) -> list[str]:
        return build_sam31_prompt_cmd(
            self._mask_command_context(projection=projection),
            images,
            masks,
            prompts=prompts,
            subtract_prompts=subtract_prompts,
            merge_mode=merge_mode,
            replace=replace,
            image_list=image_list,
        )

    def _build_mask2former_cmd(
        self,
        images: str | Path,
        masks: str | Path,
        *,
        replace: bool = False,
        projection: str | None = None,
        image_list: str | Path | None = None,
    ) -> list[str]:
        return build_mask2former_cmd(
            self._mask_command_context(projection=projection),
            images,
            masks,
            replace=replace,
            image_list=image_list,
        )

    def _build_init_masks_cmd(self, *, image_list: str | Path | None = None) -> list[str]:
        images = self._images_dir_text()
        masks = self._masks_dir_text()
        return build_init_masks_cmd(self._mask_command_context(), images, masks, image_list=image_list)

    def _run_current_image_reprocess(self) -> None:
        if self._current_reprocess_active:
            return
        if self._mask_preview_proc is not None and self._mask_preview_proc.state() != QProcess.NotRunning:
            return

        image_paths = self._selected_reprocess_image_paths()
        if not image_paths:
            self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_NO_IMAGE"))
            return
        if not self._selected_mask_tasks():
            self.mask_preview.set_status_text(i18n.t("MASK_TASK_REQUIRED"))
            return
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

        self._current_reprocess_active = True
        self._current_reprocess_queue = image_paths
        self._current_reprocess_total = len(image_paths)
        self._current_reprocess_completed = 0
        self._current_reprocess_failed = []
        self._current_reprocess_succeeded = []
        self._current_reprocess_last_success = None
        self._current_reprocess_run_id = self._new_mask_run_id("mask_reprocess")
        self._current_reprocess_settings = self._mask_settings_snapshot()
        self.mask_preview.set_current_reprocess_running(True)
        self.mask_preview.wait_for_thumbnail_rendering()
        self._start_next_current_reprocess()

    def _selected_reprocess_image_paths(self) -> list[Path]:
        paths = self.mask_preview.selected_reprocess_image_paths()
        result: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            try:
                key = str(path.resolve()).lower()
            except Exception:
                key = str(path).lower()
            if key in seen:
                continue
            if not path.exists() or not path.is_file():
                continue
            seen.add(key)
            result.append(path)
        return result

    def _start_next_current_reprocess(self) -> None:
        if not self._current_reprocess_active:
            return
        if not self._current_reprocess_queue:
            self._finish_reprocess_batch()
            return

        image_path = self._current_reprocess_queue.pop(0)
        self._set_current_reprocess_progress(image_path)
        mask_path = self._mask_output_path_for_image(image_path)
        try:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            self.mask_preview.clear_yolo_preview_mask(image_path)
            commands = self._build_current_reprocess_external_commands(image_path)
            if not commands:
                self._apply_current_image_postprocess(image_path, mask_path, replace=True)
                self._record_current_reprocess_result(success=True, image_path=image_path)
                self._queue_next_current_reprocess()
                return
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            self.mask_preview.set_status_text(str(e))
            self._record_current_reprocess_result(success=False, image_path=image_path)
            self._queue_next_current_reprocess()
            return

        self._current_reprocess_image = image_path
        self._current_reprocess_mask = mask_path
        self._current_reprocess_commands = commands
        self._start_next_current_reprocess_external_command()

    def _build_current_reprocess_external_commands(self, image_path: Path) -> ExternalCommandQueue:
        return self._build_image_external_commands(image_path, masks_root=None)

    def _build_image_external_commands(
        self,
        image_path: Path,
        *,
        masks_root: Path | None,
    ) -> ExternalCommandQueue:
        commands: ExternalCommandQueue = []
        commands.append(("yolo", self._build_yolo_current_cmd(image_path, masks_root=masks_root)))
        return commands

    def _start_next_current_reprocess_external_command(self) -> None:
        if not self._current_reprocess_commands:
            image_path = self._current_reprocess_image
            mask_path = self._current_reprocess_mask
            success = image_path is not None and mask_path is not None and mask_path.is_file()
            if success and image_path is not None and mask_path is not None:
                try:
                    self._apply_current_image_postprocess(image_path, mask_path)
                except Exception as e:
                    success = False
                    self.mask_preview.set_status_text(str(e))
            self._record_current_reprocess_result(success=bool(success), image_path=image_path)
            self._current_reprocess_image = None
            self._current_reprocess_mask = None
            self._current_reprocess_phase = ""
            self._queue_next_current_reprocess()
            return

        phase, cmd = self._current_reprocess_commands.pop(0)
        self._current_reprocess_phase = phase
        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._drain_current_reprocess_output)
        proc.errorOccurred.connect(self._on_current_reprocess_error)
        proc.finished.connect(self._on_current_reprocess_finished)
        self._current_reprocess_proc = proc
        proc.start()

    def _drain_current_reprocess_output(self) -> None:
        if self._current_reprocess_proc is not None:
            self._current_reprocess_proc.readAllStandardOutput()

    def _on_current_reprocess_error(self, _error: QProcess.ProcessError) -> None:
        self.mask_preview.set_status_text(i18n.t("MASK_REPROCESS_CURRENT_FAILED"))

    def _on_current_reprocess_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        image_path = self._current_reprocess_image
        mask_path = self._current_reprocess_mask
        success = exit_code == 0 and image_path is not None and mask_path is not None and mask_path.is_file()
        self._current_reprocess_proc = None
        if success:
            self._start_next_current_reprocess_external_command()
            return

        self._record_current_reprocess_result(success=False, image_path=image_path)
        self._current_reprocess_image = None
        self._current_reprocess_mask = None
        self._current_reprocess_commands = []
        self._current_reprocess_phase = ""
        self._queue_next_current_reprocess()

    def _record_current_reprocess_result(self, *, success: bool, image_path: Path | None) -> None:
        self._current_reprocess_completed += 1
        if success and image_path is not None:
            self._current_reprocess_last_success = image_path
            self._current_reprocess_succeeded.append(image_path)
        if not success and image_path is not None:
            self._current_reprocess_failed.append(image_path)

    def _set_current_reprocess_progress(self, image_path: Path) -> None:
        self.mask_preview.set_status_text(
            i18n.t("MASK_REPROCESS_SELECTED_PROGRESS").format(
                done=self._current_reprocess_completed + 1,
                total=self._current_reprocess_total,
                name=image_path.name,
            )
        )

    def _queue_next_current_reprocess(self) -> None:
        QTimer.singleShot(0, self._start_next_current_reprocess)

    def _finish_reprocess_batch(self) -> None:
        total = self._current_reprocess_total
        completed = self._current_reprocess_completed
        failed = len(self._current_reprocess_failed)
        succeeded_images = list(self._current_reprocess_succeeded)
        last_success = total == 1 and completed == 1 and failed == 0
        last_image = self._current_reprocess_last_success

        self._current_reprocess_active = False
        self._current_reprocess_queue = []
        self._current_reprocess_total = 0
        self._current_reprocess_completed = 0
        self._current_reprocess_last_success = None
        self._invalidate_scene_inventory_cache()
        self.mask_preview.set_current_reprocess_running(False)
        self.mask_preview.refresh_image_list(prefer_current=True)
        self.mask_preview.invalidate_thumbnail_images(succeeded_images)
        self._render_mask_preview()
        self._mask_preview_render_timer.stop()
        self._mask_preview_render_pending = False
        self._update_ready_status()
        self._record_mask_outputs(
            succeeded_images,
            mode="selected_reprocess",
            settings=self._current_reprocess_settings,
            phases=self._selected_mask_tasks(),
            run_id=self._current_reprocess_run_id,
        )

        if last_success and last_image is not None:
            self.mask_preview.set_status_text(i18n.t("MASK_REPROCESS_CURRENT_DONE").format(name=last_image.name))
        elif failed == 0:
            self.mask_preview.set_status_text(
                i18n.t("MASK_REPROCESS_SELECTED_DONE").format(done=completed, total=total)
            )
        else:
            self.mask_preview.set_status_text(
                i18n.t("MASK_REPROCESS_SELECTED_FAILED").format(failed=failed, total=total)
            )
        self._current_reprocess_failed = []
        self._current_reprocess_succeeded = []
        self._current_reprocess_run_id = ""
        self._current_reprocess_settings = None

    def _apply_current_image_postprocess(
        self,
        image_path: Path,
        mask_path: Path,
        *,
        replace: bool = False,
    ) -> None:
        apply_mask_postprocess(
            image_path,
            mask_path,
            MaskPostprocessOptions(
                projection=self._projection_for_image(image_path),
                equirect_projection=_PROJECTION_EQUIRECT,
                apply_stitch=self.run_stitch_cb.isChecked(),
                stitch_boundary_width=self._stitch_boundary_width(),
                apply_overexposure=self.run_overexp_cb.isChecked(),
                overexposure_threshold=int(self.overexp_threshold_edit.value()),
                overexposure_dilate=int(self.overexp_dilate_edit.value()),
                apply_custom=self.run_custom_cb.isChecked(),
                custom_mask_path=self._custom_mask_path_text(),
                replace=replace,
                preview_load_fail_message=i18n.t("PREVIEW_LOAD_FAIL"),
                custom_required_message=i18n.t("CUSTOM_MASK_REQUIRED"),
                custom_not_found_message=i18n.t("CUSTOM_MASK_NOT_FOUND"),
                write_failed_message=i18n.t("MASK_REPROCESS_CURRENT_FAILED"),
            ),
        )

    def _build_stitch_cmd(self, *, image_list: str | Path | None = None) -> list[str]:
        masks = self._masks_dir_text()
        return build_stitch_cmd(self._mask_command_context(), masks, image_list=image_list)

    def _build_overexposure_cmd(
        self,
        *,
        replace: bool = False,
        image_list: str | Path | None = None,
    ) -> list[str]:
        images = self._images_dir_text()
        masks = self._masks_dir_text()
        return build_overexposure_cmd(
            self._mask_command_context(),
            images,
            masks,
            replace=replace,
            image_list=image_list,
        )

    def _build_custom_cmd(
        self,
        *,
        replace: bool = False,
        image_list: str | Path | None = None,
    ) -> list[str]:
        images = self._images_dir_text()
        masks = self._masks_dir_text()
        custom_mask = self._custom_mask_path_text()
        if not custom_mask:
            raise ValueError(i18n.t("CUSTOM_MASK_REQUIRED"))
        return build_custom_cmd(
            self._mask_command_context(),
            images,
            masks,
            replace=replace,
            image_list=image_list,
        )

    # -- プログレス解析 --
