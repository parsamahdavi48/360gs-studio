"""Step 3 batch mask command orchestration."""

from __future__ import annotations

from pathlib import Path

from core.mask_source_scope import MASK_SOURCE_ALL
from core.scene_project import utc_now_iso
from gui import i18n
from gui.common.runner_types import StepCommandQueue
from gui.steps.step3_mask_manifests import write_mask_target_manifest, write_projection_manifests
from gui.steps.step3_mask_plan import (
    MASK_COMMAND_CUSTOM,
    MASK_COMMAND_INIT,
    MASK_COMMAND_OVEREXPOSURE,
    MASK_COMMAND_STITCH,
    MASK_COMMAND_YOLO,
    MaskCommandSpec,
    build_mixed_mask_command_specs,
    build_uniform_mask_command_specs,
    needs_target_manifest,
)
from gui.steps.step3_mask_records import record_mask_outputs


class Step3MaskBatchMixin:
    def phase_display_name(self, phase: str) -> str:
        labels = {
            "yolo": "MASK_PHASE_PRIMARY",
            "yolo_equirect": "MASK_PHASE_PRIMARY",
            "yolo_normal": "MASK_PHASE_PRIMARY",
            "stitch": "MASK_PHASE_STITCH",
            "stitch_equirect": "MASK_PHASE_STITCH",
            "overexposure": "MASK_PHASE_OVEREXPOSURE",
            "custom": "MASK_PHASE_CUSTOM",
            "init_masks": "MASK_PHASE_INIT",
        }
        key = labels.get(phase)
        return i18n.t(key) if key else phase

    def build_commands(self) -> StepCommandQueue:
        self._refresh_scene_inventory_cache(strict=True)
        self._sync_projection_from_project(preserve_user_quality=True)
        ready, reason = self._readiness()
        if not ready:
            raise ValueError(reason)
        requested_steps = self._selected_mask_tasks()
        if not requested_steps:
            raise ValueError(i18n.t("MASK_TASK_REQUIRED"))

        self._ensure_no_pending_drop_images()
        self._ensure_no_untracked_images()

        all_image_paths = self._scene_image_paths()
        settings = self._mask_settings_snapshot()
        image_paths = self._source_filtered_image_paths()
        if not image_paths:
            raise ValueError(i18n.t("MASK_SOURCE_EMPTY"))
        target_paths = self._mask_target_paths(image_paths, settings=settings)
        if not target_paths:
            raise ValueError(i18n.t("MASK_TARGETS_EMPTY"))

        if self._projection_mixed:
            manifests = self._write_projection_manifests(image_paths=target_paths)
            specs = build_mixed_mask_command_specs(requested_steps, manifests=manifests)
        else:
            target_manifest = (
                self._write_mask_target_manifest(target_paths)
                if needs_target_manifest(
                    source_is_all=self._selected_mask_source_key() == MASK_SOURCE_ALL,
                    target_count=len(target_paths),
                    all_image_count=len(all_image_paths),
                )
                else None
            )
            specs = build_uniform_mask_command_specs(requested_steps, target_manifest=target_manifest)

        steps = [(spec.phase, self._command_from_mask_spec(spec)) for spec in specs]
        self._mask_batch_settings = settings
        self._mask_batch_phases = [phase for phase, _cmd in steps]
        self._mask_batch_targets = target_paths
        return steps

    def _command_from_mask_spec(self, spec: MaskCommandSpec) -> list[str]:
        if spec.command == MASK_COMMAND_YOLO:
            return self._build_yolo_cmd(projection=spec.projection, image_list=spec.image_list)
        if spec.command == MASK_COMMAND_INIT:
            return self._build_init_masks_cmd(image_list=spec.image_list)
        if spec.command == MASK_COMMAND_STITCH:
            return self._build_stitch_cmd(image_list=spec.image_list)
        if spec.command == MASK_COMMAND_OVEREXPOSURE:
            return self._build_overexposure_cmd(replace=spec.replace, image_list=spec.image_list)
        if spec.command == MASK_COMMAND_CUSTOM:
            return self._build_custom_cmd(replace=spec.replace, image_list=spec.image_list)
        raise ValueError(f"Unknown mask command spec: {spec.command}")

    def _write_projection_manifests(self, *, image_paths: list[Path] | None = None) -> dict[str, Path]:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        image_paths = image_paths or self._scene_image_paths()
        if not image_paths:
            return {}
        return write_projection_manifests(
            scene_dir=self.scene_dir,
            image_paths=image_paths,
            projection_for_image=self._projection_for_image,
            mask_path_for_image=self._mask_output_path_for_image,
            run_id_factory=self._new_mask_run_id,
        )

    def _write_mask_target_manifest(self, image_paths: list[Path]) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return write_mask_target_manifest(
            scene_dir=self.scene_dir,
            image_paths=image_paths,
            projection_for_image=self._projection_for_image,
            mask_path_for_image=self._mask_output_path_for_image,
            run_id_factory=self._new_mask_run_id,
        )

    @staticmethod
    def _new_mask_run_id(prefix: str) -> str:
        return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '')}"

    def _record_mask_outputs(
        self,
        image_paths: list[Path],
        *,
        mode: str,
        settings: dict | None,
        phases: list[str],
        run_id: str | None = None,
    ) -> None:
        if not self.scene_dir or not image_paths:
            return
        record_mask_outputs(
            self.scene_dir,
            image_paths,
            mode=mode,
            settings=settings or self._mask_settings_snapshot(),
            phases=phases,
            mask_path_for_image=self._mask_output_path_for_image,
            run_id=run_id,
            run_id_factory=self._new_mask_run_id,
        )

    def on_line(self, line: str) -> tuple[int, int] | None:
        return self._progress_parser.on_line(line)

    def on_phase_finished(self, phase: str, exit_code: int, canceled: bool) -> None:
        self._progress_parser.on_phase_finished(phase, exit_code)

    def on_queue_finished(self, success: bool) -> None:
        if not success:
            self._mask_batch_settings = None
            self._mask_batch_phases = []
            self._mask_batch_targets = []
            return
        self._invalidate_scene_inventory_cache()
        self.mask_preview.clear_yolo_preview_mask()
        self.mask_preview.refresh_image_list(prefer_current=True, force_thumbnails=True)
        self._render_mask_preview()
        self._update_ready_status()
        self._record_mask_outputs(
            self._mask_batch_targets or self._scene_image_paths(),
            mode="batch",
            settings=self._mask_batch_settings,
            phases=list(self._mask_batch_phases),
        )
        self._mask_batch_settings = None
        self._mask_batch_phases = []
        self._mask_batch_targets = []
