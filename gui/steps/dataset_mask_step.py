"""Step 5 training-mask settings built from the Step 3 mask UI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QFormLayout, QWidget

from core.app_job import dataset_app_job
from core.dataset_job_spec import attach_dataset_masks_job, write_dataset_job
from core.dataset_mask_policy import (
    DATASET_MASK_CONVERT_SFM,
    DATASET_MASK_GENERATE_TRAINING,
    DATASET_MASK_NONE,
    DATASET_MASK_REUSE_EXISTING,
    normalize_dataset_mask_mode,
)
from core.mask_source_scope import MASK_SOURCE_ALL
from core.nerf_dataset_paths import find_nerf_transforms_path
from core.scene_layout import jobs_dir, scene_images_dir, scene_masks_dir
from gui import i18n
from gui.common.form_rows import add_tooltip_row
from gui.common.runner_types import StepCommandQueue
from gui.mask.mask_preview import MaskPreviewConfig
from gui.steps.step3_mask import MaskStep
from gui.steps.step3_mask_plan import (
    MASK_COMMAND_CUSTOM,
    MASK_COMMAND_INIT,
    MASK_COMMAND_OVEREXPOSURE,
    MASK_COMMAND_STITCH,
    MASK_COMMAND_YOLO,
    MASK_TASK_CUSTOM,
    MASK_TASK_OVEREXPOSURE,
    MASK_TASK_STITCH,
    MASK_TASK_YOLO,
    MaskCommandSpec,
    build_uniform_mask_command_specs,
)

_PROJECTION_EQUIRECT = "equirect"
_PROJECTION_NORMAL = "normal"


class DatasetMaskStep(MaskStep):
    """Generate masks in the active Step 5 dataset output instead of scene/masks."""

    def __init__(
        self,
        base_dir: Path,
        *,
        dataset_root_provider: Callable[[], Path],
        link_mask_paths: bool = True,
        mode_tip_keys: Mapping[str, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        self._dataset_root_provider = dataset_root_provider
        self._dataset_projection = _PROJECTION_NORMAL
        self._link_mask_paths = bool(link_mask_paths)
        self._mode_tip_keys = dict(mode_tip_keys or {})
        super().__init__(base_dir, parent)
        self.mask_preview.set_empty_messages(
            no_scene=i18n.t("DATASET_MASK_PREVIEW_NO_DATASET"),
            empty=i18n.t("DATASET_MASK_PREVIEW_EMPTY"),
        )
        self._install_dataset_mode_controls()
        self._refresh_mask_source_options()
        self._sync_preview_roots_for_mode()
        self._sync_preview_actions_for_mode()

    def _install_dataset_mode_controls(self) -> None:
        for field_name, label_name in (
            ("images_path_row", "images_path_row_label"),
            ("masks_path_label", "masks_path_row_label"),
            ("mask_source_combo", "mask_source_row_label"),
        ):
            field = getattr(self, field_name, None)
            if field is not None:
                field.setVisible(False)
            label = getattr(self, label_name, None)
            if label is not None:
                label.setVisible(False)
        self.metashape_notice.setObjectName("")
        self.metashape_notice.setVisible(False)

        form = QFormLayout()
        form.setSpacing(6)
        self.dataset_mask_mode_combo = QComboBox()
        self.dataset_mask_mode_combo.setToolTip(i18n.tip("DATASET_MASK_MODE"))
        for mode, label_key in (
            (DATASET_MASK_CONVERT_SFM, "DATASET_MASK_MODE_CONVERT_SFM"),
            (DATASET_MASK_GENERATE_TRAINING, "DATASET_MASK_MODE_GENERATE_TRAINING"),
            (DATASET_MASK_REUSE_EXISTING, "DATASET_MASK_MODE_REUSE_EXISTING"),
            (DATASET_MASK_NONE, "DATASET_MASK_MODE_NONE"),
        ):
            self.dataset_mask_mode_combo.addItem(i18n.t(label_key), mode)
            index = self.dataset_mask_mode_combo.count() - 1
            self.dataset_mask_mode_combo.setItemData(index, i18n.tip(self._mode_tip_key(mode)), Qt.ToolTipRole)
        self.dataset_mask_mode_combo.currentIndexChanged.connect(lambda _index: self._on_dataset_mask_mode_changed())
        add_tooltip_row(form, i18n.t("DATASET_MASK_MODE"), self.dataset_mask_mode_combo, i18n.tip("DATASET_MASK_MODE"))
        self.settings_layout.insertLayout(0, form)

    def _mode_tip_key(self, mode: str) -> str:
        return self._mode_tip_keys.get(mode, f"DATASET_MASK_MODE_{mode.upper()}")

    def set_dataset_projection(self, projection: str) -> None:
        projection = projection if projection in {_PROJECTION_EQUIRECT, _PROJECTION_NORMAL} else _PROJECTION_NORMAL
        self._dataset_projection = projection
        self._set_projection(projection, sync_yolo_quality=False)

    def mask_mode(self) -> str:
        return normalize_dataset_mask_mode(self.dataset_mask_mode_combo.currentData())

    def set_mask_mode(self, mode: object) -> None:
        normalized = normalize_dataset_mask_mode(mode)
        index = self.dataset_mask_mode_combo.findData(normalized)
        if index >= 0:
            self.dataset_mask_mode_combo.setCurrentIndex(index)
            self._on_dataset_mask_mode_changed()

    def primary_action_text(self) -> str:
        mode = self.mask_mode()
        if mode == DATASET_MASK_GENERATE_TRAINING:
            return i18n.t("DATASET_MASK_GENERATE_ACTION")
        if not self._link_mask_paths:
            return i18n.t("DATASET_MASK_RUN_DATASET_ACTION")
        if mode == DATASET_MASK_REUSE_EXISTING:
            return i18n.t("DATASET_MASK_ATTACH_ACTION")
        if mode == DATASET_MASK_NONE:
            return i18n.t("DATASET_MASK_CLEAR_ACTION")
        return i18n.t("DATASET_MASK_RUN_DATASET_ACTION")

    def primary_action_tooltip(self) -> str:
        mode = self.mask_mode()
        if mode == DATASET_MASK_GENERATE_TRAINING:
            return super().primary_action_tooltip()
        if not self._link_mask_paths:
            return i18n.tip("DATASET_MASK_RUN_DATASET_ACTION")
        ready, reason = self._dataset_mask_path_readiness()
        if mode == DATASET_MASK_REUSE_EXISTING:
            return i18n.tip("DATASET_MASK_ATTACH_ACTION") if ready else reason
        if mode == DATASET_MASK_NONE:
            return i18n.tip("DATASET_MASK_CLEAR_ACTION") if ready else reason
        return i18n.tip("DATASET_MASK_RUN_DATASET_ACTION")

    def primary_action_enabled(self) -> bool:
        mode = self.mask_mode()
        if mode == DATASET_MASK_GENERATE_TRAINING:
            return super().primary_action_enabled()
        if not self._link_mask_paths:
            return False
        if mode in {DATASET_MASK_REUSE_EXISTING, DATASET_MASK_NONE}:
            ready, _reason = self._dataset_mask_path_readiness()
            return ready
        return False

    def build_commands(self) -> StepCommandQueue:
        mode = self.mask_mode()
        if mode == DATASET_MASK_GENERATE_TRAINING:
            commands = [*super().build_commands()]
            if self._link_mask_paths:
                commands.append(self._attach_mask_paths_command())
            return commands
        if not self._link_mask_paths:
            return []
        if mode == DATASET_MASK_REUSE_EXISTING:
            return [self._attach_mask_paths_command()]
        if mode == DATASET_MASK_NONE:
            return [self._clear_mask_paths_command()]
        raise ValueError(i18n.t("DATASET_MASK_RUN_DATASET_ACTION"))

    def build_followup_commands(self, *, require_existing_images: bool) -> StepCommandQueue:
        mode = self.mask_mode()
        if mode == DATASET_MASK_CONVERT_SFM:
            return []
        if not self._link_mask_paths:
            if mode == DATASET_MASK_GENERATE_TRAINING:
                if require_existing_images:
                    return [*super().build_commands()]
                return [*self._build_pending_dataset_mask_commands()]
            return []
        if mode == DATASET_MASK_REUSE_EXISTING:
            return [self._attach_mask_paths_command()]
        if mode == DATASET_MASK_NONE:
            return [self._clear_mask_paths_command()]
        if require_existing_images:
            return [*super().build_commands(), self._attach_mask_paths_command()]
        return [*self._build_pending_dataset_mask_commands(), self._attach_mask_paths_command()]

    def _build_pending_dataset_mask_commands(self) -> StepCommandQueue:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        if self.run_custom_cb.isChecked():
            custom_mask = self._custom_mask_path_text()
            if not custom_mask:
                raise ValueError(i18n.t("CUSTOM_MASK_REQUIRED"))
            if not Path(custom_mask).is_file():
                raise ValueError(i18n.t("CUSTOM_MASK_NOT_FOUND").format(path=custom_mask))
        requested_steps = self._selected_dataset_mask_tasks()
        if not requested_steps:
            raise ValueError(i18n.t("MASK_TASK_REQUIRED"))
        specs = build_uniform_mask_command_specs(requested_steps, target_manifest=None)
        steps = [(spec.phase, self._command_from_mask_spec(spec)) for spec in specs]
        self._mask_batch_settings = self._mask_settings_snapshot()
        self._mask_batch_phases = [phase for phase, _cmd in steps]
        self._mask_batch_targets = []
        return steps

    def _selected_dataset_mask_tasks(self) -> list[str]:
        requested_steps = [MASK_TASK_YOLO]
        if self._dataset_projection == _PROJECTION_EQUIRECT and self.run_stitch_cb.isChecked():
            requested_steps.append(MASK_TASK_STITCH)
        if self.run_overexp_cb.isChecked():
            requested_steps.append(MASK_TASK_OVEREXPOSURE)
        if self.run_custom_cb.isChecked():
            requested_steps.append(MASK_TASK_CUSTOM)
        return requested_steps

    def _command_from_mask_spec(self, spec: MaskCommandSpec) -> list[str]:
        projection = spec.projection or self._dataset_projection
        if spec.command == MASK_COMMAND_YOLO:
            return self._build_yolo_cmd(projection=projection, image_list=spec.image_list)
        if spec.command == MASK_COMMAND_INIT:
            return self._build_init_masks_cmd(image_list=spec.image_list)
        if spec.command == MASK_COMMAND_STITCH:
            return self._build_stitch_cmd(image_list=spec.image_list)
        if spec.command == MASK_COMMAND_OVEREXPOSURE:
            return self._build_overexposure_cmd(replace=spec.replace, image_list=spec.image_list)
        if spec.command == MASK_COMMAND_CUSTOM:
            return self._build_custom_cmd(replace=spec.replace, image_list=spec.image_list)
        raise ValueError(f"Unknown mask command spec: {spec.command}")

    def _dataset_root(self) -> Path:
        return Path(self._dataset_root_provider())

    def _preview_uses_sfm_masks(self) -> bool:
        return hasattr(self, "dataset_mask_mode_combo") and self.mask_mode() == DATASET_MASK_CONVERT_SFM

    def _images_dir_for_scene(self, _scene: Path) -> Path:
        if self._preview_uses_sfm_masks():
            return scene_images_dir(_scene)
        return self._dataset_root() / "images"

    def _masks_dir_for_scene(self, _scene: Path) -> Path:
        if self._preview_uses_sfm_masks():
            return scene_masks_dir(_scene)
        return self._dataset_root() / "masks"

    def _projection_key_for_image(self, image_path: Path) -> str:
        if self.scene_dir:
            try:
                return image_path.relative_to(Path(self.scene_dir)).as_posix()
            except Exception:
                pass
        try:
            return image_path.relative_to(Path(self._images_dir_text())).as_posix()
        except Exception:
            return image_path.name

    def _mask_preview_config_from_controls(self) -> MaskPreviewConfig:
        mode = self.mask_mode() if hasattr(self, "dataset_mask_mode_combo") else DATASET_MASK_GENERATE_TRAINING
        if mode in {DATASET_MASK_CONVERT_SFM, DATASET_MASK_REUSE_EXISTING}:
            return MaskPreviewConfig(
                use_yolo=True,
                use_stitch=False,
                use_overexposure=False,
                use_sky=False,
                masks_dir=self._masks_dir_text(),
                settings_key=("dataset-mask", mode, self._images_dir_text(), self._masks_dir_text()),
            )
        if mode == DATASET_MASK_NONE:
            return MaskPreviewConfig(
                use_yolo=False,
                use_stitch=False,
                use_overexposure=False,
                use_sky=False,
                masks_dir="",
                settings_key=("dataset-mask", mode, self._images_dir_text()),
            )
        return super()._mask_preview_config_from_controls()

    def _sync_projection_from_project(self, *, preserve_user_quality: bool = False) -> None:
        self._set_projection(self._dataset_projection, sync_yolo_quality=not preserve_user_quality)
        self._projection_mixed = False
        self._projection_source = "dataset"
        self._image_projection_map = {}

    def _refresh_mask_source_options(self) -> None:
        self._syncing_mask_source_combo = True
        try:
            self.mask_source_combo.clear()
            self.mask_source_combo.addItem(i18n.t("MASK_SOURCE_ALL"), MASK_SOURCE_ALL)
            self.mask_source_combo.setCurrentIndex(0)
            self.mask_source_combo.setEnabled(False)
        finally:
            self._syncing_mask_source_combo = False

    def _ensure_no_pending_drop_images(self) -> None:
        return

    def _ensure_no_untracked_images(self) -> None:
        return

    def _record_mask_outputs(
        self,
        image_paths: list[Path],
        *,
        mode: str,
        settings: dict | None,
        phases: list[str],
        run_id: str | None = None,
    ) -> None:
        return

    def _attach_mask_paths_command(self) -> tuple[str, object]:
        return self._mask_paths_command(clear=False)

    def _clear_mask_paths_command(self) -> tuple[str, object]:
        return self._mask_paths_command(clear=True)

    def _mask_paths_command(self, *, clear: bool) -> tuple[str, object]:
        scene = Path(self.scene_dir) if self.scene_dir else self._dataset_root().parent
        payload = attach_dataset_masks_job(
            dataset_root=self._dataset_root(),
            masks_dir=self._masks_dir_text(),
            clear=clear,
        )
        job_path = jobs_dir(scene) / ("dataset_clear_mask_paths_job.json" if clear else "dataset_attach_mask_paths_job.json")
        write_dataset_job(job_path, payload)
        return "dataset_mask_paths", dataset_app_job(payload, job_path)

    def _dataset_mask_path_readiness(self) -> tuple[bool, str]:
        if not self.scene_dir:
            return False, i18n.t("SCENE_REQUIRED_ACTION_HINT")
        dataset_root = self._dataset_root()
        if not dataset_root.is_dir():
            return False, i18n.t("DATASET_MASK_READY_NO_DATASET")
        if find_nerf_transforms_path(dataset_root) is None:
            return False, i18n.t("DATASET_MASK_READY_NO_TRANSFORMS")
        if self.mask_mode() == DATASET_MASK_REUSE_EXISTING and not Path(self._masks_dir_text()).is_dir():
            return False, i18n.t("DATASET_MASK_READY_NO_MASKS")
        return True, i18n.t("MASK_READY_OK")

    def _on_dataset_mask_mode_changed(self) -> None:
        self._sync_preview_roots_for_mode()
        self._sync_preview_actions_for_mode()
        self._update_task_controls()
        self._update_ready_status()

    def _run_current_image_reprocess(self) -> None:
        if self.mask_mode() != DATASET_MASK_GENERATE_TRAINING:
            return
        super()._run_current_image_reprocess()

    def _sync_preview_roots_for_mode(self) -> None:
        if not hasattr(self, "mask_preview"):
            return
        self._invalidate_scene_inventory_cache()
        self._invalidate_readiness_cache()
        self._on_images_dir_changed(self._images_dir_text())

    def _sync_preview_actions_for_mode(self) -> None:
        if not hasattr(self, "mask_preview"):
            return
        can_save_reprocess = self.mask_mode() == DATASET_MASK_GENERATE_TRAINING
        self.mask_preview.reprocess_current_btn.setVisible(can_save_reprocess)
        self.mask_preview.reprocess_current_btn.setEnabled(can_save_reprocess)
