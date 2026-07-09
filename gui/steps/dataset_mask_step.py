"""Step 5 training-mask settings built from the Step 3 mask UI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QWidget

from core.app_job import dataset_app_job
from core.dataset_job_spec import attach_dataset_masks_job, sync_dataset_masks_job, write_dataset_job
from core.dataset_mask_policy import (
    DATASET_MASK_CONVERT_SFM,
    DATASET_MASK_GENERATE_TRAINING,
    DATASET_MASK_NONE,
    DATASET_MASK_REUSE_EXISTING,
    normalize_dataset_mask_mode,
)
from core.mask_refresh_plan import MASK_SCOPE_ALL, MASK_SCOPE_STALE
from core.mask_source_scope import MASK_SOURCE_ALL
from core.nerf_dataset_paths import find_nerf_transforms_path
from core.scene_layout import jobs_dir, scene_images_dir, scene_masks_dir, step4_meta_dir
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
    MaskCommandSpec,
)
from gui.steps.step3_mask_records import record_mask_outputs

_PROJECTION_EQUIRECT = "equirect"
_PROJECTION_NORMAL = "normal"


class DatasetMaskStep(MaskStep):
    """Generate masks in the active Step 5 dataset output instead of scene/masks."""

    def __init__(
        self,
        base_dir: Path,
        *,
        dataset_root_provider: Callable[[], Path],
        source_images_dir_provider: Callable[[], Path] | None = None,
        source_masks_dir_provider: Callable[[], Path] | None = None,
        generated_source_masks_dir_provider: Callable[[], Path] | None = None,
        link_mask_paths: bool = True,
        mode_tip_keys: Mapping[str, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        self._dataset_root_provider = dataset_root_provider
        self._source_images_dir_provider = source_images_dir_provider
        self._source_masks_dir_provider = source_masks_dir_provider
        self._generated_source_masks_dir_provider = generated_source_masks_dir_provider
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
        self._sync_preview_messages_for_mode()
        self._sync_preview_roots_for_mode()
        self._sync_preview_actions_for_mode()

    def _install_dataset_mode_controls(self) -> None:
        for field_name, label_name in (
            ("images_path_row", "images_path_row_label"),
            ("masks_path_label", "masks_path_row_label"),
            ("mask_source_combo", "mask_source_row_label"),
            ("mask_scope_combo", "mask_scope_row_label"),
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
            (DATASET_MASK_NONE, "DATASET_MASK_MODE_NONE"),
        ):
            self.dataset_mask_mode_combo.addItem(i18n.t(label_key), mode)
            index = self.dataset_mask_mode_combo.count() - 1
            self.dataset_mask_mode_combo.setItemData(index, i18n.tip(self._mode_tip_key(mode)), Qt.ToolTipRole)
        self.dataset_mask_mode_combo.currentIndexChanged.connect(lambda _index: self._on_dataset_mask_mode_changed())
        add_tooltip_row(form, i18n.t("DATASET_MASK_MODE"), self.dataset_mask_mode_combo, i18n.tip("DATASET_MASK_MODE"))

        self.dataset_mask_rebuild_all_cb = QCheckBox(i18n.t("DATASET_MASK_REBUILD_ALL"))
        self.dataset_mask_rebuild_all_cb.setToolTip(i18n.tip("DATASET_MASK_REBUILD_ALL"))
        self.dataset_mask_rebuild_all_cb.toggled.connect(lambda _checked: self._update_ready_status())
        add_tooltip_row(
            form,
            i18n.t("DATASET_MASK_REBUILD_SCOPE"),
            self.dataset_mask_rebuild_all_cb,
            i18n.tip("DATASET_MASK_REBUILD_SCOPE"),
        )
        self.dataset_mask_rebuild_all_label = form.labelForField(self.dataset_mask_rebuild_all_cb)
        self.settings_layout.insertLayout(0, form)
        self._sync_dataset_generation_controls_for_mode()

    def _mode_tip_key(self, mode: str) -> str:
        return self._mode_tip_keys.get(mode, f"DATASET_MASK_MODE_{mode.upper()}")

    def set_dataset_projection(self, projection: str) -> None:
        projection = projection if projection in {_PROJECTION_EQUIRECT, _PROJECTION_NORMAL} else _PROJECTION_NORMAL
        self._dataset_projection = projection
        self._sync_projection_from_project(preserve_user_quality=True)

    def mask_mode(self) -> str:
        return normalize_dataset_mask_mode(self.dataset_mask_mode_combo.currentData())

    def set_mask_mode(self, mode: object) -> None:
        normalized = normalize_dataset_mask_mode(mode)
        index = self.dataset_mask_mode_combo.findData(normalized)
        if index >= 0:
            self.dataset_mask_mode_combo.setCurrentIndex(index)
            self._on_dataset_mask_mode_changed()

    def _mask_scope(self) -> str:
        if (
            hasattr(self, "dataset_mask_rebuild_all_cb")
            and self.mask_mode() == DATASET_MASK_GENERATE_TRAINING
            and self.dataset_mask_rebuild_all_cb.isChecked()
        ):
            return MASK_SCOPE_ALL
        return MASK_SCOPE_STALE

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
            return self.build_source_mask_commands()
        if not self._link_mask_paths:
            return []
        if mode == DATASET_MASK_REUSE_EXISTING:
            return [self._attach_mask_paths_command()]
        if mode == DATASET_MASK_NONE:
            return [self._clear_mask_paths_command()]
        raise ValueError(i18n.t("DATASET_MASK_RUN_DATASET_ACTION"))

    def build_followup_commands(self, *, require_existing_images: bool) -> StepCommandQueue:
        _ = require_existing_images
        mode = self.mask_mode()
        if mode in {DATASET_MASK_CONVERT_SFM, DATASET_MASK_GENERATE_TRAINING}:
            return []
        if not self._link_mask_paths:
            return []
        if mode == DATASET_MASK_REUSE_EXISTING:
            return [self._attach_mask_paths_command()]
        if mode == DATASET_MASK_NONE:
            return [self._clear_mask_paths_command()]
        return []

    def build_source_mask_commands(self) -> StepCommandQueue:
        if self.mask_mode() != DATASET_MASK_GENERATE_TRAINING:
            return []
        return [*super().build_commands()]

    def build_output_mask_sync_command(self) -> tuple[str, object]:
        source_masks = self.source_mask_dir_for_dataset(require_existing=False)
        if source_masks is None:
            raise ValueError(i18n.t("DATASET_MASK_RUN_DATASET_ACTION"))
        scene = Path(self.scene_dir) if self.scene_dir else self._dataset_root().parent
        payload = sync_dataset_masks_job(
            dataset_root=self._dataset_root(),
            source_masks_dir=source_masks,
            attach=self._link_mask_paths,
        )
        job_path = jobs_dir(scene) / "dataset_sync_masks_job.json"
        write_dataset_job(job_path, payload)
        return "dataset_mask_paths", dataset_app_job(payload, job_path)

    def _command_from_mask_spec(self, spec: MaskCommandSpec) -> list[str]:
        if spec.command == MASK_COMMAND_YOLO:
            return self._build_yolo_cmd(projection=spec.projection, image_list=spec.image_list)
        if spec.command == MASK_COMMAND_INIT:
            return self._build_init_masks_cmd(image_list=spec.image_list)
        if spec.command == MASK_COMMAND_STITCH:
            return self._build_stitch_cmd(image_list=spec.image_list)
        if spec.command == MASK_COMMAND_OVEREXPOSURE:
            return self._build_overexposure_cmd(
                replace=spec.replace,
                merge_mode=spec.merge_mode,
                image_list=spec.image_list,
            )
        if spec.command == MASK_COMMAND_CUSTOM:
            return self._build_custom_cmd(
                replace=spec.replace,
                merge_mode=spec.merge_mode,
                image_list=spec.image_list,
            )
        raise ValueError(f"Unknown mask command spec: {spec.command}")

    def _dataset_root(self) -> Path:
        return Path(self._dataset_root_provider())

    def _source_images_dir(self) -> Path:
        if self._source_images_dir_provider is not None:
            return Path(self._source_images_dir_provider())
        if self.scene_dir:
            return scene_images_dir(Path(self.scene_dir))
        return Path()

    def _source_masks_dir(self) -> Path:
        if self._source_masks_dir_provider is not None:
            return Path(self._source_masks_dir_provider())
        if self.scene_dir:
            return scene_masks_dir(Path(self.scene_dir))
        return Path()

    def generated_source_masks_dir(self) -> Path:
        if self._generated_source_masks_dir_provider is not None:
            return Path(self._generated_source_masks_dir_provider())
        if self.scene_dir:
            return step4_meta_dir(Path(self.scene_dir)) / "dataset_masks" / "training_source_masks"
        return Path()

    def writes_dataset_masks_from_source(self) -> bool:
        return self.mask_mode() in {DATASET_MASK_CONVERT_SFM, DATASET_MASK_GENERATE_TRAINING}

    def source_mask_dir_for_dataset(self, *, require_existing: bool = True) -> Path | None:
        mode = self.mask_mode()
        if mode == DATASET_MASK_CONVERT_SFM:
            path = self._source_masks_dir()
        elif mode == DATASET_MASK_GENERATE_TRAINING:
            path = self.generated_source_masks_dir()
        else:
            return None
        if require_existing and not path.is_dir():
            return None
        return path

    def _mode_uses_source_images(self) -> bool:
        if not hasattr(self, "dataset_mask_mode_combo"):
            return True
        return self.mask_mode() in {DATASET_MASK_CONVERT_SFM, DATASET_MASK_GENERATE_TRAINING}

    def _images_dir_for_scene(self, _scene: Path) -> Path:
        if self._mode_uses_source_images():
            return self._source_images_dir()
        return self._dataset_root() / "images"

    def _masks_dir_for_scene(self, _scene: Path) -> Path:
        mode = self.mask_mode() if hasattr(self, "dataset_mask_mode_combo") else DATASET_MASK_CONVERT_SFM
        if mode == DATASET_MASK_CONVERT_SFM:
            return self._source_masks_dir()
        if mode == DATASET_MASK_GENERATE_TRAINING:
            return self.generated_source_masks_dir()
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
        return replace(
            super()._mask_preview_config_from_controls(),
            missing_primary_status=i18n.t("DATASET_MASK_PREVIEW_TRAINING_PENDING"),
        )

    def _mask_settings_snapshot(self) -> dict:
        return self._with_dataset_mask_settings(super()._mask_settings_snapshot())

    def _sync_projection_from_project(self, *, preserve_user_quality: bool = False) -> None:
        if self._mode_uses_source_images():
            super()._sync_projection_from_project(preserve_user_quality=preserve_user_quality)
            return
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
        if not self.scene_dir or not image_paths:
            return
        scene = Path(self.scene_dir)
        effective_settings = self._with_dataset_mask_settings(settings or self._mask_settings_snapshot())
        record_mask_outputs(
            scene,
            image_paths,
            mode=f"dataset_{mode}",
            settings=effective_settings,
            phases=phases,
            mask_path_for_image=self._mask_output_path_for_image,
            run_id=run_id,
            run_id_factory=self._new_mask_run_id,
        )

    def on_queue_finished(self, success: bool) -> None:
        if not self._mask_batch_phases and not self._mask_batch_targets:
            return
        super().on_queue_finished(success)

    def _with_dataset_mask_settings(self, settings: dict) -> dict:
        effective = dict(settings)
        scene = Path(self.scene_dir) if self.scene_dir else self._dataset_root().parent
        effective["dataset_mask"] = {
            "mode": self.mask_mode(),
            "dataset_root": _scene_relative_or_path(scene, self._dataset_root()),
            "projection": self._dataset_projection,
            "source_images_dir": _scene_relative_or_path(scene, self._source_images_dir()),
            "source_masks_dir": _scene_relative_or_path(scene, self._source_masks_dir()),
            "generated_source_masks_dir": _scene_relative_or_path(scene, self.generated_source_masks_dir()),
        }
        return effective

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
        self._sync_preview_messages_for_mode()
        self._sync_preview_roots_for_mode()
        self._sync_preview_actions_for_mode()
        self._sync_dataset_generation_controls_for_mode()
        self._sync_projection_from_project(preserve_user_quality=True)
        self._update_task_controls()
        self._update_ready_status()

    def _run_current_image_reprocess(self) -> None:
        if self.mask_mode() != DATASET_MASK_GENERATE_TRAINING:
            return
        super()._run_current_image_reprocess()

    def _sync_preview_roots_for_mode(self) -> None:
        if not hasattr(self, "mask_preview"):
            return
        self._sync_preview_messages_for_mode()
        self._invalidate_scene_inventory_cache()
        self._invalidate_readiness_cache()
        self._on_images_dir_changed(self._images_dir_text())

    def _sync_preview_actions_for_mode(self) -> None:
        if not hasattr(self, "mask_preview"):
            return
        can_save_reprocess = self.mask_mode() == DATASET_MASK_GENERATE_TRAINING
        self.mask_preview.reprocess_current_btn.setVisible(can_save_reprocess)
        self.mask_preview.reprocess_current_btn.setEnabled(can_save_reprocess)

    def _sync_dataset_generation_controls_for_mode(self) -> None:
        if not hasattr(self, "dataset_mask_rebuild_all_cb"):
            return
        generate = self.mask_mode() == DATASET_MASK_GENERATE_TRAINING
        self.dataset_mask_rebuild_all_cb.setVisible(generate)
        self.dataset_mask_rebuild_all_cb.setEnabled(generate)
        if self.dataset_mask_rebuild_all_label is not None:
            self.dataset_mask_rebuild_all_label.setVisible(generate)

    def _update_task_controls(self) -> None:
        super()._update_task_controls()
        if not hasattr(self, "dataset_mask_mode_combo"):
            return
        generate = self.mask_mode() == DATASET_MASK_GENERATE_TRAINING
        self.mask_scope_combo.setVisible(False)
        if getattr(self, "mask_scope_row_label", None) is not None:
            self.mask_scope_row_label.setVisible(False)
        self._sync_dataset_generation_controls_for_mode()
        self.mask_settings_tabs.setEnabled(generate)
        self.run_stitch_cb.setEnabled(generate and self.run_stitch_cb.isEnabled())
        self.run_overexp_cb.setEnabled(generate)
        self.run_custom_cb.setEnabled(generate)
        self.custom_mask_browse_btn.setEnabled(generate)
        self.custom_mask_clear_btn.setEnabled(generate and bool(self._custom_mask_path_text()))

    def _sync_preview_messages_for_mode(self) -> None:
        if not hasattr(self, "mask_preview"):
            return
        mode = self.mask_mode() if hasattr(self, "dataset_mask_mode_combo") else DATASET_MASK_GENERATE_TRAINING
        if mode in {DATASET_MASK_CONVERT_SFM, DATASET_MASK_GENERATE_TRAINING}:
            self.mask_preview.set_empty_messages(
                no_scene=i18n.t("MASK_PREVIEW_NO_SCENE_HELP"),
                empty=i18n.t(
                    "DATASET_MASK_PREVIEW_GENERATE_EMPTY"
                    if mode == DATASET_MASK_GENERATE_TRAINING
                    else "MASK_PREVIEW_EMPTY_HELP"
                ),
            )
            return
        self.mask_preview.set_empty_messages(
            no_scene=i18n.t("DATASET_MASK_PREVIEW_NO_DATASET"),
            empty=i18n.t("DATASET_MASK_PREVIEW_EMPTY"),
        )


def _scene_relative_or_path(scene: Path, path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(scene.resolve(strict=False)).as_posix()
    except (OSError, ValueError):
        return str(path)
