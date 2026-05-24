"""Step 3 scene/source/projection helpers."""

from __future__ import annotations

from pathlib import Path

from core.apply_frame_decisions import pending_drop_image_paths, untracked_image_paths
from core.mask_refresh_plan import MASK_SCOPE_MISSING, build_mask_refresh_plan, normalize_mask_scope
from core.mask_source_scope import MASK_SOURCE_ALL, build_mask_source_options, filter_images_by_source
from core.scene_inventory import (
    PROJECTION_EQUIRECTANGULAR,
    PROJECTION_NORMAL,
    PROJECTION_UNKNOWN,
    build_scene_inventory,
)
from core.scene_layout import scene_images_dir, scene_masks_dir, selected_frames_path
from core.scene_project import scene_relative
from gui import i18n
from gui.steps.step3_mask_plan import (
    MASK_TASK_CUSTOM,
    MASK_TASK_OVEREXPOSURE,
    MASK_TASK_STITCH,
    MASK_TASK_YOLO,
)

_PROJECTION_EQUIRECT = "equirect"
_PROJECTION_NORMAL = "normal"


class Step3MaskSceneMixin:
    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        if path:
            scene = Path(path)
            self.images_path_label.setText(str(scene_images_dir(scene)))
            self.masks_path_label.setText(str(scene_masks_dir(scene)))
        else:
            self.images_path_label.setText("-")
            self.masks_path_label.setText("-")
        self._sync_projection_from_project()
        self._refresh_mask_source_options()
        self._on_images_dir_changed(self._images_dir_text())
        self._render_mask_preview()
        self._update_ready_status()

    def primary_action_text(self) -> str:
        return i18n.t("GENERATE")

    def primary_action_tooltip(self) -> str:
        ready, reason = self._readiness()
        return i18n.tip("RUN_MASKS") if ready else reason

    def primary_action_enabled(self) -> bool:
        ready, _reason = self._readiness()
        return ready

    def on_activated(self) -> None:
        self._refresh_mask_source_options()
        self.mask_preview.refresh_image_list(prefer_current=True)
        self._render_mask_preview()
        self._update_ready_status()

    def _images_dir_text(self) -> str:
        if not self.scene_dir:
            return ""
        return str(scene_images_dir(Path(self.scene_dir)))

    def _masks_dir_text(self) -> str:
        if not self.scene_dir:
            return ""
        return str(scene_masks_dir(Path(self.scene_dir)))

    def _selected_mask_tasks(self) -> list[str]:
        requested_steps = [MASK_TASK_YOLO]
        if self._has_equirect_images() and self.run_stitch_cb.isChecked():
            requested_steps.append(MASK_TASK_STITCH)
        if self.run_overexp_cb.isChecked():
            requested_steps.append(MASK_TASK_OVEREXPOSURE)
        if self.run_custom_cb.isChecked():
            requested_steps.append(MASK_TASK_CUSTOM)
        return requested_steps

    def _mask_scope(self) -> str:
        return normalize_mask_scope(str(self.mask_scope_combo.currentData() or MASK_SCOPE_MISSING))

    def _selected_mask_source_key(self) -> str:
        return str(self.mask_source_combo.currentData() or MASK_SOURCE_ALL)

    def _on_mask_source_changed(self) -> None:
        if self._syncing_mask_source_combo:
            return
        self._render_mask_preview()
        self._update_ready_status()

    def _projection(self) -> str:
        return self._project_projection

    def _set_projection(self, projection: str) -> None:
        if projection not in {_PROJECTION_EQUIRECT, _PROJECTION_NORMAL}:
            projection = _PROJECTION_EQUIRECT
        self._project_projection = projection
        self.yolo_level_combo.setCurrentIndex(0 if projection == _PROJECTION_NORMAL else 1)
        if hasattr(self, "mask_preview"):
            self._update_preview_projection_enabled()
        self._update_task_controls()

    def _sync_projection_from_project(self) -> None:
        if not self.scene_dir:
            self._projection_mixed = False
            self._projection_source = "default"
            self._image_projection_map = {}
            self._set_projection(_PROJECTION_EQUIRECT)
            return
        inventory = build_scene_inventory(Path(self.scene_dir))
        self._image_projection_map = {image.rel_path: image.projection for image in inventory.images}
        projections = {image.projection for image in inventory.images if image.projection != PROJECTION_UNKNOWN}
        self._projection_mixed = len(projections) > 1
        self._projection_source = "project" if inventory.images else "default"
        if self._projection_mixed or PROJECTION_EQUIRECTANGULAR in projections or not projections:
            self._set_projection(_PROJECTION_EQUIRECT)
        else:
            self._set_projection(_PROJECTION_NORMAL)

    def _refresh_mask_source_options(self) -> None:
        current = self._selected_mask_source_key() if hasattr(self, "mask_source_combo") else MASK_SOURCE_ALL
        self._syncing_mask_source_combo = True
        try:
            self.mask_source_combo.clear()
            self.mask_source_combo.addItem(i18n.t("MASK_SOURCE_ALL"), MASK_SOURCE_ALL)
            if self.scene_dir and Path(self.scene_dir).is_dir():
                inventory = build_scene_inventory(Path(self.scene_dir))
                for option in build_mask_source_options(Path(self.scene_dir), inventory.images):
                    label = i18n.t("MASK_SOURCE_ITEM").format(label=option.label, count=option.image_count)
                    self.mask_source_combo.addItem(label, option.key)
            index = self.mask_source_combo.findData(current)
            self.mask_source_combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self._syncing_mask_source_combo = False

    def _refresh_image_projection_map(self) -> None:
        if not self.scene_dir:
            self._image_projection_map = {}
            return
        scene = Path(self.scene_dir)
        inventory = build_scene_inventory(scene)
        self._image_projection_map = {image.rel_path: image.projection for image in inventory.images}

    def _scene_image_paths(self) -> list[Path]:
        if not self.scene_dir:
            return []
        images = Path(self._images_dir_text())
        inventory = build_scene_inventory(Path(self.scene_dir), images_dir=images)
        return [image.path for image in inventory.images]

    def _projection_key_for_image(self, image_path: Path) -> str:
        if not self.scene_dir:
            return image_path.name
        return scene_relative(Path(self.scene_dir), image_path).replace("\\", "/")

    def _projection_for_image(self, image_path: Path | None) -> str:
        if image_path is None:
            return self._projection()
        projection = self._image_projection_map.get(self._projection_key_for_image(image_path), "")
        if projection == PROJECTION_EQUIRECTANGULAR:
            return _PROJECTION_EQUIRECT
        if projection == PROJECTION_NORMAL:
            return _PROJECTION_NORMAL
        return self._projection()

    def _has_equirect_images(self) -> bool:
        if self._projection_mixed:
            return any(value == PROJECTION_EQUIRECTANGULAR for value in self._image_projection_map.values())
        return self._projection() == _PROJECTION_EQUIRECT

    def _update_preview_projection_enabled(self) -> None:
        image_path = self.mask_preview.current_image_path() if hasattr(self, "mask_preview") else None
        projection = self._projection_for_image(image_path)
        self.mask_preview.set_perspective_enabled(projection == _PROJECTION_EQUIRECT)

    def _scene_csv_path(self) -> Path:
        return selected_frames_path(Path(self.scene_dir))

    def _has_image_files(self) -> bool:
        if not self.scene_dir:
            return False
        images = Path(self._images_dir_text())
        if not images.is_dir():
            return False
        return build_scene_inventory(Path(self.scene_dir), images_dir=images).image_count > 0

    def _readiness(self) -> tuple[bool, str]:
        if not self.scene_dir:
            return False, i18n.t("SCENE_REQUIRED_ACTION_HINT")
        if not Path(self.scene_dir).is_dir():
            return False, i18n.t("MASK_READY_SCENE_NOT_FOUND")
        images = Path(self._images_dir_text())
        if not images.is_dir():
            return False, i18n.t("MASK_READY_NO_IMAGES_DIR")
        if not self._has_image_files():
            return False, i18n.t("MASK_READY_NO_IMAGES")
        if self.run_custom_cb.isChecked():
            custom_mask = self._custom_mask_path_text()
            if not custom_mask:
                return False, i18n.t("CUSTOM_MASK_REQUIRED")
            if not Path(custom_mask).is_file():
                return False, i18n.t("CUSTOM_MASK_NOT_FOUND").format(path=custom_mask)
        if not self._selected_mask_tasks():
            return False, i18n.t("MASK_TASK_REQUIRED")
        if self._projection_mixed:
            return True, i18n.t("MASK_READY_MIXED_IMAGE_TYPES")
        if not self._scene_csv_path().is_file():
            return True, i18n.t("MASK_READY_EXTERNAL_IMAGES")
        return True, i18n.t("MASK_READY_OK")

    def _update_ready_status(self) -> None:
        self._readiness()
        self.primary_action_state_changed.emit()

    def _source_filtered_image_paths(self) -> list[Path]:
        if not self.scene_dir:
            return []
        inventory = build_scene_inventory(Path(self.scene_dir))
        images = filter_images_by_source(inventory.images, self._selected_mask_source_key())
        return [image.path for image in images]

    def _mask_target_paths(self, image_paths: list[Path], *, settings: dict) -> list[Path]:
        if not self.scene_dir:
            return []
        plan = build_mask_refresh_plan(
            scene_dir=Path(self.scene_dir),
            image_paths=image_paths,
            mask_path_for_image=self._mask_output_path_for_image,
            settings=settings,
            scope=self._mask_scope(),
        )
        return list(plan.targets)

    def _ensure_no_pending_drop_images(self) -> None:
        if not self.scene_dir:
            return
        images = self._images_dir_text()
        if not images:
            return
        scene_dir = Path(self.scene_dir)
        csv_path = selected_frames_path(scene_dir)
        if not csv_path.exists():
            return

        pending = pending_drop_image_paths(scene_dir, images_dir=Path(images))
        if not pending:
            return

        preview = "\n".join(f"- {path.name}" for path in pending[:5])
        if len(pending) > 5:
            preview += f"\n- ... +{len(pending) - 5}"
        raise ValueError(i18n.t("MASK_PENDING_DROPS_ERROR").format(n=len(pending), files=preview))

    def _ensure_no_untracked_images(self) -> None:
        if not self.scene_dir:
            return
        images = self._images_dir_text()
        if not images:
            return
        scene_dir = Path(self.scene_dir)
        csv_path = selected_frames_path(scene_dir)
        if not csv_path.exists():
            return

        untracked = untracked_image_paths(scene_dir, images_dir=Path(images))
        if not untracked:
            return

        preview = "\n".join(f"- {path.name}" for path in untracked[:5])
        if len(untracked) > 5:
            preview += f"\n- ... +{len(untracked) - 5}"
        raise ValueError(i18n.t("MASK_UNTRACKED_IMAGES_ERROR").format(n=len(untracked), files=preview))
