"""Step 3 scene/source/projection helpers."""

from __future__ import annotations

import os
from pathlib import Path

from core.apply_frame_decisions import pending_drop_image_paths, untracked_image_paths
from core.mask_refresh_plan import MASK_SCOPE_MISSING, build_mask_refresh_plan, normalize_mask_scope
from core.mask_source_scope import MASK_SOURCE_ALL, build_mask_source_options, filter_images_by_source
from core.scene_inventory import (
    PROJECTION_EQUIRECTANGULAR,
    PROJECTION_NORMAL,
    PROJECTION_UNKNOWN,
    SceneInventory,
    build_fast_scene_inventory,
    build_scene_inventory,
)
from core.scene_layout import (
    normal_camera_defaults_path,
    scene_images_dir,
    scene_masks_dir,
    selected_frames_path,
    source_image_sets_path,
    source_videos_path,
)
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
        self._invalidate_scene_inventory_cache()
        self._invalidate_readiness_cache()
        if path:
            scene = Path(path)
            self.images_path_label.setText(str(self._images_dir_for_scene(scene)))
            self.masks_path_label.setText(str(self._masks_dir_for_scene(scene)))
        else:
            self.images_path_label.setText("-")
            self.masks_path_label.setText("-")
        self._sync_projection_from_project()
        self._refresh_mask_source_options()
        self._on_images_dir_changed(self._images_dir_text())
        self._render_mask_preview()
        self._update_ready_status()
        self._scene_inventory_synced_on_scene_change = self._scene_inventory_cache is not None

    def primary_action_text(self) -> str:
        return i18n.t("GENERATE")

    def primary_action_tooltip(self) -> str:
        ready, reason = self._readiness()
        return i18n.tip("RUN_MASKS") if ready else reason

    def primary_action_enabled(self) -> bool:
        ready, _reason = self._readiness()
        return ready

    def on_activated(self) -> None:
        if self._scene_inventory_synced_on_scene_change and self._scene_inventory_cache_is_current():
            self._scene_inventory_synced_on_scene_change = False
        else:
            self._scene_inventory_synced_on_scene_change = False
            self._refresh_scene_inventory_cache()
        self._refresh_mask_source_options()
        self.mask_preview.refresh_image_list(prefer_current=True)
        self._render_mask_preview()
        self._update_ready_status()

    def _images_dir_text(self) -> str:
        if not self.scene_dir:
            return ""
        return str(self._images_dir_for_scene(Path(self.scene_dir)))

    def _masks_dir_text(self) -> str:
        if not self.scene_dir:
            return ""
        return str(self._masks_dir_for_scene(Path(self.scene_dir)))

    def _images_dir_for_scene(self, scene: Path) -> Path:
        return scene_images_dir(scene)

    def _masks_dir_for_scene(self, scene: Path) -> Path:
        return scene_masks_dir(scene)

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

    def _set_projection(self, projection: str, *, sync_yolo_quality: bool = True) -> None:
        if projection not in {_PROJECTION_EQUIRECT, _PROJECTION_NORMAL}:
            projection = _PROJECTION_EQUIRECT
        self._project_projection = projection
        if sync_yolo_quality:
            self.yolo_level_combo.setCurrentIndex(0 if projection == _PROJECTION_NORMAL else 1)
        if hasattr(self, "mask_preview"):
            self._update_preview_projection_enabled()
        self._update_task_controls()

    def _sync_projection_from_project(self, *, preserve_user_quality: bool = False) -> None:
        if not self.scene_dir:
            self._projection_mixed = False
            self._projection_source = "default"
            self._image_projection_map = {}
            self._set_projection(_PROJECTION_EQUIRECT, sync_yolo_quality=not preserve_user_quality)
            return
        inventory = self._cached_scene_inventory()
        self._image_projection_map = {image.rel_path: image.projection for image in inventory.images}
        projections = {image.projection for image in inventory.images if image.projection != PROJECTION_UNKNOWN}
        self._projection_mixed = len(projections) > 1
        self._projection_source = "project" if inventory.images else "default"
        if not projections and preserve_user_quality:
            self._set_projection(self._projection(), sync_yolo_quality=False)
            return
        if self._projection_mixed or PROJECTION_EQUIRECTANGULAR in projections or not projections:
            self._set_projection(_PROJECTION_EQUIRECT, sync_yolo_quality=not preserve_user_quality)
        else:
            self._set_projection(_PROJECTION_NORMAL, sync_yolo_quality=not preserve_user_quality)

    def _refresh_mask_source_options(self) -> None:
        current = self._selected_mask_source_key() if hasattr(self, "mask_source_combo") else MASK_SOURCE_ALL
        self._syncing_mask_source_combo = True
        try:
            self.mask_source_combo.clear()
            self.mask_source_combo.addItem(i18n.t("MASK_SOURCE_ALL"), MASK_SOURCE_ALL)
            if self.scene_dir and Path(self.scene_dir).is_dir():
                inventory = self._cached_scene_inventory()
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
        inventory = self._cached_scene_inventory()
        self._image_projection_map = {image.rel_path: image.projection for image in inventory.images}

    def _scene_image_paths(self) -> list[Path]:
        if not self.scene_dir:
            return []
        images = Path(self._images_dir_text())
        inventory = self._cached_scene_inventory(images_dir=images)
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
        return self._cached_scene_inventory(images_dir=images).image_count > 0

    def _readiness(self) -> tuple[bool, str]:
        if self._readiness_cache is not None:
            return self._readiness_cache
        if not self.scene_dir:
            return self._set_readiness_cache(False, i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        if not Path(self.scene_dir).is_dir():
            return self._set_readiness_cache(False, i18n.t("MASK_READY_SCENE_NOT_FOUND"))
        images = Path(self._images_dir_text())
        if not images.is_dir():
            return self._set_readiness_cache(False, i18n.t("MASK_READY_NO_IMAGES_DIR"))
        if not self._has_image_files():
            return self._set_readiness_cache(False, i18n.t("MASK_READY_NO_IMAGES"))
        if self.run_custom_cb.isChecked():
            custom_mask = self._custom_mask_path_text()
            if not custom_mask:
                return self._set_readiness_cache(False, i18n.t("CUSTOM_MASK_REQUIRED"))
            if not Path(custom_mask).is_file():
                return self._set_readiness_cache(False, i18n.t("CUSTOM_MASK_NOT_FOUND").format(path=custom_mask))
        if not self._selected_mask_tasks():
            return self._set_readiness_cache(False, i18n.t("MASK_TASK_REQUIRED"))
        if self._projection_mixed:
            return self._set_readiness_cache(True, i18n.t("MASK_READY_MIXED_IMAGE_TYPES"))
        if not self._scene_csv_path().is_file():
            return self._set_readiness_cache(True, i18n.t("MASK_READY_EXTERNAL_IMAGES"))
        return self._set_readiness_cache(True, i18n.t("MASK_READY_OK"))

    def _update_ready_status(self) -> None:
        self._invalidate_readiness_cache()
        self._readiness()
        self.primary_action_state_changed.emit()

    def _source_filtered_image_paths(self) -> list[Path]:
        if not self.scene_dir:
            return []
        inventory = self._cached_scene_inventory()
        images = filter_images_by_source(inventory.images, self._selected_mask_source_key())
        return [image.path for image in images]

    def _cached_scene_inventory(
        self,
        *,
        images_dir: Path | None = None,
        masks_dir: Path | None = None,
        refresh: bool = False,
        strict: bool = False,
    ) -> SceneInventory:
        scene = Path(self.scene_dir)
        images = images_dir or self._images_dir_for_scene(scene)
        masks = masks_dir or self._masks_dir_for_scene(scene)
        key = self._scene_inventory_cache_key_for(scene, images, masks, strict=strict)
        if refresh or self._scene_inventory_cache is None or self._scene_inventory_cache_key != key:
            builder = build_scene_inventory if strict else build_fast_scene_inventory
            self._scene_inventory_cache = builder(scene, images_dir=images, masks_dir=masks)
            self._scene_inventory_cache_key = key
            self._scene_inventory_refresh_token = self._scene_inventory_token(scene, images, masks)
        return self._scene_inventory_cache

    def _refresh_scene_inventory_cache(self, *, strict: bool = False) -> None:
        if self.scene_dir and Path(self.scene_dir).is_dir():
            scene = Path(self.scene_dir)
            images = self._images_dir_for_scene(scene)
            masks = self._masks_dir_for_scene(scene)
            key = self._scene_inventory_cache_key_for(scene, images, masks, strict=strict)
            token = self._scene_inventory_token(scene, images, masks)
            if (
                self._scene_inventory_cache is None
                or self._scene_inventory_cache_key != key
                or self._scene_inventory_refresh_token != token
            ):
                builder = build_scene_inventory if strict else build_fast_scene_inventory
                self._scene_inventory_cache = builder(scene, images_dir=images, masks_dir=masks)
                self._scene_inventory_cache_key = key
                self._scene_inventory_refresh_token = token
        else:
            self._invalidate_scene_inventory_cache()
        self._invalidate_readiness_cache()

    def _scene_inventory_cache_is_current(self) -> bool:
        if self._scene_inventory_cache is None or not self.scene_dir:
            return False
        scene = Path(self.scene_dir)
        if not scene.is_dir():
            return False
        images = self._images_dir_for_scene(scene)
        masks = self._masks_dir_for_scene(scene)
        return (
            self._scene_inventory_cache_key == self._scene_inventory_cache_key_for(scene, images, masks, strict=False)
            and self._scene_inventory_refresh_token == self._scene_inventory_token(scene, images, masks)
        )

    def _invalidate_scene_inventory_cache(self) -> None:
        self._scene_inventory_cache = None
        self._scene_inventory_cache_key = None
        self._scene_inventory_refresh_token = None
        self._scene_inventory_synced_on_scene_change = False

    def _invalidate_readiness_cache(self) -> None:
        self._readiness_cache = None

    def _set_readiness_cache(self, ready: bool, reason: str) -> tuple[bool, str]:
        self._readiness_cache = (ready, reason)
        return self._readiness_cache

    def _scene_inventory_cache_key_for(
        self,
        scene: Path,
        images: Path,
        masks: Path,
        *,
        strict: bool,
    ) -> tuple[str, str, str, str]:
        return (
            "strict" if strict else "fast",
            self._path_cache_key(scene),
            self._path_cache_key(images),
            self._path_cache_key(masks),
        )

    @staticmethod
    def _path_cache_key(path: Path) -> str:
        try:
            return str(path.resolve(strict=False)).replace("\\", "/").casefold()
        except OSError:
            return str(path).replace("\\", "/").casefold()

    def _scene_inventory_token(self, scene: Path, images: Path, masks: Path) -> tuple:
        metadata_paths = (
            selected_frames_path(scene),
            source_image_sets_path(scene),
            source_videos_path(scene),
            normal_camera_defaults_path(scene),
        )
        return (
            self._directory_tree_token(images),
            self._directory_tree_token(masks),
            tuple(self._path_stat_token(path) for path in metadata_paths),
        )

    def _directory_tree_token(self, root: Path) -> tuple:
        tokens = [self._path_stat_token(root)]
        if not root.is_dir():
            return tuple(tokens)
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames.sort(key=str.casefold)
                filenames.sort(key=str.casefold)
                tokens.append(
                    (
                        self._path_cache_key(Path(dirpath)),
                        tuple(filename.casefold() for filename in filenames),
                    )
                )
                for dirname in dirnames:
                    tokens.append(self._path_stat_token(Path(dirpath) / dirname))
        except OSError:
            pass
        return tuple(tokens)

    def _path_stat_token(self, path: Path) -> tuple[str, bool, int | None, int | None]:
        key = self._path_cache_key(path)
        try:
            stat = path.stat()
        except OSError:
            return key, False, None, None
        return key, True, int(stat.st_size), int(stat.st_mtime_ns)

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
