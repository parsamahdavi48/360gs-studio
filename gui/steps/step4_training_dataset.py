"""Step 4 training dataset, path, and image-count helpers."""

from __future__ import annotations

import os
from pathlib import Path

from core.nerf_dataset_paths import find_nerf_pointcloud_path, find_nerf_transforms_path, load_json_object
from core.scene_layout import project_path, scene_images_dir, scene_output_dir, step4_meta_dir
from core.scene_project import load_json
from core.workflow_artifacts import (
    DATASET_KIND_COLMAP_DATASET,
    DATASET_KIND_LICHTFELD_COLMAP,
    DATASET_KIND_NERF_JSON_PLY,
    latest_dataset_root,
)
from gui import i18n
from gui.steps.step4_contracts import (
    _LFS_STRATEGIES,
    _OUTPUT_SHAPE_EQUIRECT_3DGUT,
    _OUTPUT_SHAPE_PROJECTED,
    _SUPPORTED_TRAINING_IMAGE_EXTS,
)
from gui.steps.step4_settings import load_step4_export_settings
from gui.steps.training_backend_specs import (
    TRAINING_BACKEND_BRUSH as _TRAINING_BACKEND_BRUSH,
)
from gui.steps.training_backend_specs import (
    TRAINING_BACKEND_GSPLAT as _TRAINING_BACKEND_GSPLAT,
)
from gui.steps.training_backend_specs import (
    TRAINING_BACKEND_LICHTFELD as _TRAINING_BACKEND_LICHTFELD,
)
from gui.steps.training_backend_specs import (
    TRAINING_BACKEND_POSTSHOT as _TRAINING_BACKEND_POSTSHOT,
)
from gui.steps.training_backend_specs import (
    training_backend_default_executable,
)
from gui.steps.training_backends import TrainingDataset, lichtfeld_auto_steps_scaler


class Step4TrainingDatasetMixin:
    def _training_required_output_shape(self) -> str:
        backend = self._training_backend()
        if backend == _TRAINING_BACKEND_LICHTFELD:
            return _OUTPUT_SHAPE_EQUIRECT_3DGUT if self.lfs_gut_cb.isChecked() else _OUTPUT_SHAPE_PROJECTED
        if backend == _TRAINING_BACKEND_POSTSHOT:
            return _OUTPUT_SHAPE_PROJECTED
        if backend in {_TRAINING_BACKEND_BRUSH, _TRAINING_BACKEND_GSPLAT}:
            return _OUTPUT_SHAPE_PROJECTED
        return ""

    def _training_dataset_issue(self) -> str | None:
        required_shape = self._training_required_output_shape()
        return self._existing_training_dataset_issue(required_shape)

    def _existing_training_dataset_issue(self, required_shape: str) -> str | None:
        if not self._training_dataset_available():
            return i18n.t("TRAINING_REQUIRES_DATASET_OUTPUT")
        dataset = self._training_dataset()
        if not required_shape:
            return None
        actual_shape = self._training_dataset_export_shape(dataset.dataset_root)
        if required_shape == _OUTPUT_SHAPE_EQUIRECT_3DGUT:
            if actual_shape == _OUTPUT_SHAPE_PROJECTED:
                return i18n.t("TRAINING_DATASET_EXISTING_NOT_3DGUT")
            if not (dataset.dataset_root / "pointcloud.ply").is_file():
                return i18n.t("TRAINING_DATASET_3DGUT_NEEDS_PLY")
            return None
        if actual_shape == _OUTPUT_SHAPE_EQUIRECT_3DGUT:
            return i18n.t("TRAINING_DATASET_EXISTING_NOT_PROJECTED")
        if (
            self._training_backend() == _TRAINING_BACKEND_POSTSHOT
            and (self.postshot_camera_poses_combo.currentData() or "import") == "import"
            and dataset.colmap_sparse_dir is None
            and (dataset.transforms_json is None or not dataset.transforms_json.is_file())
        ):
            return i18n.t("POSTSHOT_IMPORT_POSES_NOT_FOUND")
        if self._training_backend() == _TRAINING_BACKEND_GSPLAT and dataset.colmap_sparse_dir is None:
            return i18n.t("GSPLAT_COLMAP_DATASET_NOT_FOUND")
        return None

    def _training_dataset_export_shape(self, dataset_root: Path) -> str:
        if not self.scene_dir:
            return ""
        settings = load_step4_export_settings(Path(self.scene_dir))
        shape = str(settings.get("output_shape", "")).strip()
        if shape in {_OUTPUT_SHAPE_PROJECTED, _OUTPUT_SHAPE_EQUIRECT_3DGUT}:
            roots: list[Path] = []
            output_dir = str(settings.get("output_dir", "")).strip()
            if output_dir:
                output_path = Path(output_dir)
                roots.append(output_path if output_path.is_absolute() else Path(self.scene_dir) / output_path)
            portable = settings.get("portable_output")
            if isinstance(portable, dict):
                portable_root = str(portable.get("root", "")).strip()
                if portable_root:
                    roots.append(Path(self.scene_dir) / portable_root)
            if not roots:
                roots.append(self._output_dir())
            for root in roots:
                try:
                    if dataset_root.resolve() == root.resolve():
                        return shape
                except OSError:
                    continue

        transforms = self._training_transforms_path(dataset_root)
        if transforms is None or not transforms.is_file():
            return ""
        data = load_json(transforms, {})
        camera_model = str(data.get("camera_model") or "").strip().upper()
        if camera_model == "EQUIRECTANGULAR":
            return _OUTPUT_SHAPE_EQUIRECT_3DGUT
        return _OUTPUT_SHAPE_PROJECTED

    def _default_training_executable(self, backend: str | None = None) -> str:
        return training_backend_default_executable(
            backend or self._training_backend(),
            windows=os.name == "nt",
        )

    def _resolve_training_executable(self) -> str:
        return self._resolve_executable(
            self.training_executable_browse.text(),
            self._default_training_executable(),
            "TRAINING_EXEC_NOT_FOUND",
        )

    def _training_output_dir(self) -> Path:
        raw = self.training_output_browse.text().strip()
        if raw:
            return Path(raw)
        return self._default_training_output_dir()

    def _training_dataset_dir(self) -> Path:
        raw = self.training_dataset_browse.text().strip()
        if raw:
            return Path(raw)
        return self._default_training_dataset_dir()

    def _default_training_output_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return scene_output_dir(Path(self.scene_dir))

    def _training_config_path(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return step4_meta_dir(Path(self.scene_dir)) / "training" / "lichtfeld_config.json"

    def _default_training_dataset_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        if not self.pipeline_stage_intent("conversion"):
            settings_output = self._settings_training_dataset_dir()
            if settings_output is not None:
                return settings_output
        return self._display_output_dir()

    def _settings_training_dataset_dir(self) -> Path | None:
        if not self.scene_dir:
            return None
        scene = Path(self.scene_dir)
        accepted_kinds = {
            DATASET_KIND_NERF_JSON_PLY,
            DATASET_KIND_COLMAP_DATASET,
            DATASET_KIND_LICHTFELD_COLMAP,
        }
        if self._training_backend() == _TRAINING_BACKEND_GSPLAT:
            accepted_kinds = {
                DATASET_KIND_COLMAP_DATASET,
                DATASET_KIND_LICHTFELD_COLMAP,
            }
        artifact_root = latest_dataset_root(
            scene,
            accepted_kinds=accepted_kinds,
        )
        if artifact_root is not None:
            return artifact_root
        settings = load_step4_export_settings(scene)
        output_dir = str(settings.get("output_dir", "")).strip()
        if output_dir:
            path = Path(output_dir)
            return path if path.is_absolute() else scene / path
        portable = settings.get("portable_output")
        if isinstance(portable, dict):
            portable_root = str(portable.get("root", "")).strip()
            if portable_root:
                return scene / portable_root
        return None

    def _default_lfs_output_name(self) -> str:
        return Path(self.scene_dir).name if self.scene_dir else ""

    def _default_postshot_project_name(self) -> str:
        if not self.scene_dir:
            return "postshot.psht"
        return f"{Path(self.scene_dir).name}.psht"

    def _default_brush_export_name(self) -> str:
        if not self.scene_dir:
            return "export_{iter}.ply"
        return f"{Path(self.scene_dir).name}_brush_{{iter}}.ply"

    def _default_gsplat_result_name(self) -> str:
        if not self.scene_dir:
            return "gsplat"
        return f"{Path(self.scene_dir).name}_gsplat"

    @staticmethod
    def _default_gsplat_script_path() -> Path:
        return Path("examples") / "simple_trainer.py"

    def _update_lfs_output_name(self, *, force: bool = False) -> None:
        if not hasattr(self, "lfs_output_name_edit"):
            return
        default_name = self._default_lfs_output_name()
        if force or not self._lfs_output_name_user_edited or not self.lfs_output_name_edit.text().strip():
            self._syncing_lfs_output_name = True
            try:
                self.lfs_output_name_edit.setText(default_name)
            finally:
                self._syncing_lfs_output_name = False

    def _update_postshot_project_name(self, *, force: bool = False) -> None:
        if not hasattr(self, "postshot_project_name_edit"):
            return
        default_name = self._default_postshot_project_name()
        if force or not self._postshot_project_name_user_edited or not self.postshot_project_name_edit.text().strip():
            self._syncing_postshot_project_name = True
            try:
                self.postshot_project_name_edit.setText(default_name)
            finally:
                self._syncing_postshot_project_name = False

    def _update_brush_export_name(self, *, force: bool = False) -> None:
        if not hasattr(self, "brush_export_name_edit"):
            return
        default_name = self._default_brush_export_name()
        if force or not self._brush_export_name_user_edited or not self.brush_export_name_edit.text().strip():
            self._syncing_brush_export_name = True
            try:
                self.brush_export_name_edit.setText(default_name)
            finally:
                self._syncing_brush_export_name = False

    def _update_gsplat_result_name(self, *, force: bool = False) -> None:
        if not hasattr(self, "gsplat_result_name_edit"):
            return
        default_name = self._default_gsplat_result_name()
        if force or not self._gsplat_result_name_user_edited or not self.gsplat_result_name_edit.text().strip():
            self._syncing_gsplat_result_name = True
            try:
                self.gsplat_result_name_edit.setText(default_name)
            finally:
                self._syncing_gsplat_result_name = False

    def _update_training_paths(self, *, force: bool = False) -> None:
        if not hasattr(self, "training_dataset_browse"):
            return
        if not self.scene_dir:
            self._syncing_training_paths = True
            try:
                self.training_dataset_browse.set_text("")
                self.training_output_browse.set_text("")
            finally:
                self._syncing_training_paths = False
            return

        self._syncing_training_paths = True
        try:
            if force or not self._training_dataset_user_edited or not self.training_dataset_browse.text():
                self.training_dataset_browse.set_text(str(self._default_training_dataset_dir()))
            if force or not self._training_output_user_edited or not self.training_output_browse.text():
                self.training_output_browse.set_text(str(self._default_training_output_dir()))
        finally:
            self._syncing_training_paths = False

    def _training_sparse_model_dir(self) -> Path | None:
        if self._is_colmap_method():
            return self._find_colmap_sparse_model()
        if self._is_spheresfm_method():
            return self._find_spheresfm_sparse_model()
        output_colmap = self._display_output_dir() / "colmap"
        return output_colmap if output_colmap.is_dir() else None

    @staticmethod
    def _dataset_sparse_model_dir(dataset_root: Path) -> Path | None:
        sparse_zero = dataset_root / "sparse" / "0"
        if sparse_zero.is_dir():
            return sparse_zero
        sparse = dataset_root / "sparse"
        return sparse if sparse.is_dir() else None

    def _training_dataset(self) -> TrainingDataset:
        dataset_root = self._training_dataset_dir()
        transforms_json = self._training_transforms_path(dataset_root)
        if self._is_colmap_method():
            images_dir = self._colmap_rig_images_dir()
            masks_dir = self._colmap_rig_masks_dir()
        else:
            images_dir = dataset_root / "images"
            masks_dir = dataset_root / "masks"
        dataset_sparse_dir = self._dataset_sparse_model_dir(dataset_root)
        return TrainingDataset(
            dataset_root=dataset_root,
            images_dir=images_dir,
            masks_dir=masks_dir,
            colmap_sparse_dir=dataset_sparse_dir or self._training_sparse_model_dir(),
            transforms_json=transforms_json,
            pointcloud_ply=self._training_pointcloud_source(dataset_root, transforms_json=transforms_json),
            output_shape=self._output_shape(),
        )

    def _training_preferred_profiles(self) -> tuple[str, ...]:
        backend = self._training_backend()
        if backend == _TRAINING_BACKEND_POSTSHOT:
            return ("postshot",)
        if backend == _TRAINING_BACKEND_LICHTFELD:
            return ("lichtfeld",)
        if backend == _TRAINING_BACKEND_BRUSH:
            return ("brush",)
        return ()

    def _training_transforms_path(self, dataset_root: Path) -> Path | None:
        return find_nerf_transforms_path(
            dataset_root,
            preferred_profiles=self._training_preferred_profiles(),
        )

    def _training_pointcloud_source(self, dataset_root: Path, *, transforms_json: Path | None = None) -> Path | None:
        if self._training_uses_external_metashape_pointcloud(dataset_root, transforms_json=transforms_json):
            return self._training_external_metashape_pointcloud(dataset_root, transforms_json=transforms_json)
        pointcloud = find_nerf_pointcloud_path(
            dataset_root,
            transforms_json=transforms_json,
            preferred_profiles=self._training_preferred_profiles(),
        )
        if pointcloud is not None:
            return pointcloud
        return self._resolve_ply_source()

    def _training_uses_external_metashape_pointcloud(
        self,
        dataset_root: Path,
        *,
        transforms_json: Path | None = None,
    ) -> bool:
        if self._training_backend() != _TRAINING_BACKEND_POSTSHOT:
            return False
        transforms = transforms_json or self._training_transforms_path(dataset_root)
        data = load_json_object(transforms) if transforms is not None and transforms.is_file() else {}
        source = data.get("source") if isinstance(data, dict) else None
        if not isinstance(source, dict):
            return False
        if source.get("type") != "metashape_xml_ply":
            return False
        axis_transform = str(source.get("axis_transform") or "").strip().lower().replace("_", "-")
        pointcloud_policy = str(source.get("pointcloud_policy") or "").strip().lower()
        return axis_transform in {"postshot", "brush"} or pointcloud_policy in {
            "raw_metashape_ply",
            "external_metashape_ply",
        }

    def _training_external_metashape_pointcloud(
        self,
        dataset_root: Path,
        *,
        transforms_json: Path | None = None,
    ) -> Path | None:
        transforms = transforms_json or self._training_transforms_path(dataset_root)
        data = load_json_object(transforms) if transforms is not None and transforms.is_file() else {}
        source = data.get("source") if isinstance(data, dict) else None
        if not isinstance(source, dict):
            return None
        raw_path = str(
            source.get("raw_metashape_pointcloud_path") or source.get("external_pointcloud_path") or ""
        ).strip()
        if raw_path:
            path = Path(raw_path)
            candidates = [path] if path.is_absolute() else [dataset_root / path]
            if self.scene_dir:
                candidates.append(Path(self.scene_dir) / path)
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
        return self._raw_metashape_ply_source()

    def _raw_metashape_ply_source(self) -> Path | None:
        ply_text = self.ms_ply_browse.text().strip() if hasattr(self, "ms_ply_browse") else ""
        if not ply_text:
            return None
        ply = Path(ply_text)
        return ply if ply.is_file() else None

    def _training_dataset_available(self) -> bool:
        if not self.scene_dir or not hasattr(self, "training_dataset_browse"):
            return False
        try:
            dataset = self._training_dataset()
        except ValueError:
            return False
        if dataset.images_dir is not None and self._count_images_in_dir(dataset.images_dir) > 0:
            return True
        return bool(dataset.transforms_json and dataset.transforms_json.is_file())

    def _selected_projection_view_count(self) -> int:
        if self._uses_direct_equirect_output() or self._uses_spheresfm_3dgut_output():
            return 1
        try:
            views = self.view_config.collect_views(include_disabled=True)
        except Exception:
            return 0
        return sum(1 for view in views if view["enabled"])

    def _count_images_in_dir(self, images_dir: Path | None) -> int:
        if images_dir is None or not images_dir.is_dir():
            return 0
        metadata_count = self._metadata_image_count_for_dir(images_dir)
        if metadata_count is not None:
            return metadata_count
        return sum(
            1
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _SUPPORTED_TRAINING_IMAGE_EXTS
        )

    def _training_image_count(self, dataset: TrainingDataset | None = None) -> int:
        if not self.scene_dir:
            return 0
        dataset = dataset or self._training_dataset()
        return self._count_images_in_dir(dataset.images_dir)

    def _metadata_image_count_for_dir(self, images_dir: Path) -> int | None:
        if not self.scene_dir:
            return None
        scene = Path(self.scene_dir)
        try:
            target = images_dir.resolve()
            output_images = (scene_output_dir(scene) / "images").resolve()
            source_images = scene_images_dir(scene).resolve()
            display_output_images = (self._display_output_dir() / "images").resolve()
            settings_output = self._settings_training_dataset_dir()
            settings_output_images = (settings_output / "images").resolve() if settings_output is not None else None
        except OSError:
            return None

        project = load_json(project_path(scene), {})
        assets = project.get("assets") if isinstance(project.get("assets"), dict) else {}
        key = ""
        if (
            target == output_images
            or target == display_output_images
            or (settings_output_images is not None and target == settings_output_images)
        ):
            key = "output_image_count"
        elif target == source_images:
            key = "source_image_count"
        if not key:
            return None
        try:
            count = int(assets.get(key))
        except (TypeError, ValueError):
            return None
        return count if count > 0 else None

    def _update_lfs_auto_steps_scaler(self) -> None:
        if not hasattr(self, "lfs_auto_steps_scaler_cb") or not self.lfs_auto_steps_scaler_cb.isChecked():
            return
        image_count = self._training_image_count() if self.scene_dir else 0
        scaler = lichtfeld_auto_steps_scaler(image_count)
        self._syncing_lfs_auto_fields = True
        try:
            self._save_lfs_active_state()
            for strategy in _LFS_STRATEGIES:
                state = self._copy_lfs_ui_state(
                    self._lfs_strategy_states.get(strategy, self._default_lfs_ui_state(strategy))
                )
                self._scale_lfs_ui_state_steps(state, scaler)
                self._lfs_strategy_states[strategy] = state
            active = getattr(self, "_lfs_active_strategy", self._lfs_current_strategy())
            self._apply_lfs_ui_state(self._lfs_strategy_states[active])
        finally:
            self._syncing_lfs_auto_fields = False
