"""Step 4 output path, reset, and artifact validation helpers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from core.colmap_rig_export import pinhole_camera_params
from core.scene_layout import (
    scene_images_dir,
    scene_masks_dir,
    scene_metashape_3dgut_dir,
    scene_metashape_cubemap_dir,
    scene_output_dir,
    scene_spheresfm_3dgut_dir,
    scene_spheresfm_cubemap_dir,
    step4_export_settings_path,
    step4_metashape_import_work_dir,
    step4_views_config_path,
)
from core.sfm_preflight import preflight_spheresfm
from gui import i18n
from gui.cubemap.view_config import _BLOCK_ENABLED_VIEWS
from gui.steps.output_reset import clear_output_dir, clear_path, dedupe_nested_paths, path_has_contents
from gui.steps.step4_contracts import (
    _AXIS_NONE,
    _GENERATED_POINTCLOUD_NAME,
    _PROFILE_LICHTFELD,
    _PROFILE_REALITYSCAN,
    _SPHERESFM_RUN_CONVERT_ONLY,
    _SPHERESFM_RUN_FULL,
    _SPHERESFM_RUN_SFM_ONLY,
    _normalize_spheresfm_quality_preset,
)


class Step4PathMixin:
    @staticmethod
    def _path_text_relative_to(path: Path, root: Path) -> str:
        try:
            return str(path.relative_to(root).as_posix())
        except ValueError:
            return str(path)

    def _output_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return scene_output_dir(Path(self.scene_dir))

    def _metashape_import_work_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return step4_metashape_import_work_dir(Path(self.scene_dir))

    def _direct_output_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return scene_metashape_3dgut_dir(Path(self.scene_dir))

    def _metashape_cubemap_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return scene_metashape_cubemap_dir(Path(self.scene_dir))

    def _realityscan_output_dir(self) -> Path:
        return self._output_dir() / "realityscan"

    def _display_output_dir(self) -> Path:
        if self._uses_direct_equirect_output():
            return self._direct_output_dir()
        if self._is_metashape_method() and self._effective_profile() == _PROFILE_REALITYSCAN:
            return self._realityscan_output_dir()
        if self._is_spheresfm_method() and self._spheresfm_runs_conversion():
            return self._spheresfm_3dgut_dir() if self._uses_spheresfm_3dgut_output() else self._spheresfm_cubemap_dir()
        if self._is_spheresfm_method():
            return self._spheresfm_project_dir()
        return self._metashape_cubemap_dir() if self._is_metashape_method() else self._colmap_rig_dir()

    def _mask_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return scene_masks_dir(Path(self.scene_dir))

    def _colmap_rig_dir(self) -> Path:
        return self._output_dir() / "colmap_rig"

    def _colmap_project_dir(self) -> Path:
        return self._colmap_rig_dir()

    def _colmap_rig_images_dir(self) -> Path:
        return self._colmap_rig_dir() / "images"

    def _colmap_rig_masks_dir(self) -> Path:
        return self._colmap_rig_dir() / "masks"

    def _colmap_database_path(self) -> Path:
        return self._colmap_rig_dir() / "database.db"

    def _colmap_sparse_dir(self) -> Path:
        return self._colmap_rig_dir() / "sparse"

    def _spheresfm_project_dir(self) -> Path:
        return self._output_dir() / "spheresfm"

    def _spheresfm_masks_dir(self) -> Path:
        return self._spheresfm_project_dir() / "masks_colmap"

    def _spheresfm_preflight_dir(self) -> Path:
        return self._spheresfm_project_dir() / "preflight"

    def _spheresfm_database_path(self) -> Path:
        return self._spheresfm_project_dir() / "database.db"

    def _spheresfm_sparse_dir(self) -> Path:
        return self._spheresfm_project_dir() / "sparse"

    def _spheresfm_equirect_dir(self) -> Path:
        return self._spheresfm_project_dir() / "equirect"

    def _spheresfm_cubemap_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return scene_spheresfm_cubemap_dir(Path(self.scene_dir))

    def _spheresfm_3dgut_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return scene_spheresfm_3dgut_dir(Path(self.scene_dir))

    def _selected_colmap_sparse_model(self) -> Path | None:
        if not hasattr(self, "colmap_sparse_browse"):
            return None
        text = self.colmap_sparse_browse.text().strip()
        if not text:
            return None
        path = Path(text)
        return path if self._has_colmap_sparse_model(path) else None

    def _selected_spheresfm_sparse_model(self) -> Path | None:
        if not hasattr(self, "spheresfm_sparse_browse"):
            return None
        text = self.spheresfm_sparse_browse.text().strip()
        if not text:
            return None
        path = Path(text)
        return path if self._has_colmap_sparse_model(path) else None

    def _auto_find_spheresfm_sparse_model(self) -> Path | None:
        sparse = self._spheresfm_sparse_dir()
        if self._has_colmap_sparse_model(sparse):
            return sparse
        if not sparse.is_dir():
            return None

        def sort_key(path: Path) -> tuple[int, int | str]:
            if path.name.isdigit():
                return (0, int(path.name))
            return (1, path.name.lower())

        candidates = [p for p in sparse.iterdir() if p.is_dir() and self._has_colmap_sparse_model(p)]
        if not candidates:
            return None

        def score(path: Path) -> tuple[int, tuple[int, int | str]]:
            images_file = path / "images.txt"
            registered = 0
            if images_file.is_file():
                try:
                    registered = (
                        sum(
                            1
                            for line in images_file.read_text(encoding="utf-8", errors="replace").splitlines()
                            if line.strip() and not line.startswith("#")
                        )
                        // 2
                    )
                except OSError:
                    registered = 0
            return (registered, sort_key(path))

        return max(candidates, key=score)

    def _find_spheresfm_sparse_model(self) -> Path | None:
        selected_text = self.spheresfm_sparse_browse.text().strip() if hasattr(self, "spheresfm_sparse_browse") else ""
        selected = self._selected_spheresfm_sparse_model()
        if selected is not None:
            return selected
        if selected_text and self._spheresfm_sparse_user_edited:
            return None
        return self._auto_find_spheresfm_sparse_model()

    def _spheresfm_sparse_model_for_conversion(self) -> Path:
        if self._spheresfm_runs_sfm():
            return self._spheresfm_sparse_dir()
        model = self._find_spheresfm_sparse_model()
        return model if model is not None else self._spheresfm_sparse_dir()

    def _auto_find_colmap_sparse_model(self) -> Path | None:
        sparse = self._colmap_sparse_dir()
        if self._has_colmap_sparse_model(sparse):
            return sparse
        if not sparse.is_dir():
            return None

        def sort_key(path: Path) -> tuple[int, int | str]:
            if path.name.isdigit():
                return (0, int(path.name))
            return (1, path.name.lower())

        for candidate in sorted((p for p in sparse.iterdir() if p.is_dir()), key=sort_key):
            if self._has_colmap_sparse_model(candidate):
                return candidate
        return None

    def _find_colmap_sparse_model(self) -> Path | None:
        selected_text = self.colmap_sparse_browse.text().strip() if hasattr(self, "colmap_sparse_browse") else ""
        selected = self._selected_colmap_sparse_model()
        if selected is not None:
            return selected
        if selected_text and self._colmap_sparse_user_edited:
            return None
        return self._auto_find_colmap_sparse_model()

    @staticmethod
    def _has_colmap_sparse_model(path: Path) -> bool:
        if not path.is_dir():
            return False
        return all((path / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin")) or all(
            (path / name).is_file() for name in ("cameras.txt", "images.txt", "points3D.txt")
        )

    def _colmap_camera_params_arg(self) -> str:
        width, height = self._planned_colmap_image_size()
        params = pinhole_camera_params(width, height, 90.0)
        return ",".join(f"{value:.12g}" for value in params)

    def _spheresfm_camera_params_arg(self) -> str:
        source = self._first_image_size(self._metashape_images_dir())
        if source is None:
            raise ValueError(f"画像フォルダに対象画像がありません: {self._metashape_images_dir()}")
        width, height = source
        return f"1,{width / 2:.12g},{height / 2:.12g}"

    def _planned_colmap_image_size(self) -> tuple[int, int]:
        if not self._writes_images():
            existing = self._first_image_size(self._colmap_rig_images_dir())
            if existing is not None:
                return existing

        source = self._first_image_size(scene_images_dir(Path(self.scene_dir))) if self.scene_dir else None
        if source is not None:
            scale = float(self.scale_combo.currentData())
            output_size = max(1, int(round(source[1] * scale)))
            return output_size, output_size

        existing = self._first_image_size(self._colmap_rig_images_dir())
        if existing is not None:
            return existing

        raise ValueError("COLMAP用の画像サイズを判定できません。images/ に画像が必要です。")

    @staticmethod
    def _first_image_size(root: Path) -> tuple[int, int] | None:
        if not root.is_dir():
            return None
        supported = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
        for path in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
            if not path.is_file() or path.suffix.lower() not in supported:
                continue
            try:
                from PIL import Image

                with Image.open(path) as img:
                    return int(img.width), int(img.height)
            except Exception:
                continue
        return None

    def _metashape_images_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return scene_images_dir(Path(self.scene_dir))

    def _validate_scene_output_dir(self, output: Path) -> None:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        scene = Path(self.scene_dir).resolve()
        try:
            resolved_output = output.resolve()
        except OSError:
            resolved_output = output.absolute()
        root = self._output_dir().resolve()
        if resolved_output == root:
            return
        try:
            resolved_output.relative_to(root)
            return
        except ValueError:
            pass
        if resolved_output.parent != scene:
            raise ValueError(f"出力フォルダがシーンフォルダ外です: {output}")

    def _3dgut_output_reset_targets(self) -> list[Path]:
        output = self._display_output_dir()
        targets = [
            output / "images",
            output / "masks",
            output / "transforms.json",
            output / "pointcloud.ply",
        ]
        return targets

    def _prepare_3dgut_output_dir(self) -> bool:
        output = self._direct_output_dir()
        self._validate_scene_output_dir(output)
        existing_targets = self._dedupe_nested_paths(
            [path for path in self._3dgut_output_reset_targets() if self._path_has_contents(path)]
        )
        if existing_targets:
            target_text = "\n".join(str(path) for path in existing_targets)
            result = QMessageBox.question(
                self,
                i18n.t("OUTPUT_PARTIAL_RESET_TITLE"),
                i18n.t("OUTPUT_PARTIAL_RESET_MESSAGE").format(paths=target_text),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                return False
            for target in existing_targets:
                self._clear_path(target)

        output.mkdir(parents=True, exist_ok=True)
        self._link_3dgut_assets(output)
        return True

    def _link_3dgut_assets(self, output: Path) -> None:
        self._link_or_copy_tree(self._metashape_images_dir(), output / "images")
        masks = self._mask_dir()
        if masks.is_dir():
            self._link_or_copy_tree(masks, output / "masks")

    @staticmethod
    def _link_or_copy_tree(source_root: Path, dest_root: Path) -> None:
        if not source_root.is_dir():
            return
        for source in sorted(source_root.rglob("*"), key=lambda path: str(path).lower()):
            relative = source.relative_to(source_root)
            dest = dest_root / relative
            if source.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            if not source.is_file():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                clear_path(dest, allowed_roots=[dest_root])
            try:
                os.link(source, dest)
            except OSError:
                shutil.copy2(source, dest)

    def _prepare_output_dir(self) -> bool:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        output = self._display_output_dir()

        self._validate_scene_output_dir(output)

        if not self._writes_any_view_assets():
            output.mkdir(parents=True, exist_ok=True)
            return True

        if self._writes_images() and self._writes_masks():
            if output.exists() and any(output.iterdir()):
                result = QMessageBox.question(
                    self,
                    i18n.t("OUTPUT_RESET_TITLE"),
                    i18n.t("OUTPUT_RESET_MESSAGE").format(path=str(output)),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if result != QMessageBox.Yes:
                    return False
                self._clear_output_dir(output)
        else:
            targets = []
            if self._writes_images():
                targets.append(output / "images")
            if self._writes_masks():
                targets.append(output / "masks")
            existing_targets = [p for p in targets if self._path_has_contents(p)]
            if existing_targets:
                target_text = "\n".join(str(p) for p in existing_targets)
                result = QMessageBox.question(
                    self,
                    i18n.t("OUTPUT_PARTIAL_RESET_TITLE"),
                    i18n.t("OUTPUT_PARTIAL_RESET_MESSAGE").format(paths=target_text),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if result != QMessageBox.Yes:
                    return False
                for target in existing_targets:
                    self._clear_path(target)

        output.mkdir(parents=True, exist_ok=True)
        return True

    def _prepare_metashape_import_work_dir(self) -> Path:
        work = self._metashape_import_work_dir()
        if self._path_has_contents(work):
            self._clear_path(work)
        work.mkdir(parents=True, exist_ok=True)
        return work

    def _prepare_colmap_rig_dir(self) -> bool:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        output = self._output_dir()
        rig_dir = self._colmap_rig_dir()

        try:
            resolved_rig = rig_dir.resolve()
        except OSError:
            resolved_rig = rig_dir.absolute()
        if resolved_rig.parent != output.resolve():
            raise ValueError(f"COLMAP Rig出力フォルダが不正です: {rig_dir}")

        if self._writes_images() and self._writes_masks():
            if rig_dir.exists() and any(rig_dir.iterdir()):
                result = QMessageBox.question(
                    self,
                    i18n.t("OUTPUT_RESET_TITLE"),
                    i18n.t("OUTPUT_RESET_MESSAGE").format(path=str(rig_dir)),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if result != QMessageBox.Yes:
                    return False
                self._clear_path(rig_dir)
        else:
            targets: list[Path] = []
            if self._writes_images():
                targets.append(self._colmap_rig_images_dir())
            if self._writes_masks():
                targets.append(self._colmap_rig_masks_dir())
            if self._colmap_sfm_intent:
                targets.extend([self._colmap_database_path(), self._colmap_sparse_dir()])
            existing_targets = [p for p in targets if self._path_has_contents(p)]
            if existing_targets:
                target_text = "\n".join(str(p) for p in existing_targets)
                result = QMessageBox.question(
                    self,
                    i18n.t("OUTPUT_PARTIAL_RESET_TITLE"),
                    i18n.t("OUTPUT_PARTIAL_RESET_MESSAGE").format(paths=target_text),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if result != QMessageBox.Yes:
                    return False
                for target in existing_targets:
                    self._clear_path(target)

        rig_dir.mkdir(parents=True, exist_ok=True)
        return True

    def _prepare_spheresfm_run_outputs(self, *, include_project: bool, include_conversion: bool) -> bool:
        self._validate_spheresfm_project_dir()
        targets: list[Path] = []
        if include_project:
            targets.append(self._spheresfm_project_dir())
        if include_conversion:
            targets.extend(self._spheresfm_conversion_reset_targets())

        existing_targets = self._dedupe_nested_paths([p for p in targets if self._path_has_contents(p)])
        if existing_targets:
            target_text = "\n".join(str(p) for p in existing_targets)
            result = QMessageBox.question(
                self,
                i18n.t("OUTPUT_PARTIAL_RESET_TITLE"),
                i18n.t("OUTPUT_PARTIAL_RESET_MESSAGE").format(paths=target_text),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                return False
            for target in existing_targets:
                self._clear_path(target)

        self._spheresfm_project_dir().mkdir(parents=True, exist_ok=True)
        if include_conversion:
            if self._uses_spheresfm_3dgut_output():
                self._spheresfm_3dgut_dir().mkdir(parents=True, exist_ok=True)
                self._link_3dgut_assets(self._spheresfm_3dgut_dir())
            else:
                self._spheresfm_cubemap_dir().mkdir(parents=True, exist_ok=True)
        return True

    def _spheresfm_conversion_reset_targets(self) -> list[Path]:
        if self._uses_spheresfm_3dgut_output():
            root = self._spheresfm_3dgut_dir()
            return [
                root / "images",
                root / "masks",
                root / "transforms.json",
                root / "pointcloud.ply",
            ]

        output = self._spheresfm_cubemap_dir()
        targets = [
            self._spheresfm_equirect_dir(),
            output / "transforms.json",
            output / "pointcloud.ply",
            step4_views_config_path(Path(self.scene_dir)),
            step4_export_settings_path(Path(self.scene_dir)),
        ]
        if self._writes_images():
            targets.append(output / "images")
        if self._writes_masks():
            targets.append(output / "masks")
        return targets

    @staticmethod
    def _dedupe_nested_paths(paths: list[Path]) -> list[Path]:
        return dedupe_nested_paths(paths)

    def _validate_spheresfm_project_dir(self) -> None:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        output = self._output_dir()
        project = self._spheresfm_project_dir()
        try:
            resolved_project = project.resolve()
        except OSError:
            resolved_project = project.absolute()
        if resolved_project.parent != output.resolve():
            raise ValueError(f"SphereSfM出力フォルダが不正です: {project}")

    @staticmethod
    def _path_has_contents(path: Path) -> bool:
        return path_has_contents(path)

    @staticmethod
    def _clear_path(path: Path) -> None:
        clear_path(path, allowed_roots=[path.parent])

    @staticmethod
    def _clear_output_dir(output: Path) -> None:
        clear_output_dir(output)

    # -- バンドル検証 --

    def _spheresfm_uses_masks(self) -> bool:
        return self.spheresfm_use_masks_cb.isChecked()

    def _spheresfm_quality_preset(self) -> str:
        return _normalize_spheresfm_quality_preset(str(self.spheresfm_quality_combo.currentData() or ""))

    def _spheresfm_run_scope(self) -> str:
        if self._spheresfm_sfm_intent and self._spheresfm_conversion_intent:
            return _SPHERESFM_RUN_FULL
        if self._spheresfm_sfm_intent:
            return _SPHERESFM_RUN_SFM_ONLY
        return _SPHERESFM_RUN_CONVERT_ONLY

    def _validate_spheresfm_export(self) -> None:
        self._validate_image_only_export()
        if self._spheresfm_runs_sfm():
            result = preflight_spheresfm(Path(self.scene_dir))
            if not result.ok:
                raise ValueError(i18n.t("SPHERESFM_PREFLIGHT_FAILED").format(details=result.error_message()))
        if self._spheresfm_runs_sfm() and self._spheresfm_uses_masks() and not self._mask_dir().is_dir():
            raise ValueError(i18n.t("SPHERESFM_MASKS_NOT_FOUND").format(path=str(self._mask_dir())))

    def _require_spheresfm_sparse_model(self) -> Path:
        model = self._find_spheresfm_sparse_model()
        if model is None:
            raise ValueError(i18n.t("SPHERESFM_CONVERT_ONLY_NO_SPARSE").format(path=str(self._spheresfm_sparse_dir())))
        return model

    def _validate_spheresfm_conversion_export(self) -> None:
        transforms_script = self.base_dir / "scripts" / "spheresfm_to_transforms.py"
        if not transforms_script.exists():
            raise FileNotFoundError(f"spheresfm_to_transforms.py が見つかりません: {transforms_script}")
        if not self._uses_spheresfm_projected_output():
            return
        cubemap_script = self.base_dir / "cubemap_transforms_json.py"
        if not cubemap_script.exists():
            raise FileNotFoundError(f"cubemap_transforms_json.py が見つかりません: {cubemap_script}")
        views = self.view_config.collect_views(include_disabled=True)
        enabled = sum(1 for v in views if v["enabled"])
        if enabled <= 0:
            raise ValueError("少なくとも1つのビューを有効にしてください")
        if enabled > _BLOCK_ENABLED_VIEWS:
            raise ValueError(f"ビュー数が多すぎます ({enabled})。{_BLOCK_ENABLED_VIEWS} 以下にしてください。")
        try:
            jpgq = int(self.jpg_quality_edit.text().strip())
        except ValueError as exc:
            raise ValueError("JPG/WebP 品質は整数で指定してください") from exc
        if not 1 <= jpgq <= 100:
            raise ValueError("JPG/WebP 品質は 1-100 の範囲で指定してください")

    def _validate_image_only_export(self) -> None:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        images = self._metashape_images_dir()
        if not images.is_dir():
            raise ValueError(f"画像フォルダが見つかりません: {images}")
        supported = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
        if not any(p.is_file() and p.suffix.lower() in supported for p in images.rglob("*")):
            raise ValueError(f"画像フォルダに対象画像がありません: {images}")

    def _validate_bundle(self) -> None:
        profile = self._effective_profile()
        source = self._resolve_ply_source()
        if source is not None:
            return
        if profile == _PROFILE_REALITYSCAN:
            return
        if profile == _PROFILE_LICHTFELD and self._preprocess_uses_ply():
            return
        if profile == _PROFILE_LICHTFELD:
            raise ValueError(
                "LichtFeldプロファイルにはpointcloud.plyが必要です。Metashapeインポート設定でPLY使用を有効にしてください。"
            )
        raise ValueError(
            "Postshot/BrushプロファイルにはMetashapeからエクスポートしたRAW PLYが必要です。"
            "LichtFeld用のpointcloud.plyは使用できません。"
        )

    def _resolve_ply_source(self) -> Path | None:
        if not self.scene_dir:
            return None
        if self._is_metashape_method() and self._effective_profile() == _PROFILE_REALITYSCAN:
            return None
        if self._axis_transform_mode() == _AXIS_NONE:
            pointcloud = self._metashape_import_work_dir() / _GENERATED_POINTCLOUD_NAME
            return pointcloud if pointcloud.is_file() else None
        ply_text = self.ms_ply_browse.text().strip() if hasattr(self, "ms_ply_browse") else ""
        if ply_text:
            ply = Path(ply_text)
            if ply.is_file():
                return ply
        return None

    # -- バンドル後処理 --
