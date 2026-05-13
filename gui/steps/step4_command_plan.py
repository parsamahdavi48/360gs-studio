"""Step 4 command planning and executable resolution."""

from __future__ import annotations

import math
import os
import shutil
import sys
from pathlib import Path

from core.orientation_correction import FINAL_ORIENTATION_LICHTFELD, FINAL_ORIENTATION_NONE
from core.scene_layout import step4_meta_dir
from gui import i18n
from gui.cubemap.view_config import _BLOCK_ENABLED_VIEWS
from gui.steps.cubemap_commands import (
    ColmapExportCommand,
    ColmapSfmCommand,
    CubemapConversionCommand,
    MetashapePreprocessCommand,
    SphereSfmCommand,
    SphereSfmTransformsCommand,
    build_colmap_export_cmd,
    build_colmap_sfm_commands,
    build_cubemap_conversion_cmd,
    build_metashape_preprocess_cmd,
    build_spheresfm_commands,
    build_spheresfm_transforms_cmd,
)
from gui.steps.step4_contracts import (
    _COLMAP_MAPPER_GLOMAP,
    _COLMAP_MAPPER_INCREMENTAL,
    _COLMAP_MATCHER_SEQUENTIAL,
    _PIPELINE_STAGE_CONVERSION,
    _PIPELINE_STAGE_SFM,
    _PROFILE_REALITYSCAN,
    _SPHERESFM_MATCHER_SPATIAL,
)


class Step4CommandPlanMixin:
    def build_commands(self) -> list[tuple[str, list[str]]]:
        if self._is_spheresfm_method():
            self._reset_spheresfm_rtx50_diagnostics()
            run_sfm = self._spheresfm_runs_sfm()
            run_conversion = self._spheresfm_runs_conversion()
            if run_sfm or run_conversion:
                self._validate_spheresfm_export()
            if run_conversion and not run_sfm:
                self._require_spheresfm_sparse_model()
                self._validate_spheresfm_conversion_export()
                if not self._prepare_spheresfm_run_outputs(include_project=False, include_conversion=True):
                    return []
                return self._build_spheresfm_conversion_commands()

            steps: list[tuple[str, list[str]]] = []
            if run_sfm:
                if not self._prepare_spheresfm_run_outputs(
                    include_project=True,
                    include_conversion=run_conversion,
                ):
                    return []
                steps.extend(self._build_spheresfm_sfm_commands())
            if run_conversion:
                self._validate_spheresfm_conversion_export()
                steps.extend(self._build_spheresfm_conversion_commands())
            return steps

        if self._is_colmap_method():
            run_conversion = self.pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION)
            run_sfm = self.pipeline_stage_intent(_PIPELINE_STAGE_SFM)
            if run_conversion or run_sfm:
                self._validate_image_only_export()
            if run_conversion and not self._prepare_colmap_rig_dir():
                return []
            steps: list[tuple[str, list[str]]] = []
            if run_conversion:
                steps.append(("colmap_rig_export", self._build_cubemap_cmd(image_only=True, colmap_rig=True)))
            if run_sfm:
                if not run_conversion and not self._colmap_rig_images_dir().is_dir():
                    raise ValueError(i18n.t("STEP4_PIPELINE_DETAIL_COLMAP_NEEDS_RIG"))
                steps.extend(self._build_colmap_sfm_commands())
            return steps

        run_conversion = self.pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION)
        if run_conversion:
            self._validate_bundle()
            preprocess_cmd = self._build_preprocess_cmd()

        if not run_conversion:
            return []

        if self._uses_direct_equirect_output():
            if not self._prepare_3dgut_output_dir():
                return []
            return [("metashape", preprocess_cmd)]

        if not self._prepare_output_dir():
            return []
        self._prepare_metashape_import_work_dir()

        steps = [("metashape", preprocess_cmd)]
        steps.append(("cubemap", self._build_cubemap_cmd()))
        if self.export_colmap_cb.isChecked():
            steps.append(("colmap", self._build_colmap_cmd()))
        return steps

    def _build_preprocess_cmd(self) -> list[str]:
        self._refresh_metashape_auto_inputs_if_empty()
        script = self.base_dir / "vendor" / "metashape_360_lfs" / "metashape_360_lfs.py"
        if not script.exists():
            raise FileNotFoundError(f"metashape_360_lfs.py が見つかりません: {script}")
        scene = Path(self.scene_dir)
        if not scene.is_dir():
            raise ValueError(f"シーンフォルダが見つかりません: {scene}")

        images = str(self._metashape_images_dir())
        xml = self.ms_xml_browse.text()
        if not images or not Path(images).is_dir():
            raise ValueError(f"Metashape画像フォルダが見つかりません: {images}")
        if not xml or not Path(xml).is_file():
            raise ValueError(f"Metashape XMLが見つかりません: {xml}")
        if self._metashape_input_output_path_issue(Path(xml)):
            raise ValueError(i18n.t("METASHAPE_INPUT_IN_OUTPUT_ERROR").format(path=xml))

        scale = float(self.ms_scale_edit.text().strip())
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("スケール係数は正の有限値である必要があります")

        ply = ""
        if self._preprocess_uses_ply():
            ply = self.ms_ply_browse.text()
            if not ply or not Path(ply).is_file():
                raise ValueError(f"PLYファイルが見つかりません: {ply}")
            if self._metashape_input_output_path_issue(Path(ply)):
                raise ValueError(i18n.t("METASHAPE_INPUT_IN_OUTPUT_ERROR").format(path=ply))
        return build_metashape_preprocess_cmd(
            MetashapePreprocessCommand(
                python_executable=sys.executable,
                script=script,
                images=Path(images),
                xml=xml,
                output=self._display_output_dir()
                if self._uses_direct_equirect_output()
                else self._metashape_import_work_dir(),
                scale=scale,
                use_ply=self._preprocess_uses_ply(),
                ply=ply,
                no_fix_rotation=self.ms_no_fix_rot_cb.isChecked(),
            )
        )

    def _build_cubemap_cmd(self, image_only: bool = False, colmap_rig: bool = False) -> list[str]:
        script = self.base_dir / "cubemap_transforms_json.py"
        if not script.exists():
            raise FileNotFoundError(f"cubemap_transforms_json.py が見つかりません: {script}")

        scene = Path(self.scene_dir)
        if not scene.is_dir():
            raise ValueError(f"シーンフォルダが見つかりません: {scene}")

        output = self._display_output_dir() if (not image_only and self._is_metashape_method()) else self._output_dir()
        input_dir = scene
        image_dir = None
        mask_dir = None
        if not image_only and self._is_metashape_method():
            input_dir = self._metashape_import_work_dir()
            image_dir = scene
            masks = self._mask_dir()
            if masks.is_dir():
                mask_dir = masks

        views = self.view_config.collect_views(include_disabled=True)
        enabled = sum(1 for v in views if v["enabled"])
        if enabled <= 0:
            raise ValueError("少なくとも1つのビューを有効にしてください")
        if enabled > _BLOCK_ENABLED_VIEWS:
            raise ValueError(f"ビュー数が多すぎます ({enabled})。{_BLOCK_ENABLED_VIEWS} 以下にしてください。")

        views_json = self._write_views_config(step4_meta_dir(scene), views)

        if colmap_rig or self._effective_profile() == _PROFILE_REALITYSCAN:
            yaw_step = 0.0
        else:
            yaw_step = float(self.yaw_per_frame_edit.value())

        out_fmt = self.output_format_combo.currentData() or "auto"
        out_depth = self.output_bit_depth_combo.currentData() or "8"

        try:
            jpgq = int(self.jpg_quality_edit.text().strip())
        except ValueError as exc:
            raise ValueError("JPG/WebP 品質は整数で指定してください") from exc
        if not 1 <= jpgq <= 100:
            raise ValueError("JPG/WebP 品質は 1-100 の範囲で指定してください")
        return build_cubemap_conversion_cmd(
            CubemapConversionCommand(
                python_executable=sys.executable,
                script=script,
                scene=input_dir,
                output=output,
                views_json=views_json,
                scale=float(self.scale_combo.currentData()),
                axis_mode=self._axis_transform_mode(),
                image_only=image_only,
                colmap_rig=colmap_rig,
                invert_masks=self.invert_masks_cb.isChecked(),
                writes_images=self._writes_images(),
                writes_masks=self._writes_masks(),
                yaw_offset_per_frame=yaw_step,
                output_format=out_fmt,
                output_bit_depth=out_depth,
                jpg_quality=jpgq,
                final_orientation=(
                    FINAL_ORIENTATION_LICHTFELD
                    if self._uses_lichtfeld_final_correction()
                    else FINAL_ORIENTATION_NONE
                ),
                image_dir=image_dir,
                mask_dir=mask_dir,
                realityscan_xmp=self._is_metashape_method() and self._effective_profile() == _PROFILE_REALITYSCAN,
                realityscan_pose_prior=self.realityscan_pose_prior_combo.currentData() or "exact",
                realityscan_calibration_prior=self.realityscan_calibration_prior_combo.currentData() or "initial",
            )
        )

    def _build_colmap_cmd(self) -> list[str]:
        script = self.base_dir / "transforms_to_colmap.py"
        if not script.exists():
            raise FileNotFoundError(f"transforms_to_colmap.py が見つかりません: {script}")

        output = self._output_dir()
        colmap_dir = output / "colmap"

        ply_path: Path | None = None
        ply = output / "pointcloud.ply"
        if ply.is_file():
            ply_path = ply
        else:
            # cubemap 出力ディレクトリ内の任意 .ply をフォールバック
            plys = sorted([p for p in output.glob("*.ply") if p.is_file()])
            if plys:
                ply_path = plys[0]
        return build_colmap_export_cmd(
            ColmapExportCommand(
                python_executable=sys.executable,
                script=script,
                output=output,
                colmap_dir=colmap_dir,
                ply=ply_path,
            )
        )

    def _default_colmap_executable(self) -> str:
        return "colmap.exe" if os.name == "nt" else "colmap"

    def _default_glomap_executable(self) -> str:
        return "glomap.exe" if os.name == "nt" else "glomap"

    def _default_spheresfm_executable(self) -> str:
        return "colmap.exe" if os.name == "nt" else "colmap"

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        return any(sep in value for sep in ("/", "\\")) or Path(value).is_absolute()

    def _resolve_executable(self, raw: str, default_name: str, message_key: str) -> str:
        value = raw.strip() or default_name
        if self._looks_like_path(value):
            path = Path(value)
            if not path.is_file():
                raise ValueError(i18n.t(message_key).format(path=value))
            return str(path)
        found = shutil.which(value)
        if not found:
            raise ValueError(i18n.t(message_key).format(path=value))
        return found

    def _resolve_colmap_executable(self) -> str:
        return self._resolve_executable(
            self.colmap_exec_browse.text(),
            self._default_colmap_executable(),
            "COLMAP_EXEC_NOT_FOUND",
        )

    def _resolve_spheresfm_executable(self) -> str:
        return self._resolve_executable(
            self.spheresfm_exec_browse.text(),
            self._default_spheresfm_executable(),
            "SPHERESFM_EXEC_NOT_FOUND",
        )

    def _resolve_glomap_executable(self) -> str:
        return self._resolve_executable(
            self.glomap_exec_browse.text(),
            self._default_glomap_executable(),
            "GLOMAP_EXEC_NOT_FOUND",
        )

    def _build_colmap_sfm_commands(self) -> list[tuple[str, list[str]]]:
        colmap = self._resolve_colmap_executable()
        rig_dir = self._colmap_rig_dir()
        images_dir = self._colmap_rig_images_dir()
        masks_dir = self._colmap_rig_masks_dir()
        database = self._colmap_database_path()
        sparse = self._colmap_sparse_dir()

        matcher = self.colmap_matcher_combo.currentData() or _COLMAP_MATCHER_SEQUENTIAL
        mapper = self.colmap_mapper_combo.currentData() or _COLMAP_MAPPER_INCREMENTAL
        glomap = (
            self._resolve_glomap_executable() if mapper == _COLMAP_MAPPER_GLOMAP else self._default_glomap_executable()
        )
        return build_colmap_sfm_commands(
            ColmapSfmCommand(
                colmap=colmap,
                glomap=glomap,
                rig_dir=rig_dir,
                images_dir=images_dir,
                masks_dir=masks_dir,
                database=database,
                sparse=sparse,
                camera_params=self._colmap_camera_params_arg(),
                writes_images=self._writes_images(),
                writes_masks=self._writes_masks(),
                matcher=matcher,
                mapper=mapper,
            )
        )

    def _build_spheresfm_sfm_commands(self) -> list[tuple[str, list[str]]]:
        preflight_script = self.base_dir / "scripts" / "spheresfm_gpu_preflight.py"
        if not preflight_script.exists():
            raise FileNotFoundError(f"spheresfm_gpu_preflight.py が見つかりません: {preflight_script}")

        prepare_script = self.base_dir / "scripts" / "prepare_spheresfm_project.py"
        if not prepare_script.exists():
            raise FileNotFoundError(f"prepare_spheresfm_project.py が見つかりません: {prepare_script}")

        matcher = self.spheresfm_matcher_combo.currentData() or _COLMAP_MATCHER_SEQUENTIAL
        pose_path = self.spheresfm_pose_browse.text().strip()
        if matcher == _SPHERESFM_MATCHER_SPATIAL and not pose_path:
            raise ValueError(i18n.t("SPHERESFM_POSE_REQUIRED"))
        if pose_path and not Path(pose_path).is_file():
            raise ValueError(i18n.t("SPHERESFM_POSE_NOT_FOUND").format(path=pose_path))

        steps = build_spheresfm_commands(
            SphereSfmCommand(
                python_executable=sys.executable,
                preflight_script=preflight_script,
                prepare_script=prepare_script,
                colmap=self._resolve_spheresfm_executable(),
                images_dir=self._metashape_images_dir(),
                source_masks_dir=self._mask_dir(),
                prepared_masks_dir=self._spheresfm_masks_dir(),
                preflight_dir=self._spheresfm_preflight_dir(),
                database=self._spheresfm_database_path(),
                sparse=self._spheresfm_sparse_dir(),
                camera_params=self._spheresfm_camera_params_arg(),
                use_masks=self._spheresfm_uses_masks(),
                matcher=matcher,
                quality_preset=self._spheresfm_quality_preset(),
                pose_path=pose_path,
            )
        )
        return steps

    def _build_spheresfm_conversion_commands(self) -> list[tuple[str, list[str]]]:
        transforms_script = self.base_dir / "scripts" / "spheresfm_to_transforms.py"
        if not transforms_script.exists():
            raise FileNotFoundError(f"spheresfm_to_transforms.py が見つかりません: {transforms_script}")

        steps: list[tuple[str, list[str]]] = []
        transforms_output = (
            self._spheresfm_3dgut_dir() if self._uses_spheresfm_3dgut_output() else self._spheresfm_equirect_dir()
        )
        image_path_mode = "images-prefix" if self._uses_spheresfm_3dgut_output() else "relative"
        steps.append(
            (
                "spheresfm_transforms",
                build_spheresfm_transforms_cmd(
                    SphereSfmTransformsCommand(
                        python_executable=sys.executable,
                        script=transforms_script,
                        sparse=self._spheresfm_sparse_model_for_conversion(),
                        output=transforms_output,
                        images_dir=self._metashape_images_dir(),
                        image_path_mode=image_path_mode,
                    )
                ),
            )
        )
        if self._uses_spheresfm_projected_output():
            steps.append(("spheresfm_cubemap", self._build_spheresfm_cubemap_cmd()))
        return steps

    def _build_spheresfm_cubemap_cmd(self) -> list[str]:
        script = self.base_dir / "cubemap_transforms_json.py"
        if not script.exists():
            raise FileNotFoundError(f"cubemap_transforms_json.py が見つかりません: {script}")

        views = self.view_config.collect_views(include_disabled=True)
        enabled = sum(1 for v in views if v["enabled"])
        if enabled <= 0:
            raise ValueError("少なくとも1つのビューを有効にしてください")
        if enabled > _BLOCK_ENABLED_VIEWS:
            raise ValueError(f"ビュー数が多すぎます ({enabled})。{_BLOCK_ENABLED_VIEWS} 以下にしてください。")

        output = self._spheresfm_cubemap_dir()
        views_json = self._write_views_config(step4_meta_dir(Path(self.scene_dir)), views)
        out_fmt = self.output_format_combo.currentData() or "auto"
        out_depth = self.output_bit_depth_combo.currentData() or "8"

        try:
            jpgq = int(self.jpg_quality_edit.text().strip())
        except ValueError as exc:
            raise ValueError("JPG/WebP 品質は整数で指定してください") from exc
        if not 1 <= jpgq <= 100:
            raise ValueError("JPG/WebP 品質は 1-100 の範囲で指定してください")

        mask_dir = self._mask_dir() if self._mask_dir().is_dir() else None
        return build_cubemap_conversion_cmd(
            CubemapConversionCommand(
                python_executable=sys.executable,
                script=script,
                scene=self._spheresfm_equirect_dir(),
                output=output,
                views_json=views_json,
                scale=float(self.scale_combo.currentData()),
                axis_mode=self._spheresfm_axis_transform_mode(),
                image_only=False,
                colmap_rig=False,
                invert_masks=self.invert_masks_cb.isChecked(),
                writes_images=self._writes_images(),
                writes_masks=self._writes_masks(),
                yaw_offset_per_frame=float(self.yaw_per_frame_edit.value()),
                output_format=out_fmt,
                output_bit_depth=out_depth,
                jpg_quality=jpgq,
                final_orientation=(
                    FINAL_ORIENTATION_LICHTFELD
                    if self._uses_spheresfm_lichtfeld_final_correction()
                    else FINAL_ORIENTATION_NONE
                ),
                image_dir=self._metashape_images_dir(),
                mask_dir=mask_dir,
            )
        )
