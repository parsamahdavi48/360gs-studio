"""Command builders for Step 4 cubemap export."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MetashapePreprocessCommand:
    python_executable: str
    script: Path
    images: Path
    xml: str
    output: Path
    scale: float
    use_ply: bool
    ply: str = ""
    no_fix_rotation: bool = False


@dataclass(frozen=True)
class CubemapConversionCommand:
    python_executable: str
    script: Path
    scene: Path
    output: Path
    views_json: Path
    scale: float
    axis_mode: str
    image_only: bool
    colmap_rig: bool
    invert_masks: bool
    writes_images: bool
    writes_masks: bool
    yaw_offset_per_frame: float
    output_format: str
    output_bit_depth: str
    jpg_quality: int
    final_orientation: str = "none"
    image_dir: Path | None = None
    mask_dir: Path | None = None


@dataclass(frozen=True)
class ColmapExportCommand:
    python_executable: str
    script: Path
    output: Path
    colmap_dir: Path
    ply: Path | None = None


@dataclass(frozen=True)
class ColmapSfmCommand:
    colmap: str
    glomap: str
    rig_dir: Path
    images_dir: Path
    masks_dir: Path
    database: Path
    sparse: Path
    camera_params: str
    writes_images: bool
    writes_masks: bool
    matcher: str
    mapper: str


@dataclass(frozen=True)
class SphereSfmCommand:
    python_executable: str
    preflight_script: Path
    prepare_script: Path
    colmap: str
    images_dir: Path
    source_masks_dir: Path
    prepared_masks_dir: Path
    preflight_dir: Path
    database: Path
    sparse: Path
    camera_params: str
    use_masks: bool
    matcher: str
    quality_preset: str
    pose_path: str = ""


@dataclass(frozen=True)
class SphereSfmTransformsCommand:
    python_executable: str
    script: Path
    sparse: Path
    output: Path
    images_dir: Path
    image_path_mode: str


def build_metashape_preprocess_cmd(options: MetashapePreprocessCommand) -> list[str]:
    cmd = [
        options.python_executable,
        "-u",
        str(options.script),
        "--images",
        str(options.images),
        "--xml",
        options.xml,
        "--output",
        str(options.output),
        "--scale",
        f"{options.scale:g}",
    ]
    if options.use_ply:
        cmd.extend(["--ply", options.ply])
    if options.no_fix_rotation:
        cmd.append("--no-fix-rotation")
    return cmd


def build_cubemap_conversion_cmd(options: CubemapConversionCommand) -> list[str]:
    cmd = [
        options.python_executable,
        "-u",
        str(options.script),
        str(options.scene),
        str(options.output),
        "--fov",
        "90",
        "--output_scale",
        f"{options.scale:g}",
        "--views-json",
        str(options.views_json),
    ]
    if options.image_only:
        cmd.append("--image-only")
        if options.colmap_rig:
            cmd.extend(["--colmap-rig", "--colmap-rig-name", "rig1"])
    else:
        if options.axis_mode == "none":
            cmd.append("--no_transform")
        if options.axis_mode == "brush":
            cmd.append("--brush")
        if options.final_orientation != "none":
            cmd.extend(["--final-orientation", options.final_orientation])
    if options.invert_masks:
        cmd.append("--invert_masks")
    if options.image_dir is not None:
        cmd.extend(["--image-dir", str(options.image_dir)])
    if options.mask_dir is not None:
        cmd.extend(["--mask_dir", str(options.mask_dir)])
    if not options.writes_images:
        cmd.append("--skip-images")
    if not options.writes_masks:
        cmd.append("--skip-masks")

    cmd.extend(["--yaw-offset-per-frame", f"{options.yaw_offset_per_frame:g}"])
    cmd.extend(["--output-format", options.output_format])
    cmd.extend(["--output-bit-depth", options.output_bit_depth])
    cmd.extend(["--jpg-quality", str(options.jpg_quality)])
    return cmd


def build_spheresfm_transforms_cmd(options: SphereSfmTransformsCommand) -> list[str]:
    return [
        options.python_executable,
        "-u",
        str(options.script),
        str(options.sparse),
        str(options.output),
        "--images-dir",
        str(options.images_dir),
        "--image-path-mode",
        options.image_path_mode,
    ]


def build_colmap_export_cmd(options: ColmapExportCommand) -> list[str]:
    cmd = [
        options.python_executable,
        "-u",
        str(options.script),
        str(options.output),
        str(options.colmap_dir),
    ]
    if options.ply is not None:
        cmd.extend(["--ply", str(options.ply)])
    return cmd


def build_colmap_sfm_commands(options: ColmapSfmCommand) -> list[tuple[str, list[str]]]:
    if not options.writes_images and not options.images_dir.is_dir():
        raise ValueError(f"COLMAP Rig画像フォルダが見つかりません: {options.images_dir}")

    options.sparse.mkdir(parents=True, exist_ok=True)
    rig_config = options.rig_dir / "rig_config.json"

    feature_cmd = [
        options.colmap,
        "feature_extractor",
        "--database_path",
        str(options.database),
        "--image_path",
        str(options.images_dir),
        "--ImageReader.single_camera_per_folder",
        "1",
        "--ImageReader.camera_model",
        "PINHOLE",
        "--ImageReader.camera_params",
        options.camera_params,
    ]
    if options.writes_masks or options.masks_dir.is_dir():
        feature_cmd.extend(["--ImageReader.mask_path", str(options.masks_dir)])

    rig_cmd = [
        options.colmap,
        "rig_configurator",
        "--database_path",
        str(options.database),
        "--rig_config_path",
        str(rig_config),
    ]

    matcher_name = "exhaustive_matcher" if options.matcher == "exhaustive" else "sequential_matcher"
    matcher_cmd = [
        options.colmap,
        matcher_name,
        "--database_path",
        str(options.database),
    ]

    if options.mapper == "global":
        mapper_cmd = [
            options.colmap,
            "global_mapper",
            "--database_path",
            str(options.database),
            "--image_path",
            str(options.images_dir),
            "--output_path",
            str(options.sparse),
        ]
    elif options.mapper == "glomap":
        mapper_cmd = [
            options.glomap,
            "mapper",
            "--database_path",
            str(options.database),
            "--image_path",
            str(options.images_dir),
            "--output_path",
            str(options.sparse),
        ]
    else:
        mapper_cmd = [
            options.colmap,
            "mapper",
            "--database_path",
            str(options.database),
            "--image_path",
            str(options.images_dir),
            "--output_path",
            str(options.sparse),
            "--Mapper.ba_refine_sensor_from_rig",
            "1",
        ]

    return [
        ("colmap_feature", feature_cmd),
        ("colmap_rig_config", rig_cmd),
        ("colmap_match", matcher_cmd),
        ("colmap_mapper", mapper_cmd),
    ]


def _spheresfm_feature_options(preset: str) -> list[str]:
    if preset == "fast":
        max_image_size = "3200"
        max_num_features = "8192"
    elif preset in {"quality", "robust"}:
        max_image_size = "5000"
        max_num_features = "32768"
    else:
        max_image_size = "4096"
        max_num_features = "16384"
    return [
        "--SiftExtraction.use_gpu",
        "1",
        "--SiftExtraction.max_image_size",
        max_image_size,
        "--SiftExtraction.max_num_features",
        max_num_features,
    ]


def _spheresfm_matching_options(preset: str) -> list[str]:
    max_num_matches = "16384" if preset == "fast" else "32768"
    options = [
        "--SiftMatching.max_error",
        "4",
        "--SiftMatching.min_num_inliers",
        "50",
        "--SiftMatching.max_num_matches",
        max_num_matches,
    ]
    if preset in {"quality", "robust"}:
        options.extend(["--SiftMatching.guided_matching", "1"])
    return options


def _spheresfm_sequential_overlap(preset: str) -> str:
    if preset == "fast":
        return "5"
    if preset in {"quality", "robust"}:
        return "15"
    return "10"


def _spheresfm_mapper_options(preset: str) -> list[str]:
    options = [
        "--Mapper.ba_refine_focal_length",
        "0",
        "--Mapper.ba_refine_principal_point",
        "0",
        "--Mapper.ba_refine_extra_params",
        "0",
        "--Mapper.sphere_camera",
        "1",
        "--Mapper.multiple_models",
        "0",
    ]
    if preset == "fast":
        options.extend(
            [
                "--Mapper.ba_local_max_num_iterations",
                "12",
                "--Mapper.ba_global_max_num_iterations",
                "25",
                "--Mapper.ba_local_max_refinements",
                "1",
                "--Mapper.ba_global_max_refinements",
                "2",
                "--Mapper.ba_global_images_ratio",
                "1.3",
                "--Mapper.ba_global_points_ratio",
                "1.3",
            ]
        )
    elif preset in {"quality", "robust"}:
        options.extend(
            [
                "--Mapper.ba_local_max_num_iterations",
                "30",
                "--Mapper.ba_global_max_num_iterations",
                "75",
                "--Mapper.ba_local_max_refinements",
                "3",
                "--Mapper.ba_global_max_refinements",
                "5",
            ]
        )
    else:
        options.extend(
            [
                "--Mapper.ba_local_max_num_iterations",
                "16",
                "--Mapper.ba_global_max_num_iterations",
                "33",
                "--Mapper.ba_local_max_refinements",
                "2",
                "--Mapper.ba_global_max_refinements",
                "2",
                "--Mapper.ba_global_images_ratio",
                "1.2",
                "--Mapper.ba_global_points_ratio",
                "1.2",
            ]
        )
    return options


def build_spheresfm_commands(options: SphereSfmCommand) -> list[tuple[str, list[str]]]:
    options.sparse.mkdir(parents=True, exist_ok=True)
    options.database.parent.mkdir(parents=True, exist_ok=True)

    preflight_cmd = [
        options.python_executable,
        "-u",
        str(options.preflight_script),
        "--colmap",
        options.colmap,
        "--images-dir",
        str(options.images_dir),
        "--work-dir",
        str(options.preflight_dir),
        "--camera-params",
        options.camera_params,
    ]

    prepare_cmd = [
        options.python_executable,
        "-u",
        str(options.prepare_script),
        "--colmap",
        options.colmap,
        "--images-dir",
        str(options.images_dir),
    ]
    if options.use_masks:
        prepare_cmd.extend(
            [
                "--use-masks",
                "--source-masks-dir",
                str(options.source_masks_dir),
                "--output-masks-dir",
                str(options.prepared_masks_dir),
            ]
        )

    database_cmd = [
        options.colmap,
        "database_creator",
        "--database_path",
        str(options.database),
    ]

    feature_cmd = [
        options.colmap,
        "feature_extractor",
        "--database_path",
        str(options.database),
        "--image_path",
        str(options.images_dir),
        "--ImageReader.camera_model",
        "SPHERE",
        "--ImageReader.camera_params",
        options.camera_params,
        "--ImageReader.single_camera",
        "1",
    ]
    if options.use_masks:
        feature_cmd.extend(["--ImageReader.mask_path", str(options.prepared_masks_dir)])
    if options.pose_path:
        feature_cmd.extend(["--ImageReader.pose_path", options.pose_path])
    feature_cmd.extend(_spheresfm_feature_options(options.quality_preset))

    if options.matcher == "spatial":
        matcher_cmd = [
            options.colmap,
            "spatial_matcher",
            "--database_path",
            str(options.database),
            "--SpatialMatching.is_gps",
            "0",
            "--SpatialMatching.max_distance",
            "50",
        ]
        matcher_cmd.extend(_spheresfm_matching_options(options.quality_preset))
    else:
        matcher_name = "exhaustive_matcher" if options.matcher == "exhaustive" else "sequential_matcher"
        matcher_cmd = [
            options.colmap,
            matcher_name,
            "--database_path",
            str(options.database),
        ]
        matcher_cmd.extend(_spheresfm_matching_options(options.quality_preset))
        if options.matcher != "exhaustive":
            matcher_cmd.extend(["--SequentialMatching.overlap", _spheresfm_sequential_overlap(options.quality_preset)])

    mapper_cmd = [
        options.colmap,
        "mapper",
        "--database_path",
        str(options.database),
        "--image_path",
        str(options.images_dir),
        "--output_path",
        str(options.sparse),
    ]
    mapper_cmd.extend(_spheresfm_mapper_options(options.quality_preset))

    return [
        ("spheresfm_preflight", preflight_cmd),
        ("spheresfm_prepare", prepare_cmd),
        ("spheresfm_database", database_cmd),
        ("spheresfm_feature", feature_cmd),
        ("spheresfm_match", matcher_cmd),
        ("spheresfm_mapper", mapper_cmd),
    ]


def views_config_payload(views: list[dict]) -> dict:
    return {
        "fov": 90.0,
        "views": [
            {"name": v["name"], "yaw": float(v["yaw"]), "pitch": float(v["pitch"]), "enabled": bool(v["enabled"])}
            for v in views
        ],
    }


def write_views_config(output_dir: Path, views: list[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "views_config.json"
    path.write_text(json.dumps(views_config_payload(views), indent=2), encoding="utf-8")
    return path
