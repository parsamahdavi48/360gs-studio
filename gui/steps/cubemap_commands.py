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
    if options.invert_masks:
        cmd.append("--invert_masks")
    if not options.writes_images:
        cmd.append("--skip-images")
    if not options.writes_masks:
        cmd.append("--skip-masks")

    cmd.extend(["--yaw-offset-per-frame", f"{options.yaw_offset_per_frame:g}"])
    cmd.extend(["--output-format", options.output_format])
    cmd.extend(["--output-bit-depth", options.output_bit_depth])
    cmd.extend(["--jpg-quality", str(options.jpg_quality)])
    return cmd


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
