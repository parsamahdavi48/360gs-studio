#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COLMAP rig SfM pipeline from exported dataset")
    parser.add_argument("output_dir", help="COLMAP output root created by colmap_rig_export.py")
    parser.add_argument("--colmap_bin", help="COLMAP executable path (default: detect from PATH)")
    parser.add_argument(
        "--matcher",
        default="exhaustive",
        choices=["exhaustive", "sequential"],
        help="Matching strategy",
    )
    parser.add_argument(
        "--run_until",
        default="mapper",
        choices=["feature", "rig", "match", "mapper"],
        help="Last pipeline stage to execute",
    )
    parser.add_argument("--use_masks", action="store_true", help="Use dataset masks for feature extraction when supported")
    parser.add_argument("--sift_max_features", type=int, default=4096, help="SIFT max features for feature extractor")
    parser.add_argument("--seq_overlap", type=int, default=6, help="Sequential matcher overlap")
    parser.add_argument(
        "--seq_loop_detection",
        action="store_true",
        help="Enable loop detection in sequential matcher when supported",
    )
    parser.add_argument("--vocab_tree_path", help="Vocabulary tree path for sequential loop detection (optional)")
    parser.add_argument(
        "--mapper_ba_global_max_iter",
        type=int,
        default=20,
        help="Mapper BA global max iterations when supported",
    )
    parser.add_argument(
        "--refine_sensor_from_rig",
        action="store_true",
        help="Enable Mapper.ba_refine_sensor_from_rig when available",
    )
    return parser.parse_args()


def _resolve_colmap_binary(path_hint: str | None) -> str:
    if path_hint:
        p = Path(path_hint).expanduser()
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"COLMAP binary not found: {p}")

    hit = shutil.which("colmap")
    if hit:
        return hit

    raise FileNotFoundError("COLMAP binary not found. Set --colmap_bin or add colmap to PATH")


def _run_command(cmd: list[str], phase: str) -> None:
    print("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip("\n"))
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"{phase} failed (exit={ret})")


def _command_help_text(colmap_bin: str, command_name: str) -> str:
    cmd = [colmap_bin, command_name, "-h"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    text = (res.stdout or "") + "\n" + (res.stderr or "")
    return text


def _supports_option(colmap_bin: str, command_name: str, option_name: str) -> bool:
    text = _command_help_text(colmap_bin, command_name)
    return option_name in text


def _load_project_paths(output_root: Path) -> dict[str, Path]:
    project_path = output_root / "colmap_project.json"
    if project_path.is_file():
        data = json.loads(project_path.read_text(encoding="utf-8"))
        return {
            "dataset": Path(data["dataset_dir"]),
            "images": Path(data["images_dir"]),
            "masks": Path(data["masks_dir"]),
            "workspace": Path(data["workspace_dir"]),
            "database": Path(data["database_path"]),
            "sparse": Path(data["sparse_dir"]),
            "rig_template": Path(data["rig_template_path"]),
            "rig_config": Path(data["rig_config_path"]),
        }

    dataset = output_root / "dataset"
    workspace = output_root / "workspace"
    return {
        "dataset": dataset,
        "images": dataset / "images",
        "masks": dataset / "masks",
        "workspace": workspace,
        "database": workspace / "database.db",
        "sparse": workspace / "sparse",
        "rig_template": dataset / "rig_template.json",
        "rig_config": dataset / "rig_config.json",
    }


def _write_rig_config_from_database(db_path: Path, rig_template_path: Path, rig_config_path: Path) -> None:
    if not db_path.is_file():
        raise FileNotFoundError(f"COLMAP database not found: {db_path}")
    if not rig_template_path.is_file():
        raise FileNotFoundError(f"rig template not found: {rig_template_path}")

    template = json.loads(rig_template_path.read_text(encoding="utf-8"))
    cameras = template.get("cameras")
    ref_view_name = str(template.get("ref_view_name", "")).strip()
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("rig_template.json must contain non-empty cameras")
    if not ref_view_name:
        raise ValueError("rig_template.json must contain ref_view_name")

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT name, camera_id FROM images").fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError("No images found in COLMAP database")

    rig_cameras: list[dict] = []
    ref_camera_id: int | None = None

    for item in cameras:
        image_prefix = str(item.get("image_prefix", "")).strip()
        view_name = str(item.get("view_name", "")).strip()
        if not image_prefix or not view_name:
            raise ValueError("Each camera entry in rig_template.json needs view_name and image_prefix")

        matched_ids = {int(camera_id) for (name, camera_id) in rows if str(name).startswith(image_prefix)}
        if not matched_ids:
            raise ValueError(
                f"No database images matched image_prefix='{image_prefix}'. "
                "Check that exported folder names match COLMAP image names."
            )
        if len(matched_ids) != 1:
            raise ValueError(
                f"image_prefix='{image_prefix}' matched multiple camera IDs: {sorted(matched_ids)}. "
                "Enable ImageReader.single_camera_per_folder=1 and keep one folder per view."
            )

        camera_id = next(iter(matched_ids))
        if view_name == ref_view_name:
            ref_camera_id = camera_id

        rig_cameras.append(
            {
                "camera_id": camera_id,
                "image_prefix": image_prefix,
                "cam_from_rig_rotation": item.get("cam_from_rig_rotation", [1.0, 0.0, 0.0, 0.0]),
                "cam_from_rig_translation": item.get("cam_from_rig_translation", [0.0, 0.0, 0.0]),
            }
        )

    if ref_camera_id is None:
        raise ValueError(f"ref_view_name '{ref_view_name}' did not match any camera")

    rig_payload = {
        "ref_camera_id": ref_camera_id,
        "cameras": rig_cameras,
    }
    rig_config_path.write_text(json.dumps(rig_payload, indent=2), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir).resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_root}")

    colmap_bin = _resolve_colmap_binary(args.colmap_bin)
    paths = _load_project_paths(output_root)

    images_dir = paths["images"]
    masks_dir = paths["masks"]
    workspace_dir = paths["workspace"]
    db_path = paths["database"]
    sparse_dir = paths["sparse"]
    rig_template_path = paths["rig_template"]
    rig_config_path = paths["rig_config"]

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Dataset images directory not found: {images_dir}")
    if not rig_template_path.is_file():
        raise FileNotFoundError(f"Rig template not found: {rig_template_path}")

    workspace_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()
        print(f"Removed existing database: {db_path}")

    feature_cmd = [
        colmap_bin,
        "feature_extractor",
        "--database_path",
        str(db_path),
        "--image_path",
        str(images_dir),
        "--ImageReader.single_camera",
        "0",
        "--ImageReader.single_camera_per_folder",
        "1",
        "--ImageReader.camera_model",
        "SIMPLE_PINHOLE",
    ]

    if args.sift_max_features > 0 and _supports_option(colmap_bin, "feature_extractor", "--SiftExtraction.max_num_features"):
        feature_cmd.extend(["--SiftExtraction.max_num_features", str(args.sift_max_features)])

    if args.use_masks and masks_dir.is_dir() and _supports_option(colmap_bin, "feature_extractor", "--ImageReader.mask_path"):
        feature_cmd.extend(["--ImageReader.mask_path", str(masks_dir)])

    _run_command(feature_cmd, "feature_extractor")
    if args.run_until == "feature":
        return

    _write_rig_config_from_database(db_path, rig_template_path, rig_config_path)
    rig_cmd = [
        colmap_bin,
        "rig_configurator",
        "--database_path",
        str(db_path),
        "--rig_config_path",
        str(rig_config_path),
    ]
    _run_command(rig_cmd, "rig_configurator")
    if args.run_until == "rig":
        return

    matcher_name = "exhaustive_matcher" if args.matcher == "exhaustive" else "sequential_matcher"
    match_cmd = [
        colmap_bin,
        matcher_name,
        "--database_path",
        str(db_path),
    ]

    if matcher_name == "sequential_matcher":
        if args.seq_overlap > 0 and _supports_option(colmap_bin, matcher_name, "--SequentialMatching.overlap"):
            match_cmd.extend(["--SequentialMatching.overlap", str(args.seq_overlap)])
        if args.seq_loop_detection and _supports_option(colmap_bin, matcher_name, "--SequentialMatching.loop_detection"):
            match_cmd.extend(["--SequentialMatching.loop_detection", "1"])
            if args.vocab_tree_path:
                vocab = Path(args.vocab_tree_path)
                if not vocab.is_file():
                    raise FileNotFoundError(f"Vocabulary tree not found: {vocab}")
                if _supports_option(colmap_bin, matcher_name, "--SequentialMatching.vocab_tree_path"):
                    match_cmd.extend(["--SequentialMatching.vocab_tree_path", str(vocab)])

    _run_command(match_cmd, matcher_name)
    if args.run_until == "match":
        return

    mapper_cmd = [
        colmap_bin,
        "mapper",
        "--database_path",
        str(db_path),
        "--image_path",
        str(images_dir),
        "--output_path",
        str(sparse_dir),
    ]

    opt_name = "--Mapper.ba_refine_sensor_from_rig"
    if _supports_option(colmap_bin, "mapper", opt_name):
        mapper_cmd.extend([opt_name, "1" if args.refine_sensor_from_rig else "0"])

    if (
        args.mapper_ba_global_max_iter > 0
        and _supports_option(colmap_bin, "mapper", "--Mapper.ba_global_max_num_iterations")
    ):
        mapper_cmd.extend(["--Mapper.ba_global_max_num_iterations", str(args.mapper_ba_global_max_iter)])

    _run_command(mapper_cmd, "mapper")


def main() -> None:
    args = parse_args()
    try:
        run_pipeline(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
