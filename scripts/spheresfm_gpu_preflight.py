"""Run a tiny isolated GPU SIFT preflight for SphereSfM."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from prepare_spheresfm_project import validate_spheresfm_colmap
except ImportError:  # pragma: no cover - used when imported as scripts.spheresfm_gpu_preflight
    from scripts.prepare_spheresfm_project import validate_spheresfm_colmap


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
PREFLIGHT_MAX_IMAGE_SIZE = "1024"
PREFLIGHT_MAX_NUM_FEATURES = "2048"


def iter_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def reset_preflight_workspace(work_dir: Path) -> Path:
    resolved = work_dir.resolve()
    if resolved.name.lower() != "preflight":
        raise ValueError(f"Preflight work folder must end with 'preflight': {work_dir}")

    images_dir = work_dir / "images"
    database = work_dir / "database.db"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    if database.exists():
        database.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def run_colmap_command(cmd: list[str], label: str) -> None:
    print("$ " + subprocess.list2cmdline(cmd), flush=True)
    try:
        result = subprocess.run(cmd, check=False)
    except OSError as exc:
        raise RuntimeError(f"{label} could not start: {exc}") from exc

    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def build_feature_command(colmap: str, database: Path, images_dir: Path, camera_params: str) -> list[str]:
    return [
        colmap,
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(images_dir),
        "--ImageReader.camera_model",
        "SPHERE",
        "--ImageReader.camera_params",
        camera_params,
        "--ImageReader.single_camera",
        "1",
        "--SiftExtraction.use_gpu",
        "1",
        "--SiftExtraction.max_image_size",
        PREFLIGHT_MAX_IMAGE_SIZE,
        "--SiftExtraction.max_num_features",
        PREFLIGHT_MAX_NUM_FEATURES,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check whether SphereSfM GPU SIFT can run on one image.")
    parser.add_argument("--colmap", required=True, help="SphereSfM colmap executable")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--camera-params", required=True)
    args = parser.parse_args(argv)

    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {args.images_dir}")
    images = iter_images(args.images_dir)
    if not images:
        raise FileNotFoundError(f"No supported images found: {args.images_dir}")

    validate_spheresfm_colmap(args.colmap)

    preflight_images = reset_preflight_workspace(args.work_dir)
    source = images[0]
    target = preflight_images / f"preflight_000001{source.suffix.lower()}"
    shutil.copy2(source, target)
    print(f"SphereSfM GPU preflight image: {source}", flush=True)

    database = args.work_dir / "database.db"
    run_colmap_command(
        [
            args.colmap,
            "database_creator",
            "--database_path",
            str(database),
        ],
        "SphereSfM preflight database_creator",
    )
    run_colmap_command(
        build_feature_command(args.colmap, database, preflight_images, args.camera_params),
        "SphereSfM preflight feature_extractor",
    )
    print("SphereSfM GPU preflight passed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
