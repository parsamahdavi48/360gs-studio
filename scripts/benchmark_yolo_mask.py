from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    args: tuple[str, ...]


CONFIGS = {
    "standard": BenchmarkConfig(
        "standard",
        ("--quality", "standard", "--projection", "equirect"),
    ),
    "high": BenchmarkConfig(
        "high",
        (
            "--quality",
            "high",
            "--projection",
            "equirect",
        ),
    ),
    "best": BenchmarkConfig(
        "best",
        (
            "--quality",
            "best",
            "--projection",
            "equirect",
        ),
    ),
}
DEFAULT_CONFIG_NAMES = ("standard", "high")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeatable YOLO mask benchmarks and optional mask comparisons.")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root containing an images/ directory.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory for benchmark outputs.")
    parser.add_argument("--label", required=True, help="Benchmark label, for example baseline or candidate.")
    parser.add_argument("--repeat", type=int, default=1, help="Runs per config (default=1).")
    parser.add_argument(
        "--config",
        action="append",
        choices=sorted([*CONFIGS.keys(), "all"]),
        help="Config to run. Repeat this option, or use all. Default: standard and high.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Copy only the first N images into each run input (default=all).")
    parser.add_argument("--compare-label", help="Compare generated masks against another label under --output-root.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the existing output directory for --label.")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python executable to use.")
    parser.add_argument(
        "--yolo-script",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "yolo_mask.py",
        help="Path to yolo_mask.py.",
    )
    return parser.parse_args(argv)


def select_configs(names: list[str] | None) -> list[BenchmarkConfig]:
    if not names:
        return [CONFIGS[name] for name in DEFAULT_CONFIG_NAMES]
    if "all" in names:
        return [CONFIGS[name] for name in CONFIGS]
    return [CONFIGS[name] for name in names]


def resolve_images_dir(dataset: Path) -> Path:
    images_dir = dataset / "images"
    if images_dir.is_dir():
        return images_dir
    if dataset.is_dir():
        return dataset
    raise FileNotFoundError(f"Dataset images directory not found: {images_dir}")


def image_files(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def prepare_run_input(images_dir: Path, run_dir: Path, limit: int) -> Path:
    if limit <= 0:
        return images_dir

    selected = image_files(images_dir)[:limit]
    if not selected:
        raise FileNotFoundError(f"No benchmark images found in {images_dir}")

    subset_dir = run_dir / "input_images"
    subset_dir.mkdir(parents=True, exist_ok=True)
    for src in selected:
        shutil.copy2(src, subset_dir / src.name)
    return subset_dir


def safe_replace_dir(path: Path, root: Path) -> None:
    path = path.resolve()
    root = root.resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Refusing to remove a directory outside output root: {path}")
    if path.exists():
        shutil.rmtree(path)


def load_profile_summary(profile_path: Path) -> dict:
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    image_totals = [
        float(image.get("timings_sec", {}).get("image.total", 0.0))
        for image in data.get("images", [])
        if "image.total" in image.get("timings_sec", {})
    ]
    stage_totals = data.get("totals", {}).get("timings_sec", {})
    return {
        "profile": str(profile_path),
        "images": int(data.get("totals", {}).get("images", len(data.get("images", [])))),
        "elapsed_sec": float(data.get("totals", {}).get("elapsed_sec", 0.0)),
        "median_image_sec": median(image_totals) if image_totals else None,
        "inference_calls": int(data.get("totals", {}).get("inference_calls", 0)),
        "yolo_boxes": int(data.get("totals", {}).get("yolo_boxes", 0)),
        "sam_masks": int(data.get("totals", {}).get("sam_masks", 0)),
        "stage_totals_sec": stage_totals,
    }


def compare_masks(reference_dir: Path, candidate_dir: Path) -> dict:
    reference_files = sorted(reference_dir.rglob("*.png"))
    candidate_files = sorted(candidate_dir.rglob("*.png"))
    candidate_rel = {path.relative_to(candidate_dir): path for path in candidate_files}
    reference_rel = {path.relative_to(reference_dir): path for path in reference_files}

    per_file = []
    missing_candidate = []
    shape_mismatch = []

    for rel, ref_path in reference_rel.items():
        candidate_path = candidate_rel.get(rel)
        if candidate_path is None:
            missing_candidate.append(str(rel))
            continue

        ref = read_mask_grayscale(ref_path)
        cand = read_mask_grayscale(candidate_path)
        if ref is None or cand is None:
            shape_mismatch.append(str(rel))
            continue
        if ref.shape != cand.shape:
            shape_mismatch.append(str(rel))
            continue

        ref_binary = ref > 0
        cand_binary = cand > 0
        intersection = int(np.logical_and(ref_binary, cand_binary).sum())
        union = int(np.logical_or(ref_binary, cand_binary).sum())
        diff_pixels = int(np.count_nonzero(ref != cand))
        white_ref = int(ref_binary.sum())
        white_cand = int(cand_binary.sum())
        per_file.append(
            {
                "file": str(rel),
                "iou": 1.0 if union == 0 else intersection / union,
                "diff_pixels": diff_pixels,
                "diff_ratio": diff_pixels / ref.size,
                "white_ref": white_ref,
                "white_candidate": white_cand,
                "white_delta": white_cand - white_ref,
                "exact": diff_pixels == 0,
            }
        )

    missing_reference = [str(rel) for rel in sorted(set(candidate_rel) - set(reference_rel))]
    if per_file:
        mean_iou = sum(item["iou"] for item in per_file) / len(per_file)
        mean_diff_ratio = sum(item["diff_ratio"] for item in per_file) / len(per_file)
        worst_by_iou = sorted(per_file, key=lambda item: item["iou"])[:10]
        worst_by_diff = sorted(per_file, key=lambda item: item["diff_ratio"], reverse=True)[:10]
    else:
        mean_iou = None
        mean_diff_ratio = None
        worst_by_iou = []
        worst_by_diff = []

    return {
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "files_compared": len(per_file),
        "exact_match_count": sum(1 for item in per_file if item["exact"]),
        "missing_candidate": missing_candidate,
        "missing_reference": missing_reference,
        "shape_mismatch": shape_mismatch,
        "mean_iou": mean_iou,
        "mean_diff_ratio": mean_diff_ratio,
        "max_diff_ratio": max((item["diff_ratio"] for item in per_file), default=None),
        "worst_by_iou": worst_by_iou,
        "worst_by_diff": worst_by_diff,
    }


def read_mask_grayscale(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def aggregate_runs(runs: list[dict]) -> dict:
    by_config: dict[str, list[dict]] = {}
    for run in runs:
        by_config.setdefault(run["config"], []).append(run)

    aggregate = {}
    for config_name, config_runs in by_config.items():
        elapsed = [run["profile_summary"]["elapsed_sec"] for run in config_runs if run.get("profile_summary")]
        aggregate[config_name] = {
            "runs": len(config_runs),
            "median_elapsed_sec": median(elapsed) if elapsed else None,
            "best_elapsed_sec": min(elapsed) if elapsed else None,
            "worst_elapsed_sec": max(elapsed) if elapsed else None,
        }
    return aggregate


def run_benchmark(args: argparse.Namespace) -> int:
    dataset = args.dataset.resolve()
    output_root = args.output_root.resolve()
    label_dir = (output_root / args.label).resolve()
    if output_root == label_dir or output_root not in label_dir.parents:
        raise ValueError(f"Refusing to write benchmark label outside output root: {args.label}")
    images_dir = resolve_images_dir(dataset)
    configs = select_configs(args.config)

    output_root.mkdir(parents=True, exist_ok=True)
    if label_dir.exists():
        if not args.overwrite:
            print(f"Output label already exists. Use --overwrite to replace it: {label_dir}", file=sys.stderr)
            return 2
        safe_replace_dir(label_dir, output_root)
    label_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for config in configs:
        for repeat_idx in range(1, max(1, args.repeat) + 1):
            run_name = f"run_{repeat_idx:02d}"
            run_dir = label_dir / config.name / run_name
            masks_dir = run_dir / "masks"
            profile_path = run_dir / "profile.json"
            run_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
            run_input_dir = prepare_run_input(images_dir, run_dir, args.limit)

            command = [
                str(args.python),
                str(args.yolo_script),
                str(run_input_dir),
                str(masks_dir),
                "--profile-json",
                str(profile_path),
                *config.args,
            ]
            print(f"[benchmark] {config.name} {run_name}", flush=True)
            started = perf_counter()
            completed = subprocess.run(command, check=False)
            wall_sec = perf_counter() - started
            if completed.returncode != 0:
                return completed.returncode

            profile_summary = load_profile_summary(profile_path)
            run_result = {
                "config": config.name,
                "run": run_name,
                "command": command,
                "wall_sec": wall_sec,
                "masks_dir": str(masks_dir),
                "profile_summary": profile_summary,
            }

            if args.compare_label:
                reference_dir = output_root / args.compare_label / config.name / "run_01" / "masks"
                if not reference_dir.is_dir():
                    raise FileNotFoundError(f"Comparison masks not found: {reference_dir}")
                comparison = compare_masks(reference_dir, masks_dir)
                comparison_path = run_dir / "mask_comparison.json"
                comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
                run_result["mask_comparison"] = comparison

            runs.append(run_result)

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "images_dir": str(images_dir),
        "output_root": str(output_root),
        "label": args.label,
        "configs": [config.name for config in configs],
        "repeat": max(1, args.repeat),
        "limit": max(0, args.limit),
        "runs": runs,
        "aggregate": aggregate_runs(runs),
    }
    summary_path = label_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[benchmark] wrote {summary_path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_benchmark(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
