from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare legacy and pair frame extraction analysis on the same video."
    )
    parser.add_argument("video", help="Input video path")
    parser.add_argument("output_dir", help="Directory for comparison outputs")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable")
    parser.add_argument("--ffprobe", default="ffprobe", help="ffprobe executable")
    parser.add_argument("--analysis-width", type=int, default=1920)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--min-gap-sec", type=float, default=0.5)
    parser.add_argument("--max-gap-sec", type=float, default=2.0)
    parser.add_argument("--pair-motion-profile", choices=["walk", "drone"], default="walk")
    parser.add_argument("--pair-drop-threshold", type=float, default=-1.0)
    parser.add_argument("--pair-add-threshold", type=float, default=-1.0)
    parser.add_argument("--image-ext", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpg-quality", type=int, default=2)
    parser.add_argument("--extract", action="store_true", help="Write images instead of estimate-only analysis")
    parser.add_argument("--use-cache", action="store_true", help="Allow extract_cache.npz reuse")
    return parser.parse_args()


def _summary_from_stdout(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("SUMMARY_JSON:"):
            return json.loads(line[len("SUMMARY_JSON:") :])
    raise RuntimeError("SUMMARY_JSON line not found")


def _run_case(args: argparse.Namespace, repo_root: Path, pipeline: str) -> dict:
    case_dir = Path(args.output_dir).resolve() / pipeline
    case_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        "-u",
        str(repo_root / "extract_frames.py"),
        str(Path(args.video).resolve()),
        str(case_dir),
        "--mode",
        "fixed",
        "--fixed-smart",
        "--analysis-pipeline",
        pipeline,
        "--pair-motion-profile",
        args.pair_motion_profile,
        "--pair-drop-threshold",
        f"{args.pair_drop_threshold:g}",
        "--pair-add-threshold",
        f"{args.pair_add_threshold:g}",
        "--interval-sec",
        f"{args.interval_sec:g}",
        "--min-gap-sec",
        f"{args.min_gap_sec:g}",
        "--max-gap-sec",
        f"{args.max_gap_sec:g}",
        "--analysis-width",
        str(args.analysis_width),
        "--image-ext",
        args.image_ext,
        "--jpg-quality",
        str(args.jpg_quality),
        "--thin-motion-threshold",
        "0",
        "--print-summary-json",
        "--ffmpeg",
        args.ffmpeg,
        "--ffprobe",
        args.ffprobe,
    ]
    if not args.extract:
        cmd.append("--estimate-only")
    if not args.use_cache:
        cmd.append("--no-cache")

    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"{pipeline} run failed with exit code {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    summary = _summary_from_stdout(proc.stdout)
    image_count = 0
    images_dir = case_dir / "images"
    if images_dir.exists():
        image_count = sum(1 for p in images_dir.iterdir() if p.is_file())

    return {
        "pipeline": pipeline,
        "elapsed_sec": elapsed,
        "output_dir": str(case_dir),
        "image_count": image_count,
        "summary": summary,
        "stdout_tail": proc.stdout.splitlines()[-20:],
    }


def _result_row(result: dict) -> dict:
    summary = result["summary"]
    analysis = summary.get("analysis", {})
    selected = summary.get("result", {})
    return {
        "pipeline": result["pipeline"],
        "elapsed_sec": round(result["elapsed_sec"], 3),
        "selected_count": selected.get("selected_count", 0),
        "selected_before_drop": selected.get("selected_before_thin", selected.get("selected_count", 0)),
        "novelty_added": selected.get("novelty_added_count", selected.get("smart_added_count", 0)),
        "redundant_drop": selected.get("redundant_drop_count", selected.get("thinned_count", 0)),
        "gap_forced": selected.get("gap_forced_count", 0),
        "motion_blur": selected.get("motion_blur_count", 0),
        "low_texture": selected.get("low_texture_count", 0),
        "weak_match": selected.get("weak_match_count", 0),
        "pair_drop_threshold": summary.get("params", {}).get("pair_drop_threshold_resolved", ""),
        "pair_add_threshold": summary.get("params", {}).get("pair_add_threshold_resolved", ""),
        "pair_gate_width": summary.get("params", {}).get("pair_gate_width", ""),
        "analysis_width": analysis.get("width", 0),
        "analysis_height": analysis.get("height", 0),
        "image_count": result["image_count"],
    }


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [_run_case(args, repo_root, "legacy"), _run_case(args, repo_root, "pair")]
    rows = [_result_row(result) for result in results]
    report = {
        "video": str(Path(args.video).resolve()),
        "mode": "extract" if args.extract else "estimate-only",
        "params": {
            "analysis_width": args.analysis_width,
            "interval_sec": args.interval_sec,
            "min_gap_sec": args.min_gap_sec,
            "max_gap_sec": args.max_gap_sec,
            "pair_motion_profile": args.pair_motion_profile,
            "pair_drop_threshold": args.pair_drop_threshold,
            "pair_add_threshold": args.pair_add_threshold,
            "use_cache": bool(args.use_cache),
        },
        "rows": rows,
        "results": results,
    }
    report_path = output_dir / "compare_extract_analysis.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("pipeline elapsed selected before_drop added dropped gap blur lowtex weak drop_thr add_thr analysis gate image_count")
    for row in rows:
        print(
            f"{row['pipeline']} {row['elapsed_sec']:.3f} {row['selected_count']} "
            f"{row['selected_before_drop']} {row['novelty_added']} {row['redundant_drop']} "
            f"{row['gap_forced']} {row['motion_blur']} {row['low_texture']} {row['weak_match']} "
            f"{row['pair_drop_threshold']} {row['pair_add_threshold']} "
            f"{row['analysis_width']}x{row['analysis_height']} {row['pair_gate_width']} {row['image_count']}"
        )
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
