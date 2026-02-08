#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import cv2
except Exception as e:  # pragma: no cover - environment-dependent import
    cv2 = None
    _CV2_IMPORT_ERROR = e
else:
    _CV2_IMPORT_ERROR = None

try:
    import numpy as np
except Exception as e:  # pragma: no cover - environment-dependent import
    np = None
    _NP_IMPORT_ERROR = e
else:
    _NP_IMPORT_ERROR = None


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    total_frames: int


def parse_fraction(value: str) -> float:
    if not value:
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        if den_f == 0:
            return 0.0
        return float(num) / den_f
    return float(value)


def ensure_python_deps() -> None:
    missing = []
    if cv2 is None:
        missing.append(f"opencv-python (cv2 import failed: {_CV2_IMPORT_ERROR})")
    if np is None:
        missing.append(f"numpy (import failed: {_NP_IMPORT_ERROR})")
    if missing:
        raise RuntimeError("Missing required Python modules: " + "; ".join(missing))


def run_cmd(cmd: List[str], capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        check=False,
    )


def ensure_binary(path: str, name: str) -> None:
    proc = run_cmd([path, "-version"], capture=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip() if proc.stderr else "not found"
        raise RuntimeError(f"Failed to execute {name}: {msg}")


def probe_video(video_path: Path, ffprobe_bin: str) -> VideoInfo:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    proc = run_cmd(cmd, capture=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")

    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No video stream found")

    stream = streams[0]
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    fps = parse_fraction(stream.get("avg_frame_rate", "0"))
    if fps <= 0:
        fps = parse_fraction(stream.get("r_frame_rate", "0"))

    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
    nb_frames_raw = stream.get("nb_frames")
    total_frames = int(nb_frames_raw) if nb_frames_raw and nb_frames_raw.isdigit() else 0

    if fps <= 0 and duration > 0 and total_frames > 0:
        fps = total_frames / duration
    if fps <= 0:
        raise RuntimeError("Could not determine FPS from video")
    if total_frames <= 0 and duration > 0:
        total_frames = max(1, int(round(duration * fps)))

    return VideoInfo(width=width, height=height, fps=fps, duration=duration, total_frames=total_frames)


def scaled_dimensions(width: int, height: int, analysis_width: int) -> Tuple[int, int]:
    if analysis_width <= 0 or analysis_width >= width:
        return width, height

    scaled_h = int(round(height * (analysis_width / float(width))))
    if scaled_h < 2:
        scaled_h = 2
    if scaled_h % 2 != 0:
        scaled_h += 1
    return analysis_width, scaled_h


def analyze_video(
    video_path: Path,
    ffmpeg_bin: str,
    src_w: int,
    src_h: int,
    analysis_width: int,
) -> Tuple[List[float], List[float], int, int]:
    out_w, out_h = scaled_dimensions(src_w, src_h, analysis_width)
    vf = f"scale={out_w}:{out_h}:flags=bilinear,format=gray"

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    assert proc.stderr is not None

    frame_size = out_w * out_h
    prev_frame: Optional[np.ndarray] = None
    blur_scores: List[float] = []
    change_scores: List[float] = []

    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if not raw:
                break
            if len(raw) < frame_size:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((out_h, out_w))
            lap_var = float(cv2.Laplacian(frame, cv2.CV_64F).var())
            blur_scores.append(lap_var)

            if prev_frame is None:
                change_scores.append(1.0)
            else:
                diff = cv2.absdiff(frame, prev_frame)
                change_scores.append(float(np.mean(diff) / 255.0))

            prev_frame = frame
    finally:
        stderr_text = proc.stderr.read().decode("utf-8", errors="replace")
        ret = proc.wait()

    if ret != 0:
        raise RuntimeError(f"ffmpeg analysis failed: {stderr_text.strip()}")
    if not blur_scores:
        raise RuntimeError("No frames decoded during analysis")

    return blur_scores, change_scores, out_w, out_h


def select_fixed(total_frames: int, fps: float, interval_sec: float) -> Tuple[List[int], int]:
    if interval_sec <= 0:
        raise ValueError("--interval-sec must be > 0")

    step = max(1, int(round(interval_sec * fps)))
    indices = list(range(0, total_frames, step))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    return indices, step


def select_change(
    change_scores: List[float],
    fps: float,
    threshold: float,
    min_gap_sec: float,
    max_gap_sec: float,
) -> Tuple[List[int], int, int]:
    if min_gap_sec <= 0 or max_gap_sec <= 0:
        raise ValueError("--min-gap-sec and --max-gap-sec must be > 0")
    if max_gap_sec < min_gap_sec:
        raise ValueError("--max-gap-sec must be >= --min-gap-sec")

    min_gap_frames = max(1, int(round(min_gap_sec * fps)))
    max_gap_frames = max(min_gap_frames, int(round(max_gap_sec * fps)))

    indices = [0]
    last = 0
    for i in range(1, len(change_scores)):
        gap = i - last
        if gap < min_gap_frames:
            continue
        if change_scores[i] >= threshold or gap >= max_gap_frames:
            indices.append(i)
            last = i

    if indices[-1] != len(change_scores) - 1:
        if len(change_scores) - 1 - indices[-1] >= max(1, min_gap_frames // 2):
            indices.append(len(change_scores) - 1)

    return indices, min_gap_frames, max_gap_frames


def compute_auto_window(fps: float, min_gap_frames: int) -> int:
    return max(1, min(int(round(0.35 * fps)), 12, max(1, int(round(0.6 * min_gap_frames)))))


def apply_blur_replacement(
    selected_indices: List[int],
    blur_scores: List[float],
    change_scores: List[float],
    blur_percentile: float,
    window_frames: int,
    min_gap_frames: int,
    change_threshold: Optional[float],
) -> List[dict]:
    if not selected_indices:
        return []

    blur_values = [blur_scores[idx] for idx in selected_indices]
    blur_threshold = float(np.percentile(blur_values, blur_percentile))
    soft_gap = max(1, min_gap_frames // 2)

    rows: List[dict] = []
    prev_final = -10**9
    n = len(blur_scores)

    for pos, original_idx in enumerate(selected_indices):
        original_blur = blur_scores[original_idx]
        next_original = selected_indices[pos + 1] if pos + 1 < len(selected_indices) else n - 1

        low = max(0, original_idx - window_frames, prev_final + soft_gap)
        high = min(n - 1, original_idx + window_frames, next_original - soft_gap)

        final_idx = original_idx
        status = "ok"

        if original_blur < blur_threshold:
            candidates: List[int] = []
            if low <= high:
                for cand in range(low, high + 1):
                    if cand != original_idx and change_threshold is not None:
                        if change_scores[cand] < change_threshold * 0.8:
                            continue
                    candidates.append(cand)

            if candidates:
                best_idx = max(candidates, key=lambda x: blur_scores[x])
                if blur_scores[best_idx] > original_blur:
                    final_idx = best_idx
                    status = "replaced"
                else:
                    status = "fallback_keep"
            else:
                status = "fallback_keep"

        prev_final = final_idx
        rows.append(
            {
                "original_index": original_idx,
                "final_index": final_idx,
                "status": status,
                "blur_threshold": blur_threshold,
            }
        )

    return rows


def build_select_expr(frame_indices: List[int]) -> str:
    return "+".join(f"eq(n\\,{idx})" for idx in frame_indices)


def extract_selected_frames(
    video_path: Path,
    ffmpeg_bin: str,
    frame_indices: List[int],
    output_dir: Path,
    image_ext: str,
    jpg_quality: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_tmp_extract"
    if tmp_dir.exists():
        for p in tmp_dir.glob("*"):
            p.unlink(missing_ok=True)
    tmp_dir.mkdir(exist_ok=True)

    select_expr = build_select_expr(frame_indices)

    quality_args: List[str] = []
    if image_ext == "jpg":
        quality_args = ["-q:v", str(jpg_quality)]

    out_pattern = str(tmp_dir / f"%08d.{image_ext}")

    # Try filter script first to avoid command-length issues.
    with tempfile.NamedTemporaryFile("w", suffix=".ffscript", delete=False, encoding="utf-8") as tf:
        tf.write(f"select='{select_expr}'\n")
        filter_script_path = tf.name

    try:
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-filter_script:v",
            filter_script_path,
            "-vsync",
            "vfr",
            *quality_args,
            out_pattern,
        ]
        proc = run_cmd(cmd, capture=True)

        if proc.returncode != 0:
            # Fallback when filter_script:v is unsupported by ffmpeg build.
            cmd = [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"select='{select_expr}'",
                "-vsync",
                "vfr",
                *quality_args,
                out_pattern,
            ]
            proc = run_cmd(cmd, capture=True)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg extraction failed: {proc.stderr.strip()}")
    finally:
        Path(filter_script_path).unlink(missing_ok=True)

    extracted_files = sorted(tmp_dir.glob(f"*.{image_ext}"))
    if len(extracted_files) != len(frame_indices):
        raise RuntimeError(
            f"Expected {len(frame_indices)} extracted files, got {len(extracted_files)}"
        )

    for seq, (src, frame_idx) in enumerate(zip(extracted_files, frame_indices), start=1):
        dst_name = f"frame_{frame_idx:06d}.{image_ext}"
        dst_path = output_dir / dst_name
        if dst_path.exists():
            dst_path.unlink()
        src.rename(dst_path)

    tmp_dir.rmdir()


def write_selected_csv(rows: List[dict], csv_path: Path, fps: float, image_ext: str) -> None:
    fieldnames = [
        "seq",
        "original_index",
        "final_index",
        "timestamp_sec",
        "change_score_original",
        "change_score_final",
        "blur_score_original",
        "blur_score_final",
        "status",
        "decision",
        "output_file",
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            final_idx = row["final_index"]
            writer.writerow(
                {
                    "seq": i,
                    "original_index": row["original_index"],
                    "final_index": final_idx,
                    "timestamp_sec": f"{final_idx / fps:.6f}",
                    "change_score_original": f"{row['change_score_original']:.6f}",
                    "change_score_final": f"{row['change_score_final']:.6f}",
                    "blur_score_original": f"{row['blur_score_original']:.6f}",
                    "blur_score_final": f"{row['blur_score_final']:.6f}",
                    "status": row["status"],
                    "decision": row.get("decision", "keep"),
                    "output_file": f"images/frame_{final_idx:06d}.{image_ext}",
                }
            )


def write_report(
    report_path: Path,
    args: argparse.Namespace,
    video_info: VideoInfo,
    analysis_w: int,
    analysis_h: int,
    selected_rows: List[dict],
    blur_percentile: float,
    window_frames: int,
    min_gap_frames: int,
) -> None:
    report = {
        "input_video": str(Path(args.input_video).resolve()),
        "mode": args.mode,
        "video": {
            "width": video_info.width,
            "height": video_info.height,
            "fps": video_info.fps,
            "duration_sec": video_info.duration,
            "total_frames": video_info.total_frames,
        },
        "analysis": {
            "width": analysis_w,
            "height": analysis_h,
            "blur_percentile": blur_percentile,
            "blur_window_frames": window_frames,
            "min_gap_frames": min_gap_frames,
        },
        "params": {
            "interval_sec": args.interval_sec,
            "change_threshold": args.change_threshold,
            "min_gap_sec": args.min_gap_sec,
            "max_gap_sec": args.max_gap_sec,
        },
        "result": {
            "selected_count": len(selected_rows),
            "replaced_count": sum(1 for r in selected_rows if r["status"] == "replaced"),
            "fallback_keep_count": sum(1 for r in selected_rows if r["status"] == "fallback_keep"),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def build_summary(
    args: argparse.Namespace,
    video_info: VideoInfo,
    analysis_w: int,
    analysis_h: int,
    selected_rows: List[dict],
    min_gap_frames: int,
    window_frames: int,
) -> dict:
    return {
        "input_video": str(Path(args.input_video).resolve()),
        "mode": args.mode,
        "video": {
            "width": video_info.width,
            "height": video_info.height,
            "fps": video_info.fps,
            "duration_sec": video_info.duration,
            "total_frames": video_info.total_frames,
        },
        "analysis": {
            "width": analysis_w,
            "height": analysis_h,
            "min_gap_frames": min_gap_frames,
            "blur_window_frames": window_frames,
        },
        "params": {
            "interval_sec": args.interval_sec,
            "change_threshold": args.change_threshold,
            "min_gap_sec": args.min_gap_sec,
            "max_gap_sec": args.max_gap_sec,
            "analysis_width": args.analysis_width,
            "blur_percentile": args.blur_percentile,
            "blur_window_frames": args.blur_window_frames,
        },
        "result": {
            "selected_count": len(selected_rows),
            "replaced_count": sum(1 for r in selected_rows if r["status"] == "replaced"),
            "fallback_keep_count": sum(1 for r in selected_rows if r["status"] == "fallback_keep"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract equirectangular frames via FFmpeg with change-based selection and blur replacement."
    )
    parser.add_argument("input_video", help="Input video file path")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=".",
        help="Output root directory (default='.')",
    )

    parser.add_argument("--mode", choices=["fixed", "change"], default="change")
    parser.add_argument("--interval-sec", type=float, default=0.5, help="Fixed mode interval in seconds")
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=0.04,
        help="Change mode threshold (0.0-1.0) based on normalized frame difference",
    )
    parser.add_argument("--min-gap-sec", type=float, default=0.25, help="Minimum gap in seconds (change mode)")
    parser.add_argument("--max-gap-sec", type=float, default=2.0, help="Maximum gap in seconds (change mode)")

    parser.add_argument(
        "--analysis-width",
        type=int,
        default=960,
        help="Analysis decode width for change/blur scoring (default=960)",
    )
    parser.add_argument(
        "--blur-percentile",
        type=float,
        default=25.0,
        help="Selected frames below this blur percentile are replacement candidates",
    )
    parser.add_argument(
        "--blur-window-frames",
        type=int,
        default=0,
        help="Neighbor search window for blur replacement; 0 means auto",
    )

    parser.add_argument("--image-ext", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpg-quality", type=int, default=2, help="JPEG quality for ffmpeg -q:v (2 is high quality)")

    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg executable")
    parser.add_argument("--ffprobe", default="ffprobe", help="Path to ffprobe executable")
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Run probe/analysis/selection and print estimated selected count without image extraction",
    )
    parser.add_argument(
        "--print-summary-json",
        action="store_true",
        help="Print one-line JSON summary prefixed with SUMMARY_JSON:",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_video = Path(args.input_video)
    if not input_video.exists():
        print(f"Error: input video not found: {input_video}")
        sys.exit(1)

    output_root = Path(args.output_dir)
    scene_dir = output_root.resolve()
    images_dir = scene_dir / "images"
    csv_path = scene_dir / "selected_frames.csv"
    report_path = scene_dir / "extract_report.json"

    try:
        ensure_python_deps()
        ensure_binary(args.ffmpeg, "ffmpeg")
        ensure_binary(args.ffprobe, "ffprobe")
        video_info = probe_video(input_video, args.ffprobe)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Input video: {input_video}")
    print(f"Video: {video_info.width}x{video_info.height} @ {video_info.fps:.3f} fps")

    try:
        blur_scores, change_scores, analysis_w, analysis_h = analyze_video(
            input_video,
            args.ffmpeg,
            video_info.width,
            video_info.height,
            args.analysis_width,
        )
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

    total_frames = len(blur_scores)
    print(f"Analyzed frames: {total_frames} ({analysis_w}x{analysis_h})")

    try:
        if args.mode == "fixed":
            selected, min_gap_frames = select_fixed(total_frames, video_info.fps, args.interval_sec)
            max_gap_frames = min_gap_frames
            change_threshold = None
        else:
            selected, min_gap_frames, max_gap_frames = select_change(
                change_scores,
                video_info.fps,
                args.change_threshold,
                args.min_gap_sec,
                args.max_gap_sec,
            )
            change_threshold = args.change_threshold
    except Exception as e:
        print(f"Error while selecting frames: {e}")
        sys.exit(1)

    if not selected:
        print("Error: no frames selected")
        sys.exit(1)

    window_frames = args.blur_window_frames
    if window_frames <= 0:
        window_frames = compute_auto_window(video_info.fps, min_gap_frames)

    rows = apply_blur_replacement(
        selected,
        blur_scores,
        change_scores,
        args.blur_percentile,
        window_frames,
        min_gap_frames,
        change_threshold,
    )

    enriched_rows: List[dict] = []
    for row in rows:
        orig = row["original_index"]
        final = row["final_index"]
        enriched_rows.append(
            {
                **row,
                "change_score_original": change_scores[orig],
                "change_score_final": change_scores[final],
                "blur_score_original": blur_scores[orig],
                "blur_score_final": blur_scores[final],
                "decision": "keep",
            }
        )

    final_indices = [r["final_index"] for r in enriched_rows]

    summary = build_summary(
        args=args,
        video_info=video_info,
        analysis_w=analysis_w,
        analysis_h=analysis_h,
        selected_rows=enriched_rows,
        min_gap_frames=min_gap_frames,
        window_frames=window_frames,
    )

    if args.estimate_only:
        print(f"Estimated selected frames: {summary['result']['selected_count']}")
        print(f"Estimated replaced frames: {summary['result']['replaced_count']}")
        print(f"Estimated fallback keep frames: {summary['result']['fallback_keep_count']}")
        if args.print_summary_json:
            print("SUMMARY_JSON:" + json.dumps(summary, ensure_ascii=False))
        return

    try:
        extract_selected_frames(
            input_video,
            args.ffmpeg,
            final_indices,
            images_dir,
            args.image_ext,
            args.jpg_quality,
        )
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)

    write_selected_csv(enriched_rows, csv_path, video_info.fps, args.image_ext)
    write_report(
        report_path,
        args,
        video_info,
        analysis_w,
        analysis_h,
        enriched_rows,
        args.blur_percentile,
        window_frames,
        min_gap_frames,
    )

    replaced_count = sum(1 for r in enriched_rows if r["status"] == "replaced")
    fallback_count = sum(1 for r in enriched_rows if r["status"] == "fallback_keep")

    print(f"Selected frames: {len(enriched_rows)}")
    print(f"Replaced blurred frames: {replaced_count}")
    print(f"Fallback keep frames: {fallback_count}")
    print(f"Images: {images_dir}")
    print(f"Selection CSV: {csv_path}")
    print(f"Report: {report_path}")
    if args.print_summary_json:
        print("SUMMARY_JSON:" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
