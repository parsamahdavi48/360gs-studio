#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
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


def sanitize_filename_prefix(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    if not text:
        return ""
    return text


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


def run_cmd_with_ffmpeg_progress(cmd: List[str], phase: str, total_items: int) -> subprocess.CompletedProcess:
    if total_items <= 0:
        return run_cmd(cmd, capture=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stderr is not None

    progress_step = max(1, total_items // 100)
    last_reported = -1
    observed_frame = 0
    stderr_lines: List[str] = []

    print(f"[progress] {phase} 0/{total_items} frames (0.0%)", flush=True)
    for raw in proc.stderr:
        line = raw.strip()
        if not line:
            continue
        if "=" not in line:
            stderr_lines.append(line)
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in {
            "frame",
            "fps",
            "stream_0_0_q",
            "bitrate",
            "total_size",
            "out_time_us",
            "out_time_ms",
            "out_time",
            "dup_frames",
            "drop_frames",
            "speed",
            "progress",
        }:
            stderr_lines.append(line)
        if key != "frame":
            continue
        try:
            frame_count = int(value)
        except ValueError:
            continue

        if frame_count < observed_frame:
            continue
        observed_frame = frame_count
        if observed_frame == 0:
            continue

        if observed_frame - last_reported >= progress_step or observed_frame >= total_items:
            shown = min(total_items, observed_frame)
            pct = min(100.0, (shown / float(total_items)) * 100.0)
            print(f"[progress] {phase} {shown}/{total_items} frames ({pct:.1f}%)", flush=True)
            last_reported = observed_frame

    proc.wait()
    if proc.returncode == 0 and last_reported < total_items:
        print(f"[progress] {phase} {total_items}/{total_items} frames (100.0%)", flush=True)

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout="",
        stderr="\n".join(stderr_lines),
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
    if duration <= 0 and total_frames > 0:
        duration = total_frames / fps
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


def analyze_video_window(
    video_path: Path,
    ffmpeg_bin: str,
    video_fps: float,
    src_w: int,
    src_h: int,
    analysis_width: int,
    start_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
    sample_fps: float = 0.0,
    progress_phase: str = "",
    progress_total_frames: int = 0,
    progress_step_frames: int = 0,
) -> Tuple[List[float], List[float], int, int, float]:
    out_w, out_h = scaled_dimensions(src_w, src_h, analysis_width)
    vf_parts = [f"scale={out_w}:{out_h}:flags=bilinear", "format=gray"]
    effective_fps = video_fps
    if sample_fps > 0:
        effective_fps = min(sample_fps, video_fps) if video_fps > 0 else sample_fps
        vf_parts.append(f"fps={effective_fps:.6f}")
    vf = ",".join(vf_parts)

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if start_sec is not None and start_sec > 0:
        cmd.extend(["-ss", f"{start_sec:.6f}"])
    cmd.extend(["-i", str(video_path)])
    if duration_sec is not None and duration_sec > 0:
        cmd.extend(["-t", f"{duration_sec:.6f}"])
    cmd.extend(
        [
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
    )

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    assert proc.stderr is not None

    frame_size = out_w * out_h
    prev_frame: Optional[np.ndarray] = None
    blur_scores: List[float] = []
    change_scores: List[float] = []
    last_progress_report = 0

    def emit_progress(processed_frames: int, force: bool = False) -> None:
        nonlocal last_progress_report
        if not progress_phase:
            return
        if not force:
            if progress_step_frames <= 0:
                return
            if processed_frames - last_progress_report < progress_step_frames:
                return
        if progress_total_frames > 0:
            pct = min(100.0, (processed_frames / float(progress_total_frames)) * 100.0)
            print(
                f"[progress] {progress_phase} {processed_frames}/{progress_total_frames} frames ({pct:.1f}%)",
                flush=True,
            )
        else:
            print(f"[progress] {progress_phase} {processed_frames} frames", flush=True)
        last_progress_report = processed_frames

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
            processed = len(change_scores)
            if processed == 1:
                emit_progress(processed, force=True)
            else:
                emit_progress(processed)
    finally:
        stderr_text = proc.stderr.read().decode("utf-8", errors="replace")
        ret = proc.wait()

    if ret != 0:
        raise RuntimeError(f"ffmpeg analysis failed: {stderr_text.strip()}")
    if not blur_scores:
        raise RuntimeError("No frames decoded during analysis")
    emit_progress(len(blur_scores), force=True)

    return blur_scores, change_scores, out_w, out_h, effective_fps


def analyze_video(
    video_path: Path,
    ffmpeg_bin: str,
    video_fps: float,
    src_w: int,
    src_h: int,
    analysis_width: int,
    progress_phase: str = "",
    progress_total_frames: int = 0,
    progress_step_frames: int = 0,
) -> Tuple[List[float], List[float], int, int]:
    blur_scores, change_scores, out_w, out_h, _ = analyze_video_window(
        video_path=video_path,
        ffmpeg_bin=ffmpeg_bin,
        video_fps=video_fps,
        src_w=src_w,
        src_h=src_h,
        analysis_width=analysis_width,
        start_sec=None,
        duration_sec=None,
        sample_fps=0.0,
        progress_phase=progress_phase,
        progress_total_frames=progress_total_frames,
        progress_step_frames=progress_step_frames,
    )
    return blur_scores, change_scores, out_w, out_h


# ===========================================================================
# 解析キャッシュ: 動画メタ情報 + analysis_width が一致すれば再計算をスキップ
# ===========================================================================

CACHE_VERSION = 1


def cache_path_for(scene_dir: Path) -> Path:
    return scene_dir / "extract_cache.npz"


def video_signature(video_path: Path) -> Tuple[int, int]:
    """動画ファイルの (size, mtime_ns) を返す。キャッシュ無効化判定用。"""
    st = video_path.stat()
    return int(st.st_size), int(st.st_mtime_ns)


def save_analysis_cache(
    cache_path: Path,
    video_path: Path,
    video_info: VideoInfo,
    analysis_w: int,
    analysis_h: int,
    blur_scores: List[float],
    change_scores: List[float],
) -> None:
    if np is None:
        return
    size, mtime_ns = video_signature(video_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        version=np.int64(CACHE_VERSION),
        video_size=np.int64(size),
        video_mtime_ns=np.int64(mtime_ns),
        video_width=np.int32(video_info.width),
        video_height=np.int32(video_info.height),
        video_fps=np.float64(video_info.fps),
        video_duration=np.float64(video_info.duration),
        video_total_frames=np.int64(video_info.total_frames),
        analysis_width=np.int32(analysis_w),
        analysis_height=np.int32(analysis_h),
        blur_scores=np.asarray(blur_scores, dtype=np.float64),
        change_scores=np.asarray(change_scores, dtype=np.float64),
    )
    print(f"[cache] saved analysis cache: {cache_path}")


def thin_stationary(
    rows: List[dict],
    change_scores: List[float],
    motion_threshold: float,
    keep_endpoints: bool = True,
) -> List[dict]:
    """累積モーションが閾値未満の連続採用フレームを間引く。

    立ち止まり区間（変化が小さい）でフレーム間隔が密集しすぎているのを削減する。
    歩行など変化が大きい区間では何も削らない。

    Args:
        rows: apply_blur_replacement の出力行（各 dict に "final_index" がある）。
        change_scores: 解析時の change_scores (per analyzed frame)。
        motion_threshold: 直前 kept フレームから次採用候補までの累積 change がこれ未満なら drop。
            0 以下なら間引きなし（全フレーム keep）。
        keep_endpoints: True なら各 stationary cluster の先頭・末尾は強制保持。

    Returns:
        各 row に "decision" と必要なら "status" を追加した同長のリスト。
        間引かれた row は decision="drop", status="thinned"。
        keep される row は decision="keep" を維持/設定。
    """
    if motion_threshold <= 0.0 or len(rows) < 2:
        for row in rows:
            row.setdefault("decision", "keep")
        return rows

    n = len(change_scores)
    out_rows = [dict(row) for row in rows]

    # 累積モーション計算: rows は final_index 順を仮定（apply_blur_replacement は順番を保つ）
    last_kept_pos = 0
    out_rows[0]["decision"] = "keep"

    for pos in range(1, len(out_rows) - (1 if keep_endpoints else 0)):
        last_idx = int(out_rows[last_kept_pos]["final_index"])
        cur_idx = int(out_rows[pos]["final_index"])
        if cur_idx <= last_idx or last_idx < 0 or cur_idx >= n:
            # インデックス異常時は安全側で keep
            out_rows[pos]["decision"] = "keep"
            last_kept_pos = pos
            continue

        # last_idx (排他) から cur_idx (含む) までの change_scores を累積
        cumulative = float(np.sum(change_scores[last_idx + 1 : cur_idx + 1]))

        if cumulative >= motion_threshold:
            out_rows[pos]["decision"] = "keep"
            last_kept_pos = pos
        else:
            out_rows[pos]["decision"] = "drop"
            existing_status = out_rows[pos].get("status", "ok")
            if existing_status == "ok":
                out_rows[pos]["status"] = "thinned"
            else:
                out_rows[pos]["status"] = f"{existing_status}+thinned"

    # 末尾は強制保持（時間カバレッジ）
    if keep_endpoints and len(out_rows) >= 2:
        out_rows[-1]["decision"] = "keep"
    elif not keep_endpoints and len(out_rows) >= 2:
        # keep_endpoints=False なら末尾も判定対象
        last_idx = int(out_rows[last_kept_pos]["final_index"])
        cur_idx = int(out_rows[-1]["final_index"])
        if cur_idx > last_idx and last_idx >= 0 and cur_idx < n:
            cumulative = float(np.sum(change_scores[last_idx + 1 : cur_idx + 1]))
            if cumulative < motion_threshold:
                out_rows[-1]["decision"] = "drop"
                existing_status = out_rows[-1].get("status", "ok")
                if existing_status == "ok":
                    out_rows[-1]["status"] = "thinned"
                else:
                    out_rows[-1]["status"] = f"{existing_status}+thinned"
            else:
                out_rows[-1]["decision"] = "keep"
        else:
            out_rows[-1]["decision"] = "keep"

    return out_rows


def load_analysis_cache(
    cache_path: Path,
    video_path: Path,
    video_info: VideoInfo,
    analysis_width: int,
) -> Optional[Tuple[List[float], List[float], int, int]]:
    """キャッシュが有効なら (blur_scores, change_scores, analysis_w, analysis_h) を返す。
    無効/不在/エラーなら None。"""
    if np is None or not cache_path.exists():
        return None
    try:
        with np.load(cache_path) as data:
            if int(data["version"]) != CACHE_VERSION:
                return None
            cur_size, cur_mtime_ns = video_signature(video_path)
            if int(data["video_size"]) != cur_size:
                return None
            if int(data["video_mtime_ns"]) != cur_mtime_ns:
                return None
            cached_aw = int(data["analysis_width"])
            # 解析幅が現在の指定と一致していること（同等の縮小寸法を生成するため厳密比較）
            cached_target_w, _ = scaled_dimensions(video_info.width, video_info.height, analysis_width)
            if cached_aw != cached_target_w:
                return None
            blur = data["blur_scores"].tolist()
            change = data["change_scores"].tolist()
            ah = int(data["analysis_height"])
            return blur, change, cached_aw, ah
    except Exception as e:
        print(f"[cache] failed to load cache (will recompute): {e}")
        return None


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


def estimate_count_range(duration_sec: float, min_gap_sec: float, max_gap_sec: float) -> Tuple[int, int]:
    if duration_sec <= 0:
        return 1, 1
    if min_gap_sec <= 0 or max_gap_sec <= 0:
        raise ValueError("--min-gap-sec and --max-gap-sec must be > 0")
    if max_gap_sec < min_gap_sec:
        raise ValueError("--max-gap-sec must be >= --min-gap-sec")

    min_count = max(1, int(math.ceil(duration_sec / max_gap_sec)))
    max_count = max(min_count, int(math.ceil(duration_sec / min_gap_sec)))
    return min_count, max_count


def build_sample_windows(
    duration_sec: float,
    segment_sec: float,
    segment_count: int,
) -> List[Tuple[float, float]]:
    if segment_sec <= 0:
        raise ValueError("--sample-segment-sec must be > 0")
    if segment_count <= 0:
        raise ValueError("--sample-segments must be > 0")
    if duration_sec <= 0:
        return []

    actual_segment = min(segment_sec, duration_sec)
    max_start = max(0.0, duration_sec - actual_segment)

    if segment_count == 1 or max_start <= 0:
        starts = [0.0]
    else:
        step = max_start / float(segment_count - 1)
        starts = [i * step for i in range(segment_count)]

    windows: List[Tuple[float, float]] = []
    seen = set()
    for start in starts:
        clamped_start = max(0.0, min(start, max_start))
        key = int(round(clamped_start * 1000))
        if key in seen:
            continue
        seen.add(key)
        seg_dur = min(actual_segment, duration_sec - clamped_start)
        if seg_dur <= 0:
            continue
        windows.append((clamped_start, seg_dur))

    if not windows:
        windows.append((0.0, actual_segment))
    return windows


def estimate_change_sampled(
    video_path: Path,
    ffmpeg_bin: str,
    video_info: VideoInfo,
    analysis_width: int,
    threshold: float,
    min_gap_sec: float,
    max_gap_sec: float,
    sample_segments: int,
    sample_segment_sec: float,
    sample_fps: float,
) -> dict:
    windows = build_sample_windows(video_info.duration, sample_segment_sec, sample_segments)
    if not windows:
        raise RuntimeError("Could not build sample windows from video duration")

    weighted_rate_sum = 0.0
    weight_sum = 0.0
    used_segments = 0
    sampled_frames = 0
    sampled_duration = 0.0

    analysis_w = 0
    analysis_h = 0
    analysis_fps = min(sample_fps, video_info.fps) if sample_fps > 0 else video_info.fps
    min_gap_frames = max(1, int(round(min_gap_sec * analysis_fps)))

    for idx, (start_sec, seg_sec) in enumerate(windows, start=1):
        _, change_scores, out_w, out_h, seg_fps = analyze_video_window(
            video_path=video_path,
            ffmpeg_bin=ffmpeg_bin,
            video_fps=video_info.fps,
            src_w=video_info.width,
            src_h=video_info.height,
            analysis_width=analysis_width,
            start_sec=start_sec,
            duration_sec=seg_sec,
            sample_fps=sample_fps,
        )

        if len(change_scores) < 2:
            print(f"[sample] segment {idx}/{len(windows)} skipped: insufficient frames")
            continue

        selected, seg_min_gap_frames, _ = select_change(
            change_scores,
            seg_fps,
            threshold,
            min_gap_sec,
            max_gap_sec,
        )
        decoded_sec = len(change_scores) / seg_fps if seg_fps > 0 else seg_sec
        if decoded_sec <= 0:
            continue

        analysis_w = out_w
        analysis_h = out_h
        analysis_fps = seg_fps
        min_gap_frames = seg_min_gap_frames

        # Each sampled segment seeds one first frame; remove that bias before extrapolation.
        segment_selected = max(0, len(selected) - 1)
        selected_per_sec = segment_selected / decoded_sec

        weighted_rate_sum += selected_per_sec * decoded_sec
        weight_sum += decoded_sec
        used_segments += 1
        sampled_frames += len(change_scores)
        sampled_duration += decoded_sec

        print(
            f"[sample] segment {idx}/{len(windows)} start={start_sec:.2f}s "
            f"dur={decoded_sec:.2f}s selected={len(selected)}"
        )

    if used_segments == 0 or weight_sum <= 0:
        raise RuntimeError("Sampled estimate failed: no valid segment data")

    selected_rate = weighted_rate_sum / weight_sum
    estimated = 1 + int(round(selected_rate * max(video_info.duration, 0.0)))
    range_min, range_max = estimate_count_range(video_info.duration, min_gap_sec, max_gap_sec)
    estimated = max(range_min, min(range_max, estimated))

    window_frames = compute_auto_window(analysis_fps, min_gap_frames)
    return {
        "selected_count": estimated,
        "replaced_count": 0,
        "fallback_keep_count": 0,
        "analysis_w": analysis_w,
        "analysis_h": analysis_h,
        "analysis_fps": analysis_fps,
        "min_gap_frames": min_gap_frames,
        "window_frames": window_frames,
        "sampled_segments_requested": len(windows),
        "sampled_segments_used": used_segments,
        "sampled_duration_sec": sampled_duration,
        "sampled_frames": sampled_frames,
        "range_min_count": range_min,
        "range_max_count": range_max,
    }


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
    filename_prefix: str,
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
            "-nostats",
            "-progress",
            "pipe:2",
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
        proc = run_cmd_with_ffmpeg_progress(cmd, phase="extract", total_items=len(frame_indices))

        if proc.returncode != 0:
            # Fallback when filter_script:v is unsupported by ffmpeg build.
            cmd = [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostats",
                "-progress",
                "pipe:2",
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
            proc = run_cmd_with_ffmpeg_progress(cmd, phase="extract", total_items=len(frame_indices))
            stderr_text = (proc.stderr or "").lower()
            if proc.returncode != 0 and "unrecognized option" in stderr_text and "progress" in stderr_text:
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

    rename_total = len(frame_indices)
    rename_step = max(1, rename_total // 100)
    last_rename_report = 0
    print(f"[progress] finalize 0/{rename_total} files (0.0%)", flush=True)
    for seq, (src, frame_idx) in enumerate(zip(extracted_files, frame_indices), start=1):
        dst_name = f"{filename_prefix}_{frame_idx:06d}.{image_ext}"
        dst_path = output_dir / dst_name
        if dst_path.exists():
            dst_path.unlink()
        src.rename(dst_path)
        if seq - last_rename_report >= rename_step or seq == rename_total:
            pct = min(100.0, (seq / float(rename_total)) * 100.0)
            print(f"[progress] finalize {seq}/{rename_total} files ({pct:.1f}%)", flush=True)
            last_rename_report = seq

    tmp_dir.rmdir()


def write_selected_csv(
    rows: List[dict],
    csv_path: Path,
    fps: float,
    image_ext: str,
    filename_prefix: str,
) -> None:
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
                    "output_file": f"images/{filename_prefix}_{final_idx:06d}.{image_ext}",
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
    filename_prefix: str,
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
            "filename_prefix": filename_prefix,
        },
        "result": {
            "selected_count": len(selected_rows),
            "replaced_count": sum(1 for r in selected_rows if r["status"] == "replaced"),
            "fallback_keep_count": sum(1 for r in selected_rows if r["status"] == "fallback_keep"),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def build_summary_from_counts(
    args: argparse.Namespace,
    video_info: VideoInfo,
    analysis_w: int,
    analysis_h: int,
    min_gap_frames: int,
    window_frames: int,
    selected_count: int,
    replaced_count: int,
    fallback_keep_count: int,
    estimate_mode: str,
    filename_prefix: str,
    estimate_meta: Optional[dict] = None,
) -> dict:
    summary = {
        "input_video": str(Path(args.input_video).resolve()),
        "mode": args.mode,
        "estimate_mode": estimate_mode,
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
            "filename_prefix": filename_prefix,
        },
        "result": {
            "selected_count": selected_count,
            "replaced_count": replaced_count,
            "fallback_keep_count": fallback_keep_count,
        },
    }
    if estimate_meta:
        summary["estimate"] = estimate_meta
    return summary


def build_summary(
    args: argparse.Namespace,
    video_info: VideoInfo,
    analysis_w: int,
    analysis_h: int,
    selected_rows: List[dict],
    min_gap_frames: int,
    window_frames: int,
    filename_prefix: str,
) -> dict:
    replaced_count = sum(1 for r in selected_rows if "replaced" in r.get("status", ""))
    fallback_keep_count = sum(1 for r in selected_rows if "fallback_keep" in r.get("status", ""))
    thinned_count = sum(
        1 for r in selected_rows
        if r.get("decision") == "drop" and "thinned" in r.get("status", "")
    )
    kept_count = sum(1 for r in selected_rows if r.get("decision", "keep") != "drop")

    summary = build_summary_from_counts(
        args=args,
        video_info=video_info,
        analysis_w=analysis_w,
        analysis_h=analysis_h,
        min_gap_frames=min_gap_frames,
        window_frames=window_frames,
        selected_count=kept_count,
        replaced_count=replaced_count,
        fallback_keep_count=fallback_keep_count,
        estimate_mode="full",
        filename_prefix=filename_prefix,
    )
    summary["result"]["thinned_count"] = thinned_count
    summary["result"]["selected_before_thin"] = len(selected_rows)
    return summary


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
        default=1920,
        help=(
            "Analysis decode width for change/blur scoring (default=1920). "
            "Higher values give more accurate Laplacian variance / change detection at the cost of "
            "analysis time. Set to 0 or a value >= source width to use full resolution."
        ),
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
    parser.add_argument(
        "--filename-prefix",
        default="",
        help="Output filename prefix. Default is input video stem.",
    )

    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg executable")
    parser.add_argument("--ffprobe", default="ffprobe", help="Path to ffprobe executable")
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Run probe/analysis/selection and print estimated selected count without image extraction",
    )
    parser.add_argument(
        "--estimate-mode",
        choices=["full", "sampled"],
        default="full",
        help="Estimate mode when --estimate-only is set (full=all frames, sampled=window sampling)",
    )
    parser.add_argument(
        "--sample-segments",
        type=int,
        default=5,
        help="Number of temporal windows for sampled estimate mode",
    )
    parser.add_argument(
        "--sample-segment-sec",
        type=float,
        default=12.0,
        help="Duration (seconds) for each sampled estimate window",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=8.0,
        help="Temporal fps used in sampled estimate windows",
    )
    parser.add_argument(
        "--print-summary-json",
        action="store_true",
        help="Print one-line JSON summary prefixed with SUMMARY_JSON:",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write the analysis cache (extract_cache.npz). Forces full re-analysis.",
    )
    parser.add_argument(
        "--thin-motion-threshold",
        type=float,
        default=0.0,
        help=(
            "Stationary thinning: drop selected frames whose cumulative change_score since the "
            "last kept frame is below this threshold. Adapts to recording style: stops are thinned, "
            "walking is preserved. Default=0.0 (disabled). 0.5-1.0 is a reasonable starting range."
        ),
    )
    parser.add_argument(
        "--no-thin-keep-endpoints",
        dest="thin_keep_endpoints",
        action="store_false",
        default=True,
        help="When thinning, allow the last frame to be dropped too (default keeps endpoints to preserve time coverage).",
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
        ensure_binary(args.ffmpeg, "ffmpeg")
        ensure_binary(args.ffprobe, "ffprobe")
        video_info = probe_video(input_video, args.ffprobe)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Input video: {input_video}")
    print(f"Video: {video_info.width}x{video_info.height} @ {video_info.fps:.3f} fps")
    resolved_prefix = sanitize_filename_prefix(args.filename_prefix)
    if not resolved_prefix:
        resolved_prefix = sanitize_filename_prefix(input_video.stem)
    if not resolved_prefix:
        resolved_prefix = "frame"
    print(f"Filename prefix: {resolved_prefix}")

    if args.estimate_only and args.mode == "fixed":
        total_frames = video_info.total_frames
        if total_frames <= 0 and video_info.duration > 0:
            total_frames = max(1, int(round(video_info.duration * video_info.fps)))
        total_frames = max(total_frames, 1)

        try:
            selected, min_gap_frames = select_fixed(total_frames, video_info.fps, args.interval_sec)
        except Exception as e:
            print(f"Error while selecting frames: {e}")
            sys.exit(1)

        analysis_w, analysis_h = scaled_dimensions(video_info.width, video_info.height, args.analysis_width)
        window_frames = compute_auto_window(video_info.fps, min_gap_frames)
        summary = build_summary_from_counts(
            args=args,
            video_info=video_info,
            analysis_w=analysis_w,
            analysis_h=analysis_h,
            min_gap_frames=min_gap_frames,
            window_frames=window_frames,
            selected_count=len(selected),
            replaced_count=0,
            fallback_keep_count=0,
            estimate_mode="fixed_exact",
            filename_prefix=resolved_prefix,
        )
        print(f"Estimated selected frames: {summary['result']['selected_count']}")
        print("Estimated replaced frames: 0")
        print("Estimated fallback keep frames: 0")
        if args.print_summary_json:
            print("SUMMARY_JSON:" + json.dumps(summary, ensure_ascii=False))
        return

    if args.estimate_only and args.mode == "change" and args.estimate_mode == "sampled":
        try:
            ensure_python_deps()
            sampled = estimate_change_sampled(
                video_path=input_video,
                ffmpeg_bin=args.ffmpeg,
                video_info=video_info,
                analysis_width=args.analysis_width,
                threshold=args.change_threshold,
                min_gap_sec=args.min_gap_sec,
                max_gap_sec=args.max_gap_sec,
                sample_segments=args.sample_segments,
                sample_segment_sec=args.sample_segment_sec,
                sample_fps=args.sample_fps,
            )
        except Exception as e:
            print(f"Error during sampled estimate: {e}")
            sys.exit(1)

        summary = build_summary_from_counts(
            args=args,
            video_info=video_info,
            analysis_w=sampled["analysis_w"],
            analysis_h=sampled["analysis_h"],
            min_gap_frames=sampled["min_gap_frames"],
            window_frames=sampled["window_frames"],
            selected_count=sampled["selected_count"],
            replaced_count=sampled["replaced_count"],
            fallback_keep_count=sampled["fallback_keep_count"],
            estimate_mode="sampled",
            filename_prefix=resolved_prefix,
            estimate_meta={
                "sampled_segments_requested": sampled["sampled_segments_requested"],
                "sampled_segments_used": sampled["sampled_segments_used"],
                "sampled_duration_sec": sampled["sampled_duration_sec"],
                "sampled_frames": sampled["sampled_frames"],
                "sampled_fps": sampled["analysis_fps"],
                "range_min_count": sampled["range_min_count"],
                "range_max_count": sampled["range_max_count"],
            },
        )

        print(f"Estimated selected frames: {summary['result']['selected_count']}")
        print(f"Estimated replaced frames: {summary['result']['replaced_count']}")
        print(f"Estimated fallback keep frames: {summary['result']['fallback_keep_count']}")
        print(
            "[sample] used segments: "
            f"{sampled['sampled_segments_used']}/{sampled['sampled_segments_requested']} "
            f"(duration={sampled['sampled_duration_sec']:.2f}s, decoded={sampled['sampled_frames']} frames)"
        )
        if args.print_summary_json:
            print("SUMMARY_JSON:" + json.dumps(summary, ensure_ascii=False))
        return

    try:
        ensure_python_deps()

        cache_path = cache_path_for(scene_dir)
        cached: Optional[Tuple[List[float], List[float], int, int]] = None
        if not args.no_cache:
            cached = load_analysis_cache(cache_path, input_video, video_info, args.analysis_width)
            if cached is not None:
                print(f"[cache] reusing analysis cache: {cache_path}")

        if cached is not None:
            blur_scores, change_scores, analysis_w, analysis_h = cached
        else:
            progress_step = max(10, video_info.total_frames // 100) if video_info.total_frames > 0 else max(
                10, int(round(video_info.fps * 2.0))
            )
            blur_scores, change_scores, analysis_w, analysis_h = analyze_video(
                input_video,
                args.ffmpeg,
                video_info.fps,
                video_info.width,
                video_info.height,
                args.analysis_width,
                progress_phase="analyze",
                progress_total_frames=video_info.total_frames,
                progress_step_frames=progress_step,
            )
            if not args.no_cache:
                try:
                    save_analysis_cache(
                        cache_path, input_video, video_info,
                        analysis_w, analysis_h, blur_scores, change_scores,
                    )
                except Exception as e:
                    print(f"[cache] failed to save cache (non-fatal): {e}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

    total_frames = len(blur_scores)
    print(f"Analyzed frames: {total_frames} ({analysis_w}x{analysis_h})")

    try:
        if args.mode == "fixed":
            selected, min_gap_frames = select_fixed(total_frames, video_info.fps, args.interval_sec)
            change_threshold = None
        else:
            selected, min_gap_frames, _ = select_change(
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

    # 立ち止まり間引き: 累積モーションが閾値未満の連続区間を drop でマーク
    if args.thin_motion_threshold > 0.0:
        enriched_rows = thin_stationary(
            enriched_rows,
            change_scores,
            motion_threshold=args.thin_motion_threshold,
            keep_endpoints=args.thin_keep_endpoints,
        )
        thinned_count = sum(
            1 for r in enriched_rows
            if r.get("decision") == "drop" and "thinned" in r.get("status", "")
        )
        kept_count = sum(1 for r in enriched_rows if r.get("decision") != "drop")
        print(
            f"Stationary thinning: dropped {thinned_count}, kept {kept_count} "
            f"(threshold={args.thin_motion_threshold:g})"
        )

    # decision=drop の行は抽出対象から除外（CSV メタとしては保持）
    final_indices = [
        r["final_index"] for r in enriched_rows if r.get("decision", "keep") != "drop"
    ]

    summary = build_summary(
        args=args,
        video_info=video_info,
        analysis_w=analysis_w,
        analysis_h=analysis_h,
        selected_rows=enriched_rows,
        min_gap_frames=min_gap_frames,
        window_frames=window_frames,
        filename_prefix=resolved_prefix,
    )

    if args.estimate_only:
        print(f"Estimated selected frames: {summary['result']['selected_count']}")
        print(f"Estimated replaced frames: {summary['result']['replaced_count']}")
        print(f"Estimated fallback keep frames: {summary['result']['fallback_keep_count']}")
        if args.print_summary_json:
            print("SUMMARY_JSON:" + json.dumps(summary, ensure_ascii=False))
        return

    if not final_indices:
        print("Error: no frames remain after thinning; skipping extraction")
        sys.exit(1)

    try:
        extract_selected_frames(
            input_video,
            args.ffmpeg,
            final_indices,
            images_dir,
            args.image_ext,
            args.jpg_quality,
            resolved_prefix,
        )
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)

    write_selected_csv(enriched_rows, csv_path, video_info.fps, args.image_ext, resolved_prefix)
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
        resolved_prefix,
    )

    replaced_count = sum(1 for r in enriched_rows if "replaced" in r.get("status", ""))
    fallback_count = sum(1 for r in enriched_rows if "fallback_keep" in r.get("status", ""))
    thinned_count = sum(
        1 for r in enriched_rows
        if r.get("decision") == "drop" and "thinned" in r.get("status", "")
    )
    kept_count = len(final_indices)

    print(f"Selected frames: {kept_count} (extracted)")
    if thinned_count > 0:
        print(f"Thinned (stationary, recorded as drop in CSV): {thinned_count}")
    print(f"Replaced blurred frames: {replaced_count}")
    print(f"Fallback keep frames: {fallback_count}")
    print(f"Images: {images_dir}")
    print(f"Selection CSV: {csv_path}")
    print(f"Report: {report_path}")
    if args.print_summary_json:
        print("SUMMARY_JSON:" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
