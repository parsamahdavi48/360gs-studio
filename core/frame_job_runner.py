from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.apply_frame_decisions import apply_decisions
from core.image_sequence_import import import_image_sequence_folder

JOB_KIND_APPLY_FRAME_DECISIONS = "apply_frame_decisions"
JOB_KIND_EXTRACT_VIDEO = "extract_video"
JOB_KIND_IMPORT_IMAGE_SEQUENCE = "import_image_sequence"


def run_frame_job_payload(job: dict[str, Any]) -> None:
    kind = str(job["kind"])
    if kind == JOB_KIND_EXTRACT_VIDEO:
        _run_extract_video(job)
        return
    if kind == JOB_KIND_IMPORT_IMAGE_SEQUENCE:
        _run_import_image_sequence(job)
        return
    if kind == JOB_KIND_APPLY_FRAME_DECISIONS:
        _run_apply_frame_decisions(job)
        return
    raise ValueError(f"Unsupported frame job kind: {kind}")


def _run_extract_video(job: dict[str, Any]) -> None:
    from core.extract_frames import main as extract_main

    argv = [
        str(job["input_video"]),
        str(job["scene_dir"]),
        "--image-ext",
        str(job.get("image_ext") or "jpg"),
        "--jpg-quality",
        str(job.get("jpg_quality", 2)),
        "--ffmpeg",
        str(job.get("ffmpeg") or "ffmpeg"),
        "--ffprobe",
        str(job.get("ffprobe") or "ffprobe"),
        "--output-mode",
        str(job.get("output_mode") or "overwrite"),
        "--interval-sec",
        f"{float(job.get('interval_sec', 0.5)):g}",
    ]
    prefix = str(job.get("filename_prefix") or "").strip()
    if prefix:
        argv.extend(["--filename-prefix", prefix])

    if bool(job.get("quick_extract")):
        argv.append("--quick-extract")
    else:
        argv.extend(
            [
                "--pair-motion-profile",
                str(job.get("pair_motion_profile") or "walk_standard"),
                "--analysis-width",
                str(job.get("analysis_width") or 1920),
            ]
        )
        if bool(job.get("fixed_smart")):
            argv.extend(
                [
                    "--fixed-smart",
                    "--min-gap-sec",
                    f"{float(job.get('min_gap_sec', 0.25)):g}",
                    "--max-gap-sec",
                    f"{float(job.get('max_gap_sec', 2.0)):g}",
                ]
            )
    if bool(job.get("allow_duplicate_video")):
        argv.append("--allow-duplicate-video")
    if bool(job.get("estimate_only")):
        argv.append("--estimate-only")
    if bool(job.get("print_summary_json")):
        argv.append("--print-summary-json")

    exit_code = int(extract_main(argv) or 0)
    if exit_code != 0:
        raise RuntimeError(f"Frame extraction failed with exit code {exit_code}")


def _run_import_image_sequence(job: dict[str, Any]) -> None:
    result = import_image_sequence_folder(
        Path(str(job["source_dir"])),
        Path(str(job["scene_dir"])),
        prefix=str(job.get("prefix") or ""),
        recursive=bool(job.get("recursive")),
    )
    print(
        "SUMMARY_JSON:"
        + json.dumps(
            {
                "source": str(result.source_dir),
                "scene": str(result.scene_dir),
                "import_id": result.import_id,
                "image_count": result.image_count,
                "output_files": list(result.output_files),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _run_apply_frame_decisions(job: dict[str, Any]) -> None:
    scene_dir = Path(str(job["scene_dir"])).resolve()
    backup_dir: Path | None = None
    if str(job.get("backup_dir") or ""):
        raw_backup = Path(str(job["backup_dir"]))
        backup_dir = raw_backup if raw_backup.is_absolute() else scene_dir / raw_backup
    apply_decisions(
        scene_dir=scene_dir,
        csv_name=str(job.get("csv") or "selected_frames.csv"),
        output_name=str(job.get("output") or "metashape_images"),
        clean_output=bool(job.get("clean_output")),
        finalize_inplace=bool(job.get("finalize_in_place")),
        backup_dir=backup_dir,
        renumber_kept_images=bool(job.get("renumber_kept_images")),
    )
