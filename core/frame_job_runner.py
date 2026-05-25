from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.apply_frame_decisions import apply_decisions
from core.cancellation import CancellationToken, raise_if_cancelled
from core.frame_job_spec import (
    JOB_KIND_APPLY_FRAME_DECISIONS,
    JOB_KIND_EXTRACT_VIDEO,
    JOB_KIND_IMPORT_IMAGE_SEQUENCE,
    load_frame_job,
    validate_frame_job_payload,
)
from core.image_sequence_import import import_image_sequence_folder


def run_frame_job_file(path: str | Path, *, cancel_event: CancellationToken | None = None) -> None:
    run_frame_job_payload(load_frame_job(path), cancel_event=cancel_event)


def run_frame_job_payload(job: dict[str, Any], *, cancel_event: CancellationToken | None = None) -> None:
    validate_frame_job_payload(job)
    raise_if_cancelled(cancel_event)
    kind = str(job["kind"])
    if kind == JOB_KIND_EXTRACT_VIDEO:
        _run_extract_video(job, cancel_event=cancel_event)
        return
    if kind == JOB_KIND_IMPORT_IMAGE_SEQUENCE:
        _run_import_image_sequence(job, cancel_event=cancel_event)
        return
    if kind == JOB_KIND_APPLY_FRAME_DECISIONS:
        _run_apply_frame_decisions(job, cancel_event=cancel_event)
        return
    raise ValueError(f"Unsupported frame job kind: {kind}")


def _run_extract_video(job: dict[str, Any], *, cancel_event: CancellationToken | None = None) -> None:
    from core.extract_frames import ExtractFramesOptions, run_extract_frames

    raise_if_cancelled(cancel_event)
    options = ExtractFramesOptions(
        input_video=Path(str(job["input_video"])),
        output_dir=Path(str(job["scene_dir"])),
        image_ext=str(job["image_ext"]),
        jpg_quality=int(job["jpg_quality"]),
        ffmpeg=str(job["ffmpeg"]),
        ffprobe=str(job["ffprobe"]),
        output_mode=str(job["output_mode"]),
        filename_prefix=str(job["filename_prefix"]).strip(),
        interval_sec=float(job["interval_sec"]),
        quick_extract=bool(job["quick_extract"]),
        pair_motion_profile=str(job["pair_motion_profile"]),
        analysis_width=int(job["analysis_width"]),
        fixed_smart=bool(job["fixed_smart"]),
        min_gap_sec=float(job["min_gap_sec"]),
        max_gap_sec=float(job["max_gap_sec"]),
        allow_duplicate_video=bool(job["allow_duplicate_video"]),
        estimate_only=bool(job["estimate_only"]),
        print_summary_json=bool(job["print_summary_json"]),
        cancel_event=cancel_event,
    )

    exit_code = int(run_extract_frames(options) or 0)
    if exit_code != 0:
        raise RuntimeError(f"Frame extraction failed with exit code {exit_code}")


def _run_import_image_sequence(job: dict[str, Any], *, cancel_event: CancellationToken | None = None) -> None:
    raise_if_cancelled(cancel_event)
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
    raise_if_cancelled(cancel_event)


def _run_apply_frame_decisions(job: dict[str, Any], *, cancel_event: CancellationToken | None = None) -> None:
    raise_if_cancelled(cancel_event)
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
    raise_if_cancelled(cancel_event)
