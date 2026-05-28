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
    JOB_KIND_IMPORT_SCENE,
    JOB_KIND_REFRESH_SCENE_ASSETS,
    load_frame_job,
    validate_frame_job_payload,
)
from core.image_sequence_import import import_image_sequence_folder
from core.scene_import import import_scene
from core.scene_import_contracts import SceneImportCancelled


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
    if kind == JOB_KIND_IMPORT_SCENE:
        _run_import_scene(job, cancel_event=cancel_event)
        return
    if kind == JOB_KIND_APPLY_FRAME_DECISIONS:
        _run_apply_frame_decisions(job, cancel_event=cancel_event)
        return
    if kind == JOB_KIND_REFRESH_SCENE_ASSETS:
        _run_refresh_scene_assets(job, cancel_event=cancel_event)
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


class _SceneImportJobCancelToken:
    def __init__(self, cancel_event: CancellationToken | None) -> None:
        self._cancel_event = cancel_event

    def request_cancel(self) -> None:
        pass

    def is_cancelled(self) -> bool:
        return bool(self._cancel_event is not None and self._cancel_event.is_set())

    def check_cancelled(self) -> None:
        if self.is_cancelled():
            raise SceneImportCancelled("Scene import canceled.")


def _run_import_scene(job: dict[str, Any], *, cancel_event: CancellationToken | None = None) -> None:
    from core.cancellation import AppJobCancelled

    raise_if_cancelled(cancel_event)
    scene = Path(str(job["scene_dir"]))
    token = _SceneImportJobCancelToken(cancel_event)
    try:
        result = import_scene(
            scene,
            cancel_token=token,
            progress_callback=lambda message: print(message, flush=True),
        )
    except SceneImportCancelled as exc:
        raise AppJobCancelled(str(exc)) from exc
    for line in result.summary_lines():
        print(line, flush=True)
    print(
        "SUMMARY_JSON:"
        + json.dumps(
            {
                "kind": "scene_import",
                "scene_dir": str(result.scene_dir),
                "import_id": result.import_id,
                "status": result.status,
                "image_count": result.image_count,
                "mask_count": result.mask_count,
                "output_image_count": result.output_image_count,
                "output_mask_count": result.output_mask_count,
                "output_shape": result.output_shape,
                "dataset_kind": result.dataset_kind,
                "warnings": list(result.warnings),
                "errors": list(result.errors),
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


def _run_refresh_scene_assets(job: dict[str, Any], *, cancel_event: CancellationToken | None = None) -> None:
    from core.scene_asset_metadata import rebuild_scene_asset_metadata

    raise_if_cancelled(cancel_event)
    scene = Path(str(job["scene_dir"])).resolve()
    payload = rebuild_scene_asset_metadata(scene, cancel_event=cancel_event)
    images = payload.get("images") if isinstance(payload.get("images"), dict) else {}
    masks = payload.get("masks") if isinstance(payload.get("masks"), dict) else {}
    print(
        "SUMMARY_JSON:"
        + json.dumps(
            {
                "scene": str(scene),
                "image_count": len(images),
                "mask_count": len(masks),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
