from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from core.cancellation import CancellationToken, raise_if_cancelled
from core.colmap_nerfstudio_dataset import export_colmap_nerfstudio_dataset
from core.dataset_job_spec import (
    JOB_KIND_ATTACH_DATASET_MASKS,
    JOB_KIND_COLMAP_NERFSTUDIO,
    JOB_KIND_METASHAPE_COLMAP,
    JOB_KIND_METASHAPE_NERF,
    JOB_KIND_REALITYSCAN_LFS_COLMAP,
    load_dataset_job,
    validate_dataset_job_payload,
)
from core.dataset_mask_paths import attach_nerf_mask_paths, clear_nerf_mask_paths
from core.metashape_colmap_dataset import export_metashape_colmap_dataset
from core.metashape_nerf_dataset import export_metashape_nerf_dataset
from core.realityscan_to_lfs_colmap import convert as convert_realityscan_to_lfs_colmap


def _progress_log_callback(cancel_event: CancellationToken | None = None) -> Callable[[int, int], None]:
    last_bucket = -1
    last_pair: tuple[int, int] | None = None

    def callback(done: int, total: int) -> None:
        nonlocal last_bucket, last_pair
        raise_if_cancelled(cancel_event)
        done = max(0, int(done))
        total = max(0, int(total))
        if total <= 0:
            return
        pair = (min(done, total), total)
        if pair == last_pair:
            return
        bucket = int((pair[0] / float(total)) * 100.0)
        if pair[0] == 0 or pair[0] >= total or bucket != last_bucket:
            print(f"[progress] {pair[0]}/{total}", flush=True)
            last_bucket = bucket
            last_pair = pair

    return callback


def run_dataset_job_file(path: str | Path, *, cancel_event: CancellationToken | None = None) -> None:
    run_dataset_job_payload(load_dataset_job(path), cancel_event=cancel_event)


def run_dataset_job_payload(job: dict, *, cancel_event: CancellationToken | None = None) -> None:
    validate_dataset_job_payload(job)
    raise_if_cancelled(cancel_event)
    kind = str(job["kind"])
    if kind == JOB_KIND_METASHAPE_COLMAP:
        _run_metashape_colmap(job, cancel_event=cancel_event)
    elif kind == JOB_KIND_METASHAPE_NERF:
        _run_metashape_nerf(job, cancel_event=cancel_event)
    elif kind == JOB_KIND_COLMAP_NERFSTUDIO:
        _run_colmap_nerfstudio(job, cancel_event=cancel_event)
    elif kind == JOB_KIND_REALITYSCAN_LFS_COLMAP:
        _run_realityscan_lfs_colmap(job, cancel_event=cancel_event)
    elif kind == JOB_KIND_ATTACH_DATASET_MASKS:
        _run_attach_dataset_masks(job, cancel_event=cancel_event)
    else:
        raise ValueError(f"Unsupported dataset job kind: {kind}")


def _enabled_views(job: dict) -> list[dict]:
    return [dict(item) for item in job.get("views", []) if isinstance(item, dict) and bool(item.get("enabled", True))]


def _run_metashape_colmap(job: dict, *, cancel_event: CancellationToken | None = None) -> None:
    raise_if_cancelled(cancel_event)
    result = export_metashape_colmap_dataset(
        scene_dir=Path(str(job["scene_dir"])),
        images_dir=Path(str(job["images_dir"])),
        masks_dir=Path(str(job["masks_dir"])) if str(job.get("masks_dir") or "") else None,
        xml_path=Path(str(job["xml_path"])),
        ply_path=Path(str(job["ply_path"])) if str(job.get("ply_path") or "") else None,
        output_dir=Path(str(job["output_dir"])),
        views=_enabled_views(job),
        output_scale=float(job.get("output_scale", 1.0)),
        output_format=str(job.get("output_format") or "jpg"),
        output_bit_depth=str(job.get("output_bit_depth") or "8"),
        jpg_quality=int(job.get("jpg_quality", 95)),
        undistort_alpha=float(job.get("undistort_alpha", 1.0)),
        axis_transform=str(job.get("axis_transform") or "none"),
        final_orientation=str(job.get("final_orientation") or "none"),
        progress_callback=_progress_log_callback(cancel_event),
    )
    print(f"Saved mixed Metashape COLMAP dataset: {result.output_dir}", flush=True)
    print(f"Images: {result.image_count}", flush=True)
    print(f"Cameras: {result.camera_count}", flush=True)
    print(f"Actions: {json.dumps(result.action_counts, sort_keys=True)}", flush=True)
    raise_if_cancelled(cancel_event)


def _run_metashape_nerf(job: dict, *, cancel_event: CancellationToken | None = None) -> None:
    raise_if_cancelled(cancel_event)
    result = export_metashape_nerf_dataset(
        scene_dir=Path(str(job["scene_dir"])),
        images_dir=Path(str(job["images_dir"])),
        masks_dir=Path(str(job["masks_dir"])) if str(job.get("masks_dir") or "") else None,
        xml_path=Path(str(job["xml_path"])),
        ply_path=Path(str(job["ply_path"])) if str(job.get("ply_path") or "") else None,
        output_dir=Path(str(job["output_dir"])),
        views=_enabled_views(job),
        output_scale=float(job.get("output_scale", 1.0)),
        output_format=str(job.get("output_format") or "jpg"),
        output_bit_depth=str(job.get("output_bit_depth") or "8"),
        jpg_quality=int(job.get("jpg_quality", 95)),
        undistort_alpha=float(job.get("undistort_alpha", 1.0)),
        axis_transform=str(job.get("axis_transform") or "none"),
        final_orientation=str(job.get("final_orientation") or "none"),
        write_images=bool(job.get("write_images", True)),
        write_masks=bool(job.get("write_masks", True)),
        progress_callback=_progress_log_callback(cancel_event),
    )
    print(f"Saved mixed Metashape NeRF dataset: {result.output_dir}", flush=True)
    print(f"transforms.json: {result.transforms_json}", flush=True)
    if result.pointcloud:
        print(f"pointcloud.ply: {result.pointcloud}", flush=True)
    print(f"Frames: {result.frame_count}", flush=True)
    print(f"Actions: {json.dumps(result.action_counts, sort_keys=True)}", flush=True)
    raise_if_cancelled(cancel_event)


def _run_realityscan_lfs_colmap(job: dict, *, cancel_event: CancellationToken | None = None) -> None:
    raise_if_cancelled(cancel_event)
    result = convert_realityscan_to_lfs_colmap(
        csv_path=Path(str(job["csv_path"])),
        output_dir=Path(str(job["output_dir"])),
        images_dir=Path(str(job["images_dir"])),
        masks_dir=Path(str(job["masks_dir"])) if str(job.get("masks_dir") or "") else None,
        ply_path=Path(str(job["ply_path"])) if str(job.get("ply_path") or "") else None,
        skip_missing_images=bool(job.get("skip_missing_images")),
        pre_undistort_distorted_images=bool(job.get("pre_undistort_distorted_images")),
        undistort_alpha=float(job.get("undistort_alpha", 1.0)),
        camera_rotation_x_deg=float(job.get("camera_rotation_x_deg", 90.0)),
        pointcloud_rotation_x_deg=float(job.get("pointcloud_rotation_x_deg", 90.0)),
        progress_callback=_progress_log_callback(cancel_event),
    )
    print(f"Wrote RealityScan LichtFeld COLMAP dataset: {result['output_dir']}", flush=True)
    print(f"Images: {result['num_images']}", flush=True)
    print(f"Cameras: {result['num_cameras']}", flush=True)
    print(f"Skipped missing images: {result['num_missing_images']}", flush=True)
    raise_if_cancelled(cancel_event)


def _run_colmap_nerfstudio(job: dict, *, cancel_event: CancellationToken | None = None) -> None:
    raise_if_cancelled(cancel_event)
    result = export_colmap_nerfstudio_dataset(
        colmap_root=Path(str(job["colmap_root"])),
        output_dir=Path(str(job["output_dir"])),
        images_dir=Path(str(job["images_dir"])) if str(job.get("images_dir") or "") else None,
        masks_dir=Path(str(job["masks_dir"])) if str(job.get("masks_dir") or "") else None,
        sparse_dir=Path(str(job["sparse_dir"])) if str(job.get("sparse_dir") or "") else None,
        require_complete_masks=bool(job.get("require_complete_masks", True)),
        progress_callback=_progress_log_callback(cancel_event),
    )
    print(f"Saved COLMAP Nerfstudio dataset: {result.output_dir}", flush=True)
    print(f"transforms.json: {result.transforms_json}", flush=True)
    print(f"pointcloud.ply: {result.pointcloud}", flush=True)
    print(f"Sparse model: {result.sparse_dir}", flush=True)
    print(f"Frames: {result.image_count}", flush=True)
    print(f"Points: {result.point_count}", flush=True)
    print(f"Masks: {result.mask_count}", flush=True)
    print(f"Actions: {json.dumps(result.action_counts, sort_keys=True)}", flush=True)
    for warning in result.warnings:
        print(f"Warning: {warning}", flush=True)
    raise_if_cancelled(cancel_event)


def _run_attach_dataset_masks(job: dict, *, cancel_event: CancellationToken | None = None) -> None:
    raise_if_cancelled(cancel_event)
    dataset_root = Path(str(job["dataset_root"]))
    transforms_json = Path(str(job["transforms_json"])) if str(job.get("transforms_json") or "") else None
    if bool(job.get("clear")):
        result = clear_nerf_mask_paths(dataset_root=dataset_root, transforms_json=transforms_json)
        print(f"Cleared dataset mask paths: {result.transforms_json}", flush=True)
        print(f"Frames: {result.frame_count}", flush=True)
        print(f"Removed: {result.missing_mask_count}", flush=True)
        raise_if_cancelled(cancel_event)
        return

    masks_dir = Path(str(job["masks_dir"])) if str(job.get("masks_dir") or "") else None
    result = attach_nerf_mask_paths(
        dataset_root=dataset_root,
        transforms_json=transforms_json,
        masks_dir=masks_dir,
    )
    print(f"Attached dataset mask paths: {result.transforms_json}", flush=True)
    print(f"Frames: {result.frame_count}", flush=True)
    print(f"Masks: {result.mask_path_count}", flush=True)
    print(f"Missing masks: {result.missing_mask_count}", flush=True)
    raise_if_cancelled(cancel_event)
