"""Batched perspective export for video and still-image sources."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2

from core.colmap_rig_export import prepare_views_for_colmap, write_rig_config_json
from core.image_io import imread_unicode
from gs360studio.domain.models import ViewSpec, atomic_write_json, utc_now
from gs360studio.engine.projection import ProjectionMapCache, project_equirectangular

ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]

_FFMPEG_INTERPOLATION = {
    "nearest": "near",
    "linear": "line",
    "cubic": "cubic",
    "lanczos": "lanc",
}


def _safe_filename(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return token or "view"


def _source_signature(path: Path) -> str:
    if path.is_dir():
        digest = hashlib.sha256(str(path.resolve()).encode("utf-8"))
        for item in sorted(candidate for candidate in path.iterdir() if candidate.is_file()):
            stat = item.stat()
            digest.update(f"\0{item.name}\0{stat.st_size}\0{stat.st_mtime_ns}".encode())
        return digest.hexdigest()
    stat = path.stat()
    identity = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def source_signature(path: str | Path) -> str:
    """Return the stable identity used by export caching and persisted jobs."""
    return _source_signature(Path(path))


def _configuration_hash(request: ExportRequest) -> str:
    payload = {
        "output_format": request.output_format,
        "frame_interval_sec": request.frame_interval_sec,
        "jpeg_quality": request.jpeg_quality,
        "video_quality": request.video_quality,
        "video_preset": request.video_preset,
        "use_nvenc": request.use_nvenc,
        "colmap_rig": request.colmap_rig,
        "views": [view.to_dict() for view in request.views],
    }
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _completed_export_matches(target: Path, request: ExportRequest) -> bool:
    manifest_path = target / "export_manifest.json"
    if not manifest_path.is_file() or not request.input_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("source_signature") == _source_signature(request.input_path) and payload.get(
        "configuration_hash"
    ) == _configuration_hash(request)


@dataclass(frozen=True, slots=True)
class ExportRequest:
    input_path: Path
    output_dir: Path
    views: tuple[ViewSpec, ...]
    output_format: str = "png"
    frame_interval_sec: float = 1.0
    jpeg_quality: int = 95
    video_quality: int = 18
    video_preset: str = "p4"
    use_nvenc: bool = False
    batch_size: int = 0
    ffmpeg_path: str = "ffmpeg"
    colmap_rig: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        fmt = self.output_format.lower()
        if fmt not in {"png", "jpeg", "video"}:
            raise ValueError("output_format must be png, jpeg, or video")
        object.__setattr__(self, "output_format", fmt)
        enabled = tuple(view for view in self.views if view.enabled)
        if not enabled:
            raise ValueError("at least one enabled view is required")
        object.__setattr__(self, "views", enabled)
        if self.colmap_rig and fmt == "video":
            raise ValueError("COLMAP rig export supports image sequences only")
        if self.colmap_rig:
            rig_shapes = {(view.width, view.height, view.hfov_deg, view.effective_vfov_deg) for view in enabled}
            if len(rig_shapes) != 1:
                raise ValueError("COLMAP rig export requires matching dimensions and FOV values for all views")
        if self.frame_interval_sec <= 0:
            raise ValueError("frame_interval_sec must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if not 0 <= self.video_quality <= 51:
            raise ValueError("video_quality must be between 0 and 51")


def estimate_batch_size(views: Sequence[ViewSpec], *, available_memory_mb: int | None = None) -> int:
    if not views:
        return 1
    largest_pixels = max(view.width * view.height for view in views)
    memory_mb = max(512, int(available_memory_mb or 2048))
    estimated_per_view_mb = max(32, math.ceil(largest_pixels * 16 / (1024 * 1024)))
    return max(1, min(len(views), 8, memory_mb // estimated_per_view_mb))


def iter_view_batches(views: Sequence[ViewSpec], batch_size: int) -> Iterable[tuple[ViewSpec, ...]]:
    size = max(1, int(batch_size))
    for start in range(0, len(views), size):
        yield tuple(views[start : start + size])


def _ffmpeg_filter(view: ViewSpec) -> str:
    return (
        "v360=input=e:output=flat:"
        f"yaw={view.yaw_deg:.6f}:pitch={view.pitch_deg:.6f}:roll={view.roll_deg:.6f}:"
        f"h_fov={view.hfov_deg:.6f}:v_fov={view.effective_vfov_deg:.6f}:"
        f"w={view.width}:h={view.height}:interp={_FFMPEG_INTERPOLATION[view.interpolation]}"
    )


def build_ffmpeg_batch_command(request: ExportRequest, batch: Sequence[ViewSpec], stage_dir: Path) -> list[str]:
    if not batch:
        raise ValueError("batch cannot be empty")
    command = [request.ffmpeg_path, "-hide_banner", "-y" if request.overwrite else "-n"]
    if request.use_nvenc:
        command.extend(["-hwaccel", "cuda"])
    command.extend(["-i", str(request.input_path)])

    input_label = "[0:v]"
    filters: list[str] = []
    if request.output_format in {"png", "jpeg"}:
        sampled = "sampled"
        filters.append(f"{input_label}fps=fps={1.0 / request.frame_interval_sec:.9f}[{sampled}]")
        input_label = f"[{sampled}]"
    split_outputs = "".join(f"[source{idx}]" for idx in range(len(batch)))
    filters.append(f"{input_label}split={len(batch)}{split_outputs}")
    for index, view in enumerate(batch):
        filters.append(f"[source{index}]{_ffmpeg_filter(view)}[view{index}]")
    command.extend(["-filter_complex", ";".join(filters)])

    prepared = prepare_views_for_colmap([view.to_legacy_view() for view in request.views]) if request.colmap_rig else []
    prepared_by_id = (
        {
            view.id: prepared[index]
            for index, view in enumerate(sorted(request.views, key=lambda item: (item.pitch_deg, item.yaw_deg, item.name)))
        }
        if request.colmap_rig
        else {}
    )
    for index, view in enumerate(batch):
        command.extend(["-map", f"[view{index}]"])
        if request.output_format == "video":
            codec = "hevc_nvenc" if request.use_nvenc else "libx265"
            quality_flag = "-cq" if request.use_nvenc else "-crf"
            command.extend(["-an", "-c:v", codec, quality_flag, str(request.video_quality), "-preset", request.video_preset])
            command.append(str(stage_dir / f"{_safe_filename(view.id)}.mp4"))
            continue
        extension = "jpg" if request.output_format == "jpeg" else "png"
        if request.colmap_rig:
            camera = prepared_by_id[view.id]["camera_name"]
            destination = stage_dir / "colmap_rig" / "images" / "rig1" / camera
            pattern = destination / f"frame_%05d.{extension}"
        else:
            destination = stage_dir / _safe_filename(view.id)
            pattern = destination / f"frame_%05d.{extension}"
        destination.mkdir(parents=True, exist_ok=True)
        if request.output_format == "jpeg":
            qscale = max(1, min(31, round(1 + (100 - request.jpeg_quality) * 30 / 99)))
            command.extend(["-qscale:v", str(qscale)])
        command.append(str(pattern))
    return command


def _commit_stage(stage_dir: Path, target_dir: Path, overwrite: bool) -> None:
    backup = target_dir.with_name(f".{target_dir.name}.previous")
    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(target_dir)
        if backup.exists():
            shutil.rmtree(backup)
        target_dir.replace(backup)
    try:
        stage_dir.replace(target_dir)
    except Exception:
        if backup.exists() and not target_dir.exists():
            backup.replace(target_dir)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _write_export_manifest(stage_dir: Path, request: ExportRequest) -> None:
    atomic_write_json(
        stage_dir / "export_manifest.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "input_path": str(request.input_path.resolve()),
            "source_signature": _source_signature(request.input_path),
            "configuration_hash": _configuration_hash(request),
            "output_format": request.output_format,
            "frame_interval_sec": request.frame_interval_sec,
            "colmap_rig": request.colmap_rig,
            "views": [view.to_dict() for view in request.views],
        },
    )


def _run_ffmpeg_cancelable(command: list[str], canceled: CancelCallback | None) -> tuple[int, str]:
    """Run FFmpeg without blocking cancellation behind a full batch."""
    with tempfile.TemporaryFile(mode="w+b") as diagnostics:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=diagnostics)
        while process.poll() is None:
            if canceled and canceled():
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
                raise InterruptedError("perspective export canceled")
            time.sleep(0.1)
        diagnostics.seek(0)
        details = diagnostics.read().decode("utf-8", errors="replace")[-4000:]
        return int(process.returncode or 0), details


def export_video_views(
    request: ExportRequest,
    *,
    progress: ProgressCallback | None = None,
    canceled: CancelCallback | None = None,
) -> Path:
    if not request.input_path.is_file():
        raise FileNotFoundError(request.input_path)
    target = request.output_dir.resolve()
    if _completed_export_matches(target, request):
        if progress:
            progress(1, 1, "Reused matching completed export")
        return target
    if target.exists() and not request.overwrite:
        raise FileExistsError(target)
    stage = target.with_name(f".{target.name}.staging-{uuid.uuid4().hex}")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        batch_size = request.batch_size or estimate_batch_size(request.views)
        batches = tuple(iter_view_batches(request.views, batch_size))
        for batch_index, batch in enumerate(batches, start=1):
            if canceled and canceled():
                raise InterruptedError("perspective export canceled")
            command = build_ffmpeg_batch_command(request, batch, stage)
            return_code, details = _run_ffmpeg_cancelable(command, canceled)
            if return_code != 0:
                raise RuntimeError(f"FFmpeg batch {batch_index} failed ({return_code}):\n{details}")
            if progress:
                progress(batch_index, len(batches), f"Exported view batch {batch_index}/{len(batches)}")
        if request.colmap_rig:
            prepared = prepare_views_for_colmap([view.to_legacy_view() for view in request.views])
            first = request.views[0]
            if any(view.width != first.width or view.height != first.height or view.hfov_deg != first.hfov_deg for view in request.views):
                raise ValueError("COLMAP rig export requires matching dimensions and horizontal FOV for all views")
            write_rig_config_json(stage, prepared, (first.width, first.height))
        _write_export_manifest(stage, request)
        _commit_stage(stage, target, request.overwrite)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise
    return target


def export_image_views(
    input_files: Sequence[str | Path],
    request: ExportRequest,
    *,
    progress: ProgressCallback | None = None,
    canceled: CancelCallback | None = None,
) -> Path:
    if request.output_format == "video":
        raise ValueError("still-image export supports png or jpeg output")
    target = request.output_dir.resolve()
    if _completed_export_matches(target, request):
        if progress:
            progress(1, 1, "Reused matching completed export")
        return target
    if target.exists() and not request.overwrite:
        raise FileExistsError(target)
    stage = target.with_name(f".{target.name}.staging-{uuid.uuid4().hex}")
    stage.mkdir(parents=True, exist_ok=False)
    cache = ProjectionMapCache(max_entries=max(1, min(32, len(request.views))))
    try:
        extension = ".jpg" if request.output_format == "jpeg" else ".png"
        total = len(input_files) * len(request.views)
        written = 0
        prepared = prepare_views_for_colmap([view.to_legacy_view() for view in request.views]) if request.colmap_rig else []
        prepared_by_id = (
            {
                view.id: prepared[index]
                for index, view in enumerate(sorted(request.views, key=lambda item: (item.pitch_deg, item.yaw_deg, item.name)))
            }
            if request.colmap_rig
            else {}
        )
        for frame_index, input_file in enumerate(input_files, start=1):
            if canceled and canceled():
                raise InterruptedError("perspective export canceled")
            source = imread_unicode(str(input_file), cv2.IMREAD_UNCHANGED)
            if source is None:
                raise ValueError(f"cannot read image: {input_file}")
            for view in request.views:
                if canceled and canceled():
                    raise InterruptedError("perspective export canceled")
                projected = project_equirectangular(source, view, cache=cache)
                if request.colmap_rig:
                    camera = prepared_by_id[view.id]["camera_name"]
                    output = stage / "colmap_rig" / "images" / "rig1" / camera / f"frame_{frame_index:05d}{extension}"
                else:
                    output = stage / _safe_filename(view.id) / f"frame_{frame_index:05d}{extension}"
                output.parent.mkdir(parents=True, exist_ok=True)
                params = [cv2.IMWRITE_JPEG_QUALITY, request.jpeg_quality] if extension == ".jpg" else []
                success, encoded = cv2.imencode(extension, projected, params)
                if not success:
                    raise RuntimeError(f"cannot encode output image: {output}")
                encoded.tofile(output)
                written += 1
                if progress:
                    progress(written, total, f"Exported {written}/{total} views")
        if request.colmap_rig:
            first = request.views[0]
            if any(view.width != first.width or view.height != first.height or view.hfov_deg != first.hfov_deg for view in request.views):
                raise ValueError("COLMAP rig export requires matching dimensions and horizontal FOV for all views")
            write_rig_config_json(stage, prepared, (first.width, first.height))
        _write_export_manifest(stage, request)
        _commit_stage(stage, target, request.overwrite)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise
    return target
