from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scene_layout import (
    mask_items_dir,
    mask_runs_path,
    project_path,
    review_runs_path,
    source_videos_path,
    step4_dataset_runs_path,
    step4_sfm_runs_path,
    step4_training_runs_path,
)

PROJECT_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(default or {})
    return data if isinstance(data, dict) else dict(default or {})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def scene_relative(scene_dir: Path, path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(scene_dir.resolve()).as_posix()
    except (OSError, ValueError):
        return str(p)


def file_identity(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return {
            "path": str(p),
            "exists": False,
            "is_file": False,
            "is_dir": False,
            "size": None,
            "mtime_ns": None,
        }
    return {
        "path": str(p),
        "exists": True,
        "is_file": p.is_file(),
        "is_dir": p.is_dir(),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def ensure_project(scene_dir: Path) -> dict[str, Any]:
    path = project_path(scene_dir)
    project = load_json(path, {"version": PROJECT_SCHEMA_VERSION})
    project.setdefault("version", PROJECT_SCHEMA_VERSION)
    project.setdefault("app", "stechdrive-3dgs-utils")
    project.setdefault("created_at", utc_now_iso())
    project["updated_at"] = utc_now_iso()
    write_json(path, project)
    return project


def update_project(scene_dir: Path, section: str, payload: dict[str, Any]) -> None:
    project = ensure_project(scene_dir)
    current = project.get(section)
    if not isinstance(current, dict):
        current = {}
    current.update(payload)
    project[section] = current
    project["updated_at"] = utc_now_iso()
    write_json(project_path(scene_dir), project)


def append_run(path: Path, key: str, record: dict[str, Any], *, max_items: int | None = None) -> None:
    data = load_json(path, {"version": 1, key: []})
    runs = data.get(key)
    if not isinstance(runs, list):
        runs = []
    runs.append(record)
    if max_items is not None and max_items > 0 and len(runs) > max_items:
        runs = runs[-max_items:]
    data["version"] = int(data.get("version") or 1)
    data[key] = runs
    write_json(path, data)


def _identity_key(identity: dict[str, Any]) -> str:
    raw = "|".join(
        [
            str(identity.get("path") or "").replace("\\", "/").casefold(),
            str(identity.get("size") or ""),
            str(identity.get("mtime_ns") or ""),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def infer_video_projection(video_info: dict[str, Any]) -> dict[str, Any]:
    width = int(video_info.get("width") or 0)
    height = int(video_info.get("height") or 0)
    tags = video_info.get("tags") if isinstance(video_info.get("tags"), dict) else {}
    format_tags = video_info.get("format_tags") if isinstance(video_info.get("format_tags"), dict) else {}
    side_data = video_info.get("side_data_list") if isinstance(video_info.get("side_data_list"), list) else []
    haystack = json.dumps(
        {"tags": tags, "format_tags": format_tags, "side_data_list": side_data},
        ensure_ascii=False,
        sort_keys=True,
    ).lower()

    if any(token in haystack for token in ("equirectangular", "spherical", "360 video", "gspherical")):
        return {
            "projection": "equirectangular",
            "confidence": "high",
            "reason": "ffprobe spherical metadata",
        }

    if width > 0 and height > 0:
        ratio = width / height
        if math.isfinite(ratio) and abs(ratio - 2.0) <= 0.04:
            return {
                "projection": "equirectangular",
                "confidence": "medium",
                "reason": "2:1 frame aspect ratio",
            }
        return {
            "projection": "normal",
            "confidence": "medium",
            "reason": "frame aspect ratio is not 2:1",
        }

    return {
        "projection": "unknown",
        "confidence": "low",
        "reason": "video dimensions unavailable",
    }


def source_video_record(video_path: Path, video_info: dict[str, Any]) -> dict[str, Any]:
    identity = file_identity(video_path)
    detected = infer_video_projection(video_info)
    return {
        "id": f"video_{_identity_key(identity)}",
        "updated_at": utc_now_iso(),
        "source": identity,
        "video": {
            "width": int(video_info.get("width") or 0),
            "height": int(video_info.get("height") or 0),
            "fps": float(video_info.get("fps") or 0.0),
            "duration_sec": float(video_info.get("duration_sec") or 0.0),
            "total_frames": int(video_info.get("total_frames") or 0),
        },
        "detected_projection": detected["projection"],
        "projection_confidence": detected["confidence"],
        "projection_reason": detected["reason"],
        "projection_override": None,
    }


def upsert_source_videos(scene_dir: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path = source_videos_path(scene_dir)
    data = load_json(path, {"version": 1, "videos": []})
    videos = data.get("videos")
    if not isinstance(videos, list):
        videos = []

    by_id: dict[str, dict[str, Any]] = {
        str(item.get("id")): item
        for item in videos
        if isinstance(item, dict) and item.get("id")
    }
    for record in records:
        existing = by_id.get(str(record.get("id")))
        if isinstance(existing, dict) and existing.get("projection_override") is not None:
            record["projection_override"] = existing.get("projection_override")
        by_id[str(record["id"])] = record

    data["version"] = 1
    data["videos"] = sorted(by_id.values(), key=lambda item: str(item.get("source", {}).get("path", "")).lower())
    write_json(path, data)
    update_project(scene_dir, "sources", {"video_count": len(data["videos"])})


def mask_item_path(scene_dir: Path, image_relpath: str) -> Path:
    normalized = image_relpath.replace("\\", "/").strip("/")
    digest = hashlib.sha1(normalized.casefold().encode("utf-8")).hexdigest()[:12]
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(normalized).stem)
    if not stem:
        stem = "mask"
    return mask_items_dir(scene_dir) / f"{stem}.{digest}.json"


def write_mask_item(
    scene_dir: Path,
    *,
    image_path: Path,
    mask_path: Path,
    settings: dict[str, Any],
    run_id: str,
    stats: dict[str, Any],
) -> None:
    image_rel = scene_relative(scene_dir, image_path)
    payload = {
        "version": 1,
        "image": image_rel,
        "mask": scene_relative(scene_dir, mask_path),
        "run_id": run_id,
        "last_generated_at": utc_now_iso(),
        "settings": settings,
        "image_file": file_identity(image_path),
        "mask_file": file_identity(mask_path),
        "stats": stats,
    }
    write_json(mask_item_path(scene_dir, image_rel), payload)


def append_mask_run(scene_dir: Path, record: dict[str, Any]) -> None:
    append_run(mask_runs_path(scene_dir), "runs", record, max_items=200)
    update_project(scene_dir, "masks", {"last_run_id": record.get("id", ""), "last_run_at": record.get("created_at", "")})


def append_review_run(scene_dir: Path, record: dict[str, Any]) -> None:
    append_run(review_runs_path(scene_dir), "runs", record, max_items=200)
    update_project(
        scene_dir,
        "review",
        {"last_run_id": record.get("id", ""), "last_run_at": record.get("created_at", "")},
    )


def append_step4_sfm_run(scene_dir: Path, record: dict[str, Any]) -> None:
    append_run(step4_sfm_runs_path(scene_dir), "runs", record, max_items=200)
    update_project(scene_dir, "step4", {"last_sfm_run_id": record.get("id", "")})


def append_step4_dataset_run(scene_dir: Path, record: dict[str, Any]) -> None:
    append_run(step4_dataset_runs_path(scene_dir), "runs", record, max_items=200)
    update_project(scene_dir, "step4", {"last_dataset_run_id": record.get("id", "")})


def append_step4_training_run(scene_dir: Path, record: dict[str, Any]) -> None:
    append_run(step4_training_runs_path(scene_dir), "runs", record, max_items=200)
    update_project(scene_dir, "step4", {"last_training_run_id": record.get("id", "")})
