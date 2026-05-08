from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from scene_layout import (
    mask_items_dir,
    mask_runs_path,
    project_path,
    review_runs_path,
    selected_frames_path,
    source_image_sets_path,
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


def infer_image_projection(width: int, height: int) -> dict[str, Any]:
    if width > 0 and height > 0:
        ratio = width / height
        if math.isfinite(ratio) and abs(ratio - 2.0) <= 0.04:
            return {
                "projection": "equirectangular",
                "confidence": "medium",
                "reason": "2:1 image aspect ratio",
            }
        return {
            "projection": "normal",
            "confidence": "medium",
            "reason": "image aspect ratio is not 2:1",
        }
    return {
        "projection": "unknown",
        "confidence": "low",
        "reason": "image dimensions unavailable",
    }


def image_header_info(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
    except Exception:
        width, height, mode = 0, 0, ""
    detected = infer_image_projection(int(width), int(height))
    return {
        "width": int(width),
        "height": int(height),
        "mode": mode,
        "detected_projection": detected["projection"],
        "projection_confidence": detected["confidence"],
        "projection_reason": detected["reason"],
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


def remove_source_videos(scene_dir: Path, video_paths: list[Path]) -> None:
    targets = {_path_lookup_key(str(path)) for path in video_paths}
    targets.discard("")
    if not targets:
        return

    path = source_videos_path(scene_dir)
    data = load_json(path, {"version": 1, "videos": []})
    videos = data.get("videos")
    if not isinstance(videos, list):
        return

    kept: list[dict[str, Any]] = []
    for item in videos:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        path_text = source.get("path") if isinstance(source, dict) else ""
        if _path_lookup_key(str(path_text or "")) in targets:
            continue
        kept.append(item)

    if len(kept) == len(videos):
        return
    data["version"] = int(data.get("version") or 1)
    data["videos"] = kept
    write_json(path, data)
    update_project(scene_dir, "sources", {"video_count": len(kept)})


def _normalize_projection(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"equirect", "equirectangular", "360", "360°"}:
        return "equirectangular"
    if text in {"normal", "perspective", "flat"}:
        return "normal"
    return "unknown"


def _projection_for_record(record: dict[str, Any]) -> str:
    override = _normalize_projection(str(record.get("projection_override") or ""))
    if override != "unknown":
        return override
    explicit = _normalize_projection(str(record.get("projection") or ""))
    if explicit != "unknown":
        return explicit
    return _normalize_projection(str(record.get("detected_projection") or ""))


def source_image_set_record(
    *,
    source_dir: Path,
    scene_dir: Path,
    imported: list[tuple[Path, Path]],
) -> dict[str, Any]:
    projections: list[str] = []
    files: list[dict[str, Any]] = []
    for index, (source_path, scene_path) in enumerate(imported, start=1):
        header = image_header_info(scene_path)
        projection = _normalize_projection(str(header.get("detected_projection") or ""))
        if projection != "unknown":
            projections.append(projection)
        files.append(
            {
                "source_path": str(source_path),
                "scene_path": scene_relative(scene_dir, scene_path),
                "sequence_index": index,
                "file": file_identity(scene_path),
                "source_file": file_identity(source_path),
                "image": {
                    "width": int(header.get("width") or 0),
                    "height": int(header.get("height") or 0),
                    "mode": str(header.get("mode") or ""),
                },
                "detected_projection": header.get("detected_projection", "unknown"),
                "projection_confidence": header.get("projection_confidence", "low"),
                "projection_reason": header.get("projection_reason", ""),
            }
        )

    unique = sorted(set(projections))
    projection = unique[0] if len(unique) == 1 else ("mixed" if unique else "unknown")
    identity = file_identity(source_dir)
    return {
        "id": f"imageset_{_identity_key(identity)}",
        "source_type": "external_images",
        "imported_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "source_dir": str(source_dir),
        "scene_images_dir": "images",
        "projection": projection,
        "projection_source": "image_header",
        "projection_override": None,
        "file_count": len(files),
        "files": files,
    }


def append_source_image_set(scene_dir: Path, record: dict[str, Any]) -> None:
    path = source_image_sets_path(scene_dir)
    data = load_json(path, {"version": 1, "image_sets": []})
    image_sets = data.get("image_sets")
    if not isinstance(image_sets, list):
        image_sets = []

    image_sets.append(record)
    data["version"] = 1
    data["image_sets"] = image_sets[-200:]
    write_json(path, data)
    update_project(scene_dir, "sources", {"image_set_count": len(data["image_sets"])})


def _source_video_projections(scene_dir: Path) -> set[str]:
    data = load_json(source_videos_path(scene_dir), {"videos": []})
    videos = data.get("videos")
    if not isinstance(videos, list):
        return set()
    return {
        projection
        for item in videos
        if isinstance(item, dict)
        for projection in [_projection_for_record(item)]
        if projection != "unknown"
    }


def _source_image_set_projections(scene_dir: Path) -> set[str]:
    data = load_json(source_image_sets_path(scene_dir), {"image_sets": []})
    image_sets = data.get("image_sets")
    if not isinstance(image_sets, list):
        return set()
    return {
        projection
        for item in image_sets
        if isinstance(item, dict)
        for projection in [_projection_for_record(item)]
        if projection != "unknown" and projection != "mixed"
    }


def scene_image_projection_map(scene_dir: Path, image_paths: list[Path] | None = None) -> dict[str, str]:
    images_dir = scene_dir / "images"
    if image_paths is None:
        image_paths = sorted(
            (
                path
                for path in images_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
            ),
            key=lambda path: str(path).lower(),
        )

    rel_keys = [scene_relative(scene_dir, path).replace("\\", "/") for path in image_paths]
    result: dict[str, str] = {}
    result.update(_image_set_file_projection_map(scene_dir))
    result.update(_selected_frame_projection_map(scene_dir))

    for key, path in zip(rel_keys, image_paths, strict=False):
        projection = _normalize_projection(result.get(key, ""))
        if projection == "unknown":
            projection = _normalize_projection(str(image_header_info(path).get("detected_projection") or ""))
        if projection == "unknown":
            projection = "equirectangular"
        result[key] = projection
    return {key: result.get(key, "equirectangular") for key in rel_keys}


def _image_set_file_projection_map(scene_dir: Path) -> dict[str, str]:
    data = load_json(source_image_sets_path(scene_dir), {"image_sets": []})
    image_sets = data.get("image_sets")
    if not isinstance(image_sets, list):
        return {}
    result: dict[str, str] = {}
    for image_set in image_sets:
        if not isinstance(image_set, dict):
            continue
        set_projection = _projection_for_record(image_set)
        if set_projection == "mixed":
            set_projection = "unknown"
        files = image_set.get("files")
        if not isinstance(files, list):
            continue
        for item in files:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("scene_path") or "").replace("\\", "/").strip("/")
            if not rel:
                continue
            projection = _projection_for_record(item)
            if projection == "unknown":
                projection = set_projection
            if projection != "unknown":
                result[rel] = projection
    return result


def _selected_frame_projection_map(scene_dir: Path) -> dict[str, str]:
    csv_path = selected_frames_path(scene_dir)
    if not csv_path.is_file():
        return {}
    video_projection = _source_video_projection_by_path(scene_dir)
    if not video_projection:
        return {}
    unique_projections = set(video_projection.values())
    fallback_projection = next(iter(unique_projections)) if len(unique_projections) == 1 else ""
    result: dict[str, str] = {}
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                output = str(row.get("output_file") or "").replace("\\", "/").strip("/")
                if not output:
                    continue
                source_video = _path_lookup_key(str(row.get("source_video") or ""))
                projection = video_projection.get(source_video) or fallback_projection
                if projection:
                    result[output] = projection
    except OSError:
        return {}
    return result


def _source_video_projection_by_path(scene_dir: Path) -> dict[str, str]:
    data = load_json(source_videos_path(scene_dir), {"videos": []})
    videos = data.get("videos")
    if not isinstance(videos, list):
        return {}
    result: dict[str, str] = {}
    for item in videos:
        if not isinstance(item, dict):
            continue
        projection = _projection_for_record(item)
        if projection == "unknown":
            continue
        source = item.get("source")
        path_text = source.get("path") if isinstance(source, dict) else ""
        key = _path_lookup_key(str(path_text or ""))
        if key:
            result[key] = projection
    return result


def _path_lookup_key(value: str) -> str:
    text = value.replace("\\", "/").strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve()).replace("\\", "/").casefold()
    except OSError:
        return text.casefold()


def _sample_image_projections(images_dir: Path, *, limit: int = 12) -> set[str]:
    if not images_dir.is_dir():
        return set()
    projections: set[str] = set()
    count = 0
    for path in sorted(images_dir.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        projection = _normalize_projection(str(image_header_info(path).get("detected_projection") or ""))
        if projection != "unknown":
            projections.add(projection)
        count += 1
        if count >= limit:
            break
    return projections


def resolve_scene_image_projection(scene_dir: Path) -> dict[str, Any]:
    video_projections = _source_video_projections(scene_dir)
    image_set_projections = _source_image_set_projections(scene_dir)
    projections = video_projections | image_set_projections
    source = "project"

    if not projections:
        projections = _sample_image_projections(scene_dir / "images")
        source = "image_header_sample"

    if len(projections) == 1:
        projection = next(iter(projections))
        return {
            "projection": projection,
            "mask_projection": "equirect" if projection == "equirectangular" else "normal",
            "source": source,
            "mixed": False,
        }
    if len(projections) > 1:
        return {
            "projection": "mixed",
            "mask_projection": "equirect",
            "source": source,
            "mixed": True,
        }
    return {
        "projection": "equirectangular",
        "mask_projection": "equirect",
        "source": "default",
        "mixed": False,
    }


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
