from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_NAME = "extract_sessions.json"


def sanitize_filename_prefix(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text


def manifest_path(scene_dir: Path) -> Path:
    return scene_dir / MANIFEST_NAME


def load_manifest(scene_dir: Path) -> dict[str, Any]:
    path = manifest_path(scene_dir)
    if not path.exists():
        return {"version": 1, "sessions": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "sessions": []}
    if not isinstance(data, dict):
        return {"version": 1, "sessions": []}
    sessions = data.get("sessions")
    if not isinstance(sessions, list):
        data["sessions"] = []
    data.setdefault("version", 1)
    return data


def save_manifest(scene_dir: Path, manifest: dict[str, Any]) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    manifest.setdefault("version", 1)
    manifest.setdefault("sessions", [])
    manifest_path(scene_dir).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def video_identity(video_path: Path) -> dict[str, Any]:
    resolved = video_path.resolve()
    st = resolved.stat()
    return {
        "path": str(resolved),
        "name": resolved.name,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _casefold_path(value: str) -> str:
    return str(Path(value)).replace("\\", "/").casefold()


def session_matches_video(session: dict[str, Any], identity: dict[str, Any]) -> bool:
    source = session.get("source_video")
    if not isinstance(source, dict):
        source = {
            "path": session.get("source_video_path", ""),
            "size": session.get("video_size"),
            "mtime_ns": session.get("video_mtime_ns"),
        }
    path = str(source.get("path") or "")
    if not path:
        return False
    if _casefold_path(path) != _casefold_path(str(identity.get("path") or "")):
        return False
    try:
        return int(source.get("size")) == int(identity.get("size")) and int(source.get("mtime_ns")) == int(
            identity.get("mtime_ns")
        )
    except Exception:
        return False


def matching_video_sessions(scene_dir: Path, video_path: Path) -> list[dict[str, Any]]:
    identity = video_identity(video_path)
    return [
        session
        for session in load_manifest(scene_dir).get("sessions", [])
        if isinstance(session, dict) and session_matches_video(session, identity)
    ]


def new_session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def build_session_record(
    *,
    session_id: str,
    input_video: Path,
    video_info: dict[str, Any],
    mode: str,
    filename_prefix: str,
    image_ext: str,
    output_files: list[str],
    selected_count: int,
    dropped_count: int,
) -> dict[str, Any]:
    return {
        "id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_video": video_identity(input_video),
        "mode": mode,
        "video": video_info,
        "filename_prefix": filename_prefix,
        "image_ext": image_ext,
        "output_files": output_files,
        "selected_count": selected_count,
        "dropped_count": dropped_count,
    }
