from __future__ import annotations

from pathlib import Path

APP_DIR_NAME = "_stechdrive"
FRAMES_DIR_NAME = "frames"
FRAME_BACKUPS_DIR_NAME = "backups"
FRAME_CACHE_DIR_NAME = "cache"
STEP4_META_DIR_NAME = "_stechdrive"

SELECTED_FRAMES_CSV = "selected_frames.csv"
SELECTED_FRAMES_KEEP_CSV = "selected_frames_keep.csv"
EXTRACT_REPORT_JSON = "extract_report.json"
EXTRACT_SESSIONS_JSON = "extract_sessions.json"
STEP4_EXPORT_SETTINGS_JSON = "export_settings.json"
STEP4_VIEWS_CONFIG_JSON = "views_config.json"


def app_dir(scene_dir: Path) -> Path:
    return scene_dir / APP_DIR_NAME


def frames_dir(scene_dir: Path) -> Path:
    return app_dir(scene_dir) / FRAMES_DIR_NAME


def frame_backups_dir(scene_dir: Path) -> Path:
    return frames_dir(scene_dir) / FRAME_BACKUPS_DIR_NAME


def frame_cache_dir(scene_dir: Path) -> Path:
    return frames_dir(scene_dir) / FRAME_CACHE_DIR_NAME


def selected_frames_path(scene_dir: Path, csv_name: str = SELECTED_FRAMES_CSV) -> Path:
    csv = Path(csv_name)
    if csv.is_absolute():
        return csv
    return frames_dir(scene_dir) / csv


def selected_frames_keep_path(scene_dir: Path) -> Path:
    return frames_dir(scene_dir) / SELECTED_FRAMES_KEEP_CSV


def extract_report_path(scene_dir: Path) -> Path:
    return frames_dir(scene_dir) / EXTRACT_REPORT_JSON


def extract_sessions_path(scene_dir: Path) -> Path:
    return frames_dir(scene_dir) / EXTRACT_SESSIONS_JSON


def step4_meta_dir(scene_dir: Path) -> Path:
    return app_dir(scene_dir)


def step4_views_config_path(scene_dir: Path) -> Path:
    return step4_meta_dir(scene_dir) / STEP4_VIEWS_CONFIG_JSON


def step4_export_settings_path(scene_dir: Path) -> Path:
    return step4_meta_dir(scene_dir) / STEP4_EXPORT_SETTINGS_JSON
