from __future__ import annotations

from pathlib import Path

APP_DIR_NAME = "_stechdrive"
PROJECT_JSON = "project.json"
SOURCES_DIR_NAME = "sources"
SOURCE_VIDEOS_JSON = "videos.json"
FRAMES_DIR_NAME = "frames"
FRAME_BACKUPS_DIR_NAME = "backups"
FRAME_CACHE_DIR_NAME = "cache"
REVIEW_DIR_NAME = "review"
REVIEW_RUNS_JSON = "review_runs.json"
MASKS_META_DIR_NAME = "masks"
MASK_RUNS_JSON = "mask_runs.json"
MASK_ITEMS_DIR_NAME = "items"
STEP4_DIR_NAME = "step4"
STEP4_META_DIR_NAME = f"{APP_DIR_NAME}/{STEP4_DIR_NAME}"

SELECTED_FRAMES_CSV = "selected_frames.csv"
SELECTED_FRAMES_KEEP_CSV = "selected_frames_keep.csv"
EXTRACT_REPORT_JSON = "extract_report.json"
EXTRACT_SESSIONS_JSON = "extract_sessions.json"
STEP4_EXPORT_SETTINGS_JSON = "export_settings.json"
STEP4_VIEWS_CONFIG_JSON = "views_config.json"


def app_dir(scene_dir: Path) -> Path:
    return scene_dir / APP_DIR_NAME


def project_path(scene_dir: Path) -> Path:
    return app_dir(scene_dir) / PROJECT_JSON


def sources_dir(scene_dir: Path) -> Path:
    return app_dir(scene_dir) / SOURCES_DIR_NAME


def source_videos_path(scene_dir: Path) -> Path:
    return sources_dir(scene_dir) / SOURCE_VIDEOS_JSON


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


def review_dir(scene_dir: Path) -> Path:
    return app_dir(scene_dir) / REVIEW_DIR_NAME


def review_runs_path(scene_dir: Path) -> Path:
    return review_dir(scene_dir) / REVIEW_RUNS_JSON


def masks_meta_dir(scene_dir: Path) -> Path:
    return app_dir(scene_dir) / MASKS_META_DIR_NAME


def mask_runs_path(scene_dir: Path) -> Path:
    return masks_meta_dir(scene_dir) / MASK_RUNS_JSON


def mask_items_dir(scene_dir: Path) -> Path:
    return masks_meta_dir(scene_dir) / MASK_ITEMS_DIR_NAME


def step4_meta_dir(scene_dir: Path) -> Path:
    return app_dir(scene_dir) / STEP4_DIR_NAME


def step4_views_config_path(scene_dir: Path) -> Path:
    return step4_meta_dir(scene_dir) / STEP4_VIEWS_CONFIG_JSON


def step4_export_settings_path(scene_dir: Path) -> Path:
    return step4_meta_dir(scene_dir) / STEP4_EXPORT_SETTINGS_JSON


def legacy_step4_views_config_path(scene_dir: Path) -> Path:
    return app_dir(scene_dir) / STEP4_VIEWS_CONFIG_JSON


def legacy_step4_export_settings_path(scene_dir: Path) -> Path:
    return app_dir(scene_dir) / STEP4_EXPORT_SETTINGS_JSON


def step4_sfm_runs_path(scene_dir: Path) -> Path:
    return step4_meta_dir(scene_dir) / "sfm_runs.json"


def step4_dataset_runs_path(scene_dir: Path) -> Path:
    return step4_meta_dir(scene_dir) / "dataset_runs.json"


def step4_training_runs_path(scene_dir: Path) -> Path:
    return step4_meta_dir(scene_dir) / "training_runs.json"
