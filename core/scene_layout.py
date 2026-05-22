from __future__ import annotations

from pathlib import Path

APP_DIR_NAME = "_stechdrive"
PROJECT_JSON = "project.json"
SOURCES_DIR_NAME = "sources"
SOURCE_VIDEOS_JSON = "videos.json"
SOURCE_IMAGE_SETS_JSON = "image_sets.json"
NORMAL_CAMERA_DEFAULTS_JSON = "normal_camera_defaults.json"
FRAMES_DIR_NAME = "frames"
FRAME_BACKUPS_DIR_NAME = "backups"
FRAME_CACHE_DIR_NAME = "cache"
REVIEW_DIR_NAME = "review"
REVIEW_RUNS_JSON = "review_runs.json"
MASKS_META_DIR_NAME = "masks"
MASK_RUNS_JSON = "mask_runs.json"
MASK_ITEMS_DIR_NAME = "items"
IMPORTS_DIR_NAME = "imports"
SCENE_IMPORTS_JSON = "scene_imports.json"
STEP4_DIR_NAME = "step4"
STEP4_META_DIR_NAME = f"{APP_DIR_NAME}/{STEP4_DIR_NAME}"
STEP4_WORK_DIR_NAME = "work"
STEP4_METASHAPE_IMPORT_WORK_DIR_NAME = "metashape_import"
ARTIFACTS_DIR_NAME = "artifacts"
SFM_ARTIFACTS_JSON = "sfm_artifacts.json"
DATASET_ARTIFACTS_JSON = "dataset_artifacts.json"
JOBS_DIR_NAME = "jobs"

SELECTED_FRAMES_CSV = "selected_frames.csv"
SELECTED_FRAMES_KEEP_CSV = "selected_frames_keep.csv"
EXTRACT_REPORT_JSON = "extract_report.json"
EXTRACT_SESSIONS_JSON = "extract_sessions.json"
STEP4_EXPORT_SETTINGS_JSON = "export_settings.json"
STEP4_VIEWS_CONFIG_JSON = "views_config.json"
STEP4_SETTINGS_VERSION = 2
SCENE_IMAGES_DIR_NAME = "images"
SCENE_MASKS_DIR_NAME = "masks"
SCENE_OUTPUT_DIR_NAME = "output"
METASHAPE_CUBEMAP_OUTPUT_DIR_NAME = "metashape_cubemap"
METASHAPE_3DGUT_OUTPUT_DIR_NAME = "metashape_3dgut"
SPHERESFM_CUBEMAP_OUTPUT_DIR_NAME = "spheresfm_cubemap"
SPHERESFM_3DGUT_OUTPUT_DIR_NAME = "spheresfm_3dgut"


def app_dir(scene_dir: Path) -> Path:
    return scene_dir / APP_DIR_NAME


def scene_images_dir(scene_dir: Path) -> Path:
    return scene_dir / SCENE_IMAGES_DIR_NAME


def scene_masks_dir(scene_dir: Path) -> Path:
    return scene_dir / SCENE_MASKS_DIR_NAME


def scene_output_dir(scene_dir: Path) -> Path:
    return scene_dir / SCENE_OUTPUT_DIR_NAME


def scene_metashape_cubemap_dir(scene_dir: Path) -> Path:
    return scene_output_dir(scene_dir) / METASHAPE_CUBEMAP_OUTPUT_DIR_NAME


def scene_metashape_3dgut_dir(scene_dir: Path) -> Path:
    return scene_output_dir(scene_dir) / METASHAPE_3DGUT_OUTPUT_DIR_NAME


def scene_spheresfm_cubemap_dir(scene_dir: Path) -> Path:
    return scene_output_dir(scene_dir) / SPHERESFM_CUBEMAP_OUTPUT_DIR_NAME


def scene_spheresfm_3dgut_dir(scene_dir: Path) -> Path:
    return scene_output_dir(scene_dir) / SPHERESFM_3DGUT_OUTPUT_DIR_NAME


def project_path(scene_dir: Path) -> Path:
    return app_dir(scene_dir) / PROJECT_JSON


def sources_dir(scene_dir: Path) -> Path:
    return app_dir(scene_dir) / SOURCES_DIR_NAME


def source_videos_path(scene_dir: Path) -> Path:
    return sources_dir(scene_dir) / SOURCE_VIDEOS_JSON


def source_image_sets_path(scene_dir: Path) -> Path:
    return sources_dir(scene_dir) / SOURCE_IMAGE_SETS_JSON


def normal_camera_defaults_path(scene_dir: Path) -> Path:
    return sources_dir(scene_dir) / NORMAL_CAMERA_DEFAULTS_JSON


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


def scene_imports_dir(scene_dir: Path) -> Path:
    return app_dir(scene_dir) / IMPORTS_DIR_NAME


def scene_imports_path(scene_dir: Path) -> Path:
    return scene_imports_dir(scene_dir) / SCENE_IMPORTS_JSON


def artifacts_dir(scene_dir: Path) -> Path:
    return app_dir(scene_dir) / ARTIFACTS_DIR_NAME


def sfm_artifacts_path(scene_dir: Path) -> Path:
    return artifacts_dir(scene_dir) / SFM_ARTIFACTS_JSON


def dataset_artifacts_path(scene_dir: Path) -> Path:
    return artifacts_dir(scene_dir) / DATASET_ARTIFACTS_JSON


def jobs_dir(scene_dir: Path) -> Path:
    return app_dir(scene_dir) / JOBS_DIR_NAME


def scene_import_backups_dir(scene_dir: Path) -> Path:
    return scene_imports_dir(scene_dir) / "backups"


def step4_meta_dir(scene_dir: Path) -> Path:
    return app_dir(scene_dir) / STEP4_DIR_NAME


def step4_work_dir(scene_dir: Path) -> Path:
    return step4_meta_dir(scene_dir) / STEP4_WORK_DIR_NAME


def step4_metashape_import_work_dir(scene_dir: Path) -> Path:
    return step4_work_dir(scene_dir) / STEP4_METASHAPE_IMPORT_WORK_DIR_NAME


def step4_views_config_path(scene_dir: Path) -> Path:
    return step4_meta_dir(scene_dir) / STEP4_VIEWS_CONFIG_JSON


def step4_export_settings_path(scene_dir: Path) -> Path:
    return step4_meta_dir(scene_dir) / STEP4_EXPORT_SETTINGS_JSON


def step4_sfm_runs_path(scene_dir: Path) -> Path:
    return step4_meta_dir(scene_dir) / "sfm_runs.json"


def step4_dataset_runs_path(scene_dir: Path) -> Path:
    return step4_meta_dir(scene_dir) / "dataset_runs.json"


def step4_training_runs_path(scene_dir: Path) -> Path:
    return step4_meta_dir(scene_dir) / "training_runs.json"
