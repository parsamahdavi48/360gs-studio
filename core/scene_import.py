from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from core.scene_import_contracts import (
    SceneImportCancelToken,
    SceneImportOptions,
    SceneImportResult,
    new_import_id,
)
from core.scene_import_outputs import (
    inspect_output_dataset,
    replace_external_dataset_run,
    write_external_step4_settings,
)
from core.scene_import_sources import (
    apply_external_mask_plan,
    backup_existing_import_metadata,
    build_external_mask_plan,
    build_source_image_set_record,
    iter_scene_images,
    remove_external_selected_frames_csv,
    replace_external_source_image_set,
    write_selected_frames_csv,
)
from core.scene_layout import (
    scene_images_dir,
    scene_imports_path,
    step4_export_settings_path,
)
from core.scene_project import load_json, scene_relative, update_project, utc_now_iso, write_json

SceneImportProgressCallback = Callable[[str], None]


def import_scene(
    scene_dir: str | Path,
    *,
    options: SceneImportOptions | None = None,
    cancel_token: SceneImportCancelToken | None = None,
    progress_callback: SceneImportProgressCallback | None = None,
) -> SceneImportResult:
    scene = Path(scene_dir)
    if not scene.is_dir():
        raise FileNotFoundError(f"Scene folder not found: {scene}")

    options = options or SceneImportOptions()
    cancel_token = cancel_token or SceneImportCancelToken()
    import_id = new_import_id()
    warnings: list[str] = []
    errors: list[str] = []

    def emit(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    def checkpoint() -> None:
        cancel_token.check_cancelled()

    def run_phase[T](label: str, func: Callable[[], T], *, cancellable_after: bool = True) -> T:
        checkpoint()
        started = perf_counter()
        emit(f"[scene import] {label}...")
        checkpoint()
        result = func()
        elapsed = perf_counter() - started
        emit(f"[scene import] {label} done ({elapsed:.1f}s)")
        if cancellable_after:
            checkpoint()
        return result

    source_images = run_phase("scan source images", lambda: iter_scene_images(scene_images_dir(scene), cancel_token))
    output_info = run_phase(
        "scan output dataset",
        lambda: inspect_output_dataset(scene, warnings, options=options, cancel_token=cancel_token),
    )
    output_images = output_info["images"]
    output_masks = output_info["masks"]

    if not source_images and not output_images:
        errors.append("No source images or output dataset images were found.")

    selected_csv: Path | None = None
    source_record = run_phase(
        "build source metadata",
        lambda: build_source_image_set_record(scene, import_id, source_images, cancel_token) if source_images else None,
    )
    mask_plan = run_phase(
        "build mask metadata",
        lambda: build_external_mask_plan(scene, import_id, source_images, warnings, cancel_token),
    )

    def apply_metadata() -> tuple[Path | None, int, Path | None]:
        backup_dir = backup_existing_import_metadata(scene, import_id)
        replace_external_source_image_set(scene, source_record)
        selected_csv: Path | None = None
        if source_images:
            selected_csv = write_selected_frames_csv(scene, import_id, source_images)
        else:
            remove_external_selected_frames_csv(scene)
        mask_count = apply_external_mask_plan(scene, mask_plan)
        write_external_step4_settings(scene, import_id, output_info)
        replace_external_dataset_run(scene, import_id, output_info)
        return backup_dir, mask_count, selected_csv

    backup_dir, mask_count, selected_csv = run_phase("apply metadata", apply_metadata, cancellable_after=False)

    status = "error" if errors else ("warning" if warnings else "ok")
    report_path = write_import_record(
        scene,
        import_id,
        status=status,
        backup_dir=backup_dir,
        source_images=source_images,
        mask_count=mask_count,
        output_info=output_info,
        warnings=warnings,
        errors=errors,
    )
    update_project_summary(scene, import_id, status, source_images, mask_count, output_info)

    return SceneImportResult(
        scene_dir=scene,
        import_id=import_id,
        status=status,
        image_count=len(source_images),
        mask_count=mask_count,
        output_image_count=len(output_images),
        output_mask_count=len(output_masks),
        output_shape=str(output_info.get("output_shape") or ""),
        dataset_kind=str(output_info.get("dataset_kind") or ""),
        warnings=tuple(warnings),
        errors=tuple(errors),
        backup_dir=backup_dir,
        report_path=report_path,
        selected_frames_csv=selected_csv,
        export_settings_json=step4_export_settings_path(scene),
    )


def write_import_record(
    scene: Path,
    import_id: str,
    *,
    status: str,
    backup_dir: Path | None,
    source_images: list[Path],
    mask_count: int,
    output_info: dict[str, Any],
    warnings: list[str],
    errors: list[str],
) -> Path:
    path = scene_imports_path(scene)
    data = load_json(path, {"version": 1, "active_import_id": "", "imports": []})
    imports = data.get("imports")
    if not isinstance(imports, list):
        imports = []
    previous_active = str(data.get("active_import_id") or "")
    record = {
        "id": import_id,
        "created_at": utc_now_iso(),
        "mode": "rescan_replace",
        "replaces_import_id": previous_active,
        "status": status,
        "source_root": ".",
        "backup_dir": scene_relative(scene, backup_dir) if backup_dir is not None else "",
        "assets": {
            "source_images": bool(source_images),
            "source_masks": mask_count > 0,
            "output_dataset": bool(output_info.get("active")),
            "pointcloud": bool(Path(output_info.get("pointcloud", "")).is_file()),
        },
        "counts": {
            "source_images": len(source_images),
            "source_masks": mask_count,
            "output_images": len(output_info.get("images") or []),
            "output_masks": len(output_info.get("masks") or []),
            "output_frames": int(output_info.get("frames_count") or 0),
        },
        "output": {
            "shape": str(output_info.get("output_shape") or ""),
            "dataset_kind": str(output_info.get("dataset_kind") or ""),
            "camera_model": str(output_info.get("camera_model") or ""),
        },
        "validation": {
            "errors": list(errors),
            "warnings": list(warnings),
            "output_image_sample_count": int(output_info.get("validation_sample_count") or 0),
            "output_image_sample_limit": int(output_info.get("validation_sample_limit") or 0),
        },
    }
    imports.append(record)
    write_json(path, {"version": 1, "active_import_id": import_id, "imports": imports[-200:]})
    return path


def update_project_summary(
    scene: Path,
    import_id: str,
    status: str,
    source_images: list[Path],
    mask_count: int,
    output_info: dict[str, Any],
) -> None:
    update_project(
        scene,
        "imports",
        {
            "last_import_id": import_id,
            "last_import_at": utc_now_iso(),
            "status": status,
            "mode": "rescan_replace",
        },
    )
    update_project(
        scene,
        "assets",
        {
            "source_image_count": len(source_images),
            "source_mask_count": mask_count,
            "output_image_count": len(output_info.get("images") or []),
            "output_mask_count": len(output_info.get("masks") or []),
            "output_shape": str(output_info.get("output_shape") or ""),
            "dataset_kind": str(output_info.get("dataset_kind") or ""),
        },
    )


__all__ = ["SceneImportResult", "import_scene"]
