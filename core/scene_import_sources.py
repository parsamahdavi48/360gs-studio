from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from core.mask_metadata import mask_file_summary, summary_size
from core.scene_import_contracts import (
    EXTERNAL_IMPORT_KIND,
    IMAGE_EXTS,
    MASK_EXTS,
    SELECTED_CSV_FIELDNAMES,
    IssueSummary,
    SceneImportCancelToken,
    import_origin,
    is_external_import_record,
)
from core.scene_layout import (
    mask_items_dir,
    mask_runs_path,
    scene_images_dir,
    scene_import_backups_dir,
    scene_imports_path,
    scene_masks_dir,
    selected_frames_path,
    source_image_sets_path,
    step4_dataset_runs_path,
    step4_export_settings_path,
)
from core.scene_project import (
    file_identity,
    image_header_info,
    load_json,
    scene_relative,
    update_project,
    utc_now_iso,
    write_json,
    write_mask_item,
)


@dataclass(frozen=True)
class ImportedMaskItem:
    image_path: Path
    mask_path: Path
    stats: dict[str, Any]


@dataclass(frozen=True)
class ExternalMaskPlan:
    kept_runs: list[dict[str, Any]]
    removed_run_ids: set[str]
    run_id: str
    settings: dict[str, Any]
    image_count: int
    items: list[ImportedMaskItem]
    created_at: str


def iter_scene_images(root: Path, cancel_token: SceneImportCancelToken | None = None) -> list[Path]:
    if not root.is_dir():
        return []
    paths: list[Path] = []
    for index, path in enumerate(root.rglob("*"), start=1):
        if cancel_token is not None and index % 256 == 0:
            cancel_token.check_cancelled()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            paths.append(path)
    if cancel_token is not None:
        cancel_token.check_cancelled()
    return sorted(paths, key=lambda path: str(path).lower())


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


def backup_existing_import_metadata(scene: Path, import_id: str) -> Path | None:
    backup_root = scene_import_backups_dir(scene) / import_id
    copied = 0
    targets = [
        (selected_frames_path(scene), "frames/selected_frames.csv"),
        (source_image_sets_path(scene), "sources/image_sets.json"),
        (mask_runs_path(scene), "masks/mask_runs.json"),
        (mask_items_dir(scene), "masks/items"),
        (step4_export_settings_path(scene), "step4/export_settings.json"),
        (step4_dataset_runs_path(scene), "step4/dataset_runs.json"),
        (scene_imports_path(scene), "imports/scene_imports.json"),
    ]
    for source, rel in targets:
        if not source.exists():
            continue
        dest = backup_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(source, dest)
        copied += 1
    return backup_root if copied else None


def build_source_image_set_record(
    scene: Path,
    import_id: str,
    image_paths: list[Path],
    cancel_token: SceneImportCancelToken | None = None,
) -> dict[str, Any]:
    projections: list[str] = []
    files: list[dict[str, Any]] = []
    for index, path in enumerate(image_paths, start=1):
        if cancel_token is not None and index % 128 == 0:
            cancel_token.check_cancelled()
        header = image_header_info(path)
        identity = file_identity(path)
        projection = str(header.get("detected_projection") or "unknown")
        if projection != "unknown":
            projections.append(projection)
        files.append(
            {
                "source_path": str(path),
                "scene_path": scene_relative(scene, path),
                "sequence_index": index,
                "file": identity,
                "source_file": identity,
                "image": {
                    "width": int(header.get("width") or 0),
                    "height": int(header.get("height") or 0),
                    "mode": str(header.get("mode") or ""),
                },
                "detected_projection": projection,
                "projection_confidence": header.get("projection_confidence", "low"),
                "projection_reason": header.get("projection_reason", ""),
                "origin": import_origin(import_id),
            }
        )
    if cancel_token is not None:
        cancel_token.check_cancelled()

    unique = sorted(set(projections))
    projection = unique[0] if len(unique) == 1 else ("mixed" if unique else "unknown")
    return {
        "id": f"imageset_{import_id}",
        "source_type": "external_images",
        "origin": import_origin(import_id),
        "registration_mode": "in_place",
        "imported_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "source_dir": str(scene_images_dir(scene)),
        "scene_images_dir": "images",
        "projection": projection,
        "projection_source": "image_header",
        "projection_override": None,
        "file_count": len(files),
        "files": files,
    }


def source_image_size_lookup(source_record: dict[str, Any] | None) -> dict[str, tuple[int, int]]:
    if not isinstance(source_record, dict):
        return {}
    files = source_record.get("files")
    if not isinstance(files, list):
        return {}
    result: dict[str, tuple[int, int]] = {}
    for item in files:
        if not isinstance(item, dict):
            continue
        rel = str(item.get("scene_path") or "").replace("\\", "/").strip("/")
        image = item.get("image")
        if not rel or not isinstance(image, dict):
            continue
        try:
            width = int(image.get("width") or 0)
            height = int(image.get("height") or 0)
        except (TypeError, ValueError):
            continue
        if width > 0 and height > 0:
            result[rel] = (width, height)
    return result


def replace_external_source_image_set(scene: Path, record: dict[str, Any] | None) -> None:
    path = source_image_sets_path(scene)
    data = load_json(path, {"version": 1, "image_sets": []})
    image_sets = data.get("image_sets")
    if not isinstance(image_sets, list):
        image_sets = []
    kept = [item for item in image_sets if not is_external_import_record(item)]
    if record is not None:
        kept.append(record)
    write_json(path, {"version": 1, "image_sets": kept})
    update_project(scene, "sources", {"image_set_count": len(kept)})


def write_selected_frames_csv(scene: Path, import_id: str, image_paths: list[Path]) -> Path:
    rows: list[dict[str, Any]] = []
    for seq, path in enumerate(image_paths, start=1):
        rows.append(
            {
                "seq": str(seq),
                "source_session": import_id,
                "source_video": "",
                "original_index": str(seq),
                "final_index": str(seq),
                "timestamp_sec": "",
                "change_score_original": "",
                "change_score_final": "",
                "blur_score_original": "",
                "blur_score_final": "",
                "sharpness_baseline": "",
                "sharpness_ratio": "",
                "status": "ok",
                "decision": "keep",
                "analysis_pipeline": EXTERNAL_IMPORT_KIND,
                "selection_reason": EXTERNAL_IMPORT_KIND,
                "review_required": "0",
                "prev_kept_index": "",
                "gap_sec": "",
                "yaw_shift_px": "",
                "yaw_shift_deg": "",
                "residual_score": "",
                "raw_change_score": "",
                "track_count": "",
                "track_coverage": "",
                "match_confidence": "",
                "risk_flags": "",
                "analysis_width": "",
                "pair_gate_width": "",
                "pair_motion_profile": "",
                "pair_threshold_mode": "",
                "pair_drop_threshold": "",
                "pair_add_threshold": "",
                "output_file": scene_relative(scene, path),
                "source_type": "external_images",
                "source_label": scene.name,
                "import_id": import_id,
            }
        )

    path = selected_frames_path(scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SELECTED_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def selected_csv_is_external_import(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return False
    return bool(rows) and all(row.get("analysis_pipeline") == EXTERNAL_IMPORT_KIND for row in rows)


def remove_external_selected_frames_csv(scene: Path) -> None:
    path = selected_frames_path(scene)
    if selected_csv_is_external_import(path):
        path.unlink(missing_ok=True)


def mask_candidates_for_image(image_path: Path, images_root: Path, masks_root: Path) -> list[Path]:
    try:
        rel_parent = image_path.resolve().relative_to(images_root.resolve()).parent
    except Exception:
        rel_parent = Path()
    return [
        masks_root / rel_parent / f"{image_path.stem}.png",
        masks_root / rel_parent / f"{image_path.name}.png",
        masks_root / f"{image_path.name}.png",
        masks_root / f"{image_path.stem}.png",
    ]


def first_existing_mask(image_path: Path, images_root: Path, masks_root: Path) -> Path | None:
    seen: set[str] = set()
    for candidate in mask_candidates_for_image(image_path, images_root, masks_root):
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file() and candidate.suffix.lower() in MASK_EXTS:
            return candidate
    return None


def mask_lookup_for_root(masks_root: Path, cancel_token: SceneImportCancelToken | None = None) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in iter_scene_images(masks_root, cancel_token):
        try:
            rel = path.relative_to(masks_root).as_posix().lower()
        except ValueError:
            rel = path.name.lower()
        lookup.setdefault(rel, path)
        lookup.setdefault(path.name.lower(), path)
        lookup.setdefault(f"{path.stem.lower()}.png", path)
    return lookup


def indexed_mask_for_image(
    image_path: Path,
    images_root: Path,
    lookup: dict[str, Path],
) -> Path | None:
    try:
        rel_parent = image_path.relative_to(images_root).parent.as_posix().lower()
    except ValueError:
        rel_parent = ""
    name = image_path.name.lower()
    stem = image_path.stem.lower()
    keys = [
        f"{rel_parent}/{stem}.png" if rel_parent else f"{stem}.png",
        f"{rel_parent}/{name}.png" if rel_parent else f"{name}.png",
        f"{name}.png",
        f"{stem}.png",
    ]
    for key in keys:
        path = lookup.get(key)
        if path is not None:
            return path
    return None


def build_external_mask_plan(
    scene: Path,
    import_id: str,
    image_paths: list[Path],
    warnings: list[str],
    cancel_token: SceneImportCancelToken | None = None,
    source_record: dict[str, Any] | None = None,
) -> ExternalMaskPlan:
    previous_runs = load_json(mask_runs_path(scene), {"version": 1, "runs": []})
    runs = previous_runs.get("runs")
    if not isinstance(runs, list):
        runs = []
    removed_run_ids = {
        str(run.get("id"))
        for run in runs
        if isinstance(run, dict) and (is_external_import_record(run) or run.get("mode") == EXTERNAL_IMPORT_KIND)
    }
    kept_runs = [run for run in runs if not (isinstance(run, dict) and str(run.get("id")) in removed_run_ids)]

    masks_root = scene_masks_dir(scene)
    run_id = f"mask_{import_id}"
    settings = {
        "mode": EXTERNAL_IMPORT_KIND,
        "origin": import_origin(import_id),
        "images_dir": "images",
        "masks_dir": "masks",
        "mask_polarity": "white_keep_black_exclude",
    }
    if not image_paths or not masks_root.is_dir():
        return ExternalMaskPlan(
            kept_runs=kept_runs,
            removed_run_ids=removed_run_ids,
            run_id=run_id,
            settings=settings,
            image_count=len(image_paths),
            items=[],
            created_at=utc_now_iso(),
        )

    mask_lookup = mask_lookup_for_root(masks_root, cancel_token)
    if not mask_lookup:
        return ExternalMaskPlan(
            kept_runs=kept_runs,
            removed_run_ids=removed_run_ids,
            run_id=run_id,
            settings=settings,
            image_count=len(image_paths),
            items=[],
            created_at=utc_now_iso(),
        )

    missing = IssueSummary("masks/ missing matching files")
    size_mismatch = IssueSummary("masks/ size mismatch")
    unreadable = IssueSummary("masks/ unreadable files")
    items: list[ImportedMaskItem] = []
    images_root = scene_images_dir(scene)
    source_sizes = source_image_size_lookup(source_record)
    for index, image_path in enumerate(image_paths, start=1):
        if cancel_token is not None and index % 128 == 0:
            cancel_token.check_cancelled()
        image_rel = scene_relative(scene, image_path)
        mask_path = indexed_mask_for_image(image_path, images_root, mask_lookup)
        if mask_path is None:
            missing.add(image_rel)
            continue
        source_size = source_sizes.get(image_rel) or image_size(image_path)
        stats = mask_file_summary(mask_path)
        candidate_size = summary_size(stats)
        if source_size is None or candidate_size is None:
            unreadable.add(scene_relative(scene, mask_path))
            continue
        if source_size != candidate_size:
            size_mismatch.add(f"{image_rel} -> {scene_relative(scene, mask_path)}")
            continue
        items.append(ImportedMaskItem(image_path=image_path, mask_path=mask_path, stats=stats))

    if cancel_token is not None:
        cancel_token.check_cancelled()

    for issue in (missing, size_mismatch, unreadable):
        message = issue.message()
        if message:
            warnings.append(message)

    return ExternalMaskPlan(
        kept_runs=kept_runs,
        removed_run_ids=removed_run_ids,
        run_id=run_id,
        settings=settings,
        image_count=len(image_paths),
        items=items,
        created_at=utc_now_iso(),
    )


def apply_external_mask_plan(scene: Path, plan: ExternalMaskPlan) -> int:
    remove_external_mask_items(scene, plan.removed_run_ids)

    generated: list[dict[str, Any]] = []
    for item in plan.items:
        write_mask_item(
            scene,
            image_path=item.image_path,
            mask_path=item.mask_path,
            settings=plan.settings,
            run_id=plan.run_id,
            stats=item.stats,
        )
        generated.append(
            {
                "image": scene_relative(scene, item.image_path),
                "mask": scene_relative(scene, item.mask_path),
                "stats": item.stats,
            }
        )

    runs = list(plan.kept_runs)
    if generated:
        runs.append(
            {
                "id": plan.run_id,
                "created_at": plan.created_at,
                "mode": EXTERNAL_IMPORT_KIND,
                "origin": plan.settings["origin"],
                "phases": [EXTERNAL_IMPORT_KIND],
                "settings": plan.settings,
                "image_count": plan.image_count,
                "mask_count": len(generated),
                "generated": generated,
            }
        )
        update_project(scene, "masks", {"last_run_id": plan.run_id, "last_run_at": plan.created_at})
    write_json(mask_runs_path(scene), {"version": 1, "runs": runs[-200:]})
    return len(generated)


def replace_external_masks(scene: Path, import_id: str, image_paths: list[Path], warnings: list[str]) -> int:
    plan = build_external_mask_plan(scene, import_id, image_paths, warnings)
    return apply_external_mask_plan(scene, plan)


def remove_external_mask_items(scene: Path, removed_run_ids: set[str]) -> None:
    root = mask_items_dir(scene)
    if not root.is_dir():
        return
    for path in root.glob("*.json"):
        data = load_json(path, {})
        settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}
        if str(data.get("run_id") or "") in removed_run_ids or is_external_import_record(settings):
            path.unlink(missing_ok=True)
