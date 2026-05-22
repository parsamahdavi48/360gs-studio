from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.extract_frames import read_selected_csv, write_selected_csv_rows
from core.extract_sessions import sanitize_filename_prefix
from core.scene_import_contracts import IMAGE_EXTS, SELECTED_CSV_FIELDNAMES, new_import_id
from core.scene_layout import scene_images_dir, selected_frames_path
from core.scene_project import (
    append_source_image_set,
    file_identity,
    image_header_info,
    scene_relative,
    utc_now_iso,
)

IMAGE_SEQUENCE_SOURCE_TYPE = "image_sequence"


@dataclass(frozen=True, slots=True)
class ImageSequenceImportResult:
    scene_dir: Path
    source_dir: Path
    import_id: str
    image_count: int
    output_files: tuple[str, ...]
    source_record: dict[str, Any]


def import_image_sequence_folder(
    source_dir: str | Path,
    scene_dir: str | Path,
    *,
    prefix: str = "",
    recursive: bool = False,
) -> ImageSequenceImportResult:
    source = Path(source_dir)
    scene = Path(scene_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Image folder was not found: {source}")
    image_paths = _source_image_files(source, recursive=recursive)
    if not image_paths:
        raise ValueError(f"No supported images found in: {source}")

    import_id = new_import_id()
    images_dir = scene_images_dir(scene)
    images_dir.mkdir(parents=True, exist_ok=True)
    resolved_prefix = _unique_prefix(images_dir, sanitize_filename_prefix(prefix or source.name) or "image")
    digits = max(4, len(str(len(image_paths))))

    imported: list[tuple[Path, Path]] = []
    for index, src in enumerate(image_paths, start=1):
        dst = images_dir / f"{resolved_prefix}_{index:0{digits}d}{src.suffix.lower()}"
        if dst.exists():
            raise FileExistsError(f"Output image already exists: {dst}")
        shutil.copy2(src, dst)
        imported.append((src, dst))
        print(f"[progress] {index}/{len(image_paths)}", flush=True)

    record = image_sequence_source_record(
        source_dir=source,
        scene_dir=scene,
        imported=imported,
        import_id=import_id,
    )
    append_source_image_set(scene, record)
    _append_selected_frames(scene, import_id, source.name, [dst for _src, dst in imported])

    return ImageSequenceImportResult(
        scene_dir=scene,
        source_dir=source,
        import_id=import_id,
        image_count=len(imported),
        output_files=tuple(scene_relative(scene, dst).replace("\\", "/") for _src, dst in imported),
        source_record=record,
    )


def image_sequence_source_record(
    *,
    source_dir: Path,
    scene_dir: Path,
    imported: list[tuple[Path, Path]],
    import_id: str,
) -> dict[str, Any]:
    projections: list[str] = []
    files: list[dict[str, Any]] = []
    for index, (source_path, scene_path) in enumerate(imported, start=1):
        header = image_header_info(scene_path)
        projection = str(header.get("detected_projection") or "unknown")
        if projection != "unknown":
            projections.append(projection)
        files.append(
            {
                "source_path": str(source_path),
                "scene_path": scene_relative(scene_dir, scene_path).replace("\\", "/"),
                "sequence_index": index,
                "file": file_identity(scene_path),
                "source_file": file_identity(source_path),
                "image": {
                    "width": int(header.get("width") or 0),
                    "height": int(header.get("height") or 0),
                    "mode": str(header.get("mode") or ""),
                },
                "detected_projection": projection,
                "projection_confidence": header.get("projection_confidence", "low"),
                "projection_reason": header.get("projection_reason", ""),
            }
        )
    unique = sorted(set(projections))
    projection = unique[0] if len(unique) == 1 else ("mixed" if unique else "unknown")
    return {
        "id": f"imageset_{import_id}",
        "source_type": IMAGE_SEQUENCE_SOURCE_TYPE,
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


def _append_selected_frames(scene: Path, import_id: str, source_label: str, image_paths: list[Path]) -> None:
    csv_path = selected_frames_path(scene)
    existing_fields, existing_rows = read_selected_csv(csv_path)
    rows = [*existing_rows]
    next_index = _next_final_index(rows)
    for offset, path in enumerate(image_paths):
        final_index = next_index + offset
        row = {field: "" for field in SELECTED_CSV_FIELDNAMES}
        row.update(
            {
                "source_session": import_id,
                "source_video": "",
                "original_index": str(offset + 1),
                "final_index": str(final_index),
                "status": "ok",
                "decision": "keep",
                "analysis_pipeline": IMAGE_SEQUENCE_SOURCE_TYPE,
                "selection_reason": IMAGE_SEQUENCE_SOURCE_TYPE,
                "review_required": "0",
                "output_file": scene_relative(scene, path).replace("\\", "/"),
                "source_type": IMAGE_SEQUENCE_SOURCE_TYPE,
                "source_label": source_label,
                "import_id": import_id,
            }
        )
        rows.append(row)
    write_selected_csv_rows(csv_path, existing_fields, rows)


def _source_image_files(source: Path, *, recursive: bool) -> list[Path]:
    iterator = source.rglob("*") if recursive else source.iterdir()
    return sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
        key=lambda path: str(path.relative_to(source)).lower(),
    )


def _unique_prefix(images_dir: Path, base: str) -> str:
    if not any(images_dir.glob(f"{base}_*")):
        return base
    for index in range(2, 1000):
        candidate = f"{base}_seq{index}"
        if not any(images_dir.glob(f"{candidate}_*")):
            return candidate
    raise RuntimeError(f"Could not allocate a unique image sequence prefix for: {base}")


def _next_final_index(rows: list[dict[str, Any]]) -> int:
    max_index = 0
    for row in rows:
        try:
            max_index = max(max_index, int(str(row.get("final_index") or row.get("seq") or "0")))
        except ValueError:
            continue
    return max_index + 1
