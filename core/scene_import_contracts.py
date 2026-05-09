from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXTERNAL_IMPORT_KIND = "external_import"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
MASK_EXTS = IMAGE_EXTS

SELECTED_CSV_FIELDNAMES = [
    "seq",
    "source_session",
    "source_video",
    "original_index",
    "final_index",
    "timestamp_sec",
    "change_score_original",
    "change_score_final",
    "blur_score_original",
    "blur_score_final",
    "sharpness_baseline",
    "sharpness_ratio",
    "status",
    "decision",
    "analysis_pipeline",
    "selection_reason",
    "review_required",
    "prev_kept_index",
    "gap_sec",
    "yaw_shift_px",
    "yaw_shift_deg",
    "residual_score",
    "raw_change_score",
    "track_count",
    "track_coverage",
    "match_confidence",
    "risk_flags",
    "analysis_width",
    "pair_gate_width",
    "pair_motion_profile",
    "pair_threshold_mode",
    "pair_drop_threshold",
    "pair_add_threshold",
    "output_file",
    "source_type",
    "source_label",
    "import_id",
]


@dataclass(frozen=True)
class SceneImportResult:
    scene_dir: Path
    import_id: str
    status: str
    image_count: int
    mask_count: int
    output_image_count: int
    output_mask_count: int
    output_shape: str
    dataset_kind: str
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    backup_dir: Path | None
    report_path: Path
    selected_frames_csv: Path | None
    export_settings_json: Path

    def summary_lines(self) -> list[str]:
        lines = [
            f"Scene import: {self.scene_dir}",
            f"Import ID: {self.import_id}",
            f"Status: {self.status}",
            f"Source images: {self.image_count}",
            f"Source masks registered: {self.mask_count}",
            f"Output images: {self.output_image_count}",
            f"Output masks: {self.output_mask_count}",
        ]
        if self.output_shape:
            lines.append(f"Output shape: {self.output_shape}")
        if self.backup_dir is not None:
            lines.append(f"Metadata backup: {self.backup_dir}")
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
            lines.extend(f"  - {warning}" for warning in self.warnings[:12])
            if len(self.warnings) > 12:
                lines.append(f"  - ... +{len(self.warnings) - 12}")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            lines.extend(f"  - {error}" for error in self.errors)
        lines.append(f"Report: {self.report_path}")
        return lines


class IssueSummary:
    def __init__(self, label: str, *, limit: int = 6) -> None:
        self.label = label
        self.limit = limit
        self.count = 0
        self.examples: list[str] = []

    def add(self, example: str) -> None:
        self.count += 1
        if len(self.examples) < self.limit:
            self.examples.append(example)

    def message(self) -> str:
        if self.count <= 0:
            return ""
        text = f"{self.label}: {self.count}"
        if self.examples:
            text += f" (examples: {', '.join(self.examples)})"
        return text


def new_import_id() -> str:
    return f"import_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"


def import_origin(import_id: str) -> dict[str, Any]:
    return {
        "kind": EXTERNAL_IMPORT_KIND,
        "import_id": import_id,
        "generated_by_app": False,
        "rerun_available": False,
    }


def is_external_import_record(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    origin = record.get("origin")
    return isinstance(origin, dict) and origin.get("kind") == EXTERNAL_IMPORT_KIND
