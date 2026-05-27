from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.dataset_writer_colmap import replace_file_with_link_or_copy


@dataclass(frozen=True, slots=True)
class NerfDatasetWriteResult:
    root: Path
    transforms_json: Path
    pointcloud: Path | None
    frame_count: int
    metadata: dict[str, Any]


def write_nerf_json_ply_dataset(
    output_dir: str | Path,
    transforms_payload: dict[str, Any],
    *,
    pointcloud_source: str | Path | None = None,
    transforms_name: str = "transforms.json",
    pointcloud_name: str = "pointcloud.ply",
    manifest: dict[str, Any] | None = None,
) -> NerfDatasetWriteResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if "frames" not in transforms_payload or not isinstance(transforms_payload["frames"], list):
        raise ValueError("NeRF dataset payload must contain a frames list")

    transforms_path = output / transforms_name
    transforms_path.write_text(json.dumps(transforms_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    pointcloud_output: Path | None = None
    if pointcloud_source:
        source = Path(pointcloud_source)
        if source.is_file():
            pointcloud_output = output / pointcloud_name
            replace_file_with_link_or_copy(source, pointcloud_output)
    elif (output / pointcloud_name).is_file():
        pointcloud_output = output / pointcloud_name

    metadata = {
        "schema_version": 1,
        "kind": "nerf_json_ply",
        "transforms_json": transforms_path.name,
        "pointcloud": pointcloud_output.name if pointcloud_output else "",
        "frame_count": len(transforms_payload["frames"]),
        **(manifest or {}),
    }
    return NerfDatasetWriteResult(
        root=output,
        transforms_json=transforms_path,
        pointcloud=pointcloud_output,
        frame_count=len(transforms_payload["frames"]),
        metadata=metadata,
    )
