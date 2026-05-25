from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.metashape_coordinates import metashape_camera_to_world, metashape_pointcloud_file_matrix
from core.metashape_model import CAMERA_MODEL_EQUIRECTANGULAR, MetashapeSensor, parse_metashape_model
from core.realityscan_to_transforms import write_transformed_ply
from core.scene_inventory import build_scene_image_label_path_lookup_with_warnings, resolve_scene_image_label

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class MetashapePreprocessResult:
    output_dir: Path
    transforms_json: Path
    pointcloud: Path | None
    num_frames: int
    num_skipped: int
    camera_model: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_legacy_summary(self) -> dict[str, Any]:
        return {
            "num_frames": self.num_frames,
            "num_skipped": self.num_skipped,
            "camera_model": self.camera_model,
            "has_pointcloud": self.pointcloud is not None,
        }


def _notify_progress(callback: ProgressCallback | None, done: int, total: int) -> None:
    if callback is not None:
        callback(max(0, int(done)), max(0, int(total)))


def export_metashape_equirectangular_dataset(
    *,
    images_dir: str | Path,
    xml_path: str | Path,
    output_dir: str | Path,
    ply_path: str | Path | None = None,
    fix_upside_down: bool = True,
    scale: float = 1.0,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> MetashapePreprocessResult:
    """Create the intermediate ERP transforms dataset used before cubemap export."""
    images_root = Path(images_dir)
    xml = Path(xml_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model = parse_metashape_model(xml)
    camera_count = len(model.cameras)
    progress_total = max(1, camera_count + 1 + (1 if ply_path else 0))
    _notify_progress(progress_callback, 0, progress_total)
    used_sensors = {camera.sensor_id: model.sensor_for_camera(camera) for camera in model.cameras}
    unsupported = sorted(
        {
            sensor.camera_model
            for sensor in used_sensors.values()
            if sensor.camera_model != CAMERA_MODEL_EQUIRECTANGULAR
        }
    )
    if unsupported:
        raise ValueError(
            "Metashape preprocess accepts equirectangular cameras only. "
            f"Use the mixed Metashape dataset writer for camera models: {', '.join(unsupported)}"
        )

    image_lookup, lookup_warnings = build_image_lookup(images_root)
    warnings = list(lookup_warnings)
    frames: list[dict[str, Any]] = []
    skipped = 0

    for camera_index, camera in enumerate(model.cameras, start=1):
        sensor = model.sensor_for_camera(camera)
        image_path = resolve_camera_image(camera.label, image_lookup)
        if image_path is None:
            skipped += 1
            warnings.append(f"Camera image not found: {camera.label or camera.camera_id}")
            if verbose:
                print(f"  Skipping {camera.label or camera.camera_id}: no matching image", flush=True)
            _notify_progress(progress_callback, camera_index, progress_total)
            continue

        transform = metashape_camera_to_world(model, camera, fix_upside_down=fix_upside_down)
        transform[:3, 3] *= float(scale)
        frame = {
            "file_path": output_file_path(image_path, images_root, output),
            "transform_matrix": transform.tolist(),
        }
        frame.update(sensor_payload(sensor))
        frames.append(frame)
        _notify_progress(progress_callback, camera_index, progress_total)

    if not frames:
        detail = f": {'; '.join(warnings)}" if warnings else ""
        raise ValueError(f"No Metashape cameras were converted{detail}")

    applied_transform = metashape_pointcloud_file_matrix(fix_upside_down=fix_upside_down, scale=1.0)[:3, :]
    payload: dict[str, Any] = {
        "camera_model": CAMERA_MODEL_EQUIRECTANGULAR,
        "frames": frames,
        "applied_transform": applied_transform.tolist(),
        "source": {
            "type": "metashape_xml_ply",
            "xml_path": str(xml),
            "images_dir": str(images_root),
            "fix_upside_down": bool(fix_upside_down),
            "scale": float(scale),
            "writer": "core.metashape_preprocess",
            "warnings": warnings,
        },
    }

    pointcloud_output: Path | None = None
    ply = Path(ply_path) if ply_path else None
    if ply is not None and ply.is_file():
        pointcloud_output = output / "pointcloud.ply"
        write_transformed_ply(
            ply,
            pointcloud_output,
            metashape_pointcloud_file_matrix(fix_upside_down=fix_upside_down, scale=scale),
        )
        payload["ply_file_path"] = pointcloud_output.name
    if ply is not None:
        _notify_progress(progress_callback, camera_count + 1, progress_total)

    transforms_path = output / "transforms.json"
    transforms_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "stechdrive_metashape_preprocess.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "metashape_preprocess",
                "source_kind": "metashape_xml_ply",
                "camera_model": CAMERA_MODEL_EQUIRECTANGULAR,
                "frames": len(frames),
                "skipped": skipped,
                "pointcloud": pointcloud_output.name if pointcloud_output else "",
                "warnings": warnings,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _notify_progress(progress_callback, progress_total, progress_total)

    if verbose:
        print(f"Processed {len(frames)} camera frames", flush=True)
        if skipped:
            print(f"Skipped {skipped} cameras", flush=True)
        print(f"Wrote {transforms_path}", flush=True)
        if pointcloud_output is not None:
            print(f"Wrote pointcloud: {pointcloud_output}", flush=True)

    return MetashapePreprocessResult(
        output_dir=output,
        transforms_json=transforms_path,
        pointcloud=pointcloud_output,
        num_frames=len(frames),
        num_skipped=skipped,
        camera_model=CAMERA_MODEL_EQUIRECTANGULAR,
        warnings=tuple(warnings),
    )


def sensor_payload(sensor: MetashapeSensor) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "w": int(sensor.width),
        "h": int(sensor.height),
    }
    for key in ("fl_x", "fl_y", "cx", "cy"):
        if key in sensor.params:
            payload[key] = float(sensor.params[key])
    return payload


def build_image_lookup(images_dir: Path) -> tuple[dict[str, Path], tuple[str, ...]]:
    # Image label semantics are owned by SceneInventory so Metashape routes and
    # mixed-source dataset routes resolve labels the same way.
    return build_scene_image_label_path_lookup_with_warnings(images_dir.parent, images_dir=images_dir)


def resolve_camera_image(label: str, image_lookup: dict[str, Path]) -> Path | None:
    return resolve_scene_image_label(label, image_lookup)


def output_file_path(image_path: Path, images_dir: Path, output_dir: Path) -> str:
    try:
        return image_path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        pass
    if images_dir.name.casefold() == "images":
        try:
            return (Path("images") / image_path.resolve().relative_to(images_dir.resolve())).as_posix()
        except ValueError:
            return (Path("images") / image_path.name).as_posix()
    return image_path.resolve().as_posix()
