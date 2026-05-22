from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.metashape_model import (
    CAMERA_MODEL_EQUIRECTANGULAR,
    CAMERA_MODEL_OPENCV,
    CAMERA_MODEL_PINHOLE,
    MetashapeCamera,
    MetashapeModel,
    MetashapeSensor,
)
from core.scene_inventory import SceneImage, SceneInventory

EXPORT_ACTION_EXPAND_ERP_TO_VIEWS = "expand_erp_to_views"
EXPORT_ACTION_LINK_PINHOLE = "link_pinhole"
EXPORT_ACTION_UNDISTORT_FRAME_TO_PINHOLE = "undistort_frame_to_pinhole"
EXPORT_ACTION_SKIP = "skip"


@dataclass(frozen=True, slots=True)
class DatasetExportPlanItem:
    action: str
    camera_id: str
    camera_label: str
    sensor_id: str
    camera_model: str
    image_rel_path: str
    mask_rel_path: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class DatasetExportPlan:
    source_kind: str
    items: tuple[DatasetExportPlanItem, ...]
    warnings: tuple[str, ...] = ()

    def items_for_action(self, action: str) -> tuple[DatasetExportPlanItem, ...]:
        return tuple(item for item in self.items if item.action == action)


def build_metashape_dataset_export_plan(
    model: MetashapeModel,
    inventory: SceneInventory,
) -> DatasetExportPlan:
    image_lookup = _build_image_lookup(inventory)
    items: list[DatasetExportPlanItem] = []
    warnings: list[str] = []
    for camera in model.cameras:
        sensor = model.sensor_for_camera(camera)
        image = _resolve_camera_image(camera, image_lookup)
        if image is None:
            warnings.append(f"Camera image not found in scene images: {camera.label or camera.camera_id}")
            items.append(_plan_item(camera, sensor, EXPORT_ACTION_SKIP, reason="missing_image"))
            continue
        mask_rel = image.mask.rel_path if image.mask is not None and image.mask.exists else ""
        items.append(
            _plan_item(
                camera,
                sensor,
                _action_for_sensor(sensor),
                image_rel_path=image.rel_path,
                mask_rel_path=mask_rel,
            )
        )
    return DatasetExportPlan(source_kind="metashape_xml_ply", items=tuple(items), warnings=tuple(warnings))


def _action_for_sensor(sensor: MetashapeSensor) -> str:
    if sensor.camera_model == CAMERA_MODEL_EQUIRECTANGULAR:
        return EXPORT_ACTION_EXPAND_ERP_TO_VIEWS
    if sensor.camera_model == CAMERA_MODEL_PINHOLE:
        return EXPORT_ACTION_LINK_PINHOLE
    if sensor.camera_model == CAMERA_MODEL_OPENCV:
        return EXPORT_ACTION_UNDISTORT_FRAME_TO_PINHOLE
    return EXPORT_ACTION_SKIP


def _plan_item(
    camera: MetashapeCamera,
    sensor: MetashapeSensor,
    action: str,
    *,
    image_rel_path: str = "",
    mask_rel_path: str = "",
    reason: str = "",
) -> DatasetExportPlanItem:
    return DatasetExportPlanItem(
        action=action,
        camera_id=camera.camera_id,
        camera_label=camera.label,
        sensor_id=sensor.sensor_id,
        camera_model=sensor.camera_model,
        image_rel_path=image_rel_path,
        mask_rel_path=mask_rel_path,
        reason=reason,
    )


def _build_image_lookup(inventory: SceneInventory) -> dict[str, SceneImage]:
    lookup: dict[str, SceneImage] = {}
    for image in inventory.images:
        path = Path(image.rel_path)
        keys = {
            image.rel_path.replace("\\", "/").casefold(),
            path.name.casefold(),
            path.stem.casefold(),
        }
        for key in keys:
            lookup.setdefault(key, image)
    return lookup


def _resolve_camera_image(camera: MetashapeCamera, lookup: dict[str, SceneImage]) -> SceneImage | None:
    label = camera.label.replace("\\", "/").strip()
    if not label:
        return None
    candidates = [
        label.casefold(),
        Path(label).name.casefold(),
        Path(label).stem.casefold(),
    ]
    for key in candidates:
        image = lookup.get(key)
        if image is not None:
            return image
    return None
