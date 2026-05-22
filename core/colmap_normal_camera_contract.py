from __future__ import annotations

import re
from dataclasses import dataclass

from core.scene_inventory import SceneImage

COLMAP_NORMAL_CAMERA_MODEL = "SIMPLE_RADIAL"

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class NormalCameraGroup:
    group_id: str
    image_dir: str
    camera_model: str
    width: int
    height: int
    source_kind: str
    source_id: str
    image_count: int = 0

    def with_image_count(self, count: int) -> NormalCameraGroup:
        return NormalCameraGroup(
            group_id=self.group_id,
            image_dir=self.image_dir,
            camera_model=self.camera_model,
            width=self.width,
            height=self.height,
            source_kind=self.source_kind,
            source_id=self.source_id,
            image_count=int(count),
        )


def normal_camera_group_for_image(image: SceneImage) -> NormalCameraGroup:
    source_kind = image.source_kind or "unknown"
    source_id = image.source_id or source_kind or "normal"
    width = image.width if image.width > 0 else 0
    height = image.height if image.height > 0 else 0
    group_id = f"{_safe_name(source_id, fallback='source')}_{width}x{height}"
    return NormalCameraGroup(
        group_id=group_id,
        image_dir=f"normal/{group_id}",
        camera_model=COLMAP_NORMAL_CAMERA_MODEL,
        width=width,
        height=height,
        source_kind=source_kind,
        source_id=source_id,
    )


def normal_camera_groups_for_images(images: list[SceneImage]) -> list[NormalCameraGroup]:
    counts: dict[str, int] = {}
    groups: dict[str, NormalCameraGroup] = {}
    for image in images:
        group = normal_camera_group_for_image(image)
        groups[group.group_id] = group
        counts[group.group_id] = counts.get(group.group_id, 0) + 1
    return [groups[group_id].with_image_count(counts[group_id]) for group_id in sorted(groups)]


def _safe_name(value: str, *, fallback: str) -> str:
    text = _SAFE_NAME_RE.sub("_", value.strip()).strip("._-")
    return text or fallback
