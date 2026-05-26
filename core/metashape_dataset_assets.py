from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.cubemap_image_io import (
    load_equirect,
    remap_with_channels,
    resolve_output_ext,
    save_image,
)
from core.cubemap_remap import (
    build_remap,
    rot4,
    rotation_matrix,
)
from core.dataset_writer_colmap import replace_file_with_link_or_copy
from core.metashape_model import MetashapeSensor


class RemapTableCache:
    def __init__(self, max_entries: int) -> None:
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[
            tuple[int, int, int, float, float, float],
            tuple[np.ndarray, np.ndarray],
        ] = OrderedDict()

    @staticmethod
    def _key(
        input_size: tuple[int, int],
        output_size: int,
        fov_deg: float,
        yaw_deg: float,
        pitch_deg: float,
    ) -> tuple[int, int, int, float, float, float]:
        return (
            int(input_size[0]),
            int(input_size[1]),
            int(output_size),
            round(float(fov_deg), 6),
            round(float(yaw_deg), 6),
            round(float(pitch_deg), 6),
        )

    def get(
        self,
        input_size: tuple[int, int],
        output_size: int,
        fov_deg: float,
        yaw_deg: float,
        pitch_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = self._key(input_size, output_size, fov_deg, yaw_deg, pitch_deg)
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return cached

        tables = build_remap(input_size, fov_deg, yaw_deg, pitch_deg, output_size)
        self._entries[key] = tables
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return tables


@dataclass(frozen=True, slots=True)
class MetashapeDatasetAsset:
    image_path: Path
    images_root: Path
    mask_path: Path | None
    masks_root: Path
    camera_model: str
    width: int
    height: int
    params: tuple[float, ...]
    c2w: np.ndarray
    action: str
    source_image: Path
    source_camera_id: str
    source_camera_label: str
    view_name: str = ""

    @property
    def image_name(self) -> str:
        return self.image_path.relative_to(self.images_root).as_posix()

    @property
    def mask_name(self) -> str:
        if self.mask_path is None:
            return ""
        return self.mask_path.relative_to(self.masks_root).as_posix()


def resolve_inventory_path(scene: Path, root: Path, rel_path: str, *, standard_root_name: str) -> Path:
    raw = Path(rel_path)
    if raw.is_absolute():
        return raw
    candidates: list[Path] = []
    parts = raw.parts
    if parts and parts[0].casefold() == standard_root_name.casefold():
        candidates.append(root / Path(*parts[1:]) if len(parts) > 1 else root)
    candidates.append(scene / raw)
    candidates.append(root / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else root / raw


def expand_erp_to_view_assets(
    *,
    source_image: Path,
    source_mask: Path | None,
    images_root: Path,
    output_images: Path,
    output_masks: Path,
    views: list[dict[str, Any]],
    output_scale: float,
    output_format: str,
    output_bit_depth: str,
    jpg_quality: int,
    fov_deg: float,
    c2w: np.ndarray,
    action: str,
    source_camera_id: str,
    source_camera_label: str,
    remap_cache: RemapTableCache | None = None,
    write_images: bool = True,
    write_masks: bool = True,
) -> tuple[MetashapeDatasetAsset, ...]:
    source = load_equirect(str(source_image)) if write_images else None
    input_size = (int(source.shape[1]), int(source.shape[0])) if source is not None else image_size(source_image)
    output_size = max(1, int(round(input_size[1] * float(output_scale))))
    out_ext = resolve_output_ext(source_image.suffix, output_format)
    mask = load_equirect(str(source_mask)) if write_masks and source_mask is not None and source_mask.is_file() else None
    rel = relative_image_path(source_image, images_root)

    assets: list[MetashapeDatasetAsset] = []
    for view in views:
        if not bool(view.get("enabled", True)):
            continue
        name = str(view["name"])
        yaw = float(view["yaw"])
        pitch = float(view["pitch"])
        output_image = output_images / rel.with_name(f"{rel.stem}_{name}{out_ext}")
        map_x: np.ndarray | None = None
        map_y: np.ndarray | None = None
        needs_remap = write_images or mask is not None
        if needs_remap:
            if remap_cache is not None:
                map_x, map_y = remap_cache.get(input_size, output_size, fov_deg, yaw, pitch)
            else:
                map_x, map_y = build_remap(input_size, fov_deg, yaw, pitch, output_size)
        if write_images:
            if source is None:
                source = load_equirect(str(source_image))
            if map_x is None or map_y is None:
                map_x, map_y = build_remap(input_size, fov_deg, yaw, pitch, output_size)
            output_image.parent.mkdir(parents=True, exist_ok=True)
            save_image(
                remap_with_channels(source, map_x, map_y),
                str(output_image),
                jpg_quality=jpg_quality,
                force_8bit=output_bit_depth == "8",
            )
        else:
            _require_existing_output(output_image, kind="image")

        output_mask: Path | None = None
        if mask is not None:
            if map_x is None or map_y is None:
                map_x, map_y = build_remap(input_size, fov_deg, yaw, pitch, output_size)
            output_mask = output_masks / output_image.relative_to(output_images).with_suffix(".png")
            output_mask.parent.mkdir(parents=True, exist_ok=True)
            converted_mask = remap_with_channels(mask, map_x, map_y)
            _threshold, binary = cv2.threshold(_ensure_gray(converted_mask), 127, 255, cv2.THRESH_BINARY)
            save_image(binary, str(output_mask), force_8bit=True)
        elif not write_images and not write_masks and source_mask is not None:
            existing_mask = output_masks / output_image.relative_to(output_images).with_suffix(".png")
            if existing_mask.is_file():
                output_mask = existing_mask

        view_c2w = c2w @ rot4(rotation_matrix(yaw, pitch, True).T)
        focal = output_size / 2.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
        principal = (output_size - 1.0) / 2.0
        assets.append(
            MetashapeDatasetAsset(
                image_path=output_image,
                images_root=output_images,
                mask_path=output_mask,
                masks_root=output_masks,
                camera_model="PINHOLE",
                width=output_size,
                height=output_size,
                params=(float(focal), float(focal), float(principal), float(principal)),
                c2w=view_c2w,
                action=action,
                source_image=source_image,
                source_camera_id=source_camera_id,
                source_camera_label=source_camera_label,
                view_name=name,
            )
        )
    return tuple(assets)


def link_pinhole_asset(
    *,
    sensor: MetashapeSensor,
    source_image: Path,
    source_mask: Path | None,
    images_root: Path,
    masks_root: Path,
    output_images: Path,
    output_masks: Path,
    c2w: np.ndarray,
    action: str,
    source_camera_id: str,
    source_camera_label: str,
    write_images: bool = True,
    write_masks: bool = True,
) -> MetashapeDatasetAsset:
    output_image = (
        linked_or_copied_output(source_image, images_root, output_images)
        if write_images
        else linked_output_path(source_image, images_root, output_images)
    )
    if not write_images:
        _require_existing_output(output_image, kind="image")
    if write_masks:
        output_mask = copy_mask_if_available(source_mask, masks_root, output_masks)
    elif not write_images and source_mask is not None:
        existing_mask = copied_mask_path(source_mask, masks_root, output_masks)
        output_mask = existing_mask if existing_mask.is_file() else None
    else:
        output_mask = None
    width, height, params = pinhole_payload(sensor, source_image)
    return MetashapeDatasetAsset(
        image_path=output_image,
        images_root=output_images,
        mask_path=output_mask,
        masks_root=output_masks,
        camera_model="PINHOLE",
        width=width,
        height=height,
        params=params,
        c2w=c2w,
        action=action,
        source_image=source_image,
        source_camera_id=source_camera_id,
        source_camera_label=source_camera_label,
    )


def undistort_frame_to_pinhole_asset(
    *,
    sensor: MetashapeSensor,
    source_image: Path,
    source_mask: Path | None,
    images_root: Path,
    output_images: Path,
    output_masks: Path,
    output_format: str,
    output_bit_depth: str,
    jpg_quality: int,
    alpha: float,
    c2w: np.ndarray,
    action: str,
    source_camera_id: str,
    source_camera_label: str,
    write_images: bool = True,
    write_masks: bool = True,
) -> MetashapeDatasetAsset:
    image = None
    if write_images:
        image = cv2.imread(str(source_image), cv2.IMREAD_UNCHANGED)
        if image is None:
            image = load_equirect(str(source_image))
    width, height, params = pinhole_payload(sensor, source_image)
    matrix = np.array([[params[0], 0.0, params[2]], [0.0, params[1], 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    matrix[1, 2] = params[3]
    distortion = distortion_coefficients(sensor)
    new_matrix, _roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (width, height), alpha, (width, height))
    needs_mask = write_masks and ((source_mask is not None and source_mask.is_file()) or alpha > 0.0)
    map_x: np.ndarray | None = None
    map_y: np.ndarray | None = None
    if write_images or needs_mask:
        map_x, map_y = cv2.initUndistortRectifyMap(
            matrix,
            distortion,
            None,
            new_matrix,
            (width, height),
            cv2.CV_32FC1,
        )

    rel = relative_image_path(source_image, images_root)
    output_ext = resolve_output_ext(source_image.suffix, output_format)
    output_image = output_images / rel.with_name(f"{rel.stem}_undistorted{output_ext}")
    if write_images:
        if image is None:
            image = load_equirect(str(source_image))
        if map_x is None or map_y is None:
            map_x, map_y = cv2.initUndistortRectifyMap(
                matrix,
                distortion,
                None,
                new_matrix,
                (width, height),
                cv2.CV_32FC1,
            )
        undistorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        output_image.parent.mkdir(parents=True, exist_ok=True)
        save_image(
            undistorted,
            str(output_image),
            jpg_quality=jpg_quality,
            force_8bit=output_bit_depth == "8",
        )
    else:
        _require_existing_output(output_image, kind="image")

    mask = load_equirect(str(source_mask)) if write_masks and source_mask is not None and source_mask.is_file() else None
    if write_masks and mask is None and alpha > 0.0:
        mask = np.full((height, width), 255, dtype=np.uint8)
    output_mask: Path | None = None
    if mask is not None:
        if map_x is None or map_y is None:
            map_x, map_y = cv2.initUndistortRectifyMap(
                matrix,
                distortion,
                None,
                new_matrix,
                (width, height),
                cv2.CV_32FC1,
            )
        output_mask = output_masks / output_image.relative_to(output_images).with_suffix(".png")
        output_mask.parent.mkdir(parents=True, exist_ok=True)
        undistorted_mask = cv2.remap(
            _ensure_gray(mask),
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        _threshold, binary = cv2.threshold(undistorted_mask, 127, 255, cv2.THRESH_BINARY)
        save_image(binary, str(output_mask), force_8bit=True)
    elif not write_images and not write_masks and source_mask is not None:
        existing_mask = output_masks / output_image.relative_to(output_images).with_suffix(".png")
        if existing_mask.is_file():
            output_mask = existing_mask

    return MetashapeDatasetAsset(
        image_path=output_image,
        images_root=output_images,
        mask_path=output_mask,
        masks_root=output_masks,
        camera_model="PINHOLE",
        width=width,
        height=height,
        params=(
            float(new_matrix[0, 0]),
            float(new_matrix[1, 1]),
            float(new_matrix[0, 2]),
            float(new_matrix[1, 2]),
        ),
        c2w=c2w,
        action=action,
        source_image=source_image,
        source_camera_id=source_camera_id,
        source_camera_label=source_camera_label,
    )


def linked_or_copied_output(source: Path, images_root: Path, output_images: Path) -> Path:
    destination = linked_output_path(source, images_root, output_images)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    replace_file_with_link_or_copy(source, destination)
    return destination


def linked_output_path(source: Path, images_root: Path, output_images: Path) -> Path:
    rel = relative_image_path(source, images_root)
    return output_images / rel


def copy_mask_if_available(source_mask: Path | None, masks_root: Path, output_masks: Path) -> Path | None:
    if source_mask is None or not source_mask.is_file():
        return None
    destination = copied_mask_path(source_mask, masks_root, output_masks)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    replace_file_with_link_or_copy(source_mask, destination)
    return destination


def copied_mask_path(source_mask: Path, masks_root: Path, output_masks: Path) -> Path:
    rel = relative_image_path(source_mask, masks_root) if masks_root else Path(source_mask.name)
    return output_masks / rel.with_suffix(".png")


def _require_existing_output(path: Path, *, kind: str) -> None:
    if not path.is_file():
        raise ValueError(f"Existing output {kind} is required when {kind} export is disabled: {path}")


def pinhole_payload(sensor: MetashapeSensor, image_path: Path) -> tuple[int, int, tuple[float, ...]]:
    width, height = image_size(image_path)
    sx = width / sensor.width if sensor.width > 0 else 1.0
    sy = height / sensor.height if sensor.height > 0 else 1.0
    fl_x = float(sensor.params.get("fl_x") or sensor.params.get("f") or width)
    fl_y = float(sensor.params.get("fl_y") or fl_x)
    cx = float(sensor.params.get("cx") or width / 2.0)
    cy = float(sensor.params.get("cy") or height / 2.0)
    return width, height, (fl_x * sx, fl_y * sy, cx * sx, cy * sy)


def distortion_coefficients(sensor: MetashapeSensor) -> np.ndarray:
    return np.array(
        [
            float(sensor.params.get("k1", 0.0)),
            float(sensor.params.get("k2", 0.0)),
            float(sensor.params.get("p1", 0.0)),
            float(sensor.params.get("p2", 0.0)),
            float(sensor.params.get("k3", 0.0)),
            float(sensor.params.get("k4", 0.0)),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )


def relative_image_path(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            return int(width), int(height)
    except Exception:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            image = load_equirect(str(path))
        height, width = image.shape[:2]
        return int(width), int(height)


def ensure_gray(image: np.ndarray) -> np.ndarray:
    return _ensure_gray(image)


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return image[..., 3]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
