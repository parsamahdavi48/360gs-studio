"""RealityScan XMP sidecar export helpers.

The exporter writes one sidecar per cubemap image.  RealityScan associates
``Image01.jpg`` with ``Image01.xmp`` when both files are in the same folder.
"""

from __future__ import annotations

import hashlib
import json
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.dataset_writer_colmap import replace_file_with_link_or_copy
from core.realityscan_layout import (
    REALITYSCAN_EXTRA_IMAGE_DIR,
    REALITYSCAN_GEOMETRY_LAYER_DIR,
    REALITYSCAN_MASK_LAYER_DIR,
    extra_geometry_dir,
    extra_mask_dir,
    mask_file_path_for_geometry,
)
from core.scene_import_contracts import IMAGE_EXTS
from core.scene_inventory import SceneImage, build_scene_inventory

REALITYSCAN_POSE_PRIORS = ("initial", "exact", "locked")
REALITYSCAN_CALIBRATION_PRIORS = ("initial", "exact", "locked")
REALITYSCAN_COORDINATE_MODES = ("auto", "absolute", "relative")
REALITYSCAN_XMP_NAMESPACE = "http://www.capturingreality.com/ns/xcr/1.1#"
_XMP_NAMESPACE = "adobe:ns:meta/"
_RDF_NAMESPACE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_UUID_NAMESPACE = uuid.UUID("3f53c4d4-f733-4892-b87c-56824f4f02fb")


@dataclass(frozen=True)
class RealityScanFrameXmp:
    image_path: Path
    xmp_path: Path
    source_file_path: str
    view_name: str
    view_index: int
    rig_guid: str | None
    rig_instance_guid: str | None
    rig_pose_index: int | None


_CUBEMAP_TO_REALITYSCAN_CAMERA = np.diag([1.0, -1.0, -1.0])


def _xcr(name: str) -> str:
    return f"{{{REALITYSCAN_XMP_NAMESPACE}}}{name}"


def _fmt_float(value: float) -> str:
    if abs(value) < 5e-16:
        value = 0.0
    return f"{value:.15g}"


def _fmt_vec(values: np.ndarray) -> str:
    return " ".join(_fmt_float(float(v)) for v in values.reshape(-1))


def _guid(name: str) -> str:
    return "{" + str(uuid.uuid5(_UUID_NAMESPACE, name)).upper() + "}"


def image_path_from_frame(output_dir: Path, frame: dict) -> Path:
    file_path = str(frame.get("file_path") or "").strip()
    if not file_path:
        raise ValueError("frame file_path is empty")
    path = Path(file_path)
    if path.is_absolute():
        return path
    return output_dir / path


def xmp_sidecar_path(image_path: Path) -> Path:
    return image_path.with_suffix(".xmp")


def mask_layer_path(image_path: Path) -> Path:
    return mask_file_path_for_geometry(image_path)


def standard_mask_path(output_dir: Path, image_path: Path) -> Path:
    return mask_file_path_for_geometry(image_path)


def write_realityscan_mask_layer(source_mask: Path, layer_path: Path) -> None:
    """Write a RealityScan mask layer from the repo's white=keep mask.

    RealityScan/RealityCapture uses the same mask polarity as this repository:
    white pixels are used in processing and black pixels are excluded.
    """

    layer_path.parent.mkdir(parents=True, exist_ok=True)
    if _path_key(source_mask) == _path_key(layer_path):
        return
    replace_file_with_link_or_copy(source_mask, layer_path)


def c2w_to_xmp_rotation_position(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a camera-to-world matrix to RealityScan XMP R/C fields.

    RealityScan's XMP camera math uses a world-to-camera rotation matrix and a
    camera center position; the corresponding translation is ``-R * position``.
    The cubemap transform matrix is already expressed in the output dataset
    coordinate profile, so no extra world-axis conversion is applied here.
    """

    if transform.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {transform.shape}")
    rotation = transform[:3, :3].T
    position = transform[:3, 3]
    return rotation, position


def cubemap_c2w_to_xmp_rotation_position(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert this tool's cubemap camera-to-world matrix to RealityScan XMP.

    Cubemap transforms in this repository are written in the NeRF/LichtFeld
    camera basis after Metashape preprocessing. RealityScan XMP expects its own
    camera basis, so flip the local Y/Z camera axes before converting to the
    world-to-camera rotation stored in XMP.
    """

    if transform.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {transform.shape}")
    adjusted = transform.copy()
    adjusted[:3, :3] = adjusted[:3, :3] @ _CUBEMAP_TO_REALITYSCAN_CAMERA
    return c2w_to_xmp_rotation_position(adjusted)


def focal_length_35mm(fl_x: float, fl_y: float, width: int, height: int) -> float:
    scale = max(1, min(int(width), int(height)))
    return ((float(fl_x) + float(fl_y)) * 0.5) * 36.0 / float(scale)


def principal_point_offset(value: float, size: int, scale: int) -> float:
    # RealityScan stores principal point offsets relative to the optical center.
    # The cubemap generator uses the pixel-center convention, so (size - 1) / 2
    # is the generated image center.
    return (float(value) - ((int(size) - 1) / 2.0)) / float(max(1, int(scale)))


def _frame_intrinsics(data: dict, frame: dict) -> tuple[int, int, float, float, float, float]:
    width = int(frame.get("w") or data.get("w") or 0)
    height = int(frame.get("h") or data.get("h") or 0)
    fl_x = float(frame.get("fl_x") or data.get("fl_x") or 0.0)
    fl_y = float(frame.get("fl_y") or data.get("fl_y") or fl_x)
    cx = float(frame.get("cx") if frame.get("cx") is not None else data.get("cx", (width - 1) / 2.0))
    cy = float(frame.get("cy") if frame.get("cy") is not None else data.get("cy", (height - 1) / 2.0))
    if width <= 0 or height <= 0 or fl_x <= 0.0 or fl_y <= 0.0:
        raise ValueError("RealityScan XMP export requires valid PINHOLE intrinsics")
    return width, height, fl_x, fl_y, cx, cy


def _calibration_key(intrinsics: tuple[int, int, float, float, float, float]) -> tuple[int, int, float, float, float, float]:
    width, height, fl_x, fl_y, cx, cy = intrinsics
    return (
        int(width),
        int(height),
        round(float(fl_x), 9),
        round(float(fl_y), 9),
        round(float(cx), 9),
        round(float(cy), 9),
    )


def _validate_prior(value: str, choices: tuple[str, ...], name: str) -> str:
    value = str(value).strip().lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}")
    return value


def realityscan_coordinates_for_pose_prior(pose_prior: str, coordinate_mode: str = "auto") -> str:
    coordinate_mode = str(coordinate_mode or "auto").strip().lower()
    if coordinate_mode not in REALITYSCAN_COORDINATE_MODES:
        raise ValueError(f"--realityscan-coordinates must be one of {', '.join(REALITYSCAN_COORDINATE_MODES)}")
    if coordinate_mode != "auto":
        return coordinate_mode
    return "relative" if pose_prior == "exact" else "absolute"


def _write_xmp(
    path: Path,
    *,
    rotation: np.ndarray,
    position: np.ndarray,
    focal_35mm: float,
    principal_u: float,
    principal_v: float,
    pose_prior: str,
    calibration_prior: str,
    rig_guid: str | None,
    rig_instance_guid: str | None,
    rig_pose_index: int | None,
    calibration_group: int,
    distortion_group: int,
    coordinates: str,
    component_guid: str | None,
) -> None:
    ET.register_namespace("x", _XMP_NAMESPACE)
    ET.register_namespace("rdf", _RDF_NAMESPACE)
    ET.register_namespace("xcr", REALITYSCAN_XMP_NAMESPACE)

    root = ET.Element(f"{{{_XMP_NAMESPACE}}}xmpmeta")
    rdf = ET.SubElement(root, f"{{{_RDF_NAMESPACE}}}RDF")
    attrs = {
        _xcr("Version"): "3",
        _xcr("PosePrior"): pose_prior,
        _xcr("Rotation"): _fmt_vec(rotation),
        _xcr("Coordinates"): coordinates,
        _xcr("DistortionModel"): "division",
        _xcr("DistortionCoeficients"): "0 0 0 0 0 0",
        _xcr("FocalLength35mm"): _fmt_float(focal_35mm),
        _xcr("Skew"): "0",
        _xcr("AspectRatio"): "1",
        _xcr("PrincipalPointU"): _fmt_float(principal_u),
        _xcr("PrincipalPointV"): _fmt_float(principal_v),
        _xcr("CalibrationPrior"): calibration_prior,
        _xcr("CalibrationGroup"): str(int(calibration_group)),
        _xcr("DistortionGroup"): str(int(distortion_group)),
        _xcr("InTexturing"): "1",
        _xcr("InMeshing"): "1",
    }
    if rig_guid is not None and rig_instance_guid is not None and rig_pose_index is not None:
        attrs[_xcr("Rig")] = rig_guid
        attrs[_xcr("RigInstance")] = rig_instance_guid
        attrs[_xcr("RigPoseIndex")] = str(int(rig_pose_index))
    if component_guid is not None:
        attrs[_xcr("ComponentId")] = component_guid
    desc = ET.SubElement(rdf, f"{{{_RDF_NAMESPACE}}}Description", attrs)
    pos = ET.SubElement(desc, _xcr("Position"))
    pos.text = _fmt_vec(position)

    path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True, short_empty_elements=False)


def write_realityscan_xmp_sidecars(
    output_dir: Path,
    *,
    transforms_name: str = "transforms.json",
    pose_prior: str = "exact",
    calibration_prior: str = "exact",
    coordinates: str = "auto",
    rig_name: str = "stechdrive-cubemap",
    include_rig: bool = False,
) -> dict:
    pose_prior = _validate_prior(pose_prior, REALITYSCAN_POSE_PRIORS, "--realityscan-pose-prior")
    calibration_prior = _validate_prior(
        calibration_prior,
        REALITYSCAN_CALIBRATION_PRIORS,
        "--realityscan-calibration-prior",
    )
    coordinates = realityscan_coordinates_for_pose_prior(pose_prior, coordinates)
    transforms_path = output_dir / transforms_name
    data = json.loads(transforms_path.read_text(encoding="utf-8"))
    if str(data.get("camera_model")) != "PINHOLE":
        raise ValueError("RealityScan XMP export expects a PINHOLE transforms.json")

    rig_guid = _guid(f"rig:{rig_name}") if include_rig else None
    component_guid = None

    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("RealityScan XMP export requires at least one frame")

    written: list[RealityScanFrameXmp] = []
    calibration_groups: dict[tuple[int, int, float, float, float, float], int] = {}
    first_focal_35mm = 0.0
    first_principal_u = 0.0
    first_principal_v = 0.0
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        image_path = image_path_from_frame(output_dir, frame)
        intrinsics = _frame_intrinsics(data, frame)
        width, height, fl_x, fl_y, cx, cy = intrinsics
        scale = min(width, height)
        focal_35mm = focal_length_35mm(fl_x, fl_y, width, height)
        principal_u = principal_point_offset(cx, width, scale)
        principal_v = principal_point_offset(cy, height, scale)
        if not written:
            first_focal_35mm = focal_35mm
            first_principal_u = principal_u
            first_principal_v = principal_v
        key = _calibration_key(intrinsics)
        if key not in calibration_groups:
            calibration_groups[key] = len(calibration_groups)
        calibration_group = calibration_groups[key]
        source_file_path = str(frame.get("source_file_path") or frame.get("file_path") or "")
        view_name = str(frame.get("view_name") or image_path.stem.rsplit("_", 1)[-1])
        view_index = int(frame.get("view_index", index))
        source_index = int(frame.get("source_image_index", index))
        rig_instance_guid = _guid(f"rig-instance:{rig_name}:{source_index}:{source_file_path}") if include_rig else None
        rig_pose_index = view_index if include_rig else None
        transform = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        rotation, position = cubemap_c2w_to_xmp_rotation_position(transform)
        xmp_path = xmp_sidecar_path(image_path)
        _write_xmp(
            xmp_path,
            rotation=rotation,
            position=position,
            focal_35mm=focal_35mm,
            principal_u=principal_u,
            principal_v=principal_v,
            pose_prior=pose_prior,
            calibration_prior=calibration_prior,
            rig_guid=rig_guid,
            rig_instance_guid=rig_instance_guid,
            rig_pose_index=rig_pose_index,
            calibration_group=calibration_group,
            distortion_group=calibration_group,
            coordinates=coordinates,
            component_guid=component_guid,
        )
        written.append(
            RealityScanFrameXmp(
                image_path=image_path,
                xmp_path=xmp_path,
                source_file_path=source_file_path,
                view_name=view_name,
                view_index=view_index,
                rig_guid=rig_guid,
                rig_instance_guid=rig_instance_guid,
                rig_pose_index=rig_pose_index,
            )
        )

    manifest = {
        "export_type": "realityscan_xmp",
        "transforms_json": transforms_name,
        "images_dir": f"images/{REALITYSCAN_GEOMETRY_LAYER_DIR}",
        "mask_layer_dir": f"images/{REALITYSCAN_MASK_LAYER_DIR}",
        "pose_prior": pose_prior,
        "calibration_prior": calibration_prior,
        "coordinates": coordinates,
        "rig_metadata": include_rig,
        "rig_name": rig_name,
        "rig_guid": rig_guid,
        "component_guid": component_guid,
        "camera_model": "PINHOLE",
        "focal_length_35mm": first_focal_35mm,
        "principal_point_u": first_principal_u,
        "principal_point_v": first_principal_v,
        "calibration_group": 0,
        "distortion_group": 0,
        "calibration_group_count": len(calibration_groups),
        "calibration_groups": [
            {
                "id": group_id,
                "width": key[0],
                "height": key[1],
                "fl_x": key[2],
                "fl_y": key[3],
                "cx": key[4],
                "cy": key[5],
            }
            for key, group_id in calibration_groups.items()
        ],
        "xmp_count": len(written),
        "mask_layer_count": 0,
        "xmp_files": [str(item.xmp_path.relative_to(output_dir).as_posix()) for item in written],
    }
    return manifest


def write_realityscan_mask_layers(output_dir: Path, *, manifest: dict | None = None) -> dict:
    transforms_path = output_dir / "transforms.json"
    data = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = data.get("frames")
    if not isinstance(frames, list):
        frames = []

    copied: list[str] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        try:
            image_path = image_path_from_frame(output_dir, frame)
        except ValueError:
            continue
        mask_path = standard_mask_path(output_dir, image_path)
        if not mask_path.is_file():
            continue
        layer_path = mask_layer_path(image_path)
        write_realityscan_mask_layer(mask_path, layer_path)
        copied.append(str(layer_path.relative_to(output_dir).as_posix()))

    if manifest is None:
        manifest = {"export_type": "realityscan_xmp"}
    manifest["mask_layer_count"] = len(copied)
    manifest["mask_layer_files"] = copied
    manifest["mask_layer_polarity"] = "white_used_black_excluded"
    manifest["source_mask_polarity"] = "white_keep_black_exclude"
    manifest["mask_layers_inverted_for_realityscan"] = False
    return manifest


def append_realityscan_unposed_scene_images(
    output_dir: Path,
    *,
    scene_dir: Path,
    exclude_source_files: list[str],
    exclude_root: Path,
    include_masks: bool = True,
    manifest: dict | None = None,
) -> dict:
    """Append scene images not present in the Metashape XML as unposed RealityScan inputs."""

    output_dir = Path(output_dir)
    scene_dir = Path(scene_dir)
    exclude_root = Path(exclude_root)
    inventory = build_scene_inventory(scene_dir)
    excluded = {
        _path_key(_resolve_source_reference(exclude_root, rel))
        for rel in exclude_source_files
        if str(rel or "").strip()
    }
    output_images_dir = extra_geometry_dir(output_dir)
    output_masks_dir = extra_mask_dir(output_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    if include_masks and inventory.masks_dir.is_dir():
        output_masks_dir.mkdir(parents=True, exist_ok=True)

    used_names = _existing_output_names(output_images_dir)
    copied_images: list[dict[str, str]] = []
    mask_layers: list[str] = []
    standard_masks: list[str] = []
    link_counts = {"hardlink": 0, "copy": 0, "same": 0}
    skipped_masks = 0

    for image in inventory.images:
        if _path_key(image.path) in excluded or _is_realityscan_layer_file(image.path):
            continue
        output_name = _unique_unposed_image_name(image, used_names)
        destination = output_images_dir / output_name
        link_kind = replace_file_with_link_or_copy(image.path, destination) or "same"
        link_counts[link_kind] = link_counts.get(link_kind, 0) + 1
        copied_images.append(
            {
                "source": image.rel_path,
                "image": str(destination.relative_to(output_dir).as_posix()),
                "link": link_kind,
                "projection": image.projection,
            }
        )
        if not include_masks or image.mask is None or not image.mask.exists:
            continue
        if not image.mask.readable or not image.mask.matches_image_size:
            skipped_masks += 1
            continue
        standard_mask = output_masks_dir / destination.with_suffix(".png").name
        mask_link_kind = replace_file_with_link_or_copy(image.mask.path, standard_mask) or "same"
        link_counts[mask_link_kind] = link_counts.get(mask_link_kind, 0) + 1
        standard_masks.append(str(standard_mask.relative_to(output_dir).as_posix()))

        mask_layers.append(str(standard_mask.relative_to(output_dir).as_posix()))

    if manifest is None:
        manifest = {"export_type": "realityscan_xmp"}

    existing_mask_layers = [str(path) for path in manifest.get("mask_layer_files") or []]
    if "cubemap_mask_layer_count" not in manifest:
        manifest["cubemap_mask_layer_count"] = len(existing_mask_layers)
    combined_mask_layers = _dedupe_text(existing_mask_layers + mask_layers)
    manifest["mask_layer_files"] = combined_mask_layers
    manifest["mask_layer_count"] = len(combined_mask_layers)
    manifest["unposed_image_count"] = len(copied_images)
    manifest["unposed_images_dir"] = f"{REALITYSCAN_EXTRA_IMAGE_DIR}/{REALITYSCAN_GEOMETRY_LAYER_DIR}"
    manifest["unposed_masks_dir"] = f"{REALITYSCAN_EXTRA_IMAGE_DIR}/{REALITYSCAN_MASK_LAYER_DIR}"
    manifest["unposed_mask_layer_count"] = len(mask_layers)
    manifest["unposed_standard_mask_count"] = len(standard_masks)
    manifest["unposed_mask_skipped_count"] = skipped_masks
    manifest["unposed_images"] = copied_images
    manifest["unposed_mask_layer_files"] = mask_layers
    manifest["unposed_standard_mask_files"] = standard_masks
    manifest["unposed_asset_links"] = link_counts
    manifest["unposed_pose"] = "none"
    manifest["unposed_source"] = "scene_images_not_in_metashape_xml"
    return manifest


def _resolve_source_reference(root: Path, value: str) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        return raw
    return root / raw


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve(strict=False)).replace("\\", "/").casefold()
    except OSError:
        return str(path).replace("\\", "/").casefold()


def _existing_output_names(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {path.name.casefold() for path in root.iterdir() if path.is_file() or path.is_symlink()}


def _is_realityscan_layer_file(path: Path) -> bool:
    name = path.name.casefold()
    return ".mask." in name or ".geometry." in name or ".texture" in name


def _unique_unposed_image_name(image: SceneImage, used_names: set[str]) -> str:
    suffix = image.path.suffix.lower()
    if suffix not in IMAGE_EXTS:
        suffix = image.suffix.lower() or ".jpg"
    rel = Path(str(image.rel_path).replace("\\", "/"))
    if rel.parts and rel.parts[0].casefold() == "images":
        rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(image.path.name)
    stem = _safe_name(rel.with_suffix("").as_posix().replace("/", "__"))
    if not stem:
        stem = _safe_name(image.path.stem) or "image"
    candidate = f"extra_{stem}{suffix}"
    if len(candidate) > 180:
        digest = hashlib.sha1(rel.as_posix().encode("utf-8")).hexdigest()[:10]
        candidate = f"extra_{stem[:140]}_{digest}{suffix}"
    base = candidate[: -len(suffix)]
    index = 2
    while candidate.casefold() in used_names:
        candidate = f"{base}_{index}{suffix}"
        index += 1
    used_names.add(candidate.casefold())
    return candidate


def _safe_name(value: str) -> str:
    chars: list[str] = []
    last_was_sep = False
    for char in value:
        if char.isascii() and (char.isalnum() or char in {"-", "_", "."}):
            chars.append(char)
            last_was_sep = False
        elif not last_was_sep:
            chars.append("_")
            last_was_sep = True
    return "".join(chars).strip("._-")


def _dedupe_text(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.replace("\\", "/").casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
