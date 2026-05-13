"""RealityScan XMP sidecar export helpers.

The exporter writes one sidecar per cubemap image.  RealityScan associates
``Image01.jpg`` with ``Image01.xmp`` when both files are in the same folder.
"""

from __future__ import annotations

import json
import shutil
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REALITYSCAN_POSE_PRIORS = ("initial", "exact", "locked")
REALITYSCAN_CALIBRATION_PRIORS = ("initial", "exact", "locked")
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
    rig_guid: str
    rig_instance_guid: str
    rig_pose_index: int


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
    return Path(f"{image_path}.mask.png")


def standard_mask_path(output_dir: Path, image_path: Path) -> Path:
    return output_dir / "masks" / f"{image_path.stem}.png"


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


def focal_length_35mm(fl_x: float, fl_y: float, width: int, height: int) -> float:
    scale = max(1, min(int(width), int(height)))
    return ((float(fl_x) + float(fl_y)) * 0.5) * 36.0 / float(scale)


def principal_point_offset(value: float, size: int, scale: int) -> float:
    # RealityScan stores principal point offsets relative to the optical center.
    # The cubemap generator uses the pixel-center convention, so (size - 1) / 2
    # is the generated image center.
    return (float(value) - ((int(size) - 1) / 2.0)) / float(max(1, int(scale)))


def _validate_prior(value: str, choices: tuple[str, ...], name: str) -> str:
    value = str(value).strip().lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}")
    return value


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
    rig_guid: str,
    rig_instance_guid: str,
    rig_pose_index: int,
    calibration_group: int,
    distortion_group: int,
) -> None:
    ET.register_namespace("x", _XMP_NAMESPACE)
    ET.register_namespace("rdf", _RDF_NAMESPACE)
    ET.register_namespace("xcr", REALITYSCAN_XMP_NAMESPACE)

    root = ET.Element(f"{{{_XMP_NAMESPACE}}}xmpmeta")
    rdf = ET.SubElement(root, f"{{{_RDF_NAMESPACE}}}RDF")
    desc = ET.SubElement(
        rdf,
        f"{{{_RDF_NAMESPACE}}}Description",
        {
            _xcr("Version"): "3",
            _xcr("PosePrior"): pose_prior,
            _xcr("Rotation"): _fmt_vec(rotation),
            _xcr("Coordinates"): "absolute",
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
            _xcr("Rig"): rig_guid,
            _xcr("RigInstance"): rig_instance_guid,
            _xcr("RigPoseIndex"): str(int(rig_pose_index)),
            _xcr("InTexturing"): "1",
            _xcr("InMeshing"): "1",
        },
    )
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
    calibration_prior: str = "initial",
    rig_name: str = "stechdrive-cubemap",
) -> dict:
    pose_prior = _validate_prior(pose_prior, REALITYSCAN_POSE_PRIORS, "--realityscan-pose-prior")
    calibration_prior = _validate_prior(
        calibration_prior,
        REALITYSCAN_CALIBRATION_PRIORS,
        "--realityscan-calibration-prior",
    )
    transforms_path = output_dir / transforms_name
    data = json.loads(transforms_path.read_text(encoding="utf-8"))
    if str(data.get("camera_model")) != "PINHOLE":
        raise ValueError("RealityScan XMP export expects a PINHOLE transforms.json")

    width = int(data.get("w", 0))
    height = int(data.get("h", 0))
    fl_x = float(data.get("fl_x", 0.0))
    fl_y = float(data.get("fl_y", fl_x))
    cx = float(data.get("cx", (width - 1) / 2.0))
    cy = float(data.get("cy", (height - 1) / 2.0))
    if width <= 0 or height <= 0 or fl_x <= 0.0 or fl_y <= 0.0:
        raise ValueError("RealityScan XMP export requires valid PINHOLE intrinsics")

    scale = min(width, height)
    focal_35mm = focal_length_35mm(fl_x, fl_y, width, height)
    principal_u = principal_point_offset(cx, width, scale)
    principal_v = principal_point_offset(cy, height, scale)
    rig_guid = _guid(f"rig:{rig_name}")

    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("RealityScan XMP export requires at least one frame")

    written: list[RealityScanFrameXmp] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        image_path = image_path_from_frame(output_dir, frame)
        source_file_path = str(frame.get("source_file_path") or frame.get("file_path") or "")
        view_name = str(frame.get("view_name") or image_path.stem.rsplit("_", 1)[-1])
        view_index = int(frame.get("view_index", index))
        source_index = int(frame.get("source_image_index", index))
        rig_instance_guid = _guid(f"rig-instance:{rig_name}:{source_index}:{source_file_path}")
        transform = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        rotation, position = c2w_to_xmp_rotation_position(transform)
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
            rig_pose_index=view_index,
            calibration_group=0,
            distortion_group=0,
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
                rig_pose_index=view_index,
            )
        )

    manifest = {
        "export_type": "realityscan_xmp",
        "transforms_json": transforms_name,
        "images_dir": "images",
        "pose_prior": pose_prior,
        "calibration_prior": calibration_prior,
        "coordinates": "absolute",
        "rig_name": rig_name,
        "rig_guid": rig_guid,
        "camera_model": "PINHOLE",
        "focal_length_35mm": focal_35mm,
        "principal_point_u": principal_u,
        "principal_point_v": principal_v,
        "calibration_group": 0,
        "distortion_group": 0,
        "xmp_count": len(written),
        "mask_layer_count": 0,
        "xmp_files": [str(item.xmp_path.relative_to(output_dir).as_posix()) for item in written],
    }
    write_realityscan_manifest(output_dir, manifest)
    return manifest


def write_realityscan_manifest(output_dir: Path, manifest: dict) -> None:
    (output_dir / "realityscan_export.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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
        layer_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mask_path, layer_path)
        copied.append(str(layer_path.relative_to(output_dir).as_posix()))

    if manifest is None:
        manifest_path = output_dir / "realityscan_export.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {"export_type": "realityscan_xmp"}
    manifest["mask_layer_count"] = len(copied)
    manifest["mask_layer_files"] = copied
    write_realityscan_manifest(output_dir, manifest)
    return manifest
