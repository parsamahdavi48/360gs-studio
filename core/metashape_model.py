from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.safe_xml import parse_xml_file

CAMERA_MODEL_EQUIRECTANGULAR = "EQUIRECTANGULAR"
CAMERA_MODEL_PINHOLE = "PINHOLE"
CAMERA_MODEL_OPENCV = "OPENCV"
CAMERA_MODEL_OPENCV_FISHEYE = "OPENCV_FISHEYE"


@dataclass(frozen=True, slots=True)
class MetashapeSensor:
    sensor_id: str
    label: str
    sensor_type: str
    camera_model: str
    width: int
    height: int
    params: dict[str, float]

    @property
    def has_distortion(self) -> bool:
        return any(abs(float(self.params.get(name, 0.0))) > 1e-12 for name in ("k1", "k2", "k3", "k4", "p1", "p2"))


@dataclass(frozen=True, slots=True)
class MetashapeCamera:
    camera_id: str
    label: str
    sensor_id: str
    component_id: str
    transform: np.ndarray


@dataclass(frozen=True, slots=True)
class MetashapeModel:
    sensors: dict[str, MetashapeSensor]
    cameras: tuple[MetashapeCamera, ...]
    components: dict[str, tuple[np.ndarray, float]]

    def sensor_for_camera(self, camera: MetashapeCamera) -> MetashapeSensor:
        return self.sensors[camera.sensor_id]


def parse_metashape_model(xml_path: str | Path) -> MetashapeModel:
    root = parse_xml_file(xml_path).getroot()
    chunk = _first_child(root, "chunk")
    if chunk is None and len(root):
        chunk = root[0]
    if chunk is None:
        raise ValueError("Metashape XML does not contain a chunk.")

    sensors = _parse_sensors(chunk)
    if not sensors:
        raise ValueError("Metashape XML does not contain calibrated sensors.")
    components = _parse_components(chunk)
    cameras = _parse_cameras(chunk)
    if not cameras:
        raise ValueError("Metashape XML does not contain cameras.")
    missing_sensors = sorted({camera.sensor_id for camera in cameras if camera.sensor_id not in sensors})
    if missing_sensors:
        raise ValueError(f"Metashape cameras reference missing sensors: {', '.join(missing_sensors)}")
    return MetashapeModel(sensors=sensors, cameras=tuple(cameras), components=components)


def _parse_sensors(chunk: ET.Element) -> dict[str, MetashapeSensor]:
    sensors_xml = chunk.find("sensors")
    if sensors_xml is None:
        return {}
    sensors: dict[str, MetashapeSensor] = {}
    for sensor in sensors_xml.iter("sensor"):
        sensor_id = str(sensor.get("id") or "")
        if not sensor_id:
            continue
        sensor_type = str(sensor.get("type") or "frame")
        resolution = sensor.find("resolution")
        if resolution is None:
            continue
        width = int(resolution.get("width") or 0)
        height = int(resolution.get("height") or 0)
        if width <= 0 or height <= 0:
            continue
        params = _parse_calibration(sensor.find("calibration"), sensor_type=sensor_type, width=width, height=height)
        sensors[sensor_id] = MetashapeSensor(
            sensor_id=sensor_id,
            label=str(sensor.get("label") or ""),
            sensor_type=sensor_type,
            camera_model=_camera_model_for_sensor(sensor_type, params),
            width=width,
            height=height,
            params=params,
        )
    return sensors


def _parse_calibration(calib: ET.Element | None, *, sensor_type: str, width: int, height: int) -> dict[str, float]:
    if calib is None:
        if sensor_type == "spherical":
            return {
                "fl_x": width / 2.0,
                "fl_y": height,
                "cx": width / 2.0,
                "cy": height / 2.0,
            }
        return {}
    f = _find_float(calib, "f", 0.0)
    params = {
        "fl_x": f,
        "fl_y": f,
        "cx": _find_float(calib, "cx", 0.0) + width / 2.0,
        "cy": _find_float(calib, "cy", 0.0) + height / 2.0,
    }
    for name in ("k1", "k2", "k3", "k4", "p1", "p2", "b1", "b2"):
        params[name] = _find_float(calib, name, 0.0)
    return params


def _parse_components(chunk: ET.Element) -> dict[str, tuple[np.ndarray, float]]:
    components_xml = chunk.find("components")
    if components_xml is None:
        return {}
    components: dict[str, tuple[np.ndarray, float]] = {}
    for component in components_xml.iter("component"):
        component_id = str(component.get("id") or "")
        if not component_id:
            continue
        transform_xml = component.find("transform")
        matrix = np.eye(4)
        scale = 1.0
        if transform_xml is not None:
            rotation = _float_array(transform_xml.findtext("rotation"), 9)
            translation = _float_array(transform_xml.findtext("translation"), 3)
            if rotation is not None:
                matrix[:3, :3] = np.array(rotation, dtype=float).reshape((3, 3))
            if translation is not None:
                matrix[:3, 3] = np.array(translation, dtype=float)
            scale = _text_float(transform_xml.findtext("scale"), 1.0)
        components[component_id] = (matrix, scale)
    return components


def _parse_cameras(chunk: ET.Element) -> list[MetashapeCamera]:
    cameras_xml = chunk.find("cameras")
    if cameras_xml is None:
        return []
    cameras: list[MetashapeCamera] = []
    for camera in cameras_xml.iter("camera"):
        transform_text = camera.findtext("transform")
        values = _float_array(transform_text, 16)
        if values is None:
            continue
        cameras.append(
            MetashapeCamera(
                camera_id=str(camera.get("id") or ""),
                label=str(camera.get("label") or ""),
                sensor_id=str(camera.get("sensor_id") or ""),
                component_id=str(camera.get("component_id") or ""),
                transform=np.array(values, dtype=float).reshape((4, 4)),
            )
        )
    return cameras


def _camera_model_for_sensor(sensor_type: str, params: dict[str, float]) -> str:
    if sensor_type == "spherical":
        return CAMERA_MODEL_EQUIRECTANGULAR
    if sensor_type == "fisheye":
        return CAMERA_MODEL_OPENCV_FISHEYE
    if sensor_type == "frame" and any(abs(params.get(name, 0.0)) > 1e-12 for name in ("k1", "k2", "k3", "p1", "p2")):
        return CAMERA_MODEL_OPENCV
    if sensor_type == "frame":
        return CAMERA_MODEL_PINHOLE
    return sensor_type.upper() or "UNKNOWN"


def _first_child(root: ET.Element, tag: str) -> ET.Element | None:
    found = root.find(tag)
    if found is not None:
        return found
    for child in root:
        if child.tag == tag:
            return child
    return None


def _find_float(parent: ET.Element, name: str, default: float) -> float:
    return _text_float(parent.findtext(name), default)


def _text_float(text: str | None, default: float) -> float:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return default


def _float_array(text: str | None, expected_count: int) -> list[float] | None:
    if not text:
        return None
    parts = str(text).split()
    if len(parts) != expected_count:
        return None
    try:
        return [float(part) for part in parts]
    except ValueError:
        return None
