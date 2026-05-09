"""Case and placement files for the AprilTag synthetic-injection dev harness."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_ROOT = REPO_ROOT / "_compare" / "apriltag_test" / "cases"


@dataclass(frozen=True)
class AprilTagDevCase:
    name: str
    case_dir: Path
    input_mode: str
    source_transforms: Path
    source_pointcloud: Path | None = None
    source_metashape_xml: Path | None = None
    image_root: Path | None = None
    tag_family: str = "tag36h11"
    tag_id: int = 7
    default_tag_size_m: float = 0.160
    true_scale: float = 0.25

    @property
    def case_json_path(self) -> Path:
        return self.case_dir / "case.json"

    @property
    def input_dir(self) -> Path:
        return self.case_dir / "input"

    @property
    def assets_dir(self) -> Path:
        return self.case_dir / "assets"

    @property
    def placements_dir(self) -> Path:
        return self.case_dir / "placements"

    @property
    def runs_dir(self) -> Path:
        return self.case_dir / "runs"

    @property
    def copied_transforms(self) -> Path:
        return self.input_dir / "transforms.json"

    def transforms_for_processing(self) -> Path:
        if self.input_mode == "copy":
            return self.copied_transforms
        return self.source_transforms


@dataclass(frozen=True)
class AprilTagPlacement:
    name: str
    tag_family: str
    tag_id: int
    tag_image: Path
    tag_size_m: float
    true_scale: float
    tag_center_sfm: tuple[float, float, float]
    tag_normal_sfm: tuple[float, float, float]
    tag_up_sfm: tuple[float, float, float]
    reference_frame: str = ""
    note: str = ""

    def run_dir_name(self) -> str:
        return safe_name(self.name) or "placement"


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned[:80]


def timestamp_case_name() -> str:
    return datetime.now().strftime("case_%Y%m%d_%H%M%S")


def unique_case_dir(case_root: Path, requested_name: str) -> Path:
    case_root.mkdir(parents=True, exist_ok=True)
    base = safe_name(requested_name) or timestamp_case_name()
    candidate = case_root / base
    if not candidate.exists():
        return candidate
    for index in range(2, 1000):
        numbered = case_root / f"{base}_{index:03d}"
        if not numbered.exists():
            return numbered
    raise RuntimeError(f"Could not create a unique case directory under {case_root}")


def _path_or_none(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    return Path(text) if text else None


def _copy_optional_file(source: Path | None, destination: Path) -> Path | None:
    if source is None:
        return None
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _copy_transforms_only(source_transforms: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_transforms, destination)


def _copy_transforms_and_images(source_transforms: Path, destination: Path) -> int:
    data = json.loads(source_transforms.read_text(encoding="utf-8"))
    source_root = source_transforms.parent
    copied = 0
    for frame in data.get("frames", []):
        file_path = frame.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            continue
        source_image = Path(file_path) if Path(file_path).is_absolute() else source_root / file_path
        if not source_image.is_file():
            continue
        if Path(file_path).is_absolute():
            relative_path = Path("images") / source_image.name
            frame["file_path"] = relative_path.as_posix()
        else:
            relative_path = Path(file_path)
        output_image = destination.parent / relative_path
        output_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, output_image)
        copied += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return copied


def create_case(
    *,
    case_root: Path = DEFAULT_CASE_ROOT,
    case_name: str = "",
    source_transforms: Path,
    source_pointcloud: Path | None = None,
    source_metashape_xml: Path | None = None,
    copy_images: bool = False,
    tag_family: str = "tag36h11",
    tag_id: int = 7,
    default_tag_size_m: float = 0.160,
    true_scale: float = 0.25,
) -> AprilTagDevCase:
    source_transforms = source_transforms.resolve()
    if not source_transforms.is_file():
        raise FileNotFoundError(f"transforms.json not found: {source_transforms}")
    source_pointcloud = _path_or_none(source_pointcloud)
    source_metashape_xml = _path_or_none(source_metashape_xml)

    case_dir = unique_case_dir(case_root, case_name)
    input_dir = case_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    if copy_images:
        _copy_transforms_and_images(source_transforms, input_dir / "transforms.json")
        input_mode = "copy"
    else:
        _copy_transforms_only(source_transforms, input_dir / "transforms.json")
        input_mode = "reference"
    _copy_optional_file(source_pointcloud, input_dir / "pointcloud.ply")
    _copy_optional_file(source_metashape_xml, input_dir / "metashape.xml")

    case = AprilTagDevCase(
        name=case_dir.name,
        case_dir=case_dir,
        input_mode=input_mode,
        source_transforms=source_transforms,
        source_pointcloud=source_pointcloud.resolve() if source_pointcloud else None,
        source_metashape_xml=source_metashape_xml.resolve() if source_metashape_xml else None,
        image_root=source_transforms.parent,
        tag_family=tag_family,
        tag_id=int(tag_id),
        default_tag_size_m=float(default_tag_size_m),
        true_scale=float(true_scale),
    )
    save_case(case)
    case.assets_dir.mkdir(parents=True, exist_ok=True)
    case.placements_dir.mkdir(parents=True, exist_ok=True)
    case.runs_dir.mkdir(parents=True, exist_ok=True)
    return case


def _path_text(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def save_case(case: AprilTagDevCase) -> None:
    data = {
        "schema_version": 1,
        "name": case.name,
        "input_mode": case.input_mode,
        "source_transforms": str(case.source_transforms),
        "source_pointcloud": _path_text(case.source_pointcloud),
        "source_metashape_xml": _path_text(case.source_metashape_xml),
        "image_root": _path_text(case.image_root),
        "tag_family": case.tag_family,
        "tag_id": case.tag_id,
        "default_tag_size_m": case.default_tag_size_m,
        "true_scale": case.true_scale,
    }
    case.case_dir.mkdir(parents=True, exist_ok=True)
    case.case_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_case(case_dir: Path) -> AprilTagDevCase:
    case_dir = case_dir.resolve()
    data = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    return AprilTagDevCase(
        name=str(data.get("name") or case_dir.name),
        case_dir=case_dir,
        input_mode=str(data.get("input_mode") or "reference"),
        source_transforms=Path(data["source_transforms"]),
        source_pointcloud=_path_or_none(data.get("source_pointcloud")),
        source_metashape_xml=_path_or_none(data.get("source_metashape_xml")),
        image_root=_path_or_none(data.get("image_root")),
        tag_family=str(data.get("tag_family") or "tag36h11"),
        tag_id=int(data.get("tag_id", 7)),
        default_tag_size_m=float(data.get("default_tag_size_m", 0.160)),
        true_scale=float(data.get("true_scale", 0.25)),
    )


def _vec3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def save_placement(case: AprilTagDevCase, placement: AprilTagPlacement) -> Path:
    case.placements_dir.mkdir(parents=True, exist_ok=True)
    path = case.placements_dir / f"{placement.run_dir_name()}.json"
    data = {
        "schema_version": 1,
        "name": placement.name,
        "tag_family": placement.tag_family,
        "tag_id": placement.tag_id,
        "tag_image": str(placement.tag_image),
        "tag_size_m": placement.tag_size_m,
        "true_scale": placement.true_scale,
        "tag_center_sfm": list(placement.tag_center_sfm),
        "tag_normal_sfm": list(placement.tag_normal_sfm),
        "tag_up_sfm": list(placement.tag_up_sfm),
        "reference_frame": placement.reference_frame,
        "note": placement.note,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_placement(path: Path) -> AprilTagPlacement:
    data = json.loads(path.read_text(encoding="utf-8"))
    return AprilTagPlacement(
        name=str(data.get("name") or path.stem),
        tag_family=str(data.get("tag_family") or "tag36h11"),
        tag_id=int(data.get("tag_id", 7)),
        tag_image=Path(data["tag_image"]),
        tag_size_m=float(data["tag_size_m"]),
        true_scale=float(data["true_scale"]),
        tag_center_sfm=_vec3(data["tag_center_sfm"], "tag_center_sfm"),
        tag_normal_sfm=_vec3(data["tag_normal_sfm"], "tag_normal_sfm"),
        tag_up_sfm=_vec3(data["tag_up_sfm"], "tag_up_sfm"),
        reference_frame=str(data.get("reference_frame") or ""),
        note=str(data.get("note") or ""),
    )


def run_dir_for_placement(case: AprilTagDevCase, placement: AprilTagPlacement) -> Path:
    return case.runs_dir / placement.run_dir_name()

