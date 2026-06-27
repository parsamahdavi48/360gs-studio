# ruff: noqa: I001
import json
import math
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QMessageBox
import core.orientation_correction as orientation_correction
import gui.steps.step4_cubemap as step4_cubemap
from core.app_job import AppJob
from core.normal_camera_metadata import save_normal_camera_default
from core.scene_layout import (
    project_path,
    source_image_sets_path,
    step4_export_settings_path,
    step4_meta_dir,
    step4_training_runs_path,
    step4_views_config_path,
)
from core.transforms_to_colmap import read_ply_points
from core.workflow_artifacts import register_dataset_artifact
from gui import i18n
from gui.common.collapsible_section import CollapsibleSection
from gui.common.perspective_preview import PREVIEW_PROJECTION_EQUIRECT, PREVIEW_PROJECTION_PERSPECTIVE
from gui.steps.sfm_route_backends import get_sfm_route_backend
from gui.steps.sfm_route_specs import (
    OUTPUT_SHAPE_EQUIRECT_3DGUT,
    OUTPUT_SHAPE_PROJECTED,
    SFM_ROUTE_COLMAP,
    SFM_ROUTE_IDS,
    SFM_ROUTE_METASHAPE,
    SFM_ROUTE_SPHERESFM,
    get_sfm_route_spec,
    normalize_sfm_route,
)
from gui.steps.step4_cubemap import CubemapStep
from gui.steps.step4_settings import STEP4_SETTINGS_VERSION
from gui.steps.step5_training import TrainingStep
from gui.steps.training_backends import lichtfeld_defaults
from tests.helpers.gui import qt_app

_IDENTITY_MATRIX_TEXT = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"


def _app():
    return qt_app()


def _workflow_job(cmd: object) -> dict:
    if isinstance(cmd, AppJob):
        return cmd.payload
    raise AssertionError(f"Expected workflow AppJob, got {cmd!r}")


def _ready_step(scene: Path, *, metashape_inputs: bool = False) -> CubemapStep:
    _app()
    scene.mkdir(exist_ok=True)
    _write_ascii_ply(scene / "pointcloud.ply", [(0.0, 0.0, 0.0)])
    if metashape_inputs:
        (scene / "images").mkdir(exist_ok=True)
        _write_test_image(scene / "images" / "frame_0001.jpg", size=(64, 32))
        _write_metashape_xml(scene / "metashape.xml")
        _write_ascii_ply(scene / "metashape.ply", [(1.0, 2.0, 3.0)])
    _app()
    _app()
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(scene))
    if metashape_inputs:
        step._approve_metashape_ply()
    return step


def _write_metashape_xml(path: Path, labels: list[str] | None = None) -> None:
    labels = labels or ["frame_0001.jpg"]
    cameras = "\n".join(
        f'        <camera id="{idx}" sensor_id="0" label="{label}">\n'
        "          <transform>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</transform>\n"
        "        </camera>"
        for idx, label in enumerate(labels)
    )
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<document version="1.2.0">\n'
        "  <chunk>\n"
        "    <sensors>\n"
        '      <sensor id="0" label="camera" type="spherical">\n'
        '        <resolution width="64" height="32" />\n'
        "      </sensor>\n"
        "    </sensors>\n"
        "    <cameras>\n"
        f"{cameras}\n"
        "    </cameras>\n"
        "  </chunk>\n"
        "</document>\n",
        encoding="utf-8",
    )


def _write_mixed_metashape_xml(path: Path) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<document>\n"
        "  <chunk>\n"
        "    <sensors>\n"
        '      <sensor id="0" type="spherical">\n'
        '        <resolution width="64" height="32" />\n'
        "      </sensor>\n"
        '      <sensor id="1" type="frame">\n'
        '        <resolution width="40" height="30" />\n'
        "        <calibration><f>35</f><cx>0</cx><cy>0</cy></calibration>\n"
        "      </sensor>\n"
        "    </sensors>\n"
        "    <cameras>\n"
        f'      <camera id="0" label="pano.jpg" sensor_id="0"><transform>{_IDENTITY_MATRIX_TEXT}</transform></camera>\n'
        f'      <camera id="1" label="frame.jpg" sensor_id="1"><transform>{_IDENTITY_MATRIX_TEXT}</transform></camera>\n'
        "    </cameras>\n"
        "  </chunk>\n"
        "</document>\n",
        encoding="utf-8",
    )


def _write_ascii_ply(path: Path, points: list[tuple[float, float, float]]) -> None:
    rows = "\n".join(f"{x:g} {y:g} {z:g}" for x, y, z in points)
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        f"{rows}\n",
        encoding="ascii",
    )


def _write_test_image(path: Path, size: tuple[int, int] = (64, 32)) -> None:
    Image.new("RGB", size, (0, 0, 0)).save(path)


def _write_output_dataset(
    scene: Path,
    *,
    output_shape: str,
    pointcloud: bool = True,
    legacy_root: bool = False,
) -> Path:
    if legacy_root:
        output = scene / "output"
    else:
        name = "metashape_3dgut" if output_shape == "equirect_3dgut" else "metashape_cubemap"
        output = scene / "output" / name
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    _write_test_image(images / "frame_0001.jpg")
    camera_model = "EQUIRECTANGULAR" if output_shape == "equirect_3dgut" else "SIMPLE_PINHOLE"
    (output / "transforms.json").write_text(json.dumps({"camera_model": camera_model, "frames": []}), encoding="utf-8")
    if pointcloud:
        _write_ascii_ply(output / "pointcloud.ply", [(0.0, 0.0, 0.0)])
    settings_path = step4_export_settings_path(scene)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            {
                "settings_version": STEP4_SETTINGS_VERSION,
                "output_shape": output_shape,
                "output_dir": str(output),
                "portable_output": {
                    "root": output.relative_to(scene).as_posix(),
                    "dataset_kind": "3dgut" if output_shape == "equirect_3dgut" else "projection_views",
                    "active": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return output


def _write_spheresfm_sparse_stub(scene: Path) -> Path:
    sparse_model = scene / "output" / "colmap_equirect" / "sparse" / "0"
    sparse_model.mkdir(parents=True, exist_ok=True)
    (sparse_model / "cameras.txt").write_text("# cameras\n", encoding="ascii")
    (sparse_model / "images.txt").write_text("# images\n", encoding="ascii")
    (sparse_model / "points3D.txt").write_text("# points\n", encoding="ascii")
    return sparse_model


def _is_descendant(widget, ancestor) -> bool:
    current = widget
    while current is not None:
        if current is ancestor:
            return True
        current = current.parentWidget()
    return False


def _ready_lichtfeld_training_step(scene: Path) -> CubemapStep:
    step = _ready_step(scene, metashape_inputs=True)
    _write_output_dataset(scene, output_shape="projected")
    fake_lfs = scene / "LichtFeld-Studio.exe"
    fake_lfs.write_text("", encoding="utf-8")
    step.run_training_cb.setChecked(True)
    step.training_executable_browse.set_text(str(fake_lfs))
    step.lfs_auto_steps_scaler_cb.setChecked(False)
    return step


__all__ = [
    "AppJob",
    "CollapsibleSection",
    "CubemapStep",
    "Image",
    "OUTPUT_SHAPE_EQUIRECT_3DGUT",
    "OUTPUT_SHAPE_PROJECTED",
    "PREVIEW_PROJECTION_EQUIRECT",
    "PREVIEW_PROJECTION_PERSPECTIVE",
    "Path",
    "QMessageBox",
    "QPoint",
    "SFM_ROUTE_COLMAP",
    "SFM_ROUTE_IDS",
    "SFM_ROUTE_METASHAPE",
    "SFM_ROUTE_SPHERESFM",
    "STEP4_SETTINGS_VERSION",
    "TrainingStep",
    "_IDENTITY_MATRIX_TEXT",
    "_app",
    "_is_descendant",
    "_ready_lichtfeld_training_step",
    "_ready_step",
    "_workflow_job",
    "_write_ascii_ply",
    "_write_metashape_xml",
    "_write_mixed_metashape_xml",
    "_write_output_dataset",
    "_write_spheresfm_sparse_stub",
    "_write_test_image",
    "get_sfm_route_backend",
    "get_sfm_route_spec",
    "i18n",
    "json",
    "lichtfeld_defaults",
    "math",
    "normalize_sfm_route",
    "np",
    "orientation_correction",
    "os",
    "project_path",
    "pytest",
    "read_ply_points",
    "register_dataset_artifact",
    "save_normal_camera_default",
    "source_image_sets_path",
    "step4_cubemap",
    "step4_export_settings_path",
    "step4_meta_dir",
    "step4_training_runs_path",
    "step4_views_config_path",
]
