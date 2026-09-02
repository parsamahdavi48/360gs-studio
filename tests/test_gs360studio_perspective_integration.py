from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from core.cancellation import AppJobCancelled
from gs360studio.domain.models import JobSpec, ViewSpec
from gs360studio.engine.perspective_export import ExportRequest, export_image_views
from gs360studio.engine.perspective_job import run_perspective_job_payload
from gs360studio.platform.job_store import list_jobs, save_job
from gui.steps.perspective_export import PerspectiveExportStep


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _write_panorama(path: Path, *, width: int = 320, height: int = 160) -> None:
    longitude = np.broadcast_to(np.arange(width, dtype=np.uint16), (height, width))
    image = np.dstack(
        (
            (longitude % 256).astype(np.uint8),
            np.full_like(longitude, 80, dtype=np.uint8),
            np.full_like(longitude, 160, dtype=np.uint8),
        )
    )
    assert cv2.imwrite(str(path), image)


def test_project_frames_flow_into_preview_and_persisted_export(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_panorama(images / "frame_00001.png")

    step = PerspectiveExportStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step.input_browse.text() == str(images)
    assert step.output_browse.text() == str(tmp_path / "output" / "perspective")
    assert step._source_frame_count == 1
    assert step.preview_label.pixmap() is not None

    for row in range(step.table.rowCount()):
        step.table.item(row, 7).setText("48")
        step.table.item(row, 8).setText("48")
        step.table.item(row, 9).setText("lanczos")
    queue = step.build_commands()
    payload = queue[0][1].payload
    assert all(view["interpolation"] == "lanczos" for view in payload["views"])

    run_perspective_job_payload(payload)

    output = tmp_path / "output" / "perspective"
    assert (output / "export_manifest.json").is_file()
    assert len(list(output.glob("*/frame_00001.png"))) == 6
    jobs = list_jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].status == "completed"
    assert jobs[0].progress_current == jobs[0].progress_total == 6
    assert jobs[0].input_signatures["source"]


def test_interrupted_export_configuration_can_be_restored_and_linked(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    source = images / "frame.png"
    _write_panorama(source)
    output = tmp_path / "output" / "restored"
    view = ViewSpec(id="restored", name="Restored", yaw_deg=45, width=40, height=32, interpolation="linear")
    interrupted = JobSpec(
        job_type="perspective-export",
        status="running",
        configuration={
            "kind": "perspective_export",
            "input_path": str(source),
            "output_dir": str(output),
            "output_format": "png",
            "frame_interval_sec": 1.0,
            "batch_size": 1,
            "views": [view.to_dict()],
        },
    )
    save_job(tmp_path, interrupted)

    step = PerspectiveExportStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    assert not step.restore_job_button.isHidden()

    step._restore_interrupted_job()
    assert step.input_browse.text() == str(source)
    assert step.output_browse.text() == str(output)
    assert step._views == [view]
    payload = step.build_commands()[0][1].payload
    assert payload["resume_job_id"] == interrupted.job_id

    run_perspective_job_payload(payload)
    jobs = list_jobs(tmp_path)
    restored = next(job for job in jobs if job.status == "completed")
    original = next(job for job in jobs if job.job_id == interrupted.job_id)
    assert original.status == "interrupted"
    assert restored.dependency_ids == [interrupted.job_id]

    step.on_queue_finished(True)
    assert step._resume_job_id == ""
    assert step.restore_job_button.isHidden()


def test_canceled_export_persists_last_known_progress(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.png"
    _write_panorama(source)
    payload = {
        "kind": "perspective_export",
        "scene_dir": str(tmp_path),
        "input_path": str(source),
        "output_dir": str(tmp_path / "out"),
        "output_format": "png",
        "views": [ViewSpec(id="front", name="Front", width=32, height=32).to_dict()],
    }

    def canceled_export(_files, _request, *, progress, canceled) -> None:
        assert not canceled()
        progress(3, 10, "Partial progress")
        raise InterruptedError("perspective export canceled")

    monkeypatch.setattr("gs360studio.engine.perspective_job.export_image_views", canceled_export)

    with pytest.raises(AppJobCancelled):
        run_perspective_job_payload(payload)

    jobs = list_jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].status == "canceled"
    assert jobs[0].progress_current == 3
    assert jobs[0].progress_total == 10


def test_image_export_cancellation_removes_partial_staging_output(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _write_panorama(source)
    target = tmp_path / "out"
    request = ExportRequest(
        input_path=source,
        output_dir=target,
        output_format="png",
        views=(
            ViewSpec(id="front", name="Front", width=64, height=64),
            ViewSpec(id="right", name="Right", yaw_deg=90, width=64, height=64),
        ),
    )
    progress_count = 0

    def progress(_current: int, _total: int, _message: str) -> None:
        nonlocal progress_count
        progress_count += 1

    with pytest.raises(InterruptedError):
        export_image_views([source], request, progress=progress, canceled=lambda: progress_count >= 1)

    assert not target.exists()
    assert list(tmp_path.glob(".out.staging-*")) == []
