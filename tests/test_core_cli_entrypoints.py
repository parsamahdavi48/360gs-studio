from __future__ import annotations

from pathlib import Path
from typing import Any

import core.image_sequence_import_cli as image_sequence_cli
import core.workflow_job_cli as workflow_cli
from core.frame_job_spec import JOB_KIND_IMPORT_IMAGE_SEQUENCE


def test_image_sequence_cli_builds_frame_job_payload(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(payload: dict[str, Any]) -> None:
        captured["payload"] = payload

    monkeypatch.setattr(image_sequence_cli, "run_frame_job_payload", fake_run)

    result = image_sequence_cli.main(
        [
            str(tmp_path / "source"),
            str(tmp_path / "scene"),
            "--prefix",
            "take",
            "--recursive",
        ]
    )

    assert result == 0
    payload = captured["payload"]
    assert payload["kind"] == JOB_KIND_IMPORT_IMAGE_SEQUENCE
    assert payload["source_dir"] == str(tmp_path / "source")
    assert payload["scene_dir"] == str(tmp_path / "scene")
    assert payload["prefix"] == "take"
    assert payload["recursive"] is True


def test_workflow_job_cli_delegates_to_runner(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, str] = {}
    job_path = tmp_path / "job.json"

    def fake_run(path: str | Path) -> None:
        captured["path"] = str(path)

    monkeypatch.setattr(workflow_cli, "run_workflow_job_file", fake_run)

    result = workflow_cli.main(["--job", str(job_path)])

    assert result == 0
    assert captured["path"] == str(job_path)
