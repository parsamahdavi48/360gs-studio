from __future__ import annotations

import sys
from pathlib import Path


def build_workflow_job_cmd(base_dir: Path, job_path: Path, *, python_executable: str | None = None) -> list[str]:
    script = base_dir / "scripts" / "run_workflow_job.py"
    if not script.exists():
        raise FileNotFoundError(f"run_workflow_job.py が見つかりません: {script}")
    return [
        python_executable or sys.executable,
        "-u",
        str(script),
        "--job",
        str(job_path),
    ]
