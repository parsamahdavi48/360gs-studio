"""Compare shared-decode batches with the historical process-per-view design."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from gs360studio.domain.models import grid_view_specs
from gs360studio.engine.perspective_export import ExportRequest, build_ffmpeg_batch_command


def _run(command: list[str]) -> float:
    started = time.perf_counter()
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace")[-2000:])
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--size", type=int, default=960)
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("FFmpeg was not found")
    views = tuple(grid_view_specs(yaw_count=args.views, size=args.size))
    with tempfile.TemporaryDirectory(prefix="360gs-benchmark-") as temp_text:
        root = Path(temp_text)
        baseline = 0.0
        for index, view in enumerate(views):
            request = ExportRequest(args.input, root / "baseline", (view,), output_format="video", overwrite=True, ffmpeg_path=ffmpeg)
            stage = root / f"single-{index}"
            stage.mkdir()
            baseline += _run(build_ffmpeg_batch_command(request, (view,), stage))
        batched_request = ExportRequest(args.input, root / "batched", views, output_format="video", overwrite=True, ffmpeg_path=ffmpeg)
        stage = root / "batch"
        stage.mkdir()
        batched = _run(build_ffmpeg_batch_command(batched_request, views, stage))
    improvement = (baseline - batched) / baseline * 100.0
    print(f"views={len(views)} baseline={baseline:.3f}s batched={batched:.3f}s improvement={improvement:.1f}%")
    return 0 if improvement >= 25.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
