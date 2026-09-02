from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from gs360studio.domain.models import ViewSpec, cubemap_view_specs
from gs360studio.engine.perspective_export import (
    ExportRequest,
    build_ffmpeg_batch_command,
    estimate_batch_size,
    export_image_views,
    export_video_views,
)
from gs360studio.engine.projection import ProjectionMapCache, build_projection_map, project_equirectangular


def _longitude_panorama(width: int = 720, height: int = 360) -> np.ndarray:
    x = np.arange(width, dtype=np.uint16)
    longitude = np.broadcast_to(x, (height, width))
    return np.dstack(((longitude % 256).astype(np.uint8), np.zeros_like(longitude, dtype=np.uint8), np.zeros_like(longitude, dtype=np.uint8)))


def test_projection_center_tracks_yaw_and_cache_is_bounded() -> None:
    source = _longitude_panorama()
    cache = ProjectionMapCache(2)
    front = ViewSpec(id="front", name="Front", yaw_deg=0, width=101, height=101)
    right = ViewSpec(id="right", name="Right", yaw_deg=90, width=101, height=101)
    back = ViewSpec(id="back", name="Back", yaw_deg=179, width=101, height=101)

    front_image = project_equirectangular(source, front, cache=cache)
    right_image = project_equirectangular(source, right, cache=cache)
    project_equirectangular(source, back, cache=cache)

    assert int(front_image[50, 50, 0]) == pytest.approx(360 % 256, abs=2)
    assert int(right_image[50, 50, 0]) == pytest.approx(540 % 256, abs=2)
    assert len(cache) == 2


def test_projection_maps_are_finite_and_inside_equirectangular_domain() -> None:
    view = ViewSpec(id="pole", name="Pole", pitch_deg=-90, roll_deg=37, hfov_deg=120, width=64, height=48)
    map_x, map_y = build_projection_map((400, 200), view)
    assert np.isfinite(map_x).all() and np.isfinite(map_y).all()
    assert map_x.min() >= 0 and map_x.max() <= 400
    assert map_y.min() >= 0 and map_y.max() <= 200


def test_ffmpeg_batch_uses_one_input_decode_and_one_filter_per_view(tmp_path: Path) -> None:
    views = tuple(cubemap_view_specs(256)[:4])
    request = ExportRequest(input_path=tmp_path / "input.mp4", output_dir=tmp_path / "out", views=views)
    command = build_ffmpeg_batch_command(request, views, tmp_path / "stage")
    graph = command[command.index("-filter_complex") + 1]
    assert command.count("-i") == 1
    assert "split=4" in graph
    assert graph.count("v360=") == 4
    assert command.count("-map") == 4
    assert estimate_batch_size(views, available_memory_mb=512) <= 4


def test_still_export_reads_frame_once_and_reuses_completed_artifact(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "pano.png"
    cv2.imwrite(str(source), _longitude_panorama(160, 80))
    views = (ViewSpec(id="a", name="A", width=32, height=32), ViewSpec(id="b", name="B", yaw_deg=90, width=32, height=32))
    request = ExportRequest(input_path=source, output_dir=tmp_path / "out", views=views, output_format="png")
    calls = 0
    original = cv2.imdecode

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr("core.image_io.cv2.imdecode", counted)
    export_image_views([source], request)
    export_image_views([source], request)

    assert calls == 1
    assert (tmp_path / "out" / "a" / "frame_00001.png").is_file()
    assert (tmp_path / "out" / "b" / "frame_00001.png").is_file()


def test_real_ffmpeg_batch_exports_two_views_from_one_decode(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg is not installed")
    source = tmp_path / "pano.mp4"
    created = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x160:rate=4:duration=1",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        capture_output=True,
        check=False,
    )
    assert created.returncode == 0, created.stderr.decode(errors="replace")
    request = ExportRequest(
        input_path=source,
        output_dir=tmp_path / "video_out",
        views=(
            ViewSpec(id="front", name="Front", width=64, height=64),
            ViewSpec(id="right", name="Right", yaw_deg=90, width=64, height=64),
        ),
        output_format="png",
        frame_interval_sec=0.5,
        ffmpeg_path=ffmpeg,
    )

    export_video_views(request)

    assert len(list((request.output_dir / "front").glob("*.png"))) == 2
    assert len(list((request.output_dir / "right").glob("*.png"))) == 2
