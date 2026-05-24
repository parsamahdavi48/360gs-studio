from __future__ import annotations

from pathlib import Path

import pytest

from core.mask_job_spec import (
    BACKEND_MASK2FORMER,
    BACKEND_SAM31,
    custom_mask_job,
    init_masks_job,
    mask_job_to_command,
    overexposure_mask_job,
    sky_mask_job,
    stitch_mask_job,
    validate_mask_job_payload,
    yolo_sam_mask_job,
)


def test_yolo_mask_job_builds_command(tmp_path: Path) -> None:
    payload = yolo_sam_mask_job(
        images=tmp_path / "images",
        masks=tmp_path / "masks",
        quality="high",
        expand=5,
        projection="equirect",
        classes=(0, 2, 3),
        extra_args=("--bottom-filter",),
        image_list=tmp_path / "targets.jsonl",
    )

    assert mask_job_to_command("python.exe", payload) == [
        "python.exe",
        "-u",
        "-m",
        "core.yolo_mask",
        str(tmp_path / "images"),
        str(tmp_path / "masks"),
        "--quality",
        "high",
        "--expand",
        "5",
        "--projection",
        "equirect",
        "--classes",
        "0,2,3",
        "--image-list",
        str(tmp_path / "targets.jsonl"),
        "--bottom-filter",
    ]


def test_sam31_mask_job_builds_safe_batch_command(tmp_path: Path) -> None:
    payload = sky_mask_job(
        images=tmp_path / "images",
        masks=tmp_path / "masks",
        backend=BACKEND_SAM31,
        projection="normal",
        quality="standard",
        inference_size=1008,
        expand=3,
        min_score=0.5,
        min_area_ratio=0.01,
        top_connected=True,
        sam_prompts=("person", "tripod"),
        sam_subtract_prompts=("logo",),
        merge_mode="replace",
        replace=True,
        safe_batch=True,
    )

    cmd = mask_job_to_command("python.exe", payload)

    assert cmd[:6] == ["python.exe", "-u", "-m", "core.sky_mask", str(tmp_path / "images"), str(tmp_path / "masks")]
    assert cmd[cmd.index("--backend") + 1] == "sam31"
    assert cmd[cmd.index("--merge-mode") + 1] == "replace"
    assert [cmd[index + 1] for index, item in enumerate(cmd) if item == "--sam-prompt"] == ["person", "tripod"]
    assert [cmd[index + 1] for index, item in enumerate(cmd) if item == "--subtract-sam-prompt"] == ["logo"]
    assert "--replace" in cmd
    assert "--safe-batch" in cmd


def test_mask2former_job_requires_labels(tmp_path: Path) -> None:
    payload = sky_mask_job(
        images=tmp_path / "images",
        masks=tmp_path / "masks",
        backend=BACKEND_MASK2FORMER,
        projection="equirect",
        quality="high",
        inference_size=768,
        expand=0,
        min_score=0.0,
        min_area_ratio=0.0,
        top_connected=False,
        labels=(),
    )

    with pytest.raises(ValueError, match="labels"):
        validate_mask_job_payload(payload)


def test_mask_helper_jobs_build_commands(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"

    assert mask_job_to_command("python.exe", init_masks_job(images=images, masks=masks)) == [
        "python.exe",
        "-u",
        "-m",
        "core.init_masks",
        str(images),
        str(masks),
    ]
    assert mask_job_to_command("python.exe", stitch_mask_job(masks=masks, boundary_width=3.5, workers=2)) == [
        "python.exe",
        "-u",
        "-m",
        "core.stitch_mask",
        str(masks),
        str(masks),
        "--boundary-width",
        "3.5",
        "--workers",
        "2",
    ]
    assert "--replace" in mask_job_to_command(
        "python.exe",
        overexposure_mask_job(images=images, masks=masks, threshold=250, dilate=2, workers=4, replace=True),
    )
    assert mask_job_to_command(
        "python.exe",
        custom_mask_job(images=images, masks=masks, custom_mask=tmp_path / "custom.png"),
    )[:7] == ["python.exe", "-u", "-m", "core.custom_mask", str(images), str(masks), str(tmp_path / "custom.png")]


def test_mask_job_rejects_invalid_ranges(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="threshold"):
        validate_mask_job_payload(
            overexposure_mask_job(
                images=tmp_path / "images",
                masks=tmp_path / "masks",
                threshold=255,
                dilate=0,
                workers=1,
            )
        )
    with pytest.raises(ValueError, match="boundary_width"):
        validate_mask_job_payload(stitch_mask_job(masks=tmp_path / "masks", boundary_width=180.0, workers=1))
    with pytest.raises(ValueError, match="classes"):
        validate_mask_job_payload(
            yolo_sam_mask_job(
                images=tmp_path / "images",
                masks=tmp_path / "masks",
                quality="high",
                expand=0,
                projection="equirect",
                classes=(-1,),
            )
        )
