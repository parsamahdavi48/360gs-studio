from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.realityscan_dataset_plan import (
    ACTION_LINK_OR_COPY_PINHOLE,
    ACTION_UNDISTORT_TO_PINHOLE,
    build_realityscan_lfs_dataset_plan,
)
from core.realityscan_to_transforms import RealityScanCameraRow


def _row(name: str, *, k1: float = 0.0) -> RealityScanCameraRow:
    return RealityScanCameraRow(
        name=name,
        x=0.0,
        y=0.0,
        z=0.0,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        f_35mm=35.0,
        px_norm=0.5,
        py_norm=0.5,
        k1=k1,
        k2=0.0,
        k3=0.0,
        k4=0.0,
        t1=0.0,
        t2=0.0,
    )


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (1, 2, 3)).save(path)


def test_realityscan_dataset_plan_classifies_distorted_rows(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    _image(images / "plain.jpg")
    _image(images / "distorted.jpg")
    _image(masks / "plain.png")

    plan = build_realityscan_lfs_dataset_plan(
        [_row("plain.jpg"), _row("distorted.jpg", k1=0.1)],
        images,
        masks,
        pre_undistort_distorted_images=True,
        skip_missing_images=False,
    )

    assert plan.issues == ()
    assert plan.action_counts == {ACTION_LINK_OR_COPY_PINHOLE: 1, ACTION_UNDISTORT_TO_PINHOLE: 1}
    assert plan.items[0].mask_path == masks / "plain.png"
