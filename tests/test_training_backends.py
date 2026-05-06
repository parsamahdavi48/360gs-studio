import json
from pathlib import Path

import pytest

from gui.steps.training_backends import (
    CustomTrainingOptions,
    LichtFeldTrainingOptions,
    PostshotTrainingOptions,
    TrainingDataset,
    build_custom_training_cmd,
    build_lichtfeld_config,
    build_lichtfeld_training_cmd,
    build_postshot_training_cmd,
    lichtfeld_auto_steps_scaler,
)


def test_lichtfeld_config_overrides_visible_training_parameters(tmp_path: Path) -> None:
    dataset = TrainingDataset(dataset_root=tmp_path / "output")
    options = LichtFeldTrainingOptions(
        executable="LichtFeld-Studio.exe",
        dataset=dataset,
        output_dir=tmp_path / "training",
        config_path=tmp_path / "config.json",
        strategy="mrnf",
        iterations=46800,
        max_gaussians=5_000_000,
        sh_degree=2,
        tile_mode=4,
        steps_scaler=1.56,
        bilateral_grid=True,
        mask_mode="segment",
        sparsity=True,
        gut=True,
        undistort=True,
        mip_filter=True,
        ppisp=True,
        background_mode="modulation",
        background_color=(0.25, 0.5, 0.75),
        config_overrides={
            "means_lr": 0.000123,
            "enable_eval": True,
            "save_steps": [5000, 30000],
        },
        headless=True,
    )

    config = build_lichtfeld_config(options)

    assert config["strategy"] == "mrnf"
    assert config["iterations"] == 30000
    assert config["max_cap"] == 5_000_000
    assert config["sh_degree"] == 2
    assert config["tile_mode"] == 4
    assert config["steps_scaler"] == pytest.approx(1.56)
    assert config["use_bilateral_grid"] is True
    assert config["mask_mode"] == "segment"
    assert config["enable_sparsity"] is True
    assert config["gut"] is True
    assert config["undistort"] is True
    assert config["mip_filter"] is True
    assert config["use_ppisp"] is True
    assert config["bg_mode"] == "modulation"
    assert config["bg_color"] == [0.25, 0.5, 0.75]
    assert config["means_lr"] == pytest.approx(0.000123)
    assert config["enable_eval"] is True
    assert config["save_steps"] == [5000, 30000]
    assert config["headless"] is True
    assert config["auto_train"] is True
    assert config["eval_steps"] == [7000, 30000]

    cmd = build_lichtfeld_training_cmd(options)

    assert cmd == [
        "LichtFeld-Studio.exe",
        "--data-path",
        str(dataset.dataset_root),
        "--output-path",
        str(options.output_dir),
        "--config",
        str(options.config_path),
        "--train",
        "--no-splash",
        "--headless",
    ]
    assert json.loads(options.config_path.read_text(encoding="utf-8"))["iterations"] == 30000


def test_lichtfeld_auto_steps_scaler_matches_image_count(tmp_path: Path) -> None:
    dataset = TrainingDataset(dataset_root=tmp_path / "output")
    options = LichtFeldTrainingOptions(
        executable="LichtFeld-Studio.exe",
        dataset=dataset,
        output_dir=tmp_path / "training",
        config_path=tmp_path / "config.json",
        strategy="mrnf",
        iterations=46800,
        max_gaussians=5_000_000,
        sh_degree=3,
        tile_mode=1,
        steps_scaler=9.99,
        image_count=468,
        auto_steps_scaler=True,
    )

    config = build_lichtfeld_config(options)

    assert lichtfeld_auto_steps_scaler(300) == pytest.approx(1.0)
    assert lichtfeld_auto_steps_scaler(468) == pytest.approx(1.56)
    assert config["steps_scaler"] == pytest.approx(1.56)
    assert config["iterations"] == 30000


def test_postshot_command_passes_images_sparse_and_project_file(tmp_path: Path) -> None:
    dataset = TrainingDataset(
        dataset_root=tmp_path / "dataset",
        images_dir=tmp_path / "dataset" / "images",
        colmap_sparse_dir=tmp_path / "dataset" / "sparse" / "0",
    )

    cmd = build_postshot_training_cmd(
        PostshotTrainingOptions(
            executable="postshot-cli.exe",
            dataset=dataset,
            output_dir=tmp_path / "training",
            project_name="scene.psht",
            ksteps=60,
            max_image_size=4096,
        )
    )

    assert cmd == [
        "postshot-cli.exe",
        "train",
        "--import",
        str(dataset.images_dir),
        str(dataset.colmap_sparse_dir),
        "--output",
        str(tmp_path / "training" / "scene.psht"),
        "-s",
        "60",
        "--max-image-size",
        "4096",
    ]


def test_custom_training_command_renders_dataset_placeholders(tmp_path: Path) -> None:
    dataset = TrainingDataset(
        dataset_root=tmp_path / "dataset",
        images_dir=tmp_path / "dataset" / "images",
        masks_dir=tmp_path / "dataset" / "masks",
        colmap_sparse_dir=tmp_path / "dataset" / "sparse" / "0",
    )

    cmd = build_custom_training_cmd(
        CustomTrainingOptions(
            executable="trainer.exe",
            dataset=dataset,
            output_dir=tmp_path / "training",
            arguments_template='--data "{dataset}" --images "{images}" --sparse "{sparse}" --out "{output}"',
        )
    )

    assert cmd == [
        "trainer.exe",
        "--data",
        str(dataset.dataset_root),
        "--images",
        str(dataset.images_dir),
        "--sparse",
        str(dataset.colmap_sparse_dir),
        "--out",
        str(tmp_path / "training"),
    ]
