"""Training backend command builders for Step 4."""
from __future__ import annotations

import json
import math
import shlex
from dataclasses import dataclass
from pathlib import Path

_LICHTFELD_REQUIRED_STRATEGIES = {"mrnf", "mcmc", "igs+"}
_LICHTFELD_MASK_MODES = {"none", "segment", "ignore", "alpha_consistent"}
_LICHTFELD_TILE_MODES = {1, 2, 4}
_LICHTFELD_BG_MODES = {"solid_color", "modulation", "image", "random"}
_LICHTFELD_BASE_IMAGE_COUNT = 300


def _mrnf_defaults() -> dict:
    return {
        "iterations": 30000,
        "sh_degree_interval": 1000,
        "means_lr": 0.00002,
        "means_lr_end": 0.0000002,
        "shs_lr": 0.002,
        "opacity_lr": 0.012,
        "scaling_lr": 0.007,
        "scaling_lr_end": 0.005,
        "rotation_lr": 0.002,
        "lambda_dssim": 0.2,
        "min_opacity": 1.0 / 255.0,
        "refine_every": 200,
        "start_refine": 0,
        "stop_refine": 28500,
        "grad_threshold": 0.003,
        "sh_degree": 3,
        "opacity_reg": 0.0,
        "scale_reg": 0.0,
        "init_opacity": 0.5,
        "init_scaling": 0.1,
        "max_cap": 5000000,
        "eval_steps": [7000, 30000],
        "save_steps": [7000, 30000],
        "strategy": "mrnf",
        "enable_eval": False,
        "enable_save_eval_images": True,
        "headless": False,
        "mip_filter": False,
        "use_bilateral_grid": False,
        "bg_modulation": False,
        "bilateral_grid_X": 16,
        "bilateral_grid_Y": 16,
        "bilateral_grid_W": 8,
        "bilateral_grid_lr": 0.002,
        "tv_loss_weight": 10.0,
        "revised_opacity": True,
        "gut": False,
        "undistort": False,
        "steps_scaler": 1.0,
        "random": False,
        "init_num_pts": 100000,
        "init_extent": 3.0,
        "tile_mode": 1,
        "mask_mode": "none",
        "invert_masks": False,
        "mask_opacity_penalty_weight": 1.0,
        "mask_opacity_penalty_power": 2.0,
        "mask_threshold": 0.5,
        "use_alpha_as_mask": True,
        "enable_sparsity": False,
        "sparsify_steps": 15000,
        "init_rho": 0.0005,
        "prune_ratio": 0.6,
        "use_ppisp": False,
        "ppisp_lr": 0.002,
        "ppisp_reg_weight": 0.001,
        "ppisp_warmup_steps": 500,
        "ppisp_freeze_from_sidecar": False,
        "ppisp_sidecar_path": "",
        "ppisp_use_controller": False,
        "ppisp_freeze_gaussians_on_distill": True,
        "ppisp_controller_activation_step": -1,
        "ppisp_controller_lr": 0.002,
        "growth_grad_threshold": 0.003,
        "grow_fraction": 0.07,
        "grow_until_iter": 15000,
        "opacity_decay": 0.004,
        "scale_decay": 0.002,
        "means_noise_weight": 50.0,
        "bounds_percentile": 0.8,
        "use_error_map": True,
        "use_edge_map": True,
        "bg_mode": "solid_color",
        "bg_color": [0.0, 0.0, 0.0],
    }


def _mcmc_defaults() -> dict:
    params = _mrnf_defaults()
    params.update(
        {
            "strategy": "mcmc",
            "means_lr": 0.000016,
            "means_lr_end": 0.00000016,
            "shs_lr": 0.0025,
            "opacity_lr": 0.025,
            "scaling_lr": 0.005,
            "scaling_lr_end": 0.005,
            "rotation_lr": 0.001,
            "min_opacity": 0.005,
            "refine_every": 100,
            "start_refine": 500,
            "stop_refine": 25000,
            "grad_threshold": 0.0002,
            "opacity_reg": 0.01,
            "scale_reg": 0.01,
            "max_cap": 1000000,
            "revised_opacity": False,
            "use_error_map": True,
            "use_edge_map": True,
        }
    )
    return params


def _igs_plus_defaults() -> dict:
    params = _mrnf_defaults()
    params.update(
        {
            "strategy": "igs+",
            "means_lr": 0.000016,
            "means_lr_end": 0.00000016,
            "shs_lr": 0.005,
            "opacity_lr": 0.025,
            "scaling_lr": 0.02,
            "scaling_lr_end": 0.005,
            "rotation_lr": 0.0015,
            "min_opacity": 0.005,
            "refine_every": 500,
            "start_refine": 500,
            "stop_refine": 15000,
            "grad_threshold": 0.0002,
            "opacity_reg": 0.0,
            "scale_reg": 0.0,
            "init_opacity": 0.1,
            "init_scaling": 0.1,
            "max_cap": 4000000,
            "tv_loss_weight": 5.0,
            "revised_opacity": True,
            "gut": False,
        }
    )
    return params


def lichtfeld_defaults(strategy: str) -> dict:
    normalized = strategy.lower().strip()
    if normalized == "mcmc":
        return _mcmc_defaults()
    if normalized == "igs+":
        return _igs_plus_defaults()
    return _mrnf_defaults()


def lichtfeld_auto_steps_scaler(image_count: int) -> float:
    if image_count <= 0:
        return 1.0
    if image_count <= _LICHTFELD_BASE_IMAGE_COUNT:
        return 1.0
    return image_count / _LICHTFELD_BASE_IMAGE_COUNT


def _round_lfs_step(value: float) -> int:
    return max(1, int(math.floor(value + 0.5)))


def _unscale_lfs_step(value: int, scaler: float) -> int:
    if scaler <= 0.0 or math.isclose(scaler, 1.0):
        return int(value)
    return _round_lfs_step(value / scaler)


@dataclass(frozen=True)
class TrainingDataset:
    dataset_root: Path
    images_dir: Path | None = None
    masks_dir: Path | None = None
    colmap_sparse_dir: Path | None = None
    transforms_json: Path | None = None
    pointcloud_ply: Path | None = None
    output_shape: str = ""


@dataclass(frozen=True)
class LichtFeldTrainingOptions:
    executable: str
    dataset: TrainingDataset
    output_dir: Path
    config_path: Path
    strategy: str
    iterations: int
    max_gaussians: int
    sh_degree: int
    tile_mode: int
    steps_scaler: float
    output_name: str = ""
    image_count: int | None = None
    auto_steps_scaler: bool = False
    bilateral_grid: bool = False
    mask_mode: str = "none"
    sparsity: bool = False
    gut: bool = False
    undistort: bool = False
    mip_filter: bool = False
    ppisp: bool = False
    background_mode: str = "solid_color"
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    background_image_path: str = ""
    dataset_resize_factor: str | None = None
    dataset_max_width: int | None = None
    dataset_use_cpu_cache: bool = True
    dataset_use_fs_cache: bool = True
    dataset_test_every: int | None = None
    config_overrides: dict[str, object] | None = None
    headless: bool = False
    no_splash: bool = True


@dataclass(frozen=True)
class PostshotTrainingOptions:
    executable: str
    dataset: TrainingDataset
    output_dir: Path
    project_name: str
    ksteps: int
    max_image_size: int
    use_imported_poses: bool = True


@dataclass(frozen=True)
class CustomTrainingOptions:
    executable: str
    dataset: TrainingDataset
    output_dir: Path
    arguments_template: str


def build_lichtfeld_config(options: LichtFeldTrainingOptions) -> dict:
    strategy = options.strategy.lower().strip()
    if strategy not in _LICHTFELD_REQUIRED_STRATEGIES:
        raise ValueError(f"Unsupported LichtFeld strategy: {options.strategy}")
    if options.tile_mode not in _LICHTFELD_TILE_MODES:
        raise ValueError("LichtFeld tile mode must be 1, 2, or 4")
    if options.mask_mode not in _LICHTFELD_MASK_MODES:
        raise ValueError(f"Unsupported LichtFeld mask mode: {options.mask_mode}")
    if strategy == "igs+" and options.gut:
        raise ValueError("LichtFeld igs+ strategy cannot be used with GUT")
    if options.iterations <= 0:
        raise ValueError("LichtFeld iterations must be greater than 0")
    if options.max_gaussians <= 0:
        raise ValueError("LichtFeld max Gaussians must be greater than 0")
    if options.sh_degree < 0 or options.sh_degree > 3:
        raise ValueError("LichtFeld SH degree must be 0, 1, 2, or 3")
    if options.steps_scaler <= 0:
        raise ValueError("LichtFeld steps scaler must be greater than 0")
    if options.background_mode not in _LICHTFELD_BG_MODES:
        raise ValueError(f"Unsupported LichtFeld background mode: {options.background_mode}")
    if len(options.background_color) != 3 or any(
        not math.isfinite(c) or c < 0.0 or c > 1.0 for c in options.background_color
    ):
        raise ValueError("LichtFeld background color must contain three values between 0 and 1")

    steps_scaler = (
        lichtfeld_auto_steps_scaler(options.image_count)
        if options.auto_steps_scaler and options.image_count is not None
        else float(options.steps_scaler)
    )
    config_iterations = _unscale_lfs_step(int(options.iterations), steps_scaler)
    config = lichtfeld_defaults(strategy)
    if options.config_overrides:
        config.update(options.config_overrides)
    config.update(
        {
            "strategy": strategy,
            "iterations": config_iterations,
            "max_cap": int(options.max_gaussians),
            "sh_degree": int(options.sh_degree),
            "tile_mode": int(options.tile_mode),
            "steps_scaler": float(steps_scaler),
            "use_bilateral_grid": bool(options.bilateral_grid),
            "mask_mode": options.mask_mode,
            "enable_sparsity": bool(options.sparsity),
            "gut": bool(options.gut),
            "undistort": bool(options.undistort),
            "mip_filter": bool(options.mip_filter),
            "use_ppisp": bool(options.ppisp),
            "bg_mode": options.background_mode,
            "bg_color": [float(c) for c in options.background_color],
            "headless": bool(options.headless),
            "auto_train": True,
            "no_splash": bool(options.no_splash),
        }
    )
    if options.background_image_path:
        config["bg_image_path"] = options.background_image_path
    if not (options.config_overrides and "eval_steps" in options.config_overrides):
        config["eval_steps"] = [min(7000, config_iterations), config_iterations]
    if not (options.config_overrides and "save_steps" in options.config_overrides):
        config["save_steps"] = [min(7000, config_iterations), config_iterations]
    return config


def write_lichtfeld_config(options: LichtFeldTrainingOptions) -> Path:
    options.config_path.parent.mkdir(parents=True, exist_ok=True)
    options.config_path.write_text(
        json.dumps(build_lichtfeld_config(options), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return options.config_path


def lichtfeld_output_name_stem(value: str) -> str:
    name = value.strip()
    if not name:
        return ""
    if any(sep in name for sep in ("/", "\\")):
        raise ValueError("LichtFeld output PLY name must be a file name, not a path")
    if name.lower().endswith(".ply"):
        name = name[:-4].strip()
    if not name:
        raise ValueError("LichtFeld output PLY name must not be empty")
    return name


def build_lichtfeld_training_cmd(options: LichtFeldTrainingOptions) -> list[str]:
    output_name = lichtfeld_output_name_stem(options.output_name)
    write_lichtfeld_config(options)
    cmd = [
        options.executable,
        "--data-path",
        str(options.dataset.dataset_root),
        "--output-path",
        str(options.output_dir),
        "--config",
        str(options.config_path),
        "--train",
    ]
    if output_name:
        cmd.extend(["--output-name", output_name])
    if options.dataset_resize_factor:
        cmd.extend(["--resize_factor", options.dataset_resize_factor])
    if options.dataset_max_width is not None:
        if options.dataset_max_width <= 0 or options.dataset_max_width > 4096:
            raise ValueError("LichtFeld max width must be between 1 and 4096")
        cmd.extend(["--max-width", str(options.dataset_max_width)])
    if not options.dataset_use_cpu_cache:
        cmd.append("--no-cpu-cache")
    if not options.dataset_use_fs_cache:
        cmd.append("--no-fs-cache")
    if options.dataset_test_every is not None:
        if options.dataset_test_every <= 0:
            raise ValueError("LichtFeld test every must be greater than 0")
        cmd.extend(["--test-every", str(options.dataset_test_every)])
    if options.no_splash:
        cmd.append("--no-splash")
    if options.headless:
        cmd.append("--headless")
    return cmd


def build_postshot_training_cmd(options: PostshotTrainingOptions) -> list[str]:
    if options.ksteps <= 0:
        raise ValueError("Postshot kSteps must be greater than 0")
    if options.max_image_size < 0:
        raise ValueError("Postshot max image size must be 0 or greater")

    output_file = options.output_dir / options.project_name
    cmd = [
        options.executable,
        "train",
        "--import",
        str(options.dataset.images_dir or options.dataset.dataset_root),
    ]
    if options.use_imported_poses and options.dataset.colmap_sparse_dir is not None:
        cmd.append(str(options.dataset.colmap_sparse_dir))
    cmd.extend(["--output", str(output_file), "-s", str(options.ksteps)])
    cmd.extend(["--max-image-size", str(options.max_image_size)])
    return cmd


def build_custom_training_cmd(options: CustomTrainingOptions) -> list[str]:
    if not options.arguments_template.strip():
        return [options.executable]
    values = {
        "dataset": str(options.dataset.dataset_root),
        "images": str(options.dataset.images_dir or ""),
        "masks": str(options.dataset.masks_dir or ""),
        "sparse": str(options.dataset.colmap_sparse_dir or ""),
        "output": str(options.output_dir),
    }
    try:
        rendered = options.arguments_template.format(**values)
    except KeyError as exc:
        raise ValueError(f"Unknown custom training placeholder: {exc}") from exc
    args = [
        arg[1:-1]
        if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in {"'", '"'}
        else arg
        for arg in shlex.split(rendered, posix=False)
    ]
    return [options.executable, *args]
