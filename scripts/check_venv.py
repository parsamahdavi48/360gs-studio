from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

try:
    from update_venv import (
        LOCKED_CORE_REQUIREMENTS,
        LOCKED_ML_REQUIREMENTS,
        LOCKED_SAM31_REQUIREMENTS,
        LOCKED_TORCH_REQUIREMENTS,
    )
except ImportError:  # pragma: no cover - used when imported as scripts.check_venv
    from scripts.update_venv import (
        LOCKED_CORE_REQUIREMENTS,
        LOCKED_ML_REQUIREMENTS,
        LOCKED_SAM31_REQUIREMENTS,
        LOCKED_TORCH_REQUIREMENTS,
    )


SMOKE_TEST = r"""
import sys

import torch
import torchvision
import torchaudio
import numpy
import cv2
import PIL
import open3d
import ultralytics
import ultralytics.nn.tasks as ultralytics_tasks
import tqdm
import PySide6
import sam3
import timm
import ftfy
import iopath

print("Python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("torchaudio", torchaudio.__version__)
print("numpy", numpy.__version__)
print("cv2", cv2.__version__)
print("Pillow", PIL.__version__)
print("open3d", open3d.__version__)
print("ultralytics", ultralytics.__version__)
print("PySide6", PySide6.__version__)
print("sam3", getattr(sam3, "__version__", "unknown"))
print("timm", timm.__version__)

if REQUIRE_CUDA and not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")

if not hasattr(ultralytics_tasks, "SemanticSegmentationModel"):
    raise SystemExit(
        "Ultralytics does not support YOLO26 semantic segmentation in this environment. "
        "Reinstall requirements/ml.txt."
    )
"""


def is_optional_sam3_numpy_conflict(line: str) -> bool:
    """Return true for the known optional sam3 NumPy metadata conflict."""
    normalized = line.strip().lower()
    return (
        normalized.startswith("sam3 ")
        and "numpy" in normalized
        and "<2" in normalized
        and ("has requirement" in normalized or "requires" in normalized)
    )


def split_pip_check_errors(text: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    ignored: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if is_optional_sam3_numpy_conflict(stripped):
            ignored.append(stripped)
        else:
            errors.append(stripped)
    return errors, ignored


def run_capture(command: list[str | Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(part) for part in command], text=True, capture_output=True)


def emit_block(title: str, text: str) -> None:
    stripped = text.strip()
    if stripped:
        print(title)
        print(stripped)


def pinned_versions(requirements: list[str]) -> dict[str, str]:
    pins: dict[str, str] = {}
    for requirement in requirements:
        if "==" not in requirement:
            continue
        name, version = requirement.split("==", 1)
        pins[name.strip()] = version.strip()
    return pins


def pinned_version_check_code(requirements: list[str]) -> str:
    expected = pinned_versions(requirements)
    return (
        "import importlib.metadata as md\n"
        f"expected = {json.dumps(expected, sort_keys=True)}\n"
        "errors = []\n"
        "for package, expected_version in expected.items():\n"
        "    try:\n"
        "        actual = md.version(package)\n"
        "    except md.PackageNotFoundError:\n"
        "        errors.append(f'{package}: not installed')\n"
        "        continue\n"
        "    if actual != expected_version:\n"
        "        errors.append(f'{package}: expected {expected_version}, found {actual}')\n"
        "if errors:\n"
        "    raise SystemExit('\\n'.join(errors))\n"
        "print('Pinned requirements: passed')\n"
    )


def venv_python(repo_root: Path) -> Path:
    return repo_root / ".venv" / "Scripts" / "python.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether the repository .venv is ready to run.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to this script's parent repository.",
    )
    parser.add_argument("--require-python", default="3.12", help="Required Python major.minor version.")
    parser.add_argument("--allow-cpu-torch", action="store_true", help="Do not fail when PyTorch CUDA is unavailable.")
    parser.add_argument(
        "--locked",
        "--use-lock",
        action="store_true",
        dest="locked",
        help="Also require runtime package versions to match requirements/ pins.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    py = venv_python(repo_root)

    print("========== Setup Check ==========")
    print(f".venv Python: {py}")

    if not py.exists():
        print("Result: .venv is not installed")
        print("=================================")
        return 1

    version_result = run_capture([py, "-c", "import sys; print(sys.version.split()[0]); print(f'{sys.version_info[0]}.{sys.version_info[1]}')"])
    if version_result.returncode != 0:
        print("Result: failed to run .venv Python")
        emit_block("Python output:", (version_result.stdout or "") + (version_result.stderr or ""))
        print("=================================")
        return 1

    version_lines = [line.strip() for line in version_result.stdout.splitlines() if line.strip()]
    full_version = version_lines[0] if version_lines else "unknown"
    major_minor = version_lines[1] if len(version_lines) > 1 else "unknown"
    print(f"Detected Python: {full_version}")

    if major_minor != args.require_python:
        print(f"Result: Python {args.require_python} is required; found {major_minor}")
        print("=================================")
        return 1

    pip_check = run_capture([py, "-m", "pip", "check"])
    if pip_check.returncode != 0:
        pip_text = (pip_check.stdout or "") + (pip_check.stderr or "")
        pip_errors, ignored_warnings = split_pip_check_errors(pip_text)
        if pip_errors:
            print("Result: pip dependency check failed")
            emit_block("pip check output:", "\n".join(pip_errors + ignored_warnings))
            print("=================================")
            return 1
        print("pip check: passed with optional SAM3.1 NumPy metadata warning")
        emit_block("Ignored optional pip check warning:", "\n".join(ignored_warnings))
    else:
        print("pip check: passed")

    if args.locked:
        pinned_requirements = (
            LOCKED_CORE_REQUIREMENTS
            + LOCKED_TORCH_REQUIREMENTS
            + LOCKED_ML_REQUIREMENTS
            + LOCKED_SAM31_REQUIREMENTS
        )
        pins = run_capture([py, "-c", pinned_version_check_code(pinned_requirements)])
        if pins.returncode != 0:
            print("Result: pinned dependency check failed")
            emit_block("Pinned dependency output:", (pins.stdout or "") + (pins.stderr or ""))
            print("=================================")
            return 1
        emit_block("Pinned dependency output:", pins.stdout)

    smoke_code = "REQUIRE_CUDA = " + repr(not args.allow_cpu_torch) + "\n" + SMOKE_TEST
    smoke = run_capture([py, "-c", smoke_code])
    if smoke.returncode != 0:
        print("Result: import/CUDA smoke test failed")
        emit_block("Smoke test output:", (smoke.stdout or "") + (smoke.stderr or ""))
        print("=================================")
        return 1

    emit_block("Smoke test output:", smoke.stdout)
    print("Result: existing .venv is ready")
    print("=================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
