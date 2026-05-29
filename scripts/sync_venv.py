from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    from update_venv import (
        SMOKE_TEST,
        TORCH_INDEX_URL,
        no_deps_requirements,
        regular_requirements,
        requirements_for_mode,
        run_pip_check,
    )
except ImportError:  # pragma: no cover - used when imported as scripts.sync_venv
    from scripts.update_venv import (
        SMOKE_TEST,
        TORCH_INDEX_URL,
        no_deps_requirements,
        regular_requirements,
        requirements_for_mode,
        run_pip_check,
    )


@dataclass(frozen=True)
class InstallStep:
    label: str
    command: tuple[str, ...]


def emit(message: str) -> None:
    print(message, flush=True)


def venv_python(repo_root: Path) -> Path:
    return repo_root / ".venv" / "Scripts" / "python.exe"


def build_pip_install_command(
    py: Path,
    requirements: list[str],
    *,
    index_url: str | None = None,
    no_deps: bool = False,
    dry_run: bool = False,
) -> tuple[str, ...]:
    cmd: list[str] = [str(py), "-m", "pip", "install", "--upgrade"]
    if dry_run:
        cmd.append("--dry-run")
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(requirements)
    if index_url:
        cmd.extend(["--index-url", index_url])
    return tuple(cmd)


def build_install_steps(py: Path, *, locked: bool, dry_run: bool) -> list[InstallStep]:
    requirements = requirements_for_mode(locked)
    steps = [
        InstallStep("pip", tuple([str(py), "-m", "pip", "install", "--upgrade", *(["--dry-run"] if dry_run else []), "pip"])),
        InstallStep(
            "core requirements",
            build_pip_install_command(py, requirements.core, dry_run=dry_run),
        ),
        InstallStep(
            "PyTorch CUDA requirements",
            build_pip_install_command(py, requirements.torch, index_url=TORCH_INDEX_URL, dry_run=dry_run),
        ),
        InstallStep(
            "ML requirements",
            build_pip_install_command(py, requirements.ml, dry_run=dry_run),
        ),
    ]

    sam31_regular = regular_requirements(requirements.sam31)
    sam31_no_deps = no_deps_requirements(requirements.sam31)
    if sam31_regular:
        steps.append(
            InstallStep(
                "SAM3.1 requirements",
                build_pip_install_command(py, sam31_regular, dry_run=dry_run),
            )
        )
    if sam31_no_deps:
        steps.append(
            InstallStep(
                "SAM3.1 source package",
                build_pip_install_command(py, sam31_no_deps, no_deps=True, dry_run=dry_run),
            )
        )
    return steps


def run_step(step: InstallStep) -> None:
    emit(f"[INFO] Updating {step.label}")
    emit("+ " + " ".join(step.command))
    subprocess.run(list(step.command), check=True)


def detect_major_minor(py: Path) -> str:
    result = subprocess.run(
        [str(py), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def run_smoke(py: Path, *, require_cuda: bool) -> None:
    smoke_code = "REQUIRE_CUDA = " + repr(require_cuda) + "\n" + SMOKE_TEST
    subprocess.run([str(py), "-c", smoke_code], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronize the existing .venv with this release's requirements.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Application folder. Defaults to this script's parent repository.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use latest compatible dependency specifiers instead of the release's pinned recommended set.",
    )
    parser.add_argument(
        "--locked",
        action="store_true",
        help="Use the release's pinned recommended set. This is the default.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Ask pip what would change without installing.")
    parser.add_argument("--allow-cpu-torch", action="store_true", help="Do not fail if torch CUDA is unavailable.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip import/CUDA smoke verification after syncing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    py = venv_python(repo_root)
    locked = not args.latest

    emit("========== Dependency Update ==========")
    emit(f"Application folder: {repo_root}")
    emit(f"Dependency mode: {'pinned recommended requirements' if locked else 'latest compatible requirements'}")

    if not py.exists():
        emit("[ERROR] .venv was not found. Run setup_windows.bat first.")
        emit("=======================================")
        return 1

    major_minor = detect_major_minor(py)
    emit(f".venv Python: {major_minor}")
    if major_minor != "3.12":
        emit(f"[ERROR] Python 3.12 is required; found {major_minor}.")
        emit("[INFO] Run setup_windows.bat --force if the environment must be rebuilt.")
        emit("=======================================")
        return 1

    for step in build_install_steps(py, locked=locked, dry_run=args.dry_run):
        run_step(step)

    if args.dry_run:
        emit("Result: dry-run completed; .venv was not changed")
        emit("=======================================")
        return 0

    run_pip_check(py)
    if not args.skip_smoke:
        run_smoke(py, require_cuda=not args.allow_cpu_torch)

    emit("Result: dependency update completed")
    emit("=======================================")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        emit(f"[ERROR] Command failed with exit code {exc.returncode}.")
        raise SystemExit(exc.returncode) from exc
