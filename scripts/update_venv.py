from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


MIN_PYTHON = (3, 12)
RELEASE_PYTHON = (3, 12)
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
PYTHON_CACHE_MAX_AGE_SEC = 7 * 24 * 60 * 60
LOG_FILE: Path | None = None
REQUIREMENTS_DIR = Path(__file__).resolve().parents[1] / "requirements"
SAM31_SOURCE_OK_REQUIREMENTS = {"iopath"}


def read_requirements_file(name: str) -> list[str]:
    path = REQUIREMENTS_DIR / name
    requirements: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


def unpin_requirement(requirement: str) -> str:
    return re.split(r"\s*(?:==|~=|!=|<=|>=|<|>)", requirement, maxsplit=1)[0].strip()


@dataclass(frozen=True)
class RequirementSet:
    core: list[str]
    torch: list[str]
    ml: list[str]
    sam31: list[str]
    test: list[str]
    locked: bool = False

    @property
    def label(self) -> str:
        return "locked pinned requirements" if self.locked else "latest compatible requirements"


LOCKED_CORE_REQUIREMENTS = read_requirements_file("core.txt")
LOCKED_TORCH_REQUIREMENTS = read_requirements_file("torch-cu128.txt")
LOCKED_ML_REQUIREMENTS = read_requirements_file("ml.txt")
LOCKED_SAM31_REQUIREMENTS = read_requirements_file("sam31.txt")
LOCKED_TEST_REQUIREMENTS = read_requirements_file("test.txt")

CORE_REQUIREMENTS = [unpin_requirement(req) for req in LOCKED_CORE_REQUIREMENTS]
TORCH_REQUIREMENTS = [unpin_requirement(req) for req in LOCKED_TORCH_REQUIREMENTS]
ML_REQUIREMENTS = [unpin_requirement(req) for req in LOCKED_ML_REQUIREMENTS]
SAM31_REQUIREMENTS = [unpin_requirement(req) for req in LOCKED_SAM31_REQUIREMENTS]
TEST_REQUIREMENTS = [unpin_requirement(req) for req in LOCKED_TEST_REQUIREMENTS]


def requirements_for_mode(locked: bool) -> RequirementSet:
    if locked:
        return RequirementSet(
            core=LOCKED_CORE_REQUIREMENTS,
            torch=LOCKED_TORCH_REQUIREMENTS,
            ml=LOCKED_ML_REQUIREMENTS,
            sam31=LOCKED_SAM31_REQUIREMENTS,
            test=LOCKED_TEST_REQUIREMENTS,
            locked=True,
        )
    return RequirementSet(
        core=CORE_REQUIREMENTS,
        torch=TORCH_REQUIREMENTS,
        ml=ML_REQUIREMENTS,
        sam31=SAM31_REQUIREMENTS,
        test=TEST_REQUIREMENTS,
        locked=False,
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
"""


def configure_log(repo_root: Path) -> None:
    global LOG_FILE
    LOG_FILE = repo_root / ".cache" / "update_venv.log"
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(
        f"update_venv.py log started {datetime.now().isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )


def write_log(text: str) -> None:
    if LOG_FILE is None:
        return
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(text)


def emit(message: str) -> None:
    print(message, flush=True)
    write_log(message + "\n")


@dataclass(frozen=True)
class PythonCandidate:
    version: tuple[int, int]
    executable: Path

    @property
    def label(self) -> str:
        return version_label(self.version)


@dataclass(frozen=True)
class CandidateReport:
    version: tuple[int, int]
    status: str
    detail: str


def version_label(version: tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]}"


def parse_version_label(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid Python version: {value}")
    return int(match.group(1)), int(match.group(2))


def version_sort_key(version: tuple[int, int]) -> tuple[int, int]:
    return version[0], version[1]


def run(
    cmd: list[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    printable = " ".join(str(part) for part in cmd)
    emit(f"+ {printable}")
    argv = [str(part) for part in cmd]
    if capture:
        result = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
        )
    else:
        output_parts: list[str] = []
        process = subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            write_log(line)
            output_parts.append(line)
        result = subprocess.CompletedProcess(argv, process.wait(), "".join(output_parts), "")
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, [str(part) for part in cmd], result.stdout, result.stderr)
    return result


def capture_cmd(cmd: list[str | os.PathLike[str]]) -> str:
    result = run(cmd, check=False, capture=True)
    if result.returncode != 0:
        return ""
    return result.stdout or ""


def parse_py_launcher_output(text: str) -> dict[tuple[int, int], list[Path]]:
    found: dict[tuple[int, int], list[Path]] = {}
    pattern = re.compile(r"-V:(\d+)\.(\d+)\s+\*?\s*(.+?python\.exe)\s*$", re.IGNORECASE)
    for line in text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        version = int(match.group(1)), int(match.group(2))
        found.setdefault(version, []).append(Path(match.group(3).strip()))
    return found


def parse_winget_python_versions(text: str) -> set[tuple[int, int]]:
    versions: set[tuple[int, int]] = set()
    for match in re.finditer(r"Python\.Python\.(\d+)\.(\d+)", text, flags=re.IGNORECASE):
        versions.add((int(match.group(1)), int(match.group(2))))
    return versions


def scan_python_install_dirs() -> list[Path]:
    paths: list[Path] = []
    roots = [
        os.environ.get("LocalAppData"),
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        "C:/",
    ]
    for root_value in roots:
        if not root_value:
            continue
        root = Path(root_value)
        for base in (root / "Programs" / "Python", root):
            if not base.exists():
                continue
            for child in base.glob("Python3*/python.exe"):
                paths.append(child)
    return paths


def known_python_paths(version: tuple[int, int]) -> list[Path]:
    major, minor = version
    tag = f"Python{major}{minor}"
    paths: list[Path] = []
    for root_name in ("LocalAppData", "ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(root_name)
        if root:
            paths.append(Path(root) / "Programs" / "Python" / tag / "python.exe")
            paths.append(Path(root) / tag / "python.exe")
    paths.append(Path(f"C:/Python{major}{minor}/python.exe"))
    return paths


def python_version(executable: Path) -> tuple[int, int] | None:
    result = run(
        [
            executable,
            "-c",
            "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
        ],
        check=False,
        capture=True,
    )
    if result.returncode != 0:
        return None
    try:
        return parse_version_label((result.stdout or "").strip())
    except argparse.ArgumentTypeError:
        return None


def add_candidate(
    candidates: dict[tuple[int, int], PythonCandidate],
    target_versions: set[tuple[int, int]],
    executable: Path,
) -> None:
    if not executable.exists():
        return
    version = python_version(executable)
    if version not in target_versions:
        return
    try:
        resolved = executable.resolve()
    except OSError:
        resolved = executable
    candidates.setdefault(version, PythonCandidate(version=version, executable=resolved))


def detect_installed_versions() -> set[tuple[int, int]]:
    versions: set[tuple[int, int]] = set()
    for version in parse_py_launcher_output(capture_cmd(["py", "-0p"])):
        versions.add(version)
    for path in scan_python_install_dirs():
        version = python_version(path)
        if version:
            versions.add(version)
    path_python = shutil.which("python")
    if path_python:
        version = python_version(Path(path_python))
        if version:
            versions.add(version)
    return versions


def detect_python_candidates(target_versions: list[tuple[int, int]]) -> list[PythonCandidate]:
    target_set = set(target_versions)
    candidates: dict[tuple[int, int], PythonCandidate] = {}

    for version, paths in parse_py_launcher_output(capture_cmd(["py", "-0p"])).items():
        if version not in target_set:
            continue
        for path in paths:
            add_candidate(candidates, target_set, path)

    for version in target_versions:
        for path in known_python_paths(version):
            add_candidate(candidates, target_set, path)

    for path in scan_python_install_dirs():
        add_candidate(candidates, target_set, path)

    path_python = shutil.which("python")
    if path_python:
        add_candidate(candidates, target_set, Path(path_python))

    return [candidates[v] for v in target_versions if v in candidates]


def cache_path(repo_root: Path) -> Path:
    return repo_root / ".cache" / "python_candidates.json"


def load_cached_winget_versions(repo_root: Path, max_age_sec: int) -> set[tuple[int, int]] | None:
    path = cache_path(repo_root)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        timestamp = float(data.get("timestamp", 0.0))
        if time.time() - timestamp > max_age_sec:
            return None
        return {parse_version_label(item) for item in data.get("versions", [])}
    except Exception:
        return None


def save_cached_winget_versions(repo_root: Path, versions: set[tuple[int, int]]) -> None:
    path = cache_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": time.time(),
        "versions": [version_label(v) for v in sorted(versions, key=version_sort_key, reverse=True)],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def query_winget_python_versions(repo_root: Path, *, refresh: bool, max_age_sec: int) -> set[tuple[int, int]]:
    if not refresh:
        cached = load_cached_winget_versions(repo_root, max_age_sec)
        if cached is not None:
            emit("[INFO] Using cached winget Python candidate list.")
            return cached

    if shutil.which("winget") is None:
        emit("[WARN] winget is not available; only installed Python versions can be used.")
        return set()

    output = capture_cmd(["winget", "search", "--source", "winget", "Python.Python"])
    versions = parse_winget_python_versions(output)
    if versions:
        save_cached_winget_versions(repo_root, versions)
    return versions


def install_python_with_winget(version: tuple[int, int]) -> bool:
    package_id = f"Python.Python.{version[0]}.{version[1]}"
    if shutil.which("winget") is None:
        emit("[WARN] winget is not available; cannot install missing Python.")
        return False
    result = run(
        [
            "winget",
            "install",
            "--id",
            package_id,
            "--source",
            "winget",
            "--accept-package-agreements",
            "--accept-source-agreements",
        ],
        check=False,
    )
    return result.returncode == 0


def build_target_versions(
    *,
    repo_root: Path,
    explicit_candidates: str,
    refresh_python_cache: bool,
    python_cache_days: int,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    if explicit_candidates:
        versions = [parse_version_label(part) for part in explicit_candidates.split(",") if part.strip()]
        return sorted(set(versions), key=version_sort_key, reverse=True), set()

    installed = detect_installed_versions()
    winget_versions = query_winget_python_versions(
        repo_root,
        refresh=refresh_python_cache,
        max_age_sec=python_cache_days * 24 * 60 * 60,
    )
    versions = installed | winget_versions | {RELEASE_PYTHON}
    versions = {v for v in versions if v >= MIN_PYTHON}
    return sorted(versions, key=version_sort_key, reverse=True), winget_versions


def pip_preflight(
    runner_python: Path,
    version: tuple[int, int],
    requirements: list[str],
    *,
    index_url: str | None = None,
    no_deps: bool = False,
) -> bool:
    py_tag = f"{version[0]}.{version[1]}"
    abi_tag = f"cp{version[0]}{version[1]}"
    cmd: list[str | os.PathLike[str]] = [
        runner_python,
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--ignore-installed",
        "--only-binary=:all:",
        "--python-version",
        py_tag,
        "--platform",
        "win_amd64",
        "--implementation",
        "cp",
        "--abi",
        abi_tag,
        "--upgrade",
    ]
    if no_deps:
        cmd.append("--no-deps")
    if index_url:
        cmd.extend(["--index-url", index_url])
    cmd.extend(requirements)
    result = run(cmd, check=False, capture=True)
    if result.returncode == 0:
        return True
    output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    if output:
        emit(output)
    return False


def no_deps_requirements(requirements: list[str]) -> list[str]:
    return [
        requirement
        for requirement in requirements
        if requirement.lower().startswith("sam3 @ ") or "github.com/facebookresearch/sam3" in requirement.lower()
    ]


def regular_requirements(requirements: list[str]) -> list[str]:
    no_deps = set(no_deps_requirements(requirements))
    return [requirement for requirement in requirements if requirement not in no_deps]


def wheel_preflight_requirements(requirements: list[str]) -> list[str]:
    return [
        requirement
        for requirement in regular_requirements(requirements)
        if "://" not in requirement and "git+" not in requirement.lower()
        and unpin_requirement(requirement).lower() not in SAM31_SOURCE_OK_REQUIREMENTS
    ]


def is_optional_sam3_numpy_conflict(line: str) -> bool:
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


def run_pip_check(py: Path) -> None:
    result = run([py, "-m", "pip", "check"], check=False, capture=True)
    output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    if result.returncode == 0:
        if output:
            emit(output)
        return

    errors, ignored = split_pip_check_errors(output)
    if errors:
        if output:
            emit(output)
        raise subprocess.CalledProcessError(result.returncode, [str(py), "-m", "pip", "check"], output, "")
    emit("[INFO] pip check passed with optional SAM3.1 NumPy metadata warning.")
    for warning in ignored:
        emit(f"[INFO] Ignored: {warning}")


def preflight_version(
    runner_python: Path,
    version: tuple[int, int],
    requirements: RequirementSet,
) -> tuple[bool, str]:
    emit(f"[INFO] Preflight for Python {version_label(version)}")
    if not pip_preflight(runner_python, version, requirements.core):
        emit(f"[WARN] Python {version_label(version)} rejected by core dependency preflight.")
        return False, "core dependency wheels are not available"
    if not pip_preflight(runner_python, version, requirements.torch, index_url=TORCH_INDEX_URL):
        emit(f"[WARN] Python {version_label(version)} rejected by PyTorch CUDA wheel preflight.")
        return False, "PyTorch CUDA wheels are not available"
    if not pip_preflight(runner_python, version, requirements.ml, no_deps=True):
        emit(f"[WARN] Python {version_label(version)} rejected by ML dependency preflight.")
        return False, "ML dependency wheels are not available"
    sam31_preflight = wheel_preflight_requirements(requirements.sam31)
    if sam31_preflight and not pip_preflight(runner_python, version, sam31_preflight, no_deps=True):
        emit(f"[WARN] Python {version_label(version)} rejected by SAM3.1 dependency preflight.")
        return False, "SAM3.1 dependency wheels are not available"
    return True, "preflight passed"


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe"


def read_python_full_version(executable: Path) -> str | None:
    if not executable.exists():
        return None
    result = run(
        [
            executable,
            "-c",
            "import sys; print(sys.version.split()[0])",
        ],
        check=False,
        capture=True,
    )
    if result.returncode != 0:
        return None
    version = (result.stdout or "").strip()
    return version or None


def read_venv_full_version(repo_root: Path) -> str | None:
    return read_python_full_version(venv_python(repo_root / ".venv"))


def full_version_major_minor(version: str | None) -> tuple[int, int] | None:
    if not version:
        return None
    match = re.match(r"(\d+)\.(\d+)", version)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def has_pytest_suite(repo_root: Path) -> bool:
    tests_dir = repo_root / "tests"
    return tests_dir.is_dir() and any(tests_dir.rglob("test_*.py"))


def create_candidate_venv(
    repo_root: Path,
    candidate: PythonCandidate,
    *,
    requirements: RequirementSet,
    require_cuda: bool,
    run_pytest: bool,
    keep_temp: bool,
) -> Path | None:
    temp_venv = repo_root / f".venv-candidate-py{candidate.version[0]}{candidate.version[1]}"
    remove_dir(temp_venv)
    run_tests = run_pytest and has_pytest_suite(repo_root)

    try:
        emit(f"[INFO] Building Python {candidate.label}: {candidate.executable}")
        run([candidate.executable, "-m", "venv", temp_venv])
        py = venv_python(temp_venv)

        run([py, "-m", "pip", "install", "--upgrade", "pip"])
        run([py, "-m", "pip", "install", "--upgrade", *requirements.core])
        run([py, "-m", "pip", "install", "--upgrade", *requirements.torch, "--index-url", TORCH_INDEX_URL])
        run([py, "-m", "pip", "install", "--upgrade", *requirements.ml])
        sam31_regular = regular_requirements(requirements.sam31)
        sam31_no_deps = no_deps_requirements(requirements.sam31)
        if sam31_regular:
            run([py, "-m", "pip", "install", "--upgrade", *sam31_regular])
        if sam31_no_deps:
            run([py, "-m", "pip", "install", "--upgrade", "--no-deps", *sam31_no_deps])
        if run_tests:
            run([py, "-m", "pip", "install", "--upgrade", *requirements.test])
        run_pip_check(py)

        smoke_code = "REQUIRE_CUDA = " + repr(require_cuda) + "\n" + SMOKE_TEST
        run([py, "-c", smoke_code])
        if run_pytest:
            if run_tests:
                run([py, "-m", "pytest", "-q"], cwd=repo_root)
            else:
                emit("[INFO] tests/ was not found; skipping pytest verification.")
    except subprocess.CalledProcessError as exc:
        emit(f"[WARN] Python {candidate.label} rejected with exit code {exc.returncode}.")
        if not keep_temp:
            remove_dir(temp_venv)
        return None

    return temp_venv


def path_is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def adopt_venv(repo_root: Path, candidate_venv: Path, *, repair_python: Path, keep_backup: bool) -> None:
    final_venv = repo_root / ".venv"
    running_python = Path(sys.executable)
    if final_venv.exists() and path_is_inside(running_python, final_venv):
        raise RuntimeError("Updater is running from .venv and cannot replace it safely.")

    backup: Path | None = None
    if final_venv.exists():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = repo_root / f".venv-backup-{timestamp}"
        emit(f"[INFO] Backing up existing .venv to {backup.name}")
        final_venv.rename(backup)

    try:
        emit("[INFO] Promoting tested environment to .venv")
        candidate_venv.rename(final_venv)
        repair_venv(final_venv, repair_python=repair_python)
    except Exception:
        if backup is not None and backup.exists() and not final_venv.exists():
            backup.rename(final_venv)
        raise

    if backup is not None and backup.exists() and not keep_backup:
        emit("[INFO] Removing old .venv backup")
        remove_dir(backup)


def repair_venv(venv_dir: Path, *, repair_python: Path) -> None:
    emit("[INFO] Repairing venv launch scripts after promotion")
    run([repair_python, "-m", "venv", "--upgrade", venv_dir])
    repair_activation_scripts(venv_dir)


def repair_activation_scripts(venv_dir: Path) -> None:
    scripts_dir = venv_dir / "Scripts"
    prompt = f"({venv_dir.name}) "
    windows_path = str(venv_dir)

    activate_bat = scripts_dir / "activate.bat"
    if activate_bat.exists():
        text = activate_bat.read_text(encoding="utf-8")
        text = re.sub(r'set "VIRTUAL_ENV=.*"', lambda _: f'set "VIRTUAL_ENV={windows_path}"', text)
        text = re.sub(r"set PROMPT=\([^)]+\) %PROMPT%", lambda _: f"set PROMPT={prompt}%PROMPT%", text)
        text = re.sub(r'set "VIRTUAL_ENV_PROMPT=.*"', lambda _: f'set "VIRTUAL_ENV_PROMPT={prompt}"', text)
        activate_bat.write_text(text, encoding="utf-8")

    activate_posix = scripts_dir / "activate"
    if activate_posix.exists():
        text = activate_posix.read_text(encoding="utf-8")
        text = re.sub(r"cygpath '.*?\.venv-candidate-py\d+'", lambda _: f"cygpath '{windows_path}'", text)
        text = re.sub(r"export VIRTUAL_ENV='.*?\.venv-candidate-py\d+'", lambda _: f"export VIRTUAL_ENV='{windows_path}'", text)
        text = re.sub(r"VIRTUAL_ENV_PROMPT='\([^)]+\) '", lambda _: f"VIRTUAL_ENV_PROMPT='{prompt}'", text)
        text = re.sub(r"\.venv-candidate-py\d+", lambda _: venv_dir.name, text)
        activate_posix.write_text(text, encoding="utf-8")


def emit_summary(
    *,
    result: str,
    existing_venv: str | None,
    final_venv: str | None,
    reports: list[CandidateReport],
) -> None:
    emit("")
    emit("========== Summary ==========")
    emit(f"Result: {result}")
    emit(f"Previous .venv Python: {existing_venv or 'not found'}")
    emit(f"Current .venv Python: {final_venv or 'not changed'}")
    if reports:
        emit("Candidates:")
        for report in reports:
            emit(f"  - Python {version_label(report.version)}: {report.status} - {report.detail}")
    if LOG_FILE is not None:
        emit(f"Log file: {LOG_FILE}")
    emit("=============================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build .venv from the newest Python candidate supported by this project's latest dependencies."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to this script's parent repository.",
    )
    parser.add_argument(
        "--candidates",
        default="",
        help="Optional comma-separated Python versions to try. When omitted, installed and winget-available versions are discovered.",
    )
    parser.add_argument(
        "--refresh-python-cache",
        action="store_true",
        help="Refresh the cached winget Python candidate list.",
    )
    parser.add_argument(
        "--python-cache-days",
        type=int,
        default=7,
        help="Days to keep the winget Python candidate cache.",
    )
    parser.add_argument(
        "--install-fallback-python",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-install-python",
        action="store_true",
        help="Do not install missing Python candidates with winget.",
    )
    parser.add_argument("--allow-cpu-torch", action="store_true", help="Do not fail if torch CUDA is unavailable.")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip pytest during candidate verification.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep failed candidate venvs for debugging.")
    parser.add_argument("--keep-backup", action="store_true", help="Keep the previous .venv backup after promotion.")
    parser.add_argument(
        "--locked",
        "--use-lock",
        action="store_true",
        dest="locked",
        help="Use pinned requirement files from requirements/ instead of resolving the latest compatible packages.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run discovery and preflight only; do not install.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    configure_log(repo_root)
    runner_python = Path(sys.executable)
    existing_venv = read_venv_full_version(repo_root)
    reports: list[CandidateReport] = []
    requirements = requirements_for_mode(args.locked)
    emit(f"[INFO] Dependency mode: {requirements.label}")

    target_versions, winget_versions = build_target_versions(
        repo_root=repo_root,
        explicit_candidates=args.candidates,
        refresh_python_cache=args.refresh_python_cache,
        python_cache_days=args.python_cache_days,
    )
    if not target_versions:
        emit("[ERROR] No Python candidates were found.")
        emit_summary(
            result="failed; no Python candidates were found",
            existing_venv=existing_venv,
            final_venv=read_venv_full_version(repo_root),
            reports=reports,
        )
        return 1

    emit("[INFO] Python candidates: " + ", ".join(version_label(v) for v in target_versions))

    for version in target_versions:
        preflight_ok, preflight_detail = preflight_version(runner_python, version, requirements)
        if not preflight_ok:
            reports.append(CandidateReport(version, "not used", preflight_detail))
            continue

        installed = detect_python_candidates([version])
        candidate = installed[0] if installed else None

        if args.dry_run:
            if candidate is not None:
                emit(f"[DRY-RUN] Python {version_label(version)} would be used: {candidate.executable}")
                reports.append(CandidateReport(version, "would be used", str(candidate.executable)))
                emit_summary(
                    result="dry-run completed; .venv was not changed",
                    existing_venv=existing_venv,
                    final_venv=read_venv_full_version(repo_root),
                    reports=reports,
                )
                return 0
            if version in winget_versions and not args.no_install_python:
                emit(f"[DRY-RUN] Python {version_label(version)} would be installed via winget.")
                reports.append(CandidateReport(version, "would be installed", "available through winget"))
                emit_summary(
                    result="dry-run completed; .venv was not changed",
                    existing_venv=existing_venv,
                    final_venv=read_venv_full_version(repo_root),
                    reports=reports,
                )
                return 0
            emit(f"[WARN] Python {version_label(version)} passed preflight but is not installed.")
            reports.append(CandidateReport(version, "not used", "preflight passed but Python is not installed"))
            continue

        if candidate is None and not args.no_install_python and version in winget_versions:
            emit(f"[INFO] Installing Python {version_label(version)} via winget.")
            if install_python_with_winget(version):
                installed = detect_python_candidates([version])
                candidate = installed[0] if installed else None
            else:
                reports.append(CandidateReport(version, "not used", "winget installation failed"))

        if candidate is None:
            emit(f"[WARN] Python {version_label(version)} passed preflight but is not installed.")
            if not any(report.version == version for report in reports):
                reports.append(CandidateReport(version, "not used", "preflight passed but Python is not installed"))
            continue

        candidate_venv = create_candidate_venv(
            repo_root,
            candidate,
            requirements=requirements,
            require_cuda=not args.allow_cpu_torch,
            run_pytest=not args.skip_pytest,
            keep_temp=args.keep_temp,
        )
        if candidate_venv is None:
            reports.append(CandidateReport(version, "not used", "candidate venv failed verification"))
            continue

        adopt_venv(repo_root, candidate_venv, repair_python=candidate.executable, keep_backup=args.keep_backup)
        emit(f"[DONE] .venv now uses Python {candidate.label}.")
        final_venv = read_venv_full_version(repo_root)
        before_minor = full_version_major_minor(existing_venv)
        if existing_venv is None:
            result = f"created .venv with Python {final_venv or candidate.label}"
        elif before_minor == candidate.version:
            result = f"rebuilt .venv with Python {final_venv or candidate.label}; Python version was unchanged"
        else:
            result = f"updated .venv from Python {existing_venv} to {final_venv or candidate.label}"
        reports.append(CandidateReport(version, "used", "passed verification and was promoted to .venv"))
        emit_summary(
            result=result,
            existing_venv=existing_venv,
            final_venv=final_venv,
            reports=reports,
        )
        return 0

    emit("[ERROR] No Python candidate passed preflight and verification.")
    emit_summary(
        result="failed; no candidate passed all checks",
        existing_venv=existing_venv,
        final_venv=read_venv_full_version(repo_root),
        reports=reports,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
