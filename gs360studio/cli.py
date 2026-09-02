"""360GS Studio command-line interface."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from gs360studio.domain.models import ViewSpec
from gs360studio.engine.perspective_export import ExportRequest, export_image_views, export_video_views
from gs360studio.platform.components import bundled_component_manager
from gs360studio.platform.diagnostics import run_diagnostics
from gs360studio.platform.project_store import load_project, migrate_legacy_project
from gs360studio.version import __version__

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"})


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _gui(args: argparse.Namespace) -> int:
    if getattr(sys, "frozen", False):
        gui_executable = Path(sys.executable).with_name("360GS Studio.exe")
        if not gui_executable.is_file():
            raise FileNotFoundError(f"desktop executable not found beside the CLI: {gui_executable}")
        command = [str(gui_executable)]
        if args.project:
            command.extend(["--scene", str(Path(args.project).resolve())])
        subprocess.Popen(command, cwd=gui_executable.parent)
        return 0
    if args.project:
        os.environ["GS360_INITIAL_PROJECT"] = str(Path(args.project).resolve())
    main = importlib.import_module("gui.app").main
    main()
    return 0


def _doctor(args: argparse.Namespace) -> int:
    diagnostics = [item.to_dict() for item in run_diagnostics()]
    if args.json:
        _print_json({"application": "360GS Studio", "version": __version__, "diagnostics": diagnostics})
    else:
        print(f"360GS Studio {__version__}")
        for item in diagnostics:
            print(f"[{item['status'].upper():11}] {item['diagnostic_id']}: {item['summary']}")
    return 0 if all(item["status"] not in {"error"} for item in diagnostics) else 1


def _load_profile(project: Path, name: str) -> dict[str, Any]:
    path = project / "_360gs" / "profiles" / f"{name}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"view profile not found: {path}") from exc
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported view profile: {path}")
    return payload


def _profile_request(project: Path, name: str) -> ExportRequest:
    manifest, _report = load_project(project)
    profile = _load_profile(project, name)
    input_value = profile.get("input_path")
    if not input_value and manifest.sources:
        input_value = manifest.sources[0].get("path")
    if not input_value:
        raise ValueError("the profile or project must identify an input video")
    input_path = Path(str(input_value))
    if not input_path.is_absolute():
        input_path = project / input_path
    output = Path(str(profile.get("output_dir") or project / "output" / "perspective" / name))
    if not output.is_absolute():
        output = project / output
    views_payload = profile.get("views")
    if not isinstance(views_payload, list):
        raise ValueError("profile must contain a views list")
    return ExportRequest(
        input_path=input_path,
        output_dir=output,
        views=tuple(ViewSpec.from_dict(item, index=index) for index, item in enumerate(views_payload)),
        output_format=str(profile.get("output_format") or "png"),
        frame_interval_sec=float(profile.get("frame_interval_sec") or 1.0),
        jpeg_quality=int(profile.get("jpeg_quality") or 95),
        video_quality=int(profile.get("video_quality") or 18),
        video_preset=str(profile.get("video_preset") or "p4"),
        use_nvenc=bool(profile.get("use_nvenc", False)),
        batch_size=int(profile.get("batch_size") or 0),
        ffmpeg_path=str(profile.get("ffmpeg_path") or "ffmpeg"),
        colmap_rig=bool(profile.get("colmap_rig", False)),
        overwrite=bool(profile.get("overwrite", False)),
    )


def _export_views(args: argparse.Namespace) -> int:
    project = Path(args.project).resolve()
    request = _profile_request(project, args.profile)

    def progress(current: int, total: int, message: str) -> None:
        print(f"[{current}/{total}] {message}")

    if request.input_path.is_dir():
        files = sorted(path for path in request.input_path.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f"no supported images found in {request.input_path}")
        export_image_views(files, request, progress=progress)
    elif request.input_path.suffix.lower() in _IMAGE_SUFFIXES:
        export_image_views([request.input_path], request, progress=progress)
    else:
        export_video_views(request, progress=progress)
    print(request.output_dir.resolve())
    return 0


def _run_stage(args: argparse.Namespace) -> int:
    if args.stage != "perspective-export":
        raise ValueError("headless v0.1 supports the 'perspective-export' stage; existing SfM stages remain available in the GUI")
    if not args.profile:
        raise ValueError("perspective-export requires --profile")
    return _export_views(argparse.Namespace(project=args.project, profile=args.profile))


def _components(args: argparse.Namespace) -> int:
    manager = bundled_component_manager(root=args.root)
    if args.component_action == "list":
        _print_json([manager.state(manifest.component_id).to_dict() for manifest in manager.manifests()])
        return 0
    if args.component_action == "verify":
        state = manager.state(args.component_id)
        _print_json(state.to_dict())
        return 0 if state.verified else 1
    if args.component_action == "install":
        state = manager.install(args.component_id, accept_license=args.accept_license)
        _print_json(state.to_dict())
        return 0
    if args.component_action == "remove":
        manager.remove(args.component_id)
        return 0
    raise ValueError(f"unsupported component action: {args.component_action}")


def _migrate(args: argparse.Namespace) -> int:
    _manifest, report = migrate_legacy_project(args.project, write=not args.dry_run)
    _print_json(report.to_dict())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="360gs-studio", description="360° media and 3DGS workstation")
    parser.add_argument("--version", action="version", version=f"360GS Studio {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    gui = sub.add_parser("gui", help="Launch the desktop application")
    gui.add_argument("--project")
    gui.set_defaults(handler=_gui)

    doctor = sub.add_parser("doctor", help="Inspect local capabilities")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(handler=_doctor)

    run = sub.add_parser("run", help="Run a saved project stage")
    run.add_argument("--project", required=True)
    run.add_argument("--stage", required=True)
    run.add_argument("--profile")
    run.set_defaults(handler=_run_stage)

    export = sub.add_parser("export-views", help="Run a saved perspective-export profile")
    export.add_argument("--project", required=True)
    export.add_argument("--profile", required=True)
    export.set_defaults(handler=_export_views)

    migrate = sub.add_parser("migrate-project", help="Create non-destructive schema-v2 metadata")
    migrate.add_argument("--project", required=True)
    migrate.add_argument("--dry-run", action="store_true")
    migrate.set_defaults(handler=_migrate)

    components = sub.add_parser("components", help="Manage optional components")
    components.add_argument("--root")
    component_sub = components.add_subparsers(dest="component_action", required=True)
    component_sub.add_parser("list")
    verify = component_sub.add_parser("verify")
    verify.add_argument("component_id")
    install = component_sub.add_parser("install")
    install.add_argument("component_id")
    install.add_argument("--accept-license", action="store_true")
    remove = component_sub.add_parser("remove")
    remove.add_argument("component_id")
    components.set_defaults(handler=_components)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
