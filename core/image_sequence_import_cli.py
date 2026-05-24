"""CLI entry point for importing still-image sequence folders."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from core.frame_job_runner import run_frame_job_payload
from core.frame_job_spec import import_image_sequence_job


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import a still-image sequence folder into a scene.")
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--recursive", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        run_frame_job_payload(
            import_image_sequence_job(
                source_dir=args.source_dir,
                scene_dir=args.scene_dir,
                prefix=args.prefix,
                recursive=args.recursive,
            )
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
