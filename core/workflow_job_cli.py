"""CLI entry point for versioned workflow jobs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from core.workflow_job_runner import run_workflow_job_file


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a versioned workflow job.")
    parser.add_argument("--job", required=True, help="Workflow job JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        run_workflow_job_file(args.job)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
