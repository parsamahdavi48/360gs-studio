from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.image_sequence_import import import_image_sequence_folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import a still-image sequence folder into a scene.")
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = import_image_sequence_folder(
            args.source_dir,
            args.scene_dir,
            prefix=args.prefix,
            recursive=args.recursive,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "SUMMARY_JSON:"
        + json.dumps(
            {
                "source": str(result.source_dir),
                "scene": str(result.scene_dir),
                "import_id": result.import_id,
                "image_count": result.image_count,
                "output_files": list(result.output_files),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
