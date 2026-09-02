"""Generate a small CycloneDX-compatible inventory from installed packages."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    components = [
        {
            "type": "library",
            "name": distribution.metadata.get("Name", distribution.name),
            "version": distribution.version,
            "purl": f"pkg:pypi/{distribution.metadata.get('Name', distribution.name).lower()}@{distribution.version}",
        }
        for distribution in importlib.metadata.distributions()
    ]
    components.sort(key=lambda item: (item["name"].lower(), item["version"]))
    payload = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {"component": {"type": "application", "name": "360GS Studio"}},
        "components": components,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
