from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.spheresfm_gpu_preflight import (  # noqa: E402,F401
    PREFLIGHT_MAX_IMAGE_SIZE,
    PREFLIGHT_MAX_NUM_FEATURES,
    build_feature_command,
    iter_images,
    main,
    reset_preflight_workspace,
    run_colmap_command,
)

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
