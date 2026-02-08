# apply_frame_decisions.py — Finalize reviewed frames

## Overview

`apply_frame_decisions.py` applies `selected_frames.csv` keep/drop decisions in one of two modes.

Primary mode:

- `--finalize-in-place`: operates directly on `scene_dir/images`
  - remove `drop` images
  - renumber `keep` images as `<prefix>_000001.*`, `<prefix>_000002.*`, ...
  - rewrite `selected_frames.csv` to keep-only rows with updated `output_file`
  - create CSV backup before rewrite

## Usage

Finalize in place (recommended):

```bash
python apply_frame_decisions.py ./scene01 --finalize-in-place
```

Legacy copy mode:

```bash
python apply_frame_decisions.py ./scene01 --output metashape_images --clean-output
```

Options:

- `scene_dir`: scene directory containing `selected_frames.csv` and extracted `images/`
- `--csv`: CSV filename (default: `selected_frames.csv`)
- `--finalize-in-place`: apply decisions directly in `images/` and rewrite CSV
- `--filename-prefix`: rename prefix for finalize mode (default: from extract report/video name)
- `--output`: copy-mode destination folder under scene directory (default: `metashape_images`)
- `--clean-output`: copy-mode only; remove existing image files in output folder before export

## Outputs

Finalize mode:

- `scene_dir/images/<prefix>_000001.*` ... (keep-only, renumbered)
- `scene_dir/selected_frames.csv` (rewritten)
- `scene_dir/selected_frames.before_finalize.csv` (backup)
- `scene_dir/selected_frames_keep.csv`

Copy mode:

- `scene_dir/<output>/*` (keep-only copy)
- `scene_dir/<output>/selected_frames_keep.csv`
