# apply_frame_decisions.py — Finalize reviewed frames

## Overview

`apply_frame_decisions.py` applies `_stechdrive/frames/selected_frames.csv` keep/drop decisions in one of two modes.

Primary mode:

- `--finalize-in-place`: operates directly on `scene_dir/images`
  - remove `drop` images
  - preserve `keep` image filenames by default
  - optionally renumber kept files in CSV order with `--renumber-kept-images`
  - rewrite `_stechdrive/frames/selected_frames.csv` to keep-only rows
  - create CSV backup before rewrite

## Usage

Finalize in place (recommended):

```bash
python -m core.apply_frame_decisions ./scene01 --finalize-in-place
```

Finalize with safety backup (snapshot images/ first):

```bash
python -m core.apply_frame_decisions ./scene01 --finalize-in-place --backup-dir _stechdrive/frames/backups/images
```

Finalize and renumber kept images before mask generation or Step 5 dataset export:

```bash
python -m core.apply_frame_decisions ./scene01 --finalize-in-place --renumber-kept-images
```

Legacy copy mode (separate output folder, originals untouched):

```bash
python -m core.apply_frame_decisions ./scene01 --output metashape_images --clean-output
```

Options:

- `scene_dir`: scene directory containing `_stechdrive/frames/selected_frames.csv` and extracted `images/`
- `--csv`: CSV filename under `_stechdrive/frames/`, or an absolute CSV path (default: `selected_frames.csv`)
- `--finalize-in-place`: apply decisions directly in `images/` and rewrite CSV
- `--backup-dir`: with `--finalize-in-place`, snapshot `images/` to this directory before modification. Existing targets are replaced only when the path is clearly a backup path such as `backups` or `images_backup`. Default empty = no backup
- `--renumber-kept-images`: with `--finalize-in-place`, rename kept images in CSV order to `images/frame_000001.ext`, update frame/source metadata, and refuse to run after downstream masks, `output/`, or SfM/dataset metadata already exist
- `--output`: copy-mode destination folder under scene directory (default: `metashape_images`)
- `--clean-output`: copy-mode only; remove existing image files in output folder before export

## Outputs

Finalize mode:

- `scene_dir/images/*` (keep-only; existing filenames are preserved unless `--renumber-kept-images` is used)
- `scene_dir/images/frame_000001.*`, `frame_000002.*`, ... (only when `--renumber-kept-images` is used; each original suffix is preserved)
- `scene_dir/_stechdrive/frames/backups/images/*` (full snapshot of pre-finalize images/ when `--backup-dir` is used)
- `scene_dir/_stechdrive/frames/selected_frames.csv` (rewritten)
- `scene_dir/_stechdrive/frames/backups/selected_frames.before_finalize.csv` (CSV backup, always created)
- `scene_dir/_stechdrive/frames/selected_frames_keep.csv`
- `_stechdrive/frames/extract_sessions.json` and `_stechdrive/sources/image_sets.json` path entries are updated when kept images are renumbered

Copy mode:

- `scene_dir/<output>/*` (keep-only copy)
- `scene_dir/<output>/selected_frames_keep.csv`

## Notes

- `--renumber-kept-images` is only valid with `--finalize-in-place`.
- Renumbering is intended before Step 3 and Step 5 dataset creation. It stops if `masks/`, mask metadata, `output/`, or `_stechdrive/step4/` already contains downstream results, because those assets may refer to the old filenames.
