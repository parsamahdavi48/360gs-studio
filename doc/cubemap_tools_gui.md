# cubemap_tools_gui.py — GUI wrapper for cubemap conversion

`cubemap_tools_gui.py` is a PySide6 wrapper around `cubemap_transforms_json.py` for preview-driven view selection.

## Purpose

Use this GUI when you want to:
- keep FOV fixed at `90°`
- define multiple pitch rows (for example `-30,0,30`)
- choose enabled/disabled view slots per pitch row
- preview extraction regions on an equirectangular image
- overlay an existing mask with adjustable opacity

## Launch

```bash
python cubemap_tools_gui.py --scene-dir ./scene01
```

On Windows (recommended):

```bat
start_cubemap_tools_gui.bat
```

## Main fields

- `Scene Directory`:
  - Directory containing `transforms.json` and `images/`.
- `Output Directory`:
  - Cubemap output directory. Default: `<scene>/cubic`.
- `Transforms JSON`:
  - JSON filename in scene dir. Default: `transforms.json`.
- `Target Profile`:
  - Preset for downstream tool.
  - `Postshot / Brush`: sets `--no_transform` OFF and preprocess `--ply` OFF.
  - `LichtFeld Studio`: sets `--no_transform` ON and preprocess `--ply` ON.
  - `Custom (manual)`: leaves both options editable.
  - In preset modes, those related checkboxes are locked to prevent accidental mismatch.
- `Preprocess`:
  - If enabled, run bundled `vendor/metashape_360_lfs/metashape_360_lfs.py` before cubemap conversion.
- `MS Images Dir`:
  - Image directory passed to `metashape_360_lfs.py --images`.
- `MS XML`:
  - Metashape XML path passed to `metashape_360_lfs.py --xml`.
- `MS PLY (optional)`:
  - PLY path used when `MS PLY Usage` is enabled.
- `MS PLY Usage`:
  - Controls whether preprocess passes `--ply`.
- `MS Scale`:
  - Scale value passed to `metashape_360_lfs.py --scale` (must be positive).
- `MS Options`:
  - `Disable rotation fix (--no-fix-rotation)` for metashape preprocess.
- `Mask Directory`:
  - Optional mask input folder used by conversion and preview overlay.
- `View Mode`:
  - `Custom Pitch/Yaw`: existing mode with pitch rows and yaw slots.
  - `Cube6 (4 sides + top/bottom)`: fixed six-face mode (FOV 90).
- `Yaw Offset (deg)`:
  - Base yaw for slot generation.
- `Yaw Slots`:
  - Number of yaw slots per pitch row (`4..8`).
  - Slot yaw = `offset + slot*(360 / yaw_slots)`.
- `Pitch Rows (deg CSV)`:
  - Pitch list. Example: `-30,0,30`.
  - Max 9 rows.
- `Cube6 Options`:
  - `Drop Top (+90deg)`: disable top face.
  - `Drop Bottom (-90deg)`: disable bottom face.
- `FOV`:
  - Fixed to `90.0` in this GUI.
- `Preview Image`:
  - Equirectangular image used for overlay preview.
  - `Auto` picks the first image from scene images.
  - `Reload` rescans scene images.
- `Preview Timeline`:
  - Slider to switch preview image from scene images (`images/` preferred, otherwise scene root).
  - Useful for checking per-view on/off decisions across different frames.
- `Mask Overlay (%)`:
  - Opacity of mask overlay in preview.
- `Preview Mask Image`:
  - Optional manual mask file for preview overlay.
  - If empty, GUI auto-searches a matching mask from `Mask Directory`.

## View selection

- After `Apply Pitch Rows`, each pitch row gets `Yaw Slots` slots.
- Checkboxes control whether each slot is exported.
- Typical setup:
  - pitch `0`: enable all slots
  - pitch `+/-30`: enable only needed slots

## Run options

- `Extract mask from alpha (--mask_from_alpha)`
- `Transforms only (--no_image)`
- `No axis transform (--no_transform)`
- `Allow duplicate (--duplicate)`
- `Invert masks (--invert_masks)`

## Execution behavior

- On run, GUI writes `<output_dir>/views_config.json`.
- If `Preprocess` is enabled, GUI runs:
  - `vendor/metashape_360_lfs/metashape_360_lfs.py --images ... --xml ... --output <scene_dir> [...]`
- Then GUI runs:
  - `cubemap_transforms_json.py --fov 90 --views-json <that file>`
- Disabled slots are written with `enabled=false` and ignored by converter.
- With `Postshot / Brush` profile, preprocess `--ply` is disabled by default to avoid point-cloud/camera mismatch.
- Mask inversion is not automatic. Enable `Invert masks (--invert_masks)` only when needed.
- After conversion, GUI packages PLY into `<output_dir>` and rewrites `<output_dir>/transforms.json` `ply_file_path`.
  - `Postshot / Brush`: copies Metashape PLY (for example `metashape.ply` / `sparse.ply`).
  - `LichtFeld Studio`: copies `pointcloud.ply`.
- If required PLY is missing for selected profile, run is blocked.

## Notes

- If no view is enabled, run is blocked.
- Soft warning appears when enabled views exceed 24.
- Run is blocked when enabled views exceed 40.
- Preview is guidance only; final conversion follows `views_config.json` exactly.
