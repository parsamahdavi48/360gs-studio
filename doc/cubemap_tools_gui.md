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
  - Directory containing `images/` and, when needed, `transforms.json`.
- `Output Directory`:
  - Cubemap output directory. Default: `<scene>/output`.
- `Export Method`:
  - `Metashape Import`: export 3DGS-oriented viewpoint images, masks, and `transforms.json` from Metashape SfM results.
  - `COLMAP Export`: export viewpoint images and masks for COLMAP SfM from extracted `images/` and `masks/`, without Metashape results.
- `Output Profile`:
  - Preset for the downstream 3DGS tool when `Metashape Import` is selected.
  - `Postshot / Brush`: applies the target coordinate preset and uses the scene PLY directly.
  - `LichtFeld Studio`: imports the Metashape point cloud as `pointcloud.ply` and writes LichtFeld-oriented camera data.
  - If advanced settings change the coordinate transform, PLY usage, or Metashape import details away from the preset value, the profile display switches to `Custom`.
- `Metashape Import Settings`:
  - If enabled, run bundled `vendor/metashape_360_lfs/metashape_360_lfs.py` before cubemap conversion.
  - `Images Folder`: fixed to scene `images/`.
  - `Camera XML`: Metashape-exported camera pose XML passed to `--xml`.
  - `Point Cloud PLY`: Metashape-exported point cloud PLY. Used automatically for LichtFeld.
  - `Advanced`: controls `--scale`, whether to pass `--ply`, and `--no-fix-rotation`.
- `View Export Settings`:
  - Shared viewpoint image export settings such as image size, per-frame yaw step, output format, bit depth, and mask inversion.
- Masks:
  - Conversion and preview automatically use matching files from scene `masks/`.
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
- `Image Size`:
  - `Full (Quality)`: output face size = input height x `1.0`; recommended for final quality.
  - `Normal`: output face size = input height x `2 / pi` (about `0.637`), matching the center angular resolution of a 90-degree view to the source image.
  - `Half (Light)`: output face size = input height x `0.5`; lightweight output that can look softer after training.
- `Preview`:
  - Automatically uses equirectangular images from the scene folder.
  - Slider to switch preview image from scene images (`images/` preferred, otherwise scene root).
  - Useful for checking per-view on/off decisions across different frames.
  - Mask overlay uses matching masks from `masks/`; the opacity slider controls its visibility.

## View selection

- After `Apply Pitch Rows`, each pitch row gets `Yaw Slots` slots.
- Checkboxes control whether each slot is exported.
- Typical setup:
  - pitch `0`: enable all slots
  - pitch `+/-30`: enable only needed slots

## Run options

- `Invert masks (--invert_masks)`
  - Usually off. Enable only when the target app expects the opposite polarity.
- `Skip Image/Mask Conversion (--no_image)`
  - Updates `transforms.json` without reconverting cubemap images or masks. Existing files in `output/` are preserved.
  - Does not overwrite `output/stechdrive_export_settings.json` saved by a normal conversion.

## Workflow tabs

- `Cubemap`:
  - Existing conversion workflow (`metashape_360_lfs.py` optional + `cubemap_transforms_json.py`).
- `COLMAP Rig SfM`:
  - Exports COLMAP rig dataset and optionally runs COLMAP SfM stages.
- `RealityScan Rig XMP`:
  - Exports a self-contained RealityScan import package.
  - `RS Output Root`: package root directory.
  - `Pose Prior`: XMP pose prior (`Draft/Exact/Locked`).
  - `Calibration Prior`: XMP calibration prior (`Fixed` is recommended default).
  - `Focal35mm Override`: optional override (empty = auto from `FOV=90` -> `18mm`).
  - `Pose Transform`: pass `--no_transform` to RS export script.
  - `Mask Export`: optional mask inversion for RS package.

## Execution behavior

- On run, GUI writes `<output_dir>/views_config.json`.
- If `Metashape Import Settings` is enabled, GUI runs:
  - `vendor/metashape_360_lfs/metashape_360_lfs.py --images ... --xml ... --output <scene_dir> [...]`
- Then GUI runs:
  - `cubemap_transforms_json.py --fov 90 --output_scale <1.0|0.6366|0.5> --views-json <that file>`
- If `Skip Image/Mask Conversion` is enabled, the GUI adds `--no_image` and does not reset existing files in `<output_dir>`.
- After a successful normal conversion, GUI saves the target profile, image size, view settings, a `views_config.json` snapshot, per-frame yaw rotation, output format, and related settings to `<output_dir>/stechdrive_export_settings.json`.
- Disabled slots are written with `enabled=false` and ignored by converter.
- With `LichtFeld Studio` profile, point cloud PLY import is enabled automatically.
- Masks normally convert with black as the ignored region. Postshot can handle interpretation through its own Mask Mode, so the GUI does not auto-invert masks for Postshot.
- After conversion, GUI packages PLY into `<output_dir>` and rewrites `<output_dir>/transforms.json` `ply_file_path`.
  - `Postshot / Brush`: copies Metashape PLY (for example `metashape.ply` / `sparse.ply`).
  - `LichtFeld Studio`: copies `pointcloud.ply`.
- With the `LichtFeld Studio` profile, the GUI applies the same final orientation correction to `transforms.json` and `pointcloud.ply` so the +X / +Z / up directions match the Metashape scene in LichtFeld.
- If required PLY is missing for selected profile, run is blocked.
- In `RealityScan Rig XMP` tab:
  - GUI writes `<rs_output_root>/views_config.json`.
  - If `Metashape Import Settings` is enabled, the Metashape import step is executed before RS export.
  - GUI runs `realityscan_rig_export.py`.
  - Exported import files are bundled under `<rs_output_root>/inputs`:
    - `<image_name>`
    - `<image_stem>.xmp`
    - `<image_name>.mask.png` (when mask exists)

## Notes

- If no view is enabled, run is blocked.
- Soft warning appears when enabled views exceed 24.
- Run is blocked when enabled views exceed 40.
- Preview is guidance only; final conversion follows `views_config.json` exactly.
