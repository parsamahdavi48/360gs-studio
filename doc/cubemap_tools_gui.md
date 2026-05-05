# Step 4 Export GUI — Cubemap and COLMAP Rig Export

Step 4 in the integrated GUI is a PySide6 wrapper around
`cubemap_transforms_json.py` for preview-driven view selection and export.

## Purpose

Use Step 4 when you want to:
- keep FOV fixed at `90°`
- define multiple pitch rows (for example `-30,0,30`)
- choose enabled/disabled view slots per pitch row
- preview extraction regions on an equirectangular image
- toggle an existing mask overlay on the preview

## Launch

```bat
run_gui.bat --scene .\scene01
```

Then open `Step 4: Export` in the workflow sidebar.

## Main fields

- `Scene Directory`:
  - Directory containing `images/` and, when needed, `transforms.json`.
- `Output Directory`:
  - Cubemap output directory. Default: `<scene>/output`.
- `Select:`:
  - `Metashape`: export 3DGS-oriented viewpoint images, masks, and `transforms.json` from Metashape SfM results.
  - `COLMAP`: export COLMAP Rig viewpoint images, masks, and `rig_config.json` from extracted `images/` and `masks/` into `output/colmap_rig/`.
- `COLMAP Run Settings`:
  - Visible as the `Conversion` settings tab when `COLMAP` is selected.
  - `Run COLMAP after export`: runs `feature_extractor` -> `rig_configurator` -> matcher -> mapper after viewpoint export. Keep this off unless you want to start the heavy SfM step.
  - `COLMAP Executable`: select the `colmap` executable for this machine. Leave empty to resolve `colmap` from PATH.
  - `Matcher`: `Sequential` is fast and suited to ordered video frames. `Exhaustive` can improve coverage but compares all pairs and can become tens-of-hours slow on large sets.
  - `Mapper`: `Global` is the COLMAP 4.0+ integrated GLOMAP-style global SfM path and is the recommended default. `Incremental` is the classic `colmap mapper`; `GLOMAP` uses an external `glomap` executable.
- `Output Preset`:
  - Preset for the downstream 3DGS tool when `Metashape` is selected.
  - `Postshot / Brush`: applies the target coordinate preset and uses the scene PLY directly.
  - `LichtFeld Studio`: imports the Metashape point cloud as `pointcloud.ply` and writes LichtFeld-oriented camera data.
  - If advanced settings change the coordinate transform, PLY usage, or Metashape import details away from the preset value, the preset display switches to `Custom`.
- `Output Shape`:
  - `Convert to Projection Views`: standard path. Exports cubemap/view images and masks into `output/` using the `Projection Views` tab.
  - `3DGUT (LichtFeld)`: comparison path for LichtFeld 3DGUT. Uses the Metashape source equirectangular `images/` and `masks/` directly, skips view image and converted mask export, and updates scene-root `transforms.json` plus `pointcloud.ply`.
  - Direct 3DGUT mode requires the LichtFeld-oriented coordinate settings and PLY import. The `Projection Views` tab and view image/mask output toggles are disabled while it is active.
- `Metashape Import Settings`:
  - Visible as the `Conversion` settings tab when `Metashape` is selected.
  - In `Metashape` mode, the GUI runs bundled `vendor/metashape_360_lfs/metashape_360_lfs.py` before cubemap conversion.
  - `Images Folder`: fixed to scene `images/`.
  - `Camera XML`: Metashape-exported camera pose XML passed to `--xml`.
  - `Point Cloud PLY`: Metashape-exported point cloud PLY. Used automatically for LichtFeld.
  - `Add COLMAP Text Model`: creates `cameras.txt` / `images.txt` / `points3D.txt` under `output/colmap/` from `output/transforms.json` and PLY. This is not a COLMAP SfM image export.
  - `Advanced`: controls `--scale`, whether to pass `--ply`, and `--no-fix-rotation`.
- `Output`:
  - Always visible. Independently toggles `Images` and `Masks`. Turn `Images` off and `Masks` on when you only rebuilt masks.
- `Projection Views`:
  - Always visible as the `Projection Views` settings tab.
  - Shared viewpoint image export settings such as view preset, yaw offset, image size, per-frame yaw step, output format, bit depth, and mask inversion.
  - Not used when `Output Shape` is `3DGUT (LichtFeld)`.
- Masks:
  - Conversion and preview automatically use matching files from scene `masks/`.
- `Preset`:
  - `Cube6`: six-face preset on a 4 x 3 grid. By default it uses 4 yaw slots, pitch rows `-90,0,90`, all four pitch `0` slots, and `S3` for top/bottom.
  - `Custom Grid`: freely edit pitch rows and yaw slots. Editing the Cube6 grid automatically switches it to custom.
- `Yaw Offset (deg)`:
  - Base yaw for slot generation.
- `Yaw Slots`:
  - Number of yaw slots per pitch row (`4..8`). Use the `-` / `+` controls above Output Views to remove or add columns.
  - Existing column on/off states and pitch values are preserved when resizing; newly added columns start enabled.
  - Slot yaw = `offset + slot*(360 / yaw_slots)`.
- `Pitch Rows`:
  - Pitch list. `Cube6` uses `-90,0,90`; `Custom Grid` starts from `-45,0,45`.
  - Range is `-90..90`, max 5 rows. Use the `+` control beside the pitch rows to add a row, and the delete button at the left of each pitch row to remove it.
  - Remaining row pitch values and on/off states are preserved by row order when rows are added or removed.
- `Cube6`:
  - Always exports all six faces. Top and bottom faces are treated as valid fixed-camera observations.
- `FOV`:
  - Fixed to `90.0` in this GUI.
- `Image Size`:
  - `Full`: output face size = input height x `1.0`; recommended for final quality.
  - `Normal`: output face size = input height x `2 / pi` (about `0.637`), matching the center angular resolution of a 90-degree view to the source image.
  - `Half`: output face size = input height x `0.5`; lightweight output that can look softer after training.
- `Preview`:
  - Automatically uses equirectangular images from the scene folder.
  - Slider to switch preview image from scene images (`images/` preferred, otherwise scene root).
  - Useful for checking per-view on/off decisions across different frames.
  - Mask overlay uses matching masks from `masks/`; the mask display button toggles it on/off.
  - The projection button switches to a square 90° FOV perspective view. Drag the preview to look around; the mouse wheel still performs 2D zoom on the displayed image. Mask overlays and view outlines are reflected in the perspective view. Hovering the view grid aligns the perspective view to that export direction. The perspective view uses OpenGL/GPU acceleration when available and falls back to CPU rendering otherwise.

## View selection

- Checkboxes control whether each slot is exported.
- Output Views is always visible. Its toolbar provides select all/deselect all and yaw column add/remove controls; pitch rows are added beside the pitch row list.
- In the default Cube6 preset, top and bottom are assigned to `S3`; with the default 45-degree yaw offset, `S3=-45°`.
- Typical setup:
  - pitch `0`: enable all slots
  - pitch `+/-30`: enable only needed slots

## Run options

- `Invert masks (--invert_masks)`
  - Usually off. Enable only when the target app expects the opposite polarity.
- `Output`
  - Turning `Images` off adds `--skip-images`; turning `Masks` off adds `--skip-masks`.
  - If both are off, the GUI updates camera metadata only and preserves existing files under `output/`.

## Execution behavior

- On run, GUI writes `<output_dir>/views_config.json`.
- In `Metashape` mode, GUI first runs:
  - `vendor/metashape_360_lfs/metashape_360_lfs.py --images ... --xml ... --output <scene_dir> [...]`
- Then GUI runs:
  - `cubemap_transforms_json.py --fov 90 --output_scale <1.0|0.6366|0.5> --views-json <that file>`
- When `Output Shape` is `3DGUT (LichtFeld)`, the GUI runs only the Metashape import step above and does not call `cubemap_transforms_json.py`.
  - Transforms: `<scene_dir>/transforms.json`
  - Point cloud: `<scene_dir>/pointcloud.ply`
  - Images: referenced in place from `<scene_dir>/images/`
  - Masks: referenced in place from `<scene_dir>/masks/`
  - Export settings: `<scene_dir>/stechdrive_export_settings.json`
- In `COLMAP` mode, GUI runs `cubemap_transforms_json.py --image-only --colmap-rig --yaw-offset-per-frame 0 ...`.
  - Images: `<output_dir>/colmap_rig/images/rig1/camXX/frame_00001.<ext>`
  - Masks: `<output_dir>/colmap_rig/masks/rig1/camXX/frame_00001.<ext>.png`
  - Rig config: `<output_dir>/colmap_rig/rig_config.json`
  - Per-frame yaw rotation is forced to 0 degrees for COLMAP Rig because changing yaw per frame breaks the fixed rig sensor assumption.
- When `Run COLMAP after export` is enabled, GUI then runs `feature_extractor`, `rig_configurator`, matcher, and mapper.
- If `Images` or `Masks` is disabled in Export Targets, the GUI adds `--skip-images` or `--skip-masks`.
- After a successful normal conversion, GUI saves the target profile, image size, view settings, a `views_config.json` snapshot, per-frame yaw rotation, output format, and related settings to `<output_dir>/stechdrive_export_settings.json`.
- Disabled slots are written with `enabled=false` and ignored by converter.
- With `LichtFeld Studio` profile, point cloud PLY import is enabled automatically.
- Masks normally convert with black as the ignored region. Postshot can handle interpretation through its own Mask Mode, so the GUI does not auto-invert masks for Postshot.
- After conversion, GUI packages PLY into `<output_dir>` and rewrites `<output_dir>/transforms.json` `ply_file_path`.
  - `Postshot / Brush`: copies Metashape PLY (for example `metashape.ply` / `sparse.ply`).
  - `LichtFeld Studio`: copies `pointcloud.ply`.
- With the `LichtFeld Studio` profile, the GUI applies the same final orientation correction to `transforms.json` and `pointcloud.ply` so the +X / +Z / up directions match the Metashape scene in LichtFeld.
- If required PLY is missing for selected profile, run is blocked.

## Notes

- If no view is enabled, run is blocked.
- Soft warning appears when enabled views exceed 24.
- Run is blocked when enabled views exceed 40.
- Preview is guidance only; final conversion follows `views_config.json` exactly.
