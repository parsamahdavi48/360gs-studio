# realityscan_rig_export.py - RealityScan rig package exporter

`realityscan_rig_export.py` exports perspective crops from equirectangular inputs and writes matching XMP sidecar files for RealityScan rig import.

## Purpose

Use this script when you want:
- per-view perspective images from 360 input frames
- XMP sidecars paired with each exported image
- fixed rig metadata (`Rig`, `RigInstance`, `RigPoseIndex`) for RealityScan
- optional mask export in the same import folder (`<image_name>.mask.png`)

## Basic usage

```bash
python realityscan_rig_export.py <scene_dir> <rs_output_root> --views-json <views_config.json>
```

Example:

```bash
python realityscan_rig_export.py ./scene01 ./scene01/realityscan_rig --views-json ./scene01/realityscan_rig/views_config.json --mask_dir ./scene01/masks --mask_from_alpha
```

## Input requirements

- `<scene_dir>/<transforms.json>` must exist (or pass `--json`)
- `camera_model` must be `EQUIRECTANGULAR`
- `frames[].file_path` must point to readable image files
- each frame needs a valid `4x4` `transform_matrix`

## Output layout

All outputs are written under `<rs_output_root>`:

- `inputs/`
  - exported perspective images
  - matching XMP files (`<image_name>.xmp`)
  - optional mask layers (`<image_name>.mask.png`)
- `views_config.json`
- `manifest.csv`
- `realityscan_project.json`

Use `inputs/` as the import folder in RealityScan.

## Key options

- `--views-json`: custom enabled view list (`name`, `yaw`, `pitch`)
- `--fov`: crop FOV (default `90`)
- `--pose_prior`: XMP pose prior (`initial|exact|locked`, default `exact`)
- `--calibration_prior`: XMP calibration prior (`initial|fixed|exact|locked`, default `fixed`)
- `--focal35mm`: override focal length in XMP (default is derived from FOV; `90deg -> 18mm`)
- `--mask_dir`: source mask folder
- `--mask_from_alpha`: generate masks from alpha channel when source is RGBA
- `--invert_masks`: invert exported mask pixels
- `--no_transform`: disable axis transform on input camera poses

## Notes

- Mask output is merged from alpha-derived and source mask when both exist.
- `Rig` is shared for the whole export; `RigInstance` is generated per source frame.
- The script writes deterministic IDs so repeated exports with the same view layout stay consistent.
