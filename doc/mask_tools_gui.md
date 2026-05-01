# 3DGS Studio Step 3 — Mask Generation GUI

## Overview

3DGS Studio Step 3 is a PySide6 wrapper around:

- `yolo_mask.py` (person mask generation)
- `stitch_mask.py` (stitch-region masking)
- `overexposure_mask.py` (overexposure masking)

It is intended for the frame-extraction workflow (`images/` -> `masks/`).

## Usage

```bat
start_gui.bat --scene ./scene01
```

## Main Fields

- `Scene Folder`: base folder. Step 3 fills `images` and `masks` from it.
- `Images Folder`: input images for YOLO/SAM masking.
- `Masks Folder`: output masks; also stitch input/output.
- `YOLO Level`: forwarded to `yolo_mask.py --level` (0-3).
- `YOLO Expand (px)`: forwarded to `yolo_mask.py --expand`.
  - Drag horizontally on the number field to adjust.
- `YOLO Add Ext`: forwarded to `yolo_mask.py --add-ext`.
- `YOLO Classes`: collapsed picker for class selection.
  - Choose classes by checkbox labels (`id: name`) instead of memorizing numeric ids.
  - Default preset is `person` only (`id=0`).
  - Presets: `Person only`, `People + Vehicles`, `All`, `Clear`.
  - Forwarded to `yolo_mask.py --classes`.
- `Boundary Mask Width (deg)`: forwarded to `stitch_mask.py --boundary-width`.
  - Drag horizontally on the number field to adjust.
  - The GUI clamps the value to `0.0-30.0` degrees for safety.
- `Stitch Workers`: forwarded to `stitch_mask.py --workers`.
  - Drag horizontally on the number field to adjust.

## Actions

Select `YOLO Detection`, `Stitch Seam`, and/or `Overexposure`, then press `Run Mask Creation`.
When multiple tasks are selected, they run in this order: YOLO detection, stitch seam, overexposure.

## Notes

- The GUI runs scripts as subprocesses, so behavior stays aligned with CLI.
- Logs from each step are shown in the integrated log panel.
