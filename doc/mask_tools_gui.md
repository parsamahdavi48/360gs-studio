# mask_tools_gui.py — GUI wrapper for mask generation

## Overview

`mask_tools_gui.py` is a PySide6 wrapper around:

- `yolo_mask.py` (person mask generation)
- `stitch_mask.py` (stitch-region masking)

It is intended for the frame-extraction workflow (`images/` -> `masks/`).

## Usage

```bash
python mask_tools_gui.py --scene-dir ./scene01
```

Windows launcher:

```bat
start_mask_tools_gui.bat
```

## Main Fields

- `Scene Directory`: base folder for quick path fill.
- `Images Directory`: input images for YOLO/SAM masking.
- `Masks Directory`: output masks; also stitch input/output.
- `YOLO Level`: forwarded to `yolo_mask.py --level` (0-3).
- `YOLO Expand (px)`: forwarded to `yolo_mask.py --expand`.
- `YOLO Add Ext`: forwarded to `yolo_mask.py --add_ext`.
- `Stitch FOV (deg)`: forwarded to `stitch_mask.py --fov`.
- `Stitch Workers`: forwarded to `stitch_mask.py --workers`.

## Actions

1. `Run YOLO Mask`: runs `yolo_mask.py`.
2. `Run Stitch Mask`: runs `stitch_mask.py` on `Masks Directory`.
3. `Run YOLO + Stitch`: runs both in sequence.

## Notes

- The GUI runs scripts as subprocesses, so behavior stays aligned with CLI.
- Logs from each step are shown in the integrated log panel.
