# STechDrive 3DGS Utils Step 3 — Mask Generation GUI

## Overview

Step 3 in STechDrive 3DGS Utils is a PySide6 wrapper around:

- `yolo_mask.py` (person mask generation)
- `stitch_mask.py` (stitch-region masking)
- `overexposure_mask.py` (overexposure masking)
- `custom_mask.py` (user-provided static mask merging)

It is intended for extracted frames or externally prepared image sequences (`images/` -> `masks/`).

## Usage

```bat
run_gui.bat --scene ./scene01
```

## Main Fields

- `Scene Folder`: base folder. Step 3 fills `images` and `masks` from it.
- `Images Folder`: input images for YOLO/SAM masking.
- `Masks Folder`: output masks; also stitch input/output.
- `Image Type`:
  - `360°`: equirectangular 360° images. Enables stitch seam masking and 360° bottom-view re-detection.
  - `Normal`: normal video frames or still-camera image sequences. Disables stitch seam masking and 360° bottom-view re-detection.
- `YOLO Level`: forwarded to `yolo_mask.py --level` (0-3).
  - For 360° images, start with `2 Quality`.
  - Use `1 Standard` for faster checks, and `3 Best` only if people still leak through.
  - For normal images, start with `1 Standard`.
- `YOLO Expand`: forwarded to `yolo_mask.py --expand`.
  - Default is `2px`; drag horizontally on the number field to adjust.
  - Clamped to `-16..32px` for safety.
- `Bottom Enhance`: preset for missed masks near the bottom of equirectangular 360° images.
  - `Standard`: use when the bottom is already masked well and you want to avoid extra floor/ground masking.
  - `High`: use when top-down photographers, tripods, or hands remain near the bottom.
  - `Max`: use only when bottom leaks remain after `High`; it is slower and more likely to mask extra floor or ground.
  - Not used when `Image Type` is `Normal`.
- `YOLO Classes`: collapsed picker for class selection.
  - Choose classes by checkbox labels (`id: name`) instead of memorizing numeric ids.
  - Default preset is `person` only (`id=0`).
  - Forwarded to `yolo_mask.py --classes`.
- `Boundary Mask Width (deg)`: forwarded to `stitch_mask.py --boundary-width`.
  - Not used when `Image Type` is `Normal`.
  - Drag horizontally on the number field to adjust.
  - The GUI clamps the value to `0.0-30.0` degrees for safety.
- `Stitch Workers`: forwarded to `stitch_mask.py --workers`.
  - Drag horizontally on the number field to adjust.
- `Custom Mask`:
  - AND-merges a user-provided static mask into every mask in `masks/` as the final step.
  - Input must be PNG. 8-bit/16-bit grayscale, RGB, and RGBA inputs are accepted. RGB/RGBA inputs are converted to grayscale and alpha is ignored.
  - 8-bit values >=128 and 16-bit values >=32768 become white during 0/255 binarization. White means keep and black means exclude.
  - The custom mask applies only to source images with matching dimensions. Mismatches are skipped without automatic resizing.
  - If every image is skipped because none match the custom mask size, the custom step fails.
  - If `Custom` is turned on before a file is selected, the file picker opens automatically. You can also select the file first with `Load`.
- `YOLO/SAM Preview`: runs `yolo_mask.py` only for the currently displayed preview image.
  - The result is shown as a red overlay and is not saved to `masks/`.
  - It uses the current `Image Type`, `YOLO Level`, `YOLO Expand`, `Bottom Enhance`, and `YOLO Classes` settings.
  - Use the main `Generate` action when you want to write masks for all frames.
- `Reprocess Current`: rebuilds and saves the mask for only the currently displayed preview image.
  - If `YOLO Detection` is enabled, it reruns YOLO/SAM for that single image. If `Stitch Seam`, `Overexposure`, or `Custom` is enabled, those masks are merged into the same output.
  - Use it to fix misses found in preview without regenerating the whole set.

## Actions

Select `YOLO Detection`, `Stitch Seam`, `Overexposure`, and/or `Custom`, then press `Generate`.
When multiple tasks are selected, they run in this order: YOLO detection, stitch seam, overexposure, custom.

If `selected_frames.csv` is not present, Step 3 can still generate masks as long as `images/` contains supported images.
In that external-image mode, Step 2 keep/drop validation is skipped.

## Notes

- The GUI runs scripts as subprocesses, so behavior stays aligned with CLI.
- Logs from each step are shown in the integrated log panel.
