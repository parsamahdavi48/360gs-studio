# STechDrive 3DGS Utils Step 3 — Mask Generation GUI

## Overview

Step 3 in STechDrive 3DGS Utils is a PySide6 wrapper around:

- `yolo_mask.py` (person mask generation)
- `sky_mask.py` (Mask2Former ADE20K sky masking)
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
- Settings tabs:
  - `YOLO`: YOLO/SAM strength, expansion, bottom enhancement, and detection classes.
  - `Stitch/Overexp.`: stitch seam and overexposure mask settings.
  - `Sky`: Mask2Former sky detection settings.
  - `Custom Mask`: load or clear the user-provided PNG mask.
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
- `Sky`:
  - Runs `sky_mask.py` and AND-merges detected sky into `masks/`.
  - `Hybrid` is the default for 360° images. It combines direct detection and a top projection view.
  - Increase `Size` for more detailed inference; use `Min Score`, `Min Area`, `Expand`, and `Top-connected only` when tuning false positives or missed sky.
- `Custom Mask`:
  - AND-merges a user-provided static mask into every mask in `masks/` as the final step.
  - Input must be PNG. 8-bit/16-bit grayscale, RGB, and RGBA inputs are accepted. RGB/RGBA inputs are converted to grayscale and alpha is ignored.
  - 8-bit values >=128 and 16-bit values >=32768 become white during 0/255 binarization. White means keep and black means exclude.
  - The custom mask applies only to source images with matching dimensions. Mismatches are skipped without automatic resizing.
  - If every image is skipped because none match the custom mask size, the custom step fails.
  - If `Custom` is turned on before a file is selected, the file picker opens automatically. You can also select the file first with `Load`.
- `Mask Preview` button: builds a temporary mask for the displayed image using the currently enabled `YOLO`, `Stitch`, `Overexp`, `Sky`, and `Custom` steps.
  - Existing files in `masks/` are not used as the base, so results from steps that are now off are not mixed into the preview.
  - The result is shown as a red overlay and is not saved to `masks/`.
  - In thumbnail mode, it switches the currently selected image to single-preview mode and shows the temporary result there.
  - While a temporary preview is active, the button changes to `Clear Preview`. Press it to discard the temporary preview and return to the saved mask display.
  - If `YOLO` or `Sky` is enabled, the first run shows the relevant third-party model/license notice.
- `Regenerate Current`: rebuilds and saves the mask for only the currently displayed preview image.
  - If `YOLO` or `Sky` is enabled, it reruns the matching model step for that single image. If `Stitch`, `Overexp`, or `Custom` is enabled, those masks are merged into the same output.
  - Results from steps that are now off are not kept, so use it to fix misses found in preview without regenerating the whole set.
- `Mask Preview`:
  - Use the icons at the right side of the preview header to switch between single preview and thumbnail list.
  - Thumbnail mode shows existing masks as red overlays. The mask display icon toggles the overlay on and off without rebuilding thumbnails.
  - Double-clicking a thumbnail returns to single preview on that image.
  - `Ctrl` click, `Ctrl+Shift` click, and `Shift` click follow Windows Explorer-style multi-selection.
  - Switching to thumbnail mode focuses the thumbnail list, so arrow keys move the visible thumbnail selection.
  - In thumbnail mode, `Regenerate N Selected` rebuilds only the selected images with the current mask-generation settings and saves them to `masks/`.
  - The status text stays on one line and elides when space is tight; the full text is available as a tooltip.
  - Thumbnail rendering is lazy, prioritizes the visible rows, and reuses cached thumbnails across step switches so large image sets do not rebuild in full on every view change.

## Actions

Select `YOLO`, `Stitch`, `Overexp`, `Sky`, and/or `Custom`, then press `Generate`.
When multiple tasks are selected, they run in this order: YOLO, stitch seam, overexposure, sky, custom.
Existing masks are regenerated from the currently enabled steps. Results from steps that are now off are not kept.

If `selected_frames.csv` is not present, Step 3 can still generate masks as long as `images/` contains supported images.
In that external-image mode, Step 2 keep/drop validation is skipped.

## Notes

- The GUI runs scripts as subprocesses, so behavior stays aligned with CLI.
- YOLO/SAM and sky masking use third-party libraries and model weights with separate license terms. See [../THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md).
- Logs from each step are shown in the integrated log panel.
