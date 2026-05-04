# STechDrive 3DGS Utils Step 3 — Mask Generation GUI

## Overview

Step 3 in STechDrive 3DGS Utils is a PySide6 wrapper around:

- `yolo_mask.py` (YOLO/SAM2.1 primary mask generation)
- `sky_mask.py` (Mask2Former ADE20K or local SAM3.1 prompt-based primary masks)
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
- `Images Folder`: input images for primary and extra mask generation.
- `Masks Folder`: output masks; also stitch input/output.
- `Image Type`:
  - `360°`: equirectangular 360° images. Enables stitch seam masking and 360° projection assist.
  - `Normal`: normal video frames or still-camera image sequences. Disables stitch seam masking and 360° pole projection assist.
- `Extra Masks`: optional mask passes added after the primary mask.
  - `Stitch`: stitch seam masks for equirectangular 360° images.
  - `Overexp`: overexposure masks.
  - `Custom`: user-provided static PNG mask.
- Settings tabs:
  - `Mask Settings`: primary model, target classes/prompts, expansion, and projection assist.
  - `Stitch/Overexp.`: stitch seam and overexposure mask settings.
  - `Custom Mask`: load or clear the user-provided PNG mask.
- `Model`: selects the primary mask backend.
  - `YOLO/SAM2.1`: default path. YOLO detects people or selected classes, then SAM2.1 refines the mask.
  - `Mask2Former`: ADE20K semantic segmentation. The GUI passes selected ADE20K class names to `sky_mask.py --backend mask2former --labels ...`.
  - `SAM3.1`: local prompt path. It runs `sky_mask.py --backend sam31` when `models/sam3.1/sam3.1_multiplex.pt` exists.
  - All primary models share the same `Quality` input-view recipe.
- `Quality`: forwarded as `--quality standard|high|best`.
  - `Standard`: direct full-image inference. For 360° images, also runs a light bottom-pole pass.
  - `High`: recommended default. Adds person-oriented tiles and, for 360° images, top/bottom projection assist.
  - `Best`: denser tiles and stronger bottom-pole settings for difficult source images.
  - Normal images use direct inference and whole-image tiling; 360° pole projection is skipped.
- `Mask Expand`: forwarded to the selected primary backend as `--expand`.
  - Default is `0px`; drag horizontally on the number field to adjust.
  - Clamped to `-16..32px` for safety.
- `Detection Classes`: collapsed picker for class selection in `YOLO/SAM2.1`.
  - Choose classes by checkbox labels (`id: name`) instead of memorizing numeric ids.
  - Default preset is `person` only (`id=0`).
  - Forwarded to `yolo_mask.py --classes`.
- `ADE20K Class List`: collapsed picker used by `Mask2Former`.
  - Defaults to `sky` and `person`.
  - Multiple classes are resolved in one model inference and merged into the output mask.
- `Detection Targets`: SAM3.1 prompt presets and a custom English prompt field.
  - Defaults to `person` and `sky`.
  - Multiple prompts are run one at a time and OR-merged into the output mask.
- `Inference Size`: controls Mask2Former inference size. SAM3.1 currently uses fixed `1008`.
- `Model Details`: contains Mask2Former-specific `Min Score` (`0.00-1.00`, `0` disables it).
- `Sky Postprocess`: contains sky-only filters.
  - `Min Area` removes small sky components by image-area ratio.
  - `Top-connected sky only` keeps only sky components touching the top image edge.
  - These filters apply only to sky labels/prompts, not to people or other selected targets.
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
- `Mask Preview` button: builds a temporary mask for the displayed image using the current primary model and enabled extra masks.
  - Existing files in `masks/` are not used as the base, so results from steps that are now off are not mixed into the preview.
  - The result is shown as a red overlay and is not saved to `masks/`.
  - In thumbnail mode, it switches the currently selected image to single-preview mode and shows the temporary result there.
  - While a temporary preview is active, the button changes to `Clear Preview`. Press it to discard the temporary preview and return to the saved mask display.
  - The first run with a third-party model shows the relevant model/license notice.
- `Regenerate Current`: rebuilds and saves the mask for only the currently displayed preview image.
  - It reruns the selected primary model for that single image. If `Stitch`, `Overexp`, or `Custom` is enabled, those masks are merged into the same output.
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

Choose the primary model in `Mask Settings`, enable any `Extra Masks`, then press `Generate`.
The primary mask always runs first. Extra masks run in this order: stitch seam, overexposure, custom.
Existing masks are regenerated from the current primary model and enabled extra masks. Results from extra masks that are now off are not kept.

If `selected_frames.csv` is not present, Step 3 can still generate masks as long as `images/` contains supported images.
In that external-image mode, Step 2 keep/drop validation is skipped.

## Notes

- The GUI runs scripts as subprocesses, so behavior stays aligned with CLI.
- Primary semantic/person masking uses third-party libraries and model weights with separate license terms. See [../THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md).
- Logs from each step are shown in the integrated log panel.
