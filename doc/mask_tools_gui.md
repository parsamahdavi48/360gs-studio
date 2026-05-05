# STechDrive 3DGS Utils Step 3 — Mask Generation GUI

## Overview

Step 3 in STechDrive 3DGS Utils is a PySide6 wrapper around:

- `yolo_mask.py` (YOLO/SAM2.1 model-based mask generation)
- `sky_mask.py` (Mask2Former ADE20K or SAM3.1 prompt-based masks)
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
- `Images Folder`: input images for mask generation. Use the `+` icon at the right side of the row to copy supported images from another folder into the current scene's `images/`.
  - If `Scene Folder` is not set, the GUI asks you to choose one first.
  - Only files directly inside the selected folder are copied; subfolders are not scanned.
  - Existing filenames in `images/` are skipped and never overwritten.
  - If `selected_frames.csv` already exists, newly copied files are not added to the CSV automatically. Step 3 will report them as untracked before generation, so resolve the CSV/image mismatch or use a separate scene for external sequences.
- `Masks Folder`: output masks; also stitch input/output.
- `Image Type`:
  - `360°`: choose this when processing equirectangular 360° images.
  - `Normal`: choose this when processing normal video frames or still images.
  - Split mixed image types into separate folders and process each type separately.
- `Options`: optional mask passes added after the model-based mask.
  - `Stitch`: stitch seam masks for equirectangular 360° images.
  - `Overexp`: overexposure masks.
  - `Custom`: user-provided static PNG mask.
- Settings tabs:
  - `Mask Settings`: mask model, target classes/prompts, expansion, and projection assist.
  - `Options`: stitch seam, overexposure, and custom PNG mask settings.
- `Model`: selects the mask backend.
  - `YOLO/SAM2.1`: default path. YOLO detects people or selected classes, then SAM2.1 refines the mask.
  - `Mask2Former`: ADE20K semantic segmentation. The GUI passes selected ADE20K class names to `sky_mask.py --backend mask2former --labels ...`.
  - `SAM3.1`: prompt-based masks. If `models/sam3.1/sam3.1_multiplex.pt` is missing, the GUI can download it after Hugging Face access approval.
  - All models share the same `Quality` input-view recipe.
- `Quality`: chooses the balance between accuracy and processing time, forwarded as `--quality standard|high|best`.
  - `Standard`: direct full-image inference. For 360° images, also runs a light bottom-pole pass.
  - `High`: recommended default. Adds person-oriented tiles and, for 360° images, top/bottom projection assist.
  - `Best`: denser tiles and stronger bottom-pole settings for difficult source images.
  - Normal images use direct inference and whole-image tiling; 360° pole projection is skipped.
  - Start with `Standard` or `High`, then select only images with missed areas and regenerate them at a higher quality.
- `Mask Expand`: forwarded to the selected backend as `--expand`.
  - Positive values mask a wider area; negative values make the boundary tighter.
  - Default is `0px`; drag horizontally on the number field to adjust.
  - Clamped to `-16..32px` for safety.
- `Detection Targets`: collapsed picker for class selection in `YOLO/SAM2.1`.
  - Choose classes by checkbox labels (`id: name`) instead of memorizing numeric ids.
  - Default preset is `person` only (`id=0`).
  - Forwarded to `yolo_mask.py --classes`.
- `Detection Targets`: collapsed picker used by `Mask2Former`.
  - Defaults to `sky` and `person`.
  - Multiple classes are resolved in one model inference and merged into the output mask.
- `Detection Targets`: SAM3.1 prompt presets and a custom English prompt field.
  - Defaults to `person` and `sky`.
  - Multiple prompts are run one at a time and OR-merged into the output mask.
  - Custom prompt fields accept comma or semicolon separators. Spaces around separators are ignored; spaces inside prompts are kept.
  - The subtract prompt field removes matching SAM3.1 detections from the positive prompt result.
- `Op.` is SAM3.1-only and appears in the same row as `Qual.` and `Exp.`. `Replace` rewrites the mask, `Add` blackens detected regions in the existing mask, and `Subtract` turns detected regions white in the existing mask.
- `Inference Size`: larger values can improve detail and boundaries but use more GPU memory and time. Only Mask2Former can be changed in the GUI; YOLO/SAM2.1 and SAM3.1 use fixed processing sizes.
- `Model Details`: contains Mask2Former-specific `Min Score` (`0.00-1.00`, `0` disables it).
- `Sky Mask`: contains sky-only filters.
  - `Min Area` removes small sky components by image-area ratio.
  - `Top edge only` keeps only sky components touching the top image edge.
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
  - If `Custom` is turned on before a file is selected, the file picker opens automatically. You can also select a file with the file icon at the right side of the row, or clear it with the delete icon.
- `Mask Preview` button: builds a temporary mask for the displayed image using the current model and enabled extra masks.
  - Existing files in `masks/` are not used as the base, except in SAM3.1 `Add`/`Subtract` mode where the saved mask is copied into the temporary preview so the correction can be inspected.
  - The result is shown as a red overlay and is not saved to `masks/`.
  - In thumbnail mode, it switches the currently selected image to single-preview mode and shows the temporary result there.
  - `Show Preview` toggles between the generated temporary preview and the saved mask in `masks/` without deleting the temporary preview.
  - The first run with a third-party model shows the relevant model/license notice.
- `Regenerate Mask`: rebuilds and saves the mask for only the currently displayed preview image.
  - It reruns the selected model for that single image. If `Stitch`, `Overexp`, or `Custom` is enabled, those masks are merged into the same output.
  - In SAM3.1 `Add`/`Subtract` mode, it applies the current prompt result to the existing saved mask instead of replacing the whole mask.
  - Results from steps that are now off are not kept, so use it to fix misses found in preview without regenerating the whole set.
- `Mask Preview`:
  - Use the icons at the right side of the preview header to switch between single preview and thumbnail list.
  - Thumbnail mode shows existing masks as red overlays. The mask display icon toggles the overlay on and off without rebuilding thumbnails.
  - Double-clicking a thumbnail returns to single preview on that image.
  - `Ctrl` click, `Ctrl+Shift` click, and `Shift` click follow Windows Explorer-style multi-selection.
  - Switching to thumbnail mode focuses the thumbnail list, so arrow keys move the visible thumbnail selection.
  - In thumbnail mode, `Regenerate N Masks` rebuilds only the selected images with the current mask-generation settings and saves them to `masks/`.
  - The status text stays on one line and elides when space is tight; the full text is available as a tooltip.
  - Thumbnail rendering is lazy, prioritizes the visible rows, and reuses cached thumbnails across step switches so large image sets do not rebuild in full on every view change.

## Actions

Choose the mask model in `Mask Settings`, enable any `Options`, then press `Generate`.
The model-based mask always runs first. Extra masks run in this order: stitch seam, overexposure, custom.
Existing masks are regenerated from the current model and enabled extra masks. Results from extra masks that are now off are not kept.

If `selected_frames.csv` is not present, Step 3 can still generate masks as long as `images/` contains supported images.
In that external-image mode, Step 2 keep/drop validation is skipped.
If `selected_frames.csv` is present, Step 3 validates `images/` against it and stops when pending drops or untracked files are found.

## Notes

- The GUI runs scripts as subprocesses, so behavior stays aligned with CLI.
- Semantic/person masking uses third-party libraries and model weights with separate license terms. See [../THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md).
- Logs from each step are shown in the integrated log panel.
