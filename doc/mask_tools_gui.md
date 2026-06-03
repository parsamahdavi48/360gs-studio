# Step 3 Mask Generation GUI

Step 3 creates `masks/` from the `images/` kept in Step 2. These masks exclude regions you do not want SfM or 3DGS to use, such as people, the camera operator, tripods, hands, vehicles, sky, stitch seams, and blown-out highlights.

Mask polarity is **white = use, black = exclude**. Creating masks before Metashape, COLMAP, SphereSfM, or another SfM route helps avoid unwanted features from people, sky, or other unstable regions.

## First Choice

| Goal | Recommended settings |
| --- | --- |
| Mask people or the camera operator in 360° frames | Confirm `Image Type: 360°`, then use `Model: YOLO/SAM2.1`, `Quality: High` |
| Get higher-accuracy person or sky masks | `Model: SAM3.1` |
| Mask sky, roads, vehicles, vegetation, or other urban street-scene classes | `Model: YOLO26-sem`, choose Cityscapes targets |
| Process normal photos or normal video frames | Add or copy them into the scene and confirm `Image Type: Normal` |
| Process only the source you just added | Choose that video or still folder in `Source` |
| Keep existing masks untouched | Choose `Scope: Images without masks only` |
| Rebuild only masks affected by changed settings | Choose `Scope: Missing + changed settings` |
| Exclude visible nearby stitch seams | `Stitch` ON |
| Exclude blown-out windows or lights | `Overexp` ON |
| Apply your own fixed mask to every image | `Custom` ON |

For urban roadside 360° images, start with `YOLO26-sem` + `High` and the Cityscapes targets you need. Keep `YOLO/SAM2.1` available when you specifically want fast person-focused COCO object detection with SAM2.1 refinement.

## Basic Flow

1. Confirm that `Images Folder` points to the scene `images/`.
2. Check `Source`. Leave it on `All` for normal runs. Choose a specific video or still folder when you only want to process newly added material.
3. Choose `Scope`. Use `Images without masks only` to preserve existing masks, `Missing + changed settings` to update only images affected by setting changes, and `Regenerate all images` only when you intentionally want a full rebuild.
4. Confirm the `Image Type` status. Step 3 detects it from Step 1 records, external image registration, or image headers, and splits mixed inputs by image type automatically.
5. Choose the mask `Model` and detection targets.
6. Enable `Stitch`, `Overexp`, or `Custom` only when needed.
7. Run `Mask Preview` on one image and confirm that the regions to be masked are shown with a red overlay.
8. If the preview looks reasonable, press `Generate` for the selected range.
9. Scan the thumbnail list for missed regions or false positives.
10. Select only problem images, adjust settings, and run `Regenerate Mask`.

Step 3 stops before running if images marked for removal still remain, or if unrelated images are mixed into the scene. Apply Step 2 decisions first, or clean up `images/` so it contains only the images you want to mask.

## Model Choice

| Model | Best for |
| --- | --- |
| `YOLO/SAM2.1` | Legacy COCO object detection refined with SAM2.1, mainly for people |
| `YOLO26-sem` | Urban street scenes. Fast Cityscapes semantic masks for sky, roads, buildings, vehicles, and vegetation |
| `SAM3.1` | Higher-accuracy English-prompt masks for people, sky, tripods, hands, phones, and cleanup |

### YOLO/SAM2.1

This is the recommended route when you want fast person masks. YOLO detects the selected targets, then SAM2.1 refines their shapes. The usual target is `person` only. Add vehicles or other available targets only when you also want to mask them.

### YOLO26-sem

YOLO26-sem uses the 19-class semantic model trained on Cityscapes for urban street scenes. It predicts a class ID for each pixel, so the final black/white mask can be built directly without a separate SAM refinement step. It can quickly create mask candidates for classes such as sky, roads, buildings, vehicles, and vegetation. Turn on `vegetation` when moving trees or leaves are hurting SfM; leave static classes such as `building`, `road`, and `sidewalk` off unless you intentionally want to remove them.

For 360° images, the quality setting controls projection assist. High quality combines direct equirectangular inference with top and bottom views to reduce misses caused by ERP distortion near the vertical extremes.

### SAM3.1

SAM3.1 is prompt-driven rather than a fixed class list. You describe what to mask with English prompts. Single words such as `person`, `sky`, or `tripod` work, and short natural-language phrases such as `selfie stick`, `cell phone`, `person wearing a red jacket`, `hand holding a phone`, or `tripod legs near the floor` can also be used. The checkboxes are shortcuts for common prompts, so you do not have to type them manually. Add any missing targets in the extra prompt field.

After generation, select only images with misses or false detections and combine prompts with Method to add regions to the current mask, or exclude unwanted targets such as `logo` or `sign` inside SAM3.1 candidate generation.

Use add prompts to pick up targets that were not masked. For example, enter `tripod legs near the floor` for a missed tripod near the feet, or `hand holding a phone` for a missed phone, set Method to `Add`, and regenerate only the affected images. Use exclude prompts to remove targets from SAM3.1's candidate mask. For example, enter `male icon`, `female icon`, `logo`, or `sign` when those should not be part of the generated candidate.

On first use, if `models/sam3.1/sam3.1_multiplex.pt` is missing, Hugging Face access approval and SAM License acceptance are required. GUI downloads use a `Read` token from the approved account. This app does not save the token.

## Quality And Expansion

`Quality` controls the balance between accuracy and processing time.

| Quality | Use when |
| --- | --- |
| `Standard` | Quick checks. Also a reasonable starting point for normal images |
| `High` | Recommended starting point for 360° images. Adds ERP tiles and top/bottom projection assist |
| `Best` | Targeted fixes for images that still leak. Slower |

`Mask Expand` grows or shrinks mask boundaries. Positive values make black excluded regions larger; negative values make them tighter. Increase it when silhouettes leak through, and decrease it when masks remove too much.

## Method

`Method` chooses how the candidate mask from this run is applied to existing masks.

| Method | Behavior |
| --- | --- |
| `Replace` | Rebuild the mask without using the existing mask |
| `Add` | Add detected regions as exclusions to the existing mask |
| `Restore` | Turn detected regions back to white in the existing mask |

SAM3.1 add/exclude prompts are internal instructions for building the candidate mask. `Method` is the global setting that decides how that candidate is applied to the saved mask.

## Optional Masks

After the model-based mask, Step 3 can merge extra masks. The order is always model -> stitch seam -> overexposure -> custom.

| Option | Use when |
| --- | --- |
| `Stitch` | Excluding stitch seams in 360° images |
| `Overexp` | Excluding blown-out windows, lights, or reflections |
| `Custom` | Applying your own PNG mask to every output |

If stitch seams are barely visible, keep stitch masks off or use a narrow seam mask first. Usually keep it off for stabilized, direction-locked, or AI-stitched footage where seam positions move.

Custom masks use PNG input. White means use, black means exclude. They apply only to images with matching dimensions.

## Preview And Regeneration

### Mask Preview

`Mask Preview` builds a temporary mask for the currently displayed image using the current settings. It does not save to `masks/`. Use it first to check missed regions or over-masking.

`Show Preview` switches between the temporary preview and the saved mask display. For 360° images, perspective preview makes it easier to inspect feet, tripods, silhouettes, and stitch-seam regions.

### Regenerate Mask

`Regenerate Mask` saves a mask to `masks/` for the current image, or for selected images in thumbnail mode, using the current settings and Method.

It is usually faster to generate the full set at Standard/High quality, then regenerate only images with visible misses.

## Outputs

| Output | Meaning |
| --- | --- |
| `masks/` | Mask images corresponding to `images/` |
| Temporary preview | Visual check only; not a saved mask |
| Run log | Shown in the integrated log panel |

Generated `masks/` can be imported into Metashape, COLMAP, or SphereSfM, and Step 5 converts them to match training images when creating datasets. Polarity remains white = use, black = exclude.

## Common Decisions

- In urban street scenes, start with `YOLO26-sem` + `High` for the Cityscapes targets you need, such as sky, vehicles, and vegetation.
- Keep `YOLO/SAM2.1` for the legacy COCO person/object route when it works better on a specific scene.
- Use `SAM3.1` when you need tripods, hands, selfie sticks, phones, or prompt-based cleanup.
- If the camera operator or tripod remains near the bottom of a 360° image, raise quality or use SAM3.1 prompts such as `tripod`, `hand`, or `selfie stick` to add those areas to the mask.
- Turn `Overexp` on only for footage where blown-out areas are actually harmful; it can remove too much in some interiors.
- Normal images do not use stitch seam masks or 360° pole projection assist.
- Use `Images without masks only` when you want to keep masks you already reviewed.
- Use `Missing + changed settings` after changing mask settings and updating only affected images.
- After adding a new source, choose that source in `Source` to keep the run and review focused.
- If SAM3.1 stops due to GPU memory, completed masks remain saved. Rerun with the same settings to continue from unfinished images.

## Notes

- Semantic/person masking uses third-party libraries and model weights with separate license terms. See [../THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md).
