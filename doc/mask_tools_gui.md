# Step 3 Mask Generation GUI

Step 3 creates `masks/` from the `images/` kept in Step 2. These masks exclude regions you do not want SfM or 3DGS to use, such as people, the camera operator, tripods, hands, vehicles, sky, stitch seams, and blown-out highlights.

Mask polarity is **white = use, black = exclude**. Creating masks before Metashape, COLMAP, SphereSfM, or another SfM route helps avoid unwanted features from people, sky, or other unstable regions.

## First Choice

| Goal | Recommended settings |
| --- | --- |
| Mask people or the camera operator in 360° frames | Confirm `Image Type: 360°`, then use `Model: YOLO/SAM2.1`, `Quality: High` |
| Get higher-accuracy person or sky masks | `Model: SAM3.1` |
| Try sky masks without SAM3.1 | `Model: Mask2Former`, target `sky` |
| Process normal photos or normal video frames | Add or copy them into the scene and confirm `Image Type: Normal` |
| Process only the source you just added | Choose that video or still folder in `Source` |
| Keep existing masks untouched | Choose `Scope: Images without masks only` |
| Rebuild only masks affected by changed settings | Choose `Scope: Missing + changed settings` |
| Exclude visible nearby stitch seams | `Stitch` ON |
| Exclude blown-out windows or lights | `Overexp` ON |
| Apply your own fixed mask to every image | `Custom` ON |

When unsure, start with `YOLO/SAM2.1` + `High` + `person` for 360° images, then regenerate only problem images with higher quality or a different model.

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
| `YOLO/SAM2.1` | Recommended route for fast person masking |
| `SAM3.1` | Higher-accuracy English-prompt masks for people, sky, tripods, hands, phones, and cleanup |
| `Mask2Former` | Trying semantic masks such as sky without SAM3.1 |

### YOLO/SAM2.1

This is the recommended route when you want fast person masks. YOLO detects the selected targets, then SAM2.1 refines their shapes. The usual target is `person` only. Add vehicles or other available targets only when you also want to mask them.

### SAM3.1

SAM3.1 is prompt-driven rather than a fixed class list. You describe what to mask with English prompts. Single words such as `person`, `sky`, or `tripod` work, and short natural-language phrases such as `selfie stick`, `cell phone`, `person wearing a red jacket`, `hand holding a phone`, or `tripod legs near the floor` can also be used. The checkboxes are shortcuts for common prompts, so you do not have to type them manually. Add any missing targets in the extra prompt field.

After generation, select only images with misses or false detections and combine prompts with the mask regeneration mode to add regions to the current mask or remove unwanted targets such as `logo` or `sign` from the masked area.

Use add prompts to pick up targets that were not masked. For example, enter `tripod legs near the floor` for a missed tripod near the feet, or `hand holding a phone` for a missed phone, set the mode to `Add`, and regenerate only the affected images. Use subtract prompts when areas you do not want masked were picked up by mistake. For example, enter `male icon`, `female icon`, `logo`, or `sign`, set the mode to `Subtract`, and remove those targets from the masked area.

On first use, if `models/sam3.1/sam3.1_multiplex.pt` is missing, Hugging Face access approval and SAM License acceptance are required. GUI downloads use a `Read` token from the approved account. This app does not save the token.

### Mask2Former

Mask2Former uses ADE20K semantic classes. Use it when you want to try semantic targets such as sky without setting up SAM3.1.

## Quality And Expansion

`Quality` controls the balance between accuracy and processing time.

| Quality | Use when |
| --- | --- |
| `Standard` | Quick checks. Also a reasonable starting point for normal images |
| `High` | Recommended starting point for 360° images. Adds person-oriented tiles and top/bottom assist |
| `Best` | Targeted fixes for images that still leak. Slower |

`Mask Expand` grows or shrinks mask boundaries. Positive values make black excluded regions larger; negative values make them tighter. Increase it when silhouettes leak through, and decrease it when masks remove too much.

## Optional Masks

After the model-based mask, Step 3 can merge extra masks. The order is always model -> stitch seam -> overexposure -> custom.

| Option | Use when |
| --- | --- |
| `Stitch` | Excluding stitch seams in 360° images |
| `Overexp` | Excluding blown-out windows, lights, or reflections |
| `Custom` | AND-merging your own PNG mask into every output |

If stitch seams are barely visible, keep stitch masks off or use a narrow seam mask first. Usually keep it off for stabilized, direction-locked, or AI-stitched footage where seam positions move.

Custom masks use PNG input. White means use, black means exclude. They apply only to images with matching dimensions.

## Preview And Regeneration

### Mask Preview

`Mask Preview` builds a temporary mask for the currently displayed image using the current settings. It does not save to `masks/`. Use it first to check missed regions or over-masking.

`Show Preview` switches between the temporary preview and the saved mask display. For 360° images, perspective preview makes it easier to inspect feet, tripods, silhouettes, and stitch-seam regions.

### Regenerate Mask

`Regenerate Mask` saves a new mask to `masks/` for the current image, or for selected images in thumbnail mode.

With SAM3.1, choose how prompt detections are applied to the saved mask:

- `Replace`: rebuild the mask from current settings
- `Add`: add detected targets to the masked area
- `Subtract`: remove detected targets from the masked area

It is usually faster to generate the full set at Standard/High quality, then regenerate only images with visible misses.

## Outputs

| Output | Meaning |
| --- | --- |
| `masks/` | Mask images corresponding to `images/` |
| Temporary preview | Visual check only; not a saved mask |
| Run log | Shown in the integrated log panel |

Generated `masks/` can be imported into Metashape, COLMAP, or SphereSfM, and Step 5 converts them to match training images when creating datasets. Polarity remains white = use, black = exclude.

## Common Decisions

- Start with `YOLO/SAM2.1` + `High` + `person`.
- Use `SAM3.1` when you also need sky, tripods, hands, or prompt-based cleanup.
- Use `Mask2Former` for a quick sky-mask test without SAM3.1.
- If the camera operator or tripod remains near the bottom of a 360° image, raise quality or use SAM3.1 prompts such as `tripod`, `hand`, or `selfie stick` to add those areas to the mask.
- Turn `Overexp` on only for footage where blown-out areas are actually harmful; it can remove too much in some interiors.
- Normal images do not use stitch seam masks or 360° pole projection assist.
- Use `Images without masks only` when you want to keep masks you already reviewed.
- Use `Missing + changed settings` after changing mask settings and updating only affected images.
- After adding a new source, choose that source in `Source` to keep the run and review focused.
- If SAM3.1 stops due to GPU memory, completed masks remain saved. Rerun with the same settings to continue from unfinished images.

## Notes

- Semantic/person masking uses third-party libraries and model weights with separate license terms. See [../THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md).
