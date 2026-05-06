# Step 3 Mask Generation GUI

Step 3 creates `masks/` from the `images/` kept in Step 2. These masks exclude regions you do not want SfM or 3DGS to use, such as people, the camera operator, tripods, hands, vehicles, sky, stitch seams, and blown-out highlights.

Mask polarity is **white = use, black = exclude**. Creating masks before Metashape helps avoid unwanted features from people, sky, or other unstable regions.

## Launch

```bat
run_gui.bat --scene .\scene01
```

Then open `Step 3: Mask Generation` in the workflow sidebar.

## First Choice

| Goal | Recommended settings |
| --- | --- |
| Mask people or the camera operator in 360° frames | `Image Type: 360°`, `Model: YOLO/SAM2.1`, `Quality: High` |
| Get higher-accuracy person or sky masks | `Model: SAM3.1` |
| Try sky masks without SAM3.1 | `Model: Mask2Former`, target `sky` |
| Process normal photos or normal video frames | `Image Type: Normal` |
| Exclude visible nearby stitch seams | `Stitch` ON |
| Exclude blown-out windows or lights | `Overexp` ON |
| Apply your own fixed mask to every image | `Custom` ON |

When unsure, start with `YOLO/SAM2.1` + `High` + `person` for 360° images, then regenerate only problem images with higher quality or a different model.

## Basic Flow

1. Confirm that `Images Folder` points to the scene `images/`.
2. Choose `Image Type`: `360°` for equirectangular images or `Normal` for normal photos/video frames.
3. Choose the mask `Model` and detection targets.
4. Enable `Stitch`, `Overexp`, or `Custom` only when needed.
5. Run `Mask Preview` on one image and inspect the red overlay.
6. If the preview looks reasonable, press `Generate` for the full set.
7. Scan the thumbnail list for missed regions or false positives.
8. Select only problem images, adjust settings, and run `Regenerate Mask`.

If `_stechdrive/frames/selected_frames.csv` exists, Step 3 validates it against `images/`. If pending drops or untracked images remain, apply Step 2 decisions or resolve the mismatch before generating masks.

## Model Choice

| Model | Best for |
| --- | --- |
| `YOLO/SAM2.1` | Fast default person masking |
| `SAM3.1` | Higher-accuracy prompt-based masks for people, sky, tripods, hands, phones, and cleanup |
| `Mask2Former` | Trying semantic masks such as sky without SAM3.1 |

### YOLO/SAM2.1

This is the default fast route. YOLO detects the selected targets, then SAM2.1 refines their shapes. Start with `person` only. Add vehicle or other classes only when you also want to remove them.

### SAM3.1

SAM3.1 uses English prompts. You can add missed targets such as `person`, `sky`, `tripod`, `hand`, `selfie stick`, or `cell phone`, and subtract false positives such as `logo` or `sign`.

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

With SAM3.1, choose the write mode:

- `Replace`: rebuild the mask from current settings
- `Add`: add detected regions to the existing mask as black excluded areas
- `Subtract`: remove detected regions from the existing mask by turning them white

It is usually faster to generate the full set at Standard/High quality, then regenerate only images with visible misses.

## Outputs

| Output | Meaning |
| --- | --- |
| `masks/` | Mask images corresponding to `images/` |
| Temporary preview | Visual check only; not a saved mask |
| Run log | Shown in the integrated log panel |

Generated `masks/` can be imported into Metashape or converted with projection views in Step 4. Polarity remains white = use, black = exclude.

## Common Decisions

- Start with `YOLO/SAM2.1` + `High` + `person`.
- Use `SAM3.1` when you also need sky, tripods, hands, or prompt-based cleanup.
- Use `Mask2Former` for a quick sky-mask test without SAM3.1.
- If the camera operator or tripod remains near the bottom of a 360° image, raise quality or use SAM3.1 prompts such as `tripod`, `hand`, or `selfie stick`.
- Turn `Overexp` on only for footage where blown-out areas are actually harmful; it can remove too much in some interiors.
- Normal images do not use stitch seam masks or 360° pole projection assist.
- If SAM3.1 stops due to GPU memory, completed masks remain saved. Rerun with the same settings to continue from unfinished images.

## Notes

- The GUI runs the CLI scripts as subprocesses, so behavior stays aligned with CLI.
- Semantic/person masking uses third-party libraries and model weights with separate license terms. See [../THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md).
