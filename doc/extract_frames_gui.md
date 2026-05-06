# Step 1 Frame Extraction GUI

Step 1 turns 360° video into equirectangular still images for SfM and 3DGS. The `images/` folder and `_stechdrive/frames/selected_frames.csv` created here become the input for Step 2 review, Step 3 mask generation, and either Metashape or SphereSfM.

In the common workflow, you choose a video, choose a scene folder, and extract frames on a fixed interval. When `Motion` is enabled, the GUI can drop near-duplicate candidates and add extra candidates where viewpoint change is useful.

## Launch

```bat
run_gui.bat --scene .\scene01
```

Then open `Step 1: Frame Extraction` in the workflow sidebar.

The `?` help icon at the right edge of the center-panel header opens this step's GitHub documentation. It opens the Japanese or English page to match the active UI language.

## First Choice

| Goal | Recommended settings |
| --- | --- |
| Create normal SfM-ready frames | `Interval 1.0 sec`, `Motion ON` |
| Quickly cut frames without analysis | `Quick extract ON` |
| Walking or indoor footage with nearby structure | `Scene Distance: Near / Walking` |
| Aerial, plaza, coast, mountain, or distant scenes | `Scene Distance: Distant / Aerial` |
| Rebuild the same video with new settings | `Output Mode: Reset and Overwrite` |
| Add multiple videos into one scene | `Output Mode: Add New Only` |

The GUI stops before running when the scene folder path contains non-ASCII characters, an extremely long path, control characters, or `"`. Use a short ASCII working path because external tools often fail on problematic paths.

## Basic Flow

1. Select `Input Video`. Multiple videos can be selected.
2. Confirm `Scene Folder`. Output images are written under `images/` inside it.
3. Choose `Interval`. Start with `1.0` second when unsure.
4. Keep `Motion` on for normal extraction. Turn `Quick extract` on only when you want a fast fixed-interval cut.
5. Choose `Output Mode`. `Add New Only` is fine for the first run.
6. When the preflight status says the run is ready, press `Extract Frames`.
7. After extraction finishes, continue to Step 2.

Step 1 separates analysis from image export. Analysis uses grayscale frames resized to `Analysis Width`; files written to `images/` keep the source video resolution.

## Fixed Interval And Motion

### Fixed Interval

`Interval` is the baseline spacing between extracted candidates. At 30fps, `1.0` second means roughly one candidate every 30 frames.

Increasing the value reduces the frame count. Decreasing it increases the count. For SfM, a stable fixed cadence is easier to reason about than a fully variable extraction interval.

### Motion

`Motion` adds pair-analysis decisions on top of the fixed interval. It compares candidates with the last kept frame and can:

- drop candidates as `Drop: similar frame` when they are too redundant
- add candidates as `Added: viewpoint change` before the next fixed-cadence point
- keep safety candidates as `Added: preserved spacing` when the gap would become too large
- flag possible blur, low texture, or weak feature tracking for Step 2 review

### Quick Extract

`Quick extract` skips analysis and cuts frames directly at the requested `Interval`. It is fast, but it does not create motion-adjustment decisions or Step 2 review labels.

Use it for a fast content check or when you only need frames immediately. For production Metashape or SphereSfM input, normal extraction with `Motion` is usually safer.

## Interval Settings

| Setting | Meaning | Starting point |
| --- | --- | --- |
| `Interval` | Baseline candidate spacing | `1.0` sec |
| `Min` | Minimum spacing for inserted candidates | `0.5` sec |
| `Max` | Safety spacing so low-motion sections do not become too sparse | `2.0` sec |

If the output has too many frames, raise `Interval`. If camera motion is fast and useful viewpoints are missing, lower `Interval` or `Min`.

## Scene Distance

`Scene Distance` chooses the assumption used by the automatic motion thresholds.

- `Near / Walking`: interiors, buildings, columns, vegetation, furniture, and other nearby structure.
- `Distant / Aerial`: aerial footage, plazas, mountains, coastlines, and other distant-view scenes where the same movement creates weaker image change.

When unsure, use `Near / Walking` for walking or architectural footage and `Distant / Aerial` for drones or open outdoor scenes.

## Analysis Width And JPEG Quality

`Analysis Width` is the horizontal width used for motion analysis, blur checks, and feature tracking. Larger values can see finer detail but are slower. The default is usually sufficient.

`JPEG Quality` is ffmpeg's `-q:v` value. Lower values mean higher quality and larger files. The default `2` is high quality.

## Outputs

| Output | Meaning |
| --- | --- |
| `images/` | Extracted equirectangular still images |
| `_stechdrive/frames/selected_frames.csv` | Keep/drop candidates and analysis metadata for Step 2 |
| `_stechdrive/frames/extract_report.json` | Extraction settings and run summary |
| `extract_cache.npz` | Cache used to speed up re-analysis |

Step 2 turns `_stechdrive/frames/selected_frames.csv` decisions into visible review labels. If there are too many added, dropped, or review-target frames, adjust Step 1 interval or scene distance and extract again.

## Common Decisions

- Start with `Interval 1.0 sec`, `Min 0.5 sec`, `Max 2.0 sec`, and `Motion ON`.
- If there are too many frames, raise `Interval`.
- If many frames are similar, review examples in Step 2, then consider raising `Interval` or trying `Distant / Aerial`.
- If many frames are flagged for blur, inspect the source footage. Step 2 can still keep frames that look acceptable.
- Use `Reset and Overwrite` when rebuilding the same video with new settings.
- `Quick extract` is convenient, but normal extraction is better for production selection because it creates Step 2 review labels.
