# Step 1 Frame Extraction GUI

Step 1 turns 360° video into equirectangular still images for SfM and 3DGS. The `images/` folder and `_stechdrive/frames/selected_frames.csv` created here become the input for Step 2 review, Step 3 mask generation, and either Metashape or SphereSfM.

In the common workflow, you choose a video, choose a scene folder, and extract frames on a fixed interval. When `Motion` is enabled, the GUI can drop near-duplicate candidates and add extra candidates where viewpoint change is useful.

## Extraction Approach

This step is not meant to create as many still images as possible from a 360° video. It is a preprocessing step for creating an equirectangular image set with enough frames, but not excessive frames, for Metashape or SphereSfM.

The source footage quality matters most. If the video is strongly blurred during capture, badly exposed, low in usable features, or filmed along a poor path, frame extraction cannot fundamentally fix it. This step can only select SfM-friendly candidates from good footage and make suspicious frames easier to review.

The goal is to preserve the viewpoint change and coverage SfM needs while reducing near-duplicate frames and keeping compute cost under control. A fixed interval gives stable whole-video coverage. Motion adjustment can then drop redundant candidates in slow sections and add candidates where movement is faster.

Choosing only the sharpest nearby frame is not enough for SfM. A frame can be sharp but still too similar to the last kept frame, weak in features, or poorly overlapped for reconstruction. This step therefore considers sharpness together with change from the last kept frame, sparse feature tracking, and low-texture checks.

The design focuses expensive decisions where they matter. It does not run a high-resolution, high-cost search for the sharpest image everywhere. Instead, it starts from a fixed interval and pays extra attention to candidates that look too similar, lack viewpoint change, or may be blurred. This keeps candidate quality reviewable while helping reduce processing time, output count, and downstream SfM cost.

The main capture assumptions are walking footage and drone footage. Walking footage often contains nearby structure, so the same camera movement creates stronger parallax. Drone and distant-view footage tends to produce weaker image change for the same movement. `Scene Distance` chooses whether the motion thresholds should assume near / walking footage or distant / aerial footage.

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

The fixed interval is not the only quality decision. It is the baseline that covers the whole video consistently; motion analysis then adjusts candidates based on actual viewpoint change and blur risk.

### Motion

`Motion` adds SfM-oriented decisions on top of the fixed interval. It compares candidates with the last kept frame and looks at the remaining image change after yaw alignment, rather than treating pure camera heading changes as useful motion. It can:

- drop candidates as `Drop: similar frame` when they are too redundant
- add candidates as `Added: viewpoint change` before the next fixed-cadence point
- replace dropped blur candidates from the range up to `Max` and mark them as `Added: blur replacement`
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

`Scene Distance` chooses the assumption used by the automatic motion thresholds. It does not lock the workflow to a capture genre; it tells the analyzer how much useful image change to expect for the same camera movement.

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
- If many frames are flagged for blur, inspect the source footage first. Extraction searches nearby replacement candidates, but footage that is blurred overall cannot be fundamentally rescued. Step 2 can still keep frames that look acceptable.
- Use `Reset and Overwrite` when rebuilding the same video with new settings.
- `Quick extract` is convenient, but normal extraction is better for production selection because it creates Step 2 review labels.
