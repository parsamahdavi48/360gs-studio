# Frame Extraction GUI Parameters

This GUI extracts equirectangular still images from 360-degree video for Metashape SfM. The current extraction baseline is `Fixed Interval`. When `Motion` is enabled, the extractor keeps the fixed cadence but skips low-change candidates and inserts extra candidates in high-motion ranges.

## Assumptions

- Analysis and image export are separate phases.
- Analysis uses grayscale frames scaled to `Analysis Width`.
- Images written to `images/` keep the source video resolution.
- `Instant Estimate` is the fixed-interval baseline count. With `Motion` enabled, the final count can increase or decrease after analysis.
- With `Quick extract` enabled, the GUI skips analysis and writes images at the specified interval quickly. Motion adjustment and pair-analysis review metadata are not produced.
- The integrated GUI stops before running when the scene folder path contains non-ASCII characters, an extremely long path, control characters, or `"`. Use a short ASCII working path.

## Fixed Interval

### `Interval`

- Unit: seconds
- Meaning: baseline spacing between extracted frame candidates
- Internal calculation: `step = round(interval_sec * fps)`
- UI range: `0.05-60.0` seconds
- Control: horizontal drag on the numeric field
- Example: at 30fps, `1.0` seconds is about every 30 frames

Increasing the value reduces the baseline count. Decreasing it increases the count. For 3DGS-oriented SfM, a stable fixed cadence is easier to reason about than a fully variable interval.

### `Quick extract`

Use this fixed-interval option when you want to skip analysis and cut the video quickly at the requested `Interval`. Step 2 does not receive extra quality-review flags from automatic scoring.

Enabling it turns `Motion` off. Turn it off again when you want motion adjustment.

## Motion Adjustment

`Motion` is an additional option for fixed interval extraction. In the GUI, it uses pair analysis against the last kept frame. The fixed cadence remains the baseline, then analysis applies:

- Yaw-compensated residual: estimate the horizontal shift in the 360 equirectangular frame and measure the residual after removing changes explained by pure yaw.
- Redundant drops: fixed-cadence candidates below the `drop` residual threshold are marked `redundant_drop`.
- Novelty additions: frames before the next fixed-cadence point can be added as `novelty_added` when residual change reaches the `add` threshold.
- Safety keeps: if the kept-frame gap reaches `Max`, the candidate is kept as `gap_forced`.
- Candidate pair tracking: sparse feature tracking runs only at keep/drop decision points. Weak pairs are flagged as `weak_match` for Step 2.
- Candidate risk checks: sharpness is measured only at keep/drop decision points. Sudden sharpness collapse is flagged as `motion_blur`; persistently weak low-texture candidates are flagged as `low_texture` for Step 2.

### `Min`

- Unit: seconds
- Meaning: minimum spacing for inserted candidates
- Internal calculation: `min_gap_frames = round(min_gap_sec * fps)`
- Default: `0.5` seconds

Extra candidates are never inserted closer than this, even when motion is high.

### `Max`

- Unit: seconds
- Meaning: safety spacing for low-change skipping
- Internal calculation: `max_gap_frames = round(max_gap_sec * fps)`
- Default: `2.0` seconds

Low-change skipping will keep a candidate when dropping it would make the kept-frame gap too large.

## Analysis Width

### `Analysis Width`

- Unit: pixels
- Meaning: horizontal decode width used for yaw-compensated residuals, candidate pair tracking, and candidate-only blur/low-texture checks
- Default: `1920`

Higher values can improve fine-feature detection but increase analysis time. Lower values are faster but can miss subtle motion and feature detail. Yaw-compensated residual monitoring is internally capped to a `1280px` gate width, while feature tracking and blur checks use this analysis width.

## Pair-Analysis Review Metadata

The GUI's normal extraction path uses pair analysis instead of scoring individual frames in isolation. For carefully captured video, the useful question is less "which individual frame is prettier?" and more "does this pair still overlap while adding non-redundant viewpoint change?"

The CSV records these Step 2 review fields:

- `residual_score`: residual change after yaw compensation
- `raw_change_score`: luma difference before yaw compensation
- `yaw_shift_deg`: estimated horizontal yaw adjustment
- `track_count`: tracked feature count for the candidate pair
- `track_coverage`: screen coverage of tracked points
- `match_confidence`: review confidence from tracked count and coverage
- `blur_score_final`: candidate-only sharpness value
- `sharpness_ratio`: ratio against the recent kept-frame sharpness median
- `pair_gate_width`: internal gate width used for yaw-compensated residuals
- `pair_drop_threshold` / `pair_add_threshold`: interval-derived thresholds for redundant drops and novelty additions

Use `Scene Distance` in Advanced Settings to choose the assumption behind the automatic thresholds.

- `Near / Walking`: for scenes with nearby structures such as buildings, interiors, columns, or vegetation. Its 1.0-second reference thresholds are `drop=0.035` and `add=0.090`.
- `Distant / Aerial`: for aerial, plaza, mountain, coast, or distant-view scenes. Distant scenes often have weaker residual parallax because features are farther away, so this profile uses lower reference thresholds.

When the interval changes, thresholds scale gently by `sqrt(interval_sec / reference_interval)`. The clamp bounds are calculated by applying the same formula to each profile's practical fixed-interval range. This keeps short intervals from becoming too sensitive and long intervals from becoming too insensitive.

## Image Format

### `JPEG Quality`

- Unit: ffmpeg `-q:v`
- UI range: `1-31`
- Control: horizontal drag on the numeric field
- Important: lower values mean higher quality and larger files
- Default: `2`

## Review Status

Step 2 translates extraction decisions recorded in the CSV into visible review labels and summary counts. The main places to look are:

- Colored advisory label: why the currently displayed frame was marked that way
- `Review n | +a / -d / gap g / blur b / tex l / weak w | target/normal`: total review counts and whether the current frame is a review target
- `extract problem ← (Shift+F)` / `extract problem → (F)`: jump through review targets in capture order
- `Decision: Keep/Drop`: whether Step 2 will leave the frame in the image folder

Colored labels use red for dropped frames, yellow for kept frames that need review, blue for added frames, purple for quick extraction, and green for OK frames. Thumbnail view uses the same category color in the bottom ribbon with a shorter label. The labels mean:

- `OK: kept`: a normal kept frame from fixed-interval or standard extraction (`ok`).
- `Quick: extracted at the specified interval`: a quick-extract frame cut at the requested interval without analysis (`analysis_pipeline=quick`).
- `Added: viewpoint change`: an extra frame inserted before the next fixed-cadence point because viewpoint change was useful (`novelty_added`).
- `Added: preserved spacing`: a safety keep that prevents low-change skipping from exceeding the `Max` gap (`gap_forced`).
- `Review: possible blur`: kept, but sharpness fell sharply and SfM matches may be weak (`motion_blur`).
- `Review: low texture`: not necessarily blurred, but low texture may weaken SfM matches (`low_texture`).
- `Review: weak feature tracking`: kept, but flagged because candidate-pair feature tracking is weak (`weak_match`).
- `Drop: possible blur`: dropped because blur may weaken SfM (`motion_blur`).
- `Drop: similar frame`: dropped because the fixed-cadence candidate is too similar to the last kept frame (`redundant_drop`).
- `Drop: manually excluded`: manually switched to Drop in Step 2.

A practical starting point is `1.0` second, `Min 0.5`, `Max 2.0`, with `Motion` enabled, then review the `+`, `-`, `gap`, `blur`, `tex`, and `weak` counts and examples in Step 2.
