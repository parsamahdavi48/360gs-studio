# Frame Extraction GUI Parameters

This GUI extracts equirectangular still images from 360-degree video for Metashape SfM. The current extraction baseline is `Fixed Interval`. When `Motion` is enabled, the extractor keeps the fixed cadence but skips low-change candidates and inserts extra candidates in high-motion ranges.

## Assumptions

- Analysis and image export are separate phases.
- Analysis uses grayscale frames scaled to `Analysis Width`.
- Images written to `images/` keep the source video resolution.
- `Instant Estimate` is the fixed-interval baseline count. With `Motion` enabled, the final count can increase or decrease after analysis.
- With `Quick extract` enabled, the GUI prioritizes fixed-interval output for short test SfM runs. Motion adjustment and pair-analysis review metadata are not produced.
- The integrated GUI stops before running when the scene folder path contains non-ASCII characters, an extremely long path, control characters, or `"`. Use a short ASCII working path.

## Fixed Interval

### `Interval`

- Unit: seconds
- Meaning: baseline spacing between extracted frame candidates
- Internal calculation: `step = round(interval_sec * fps)`
- UI range: `0.05-60.0` seconds
- Control: horizontal drag on the numeric field
- Example: at 30fps, `0.8` seconds is about every 24 frames

Increasing the value reduces the baseline count. Decreasing it increases the count. For 3DGS-oriented SfM, a stable fixed cadence is easier to reason about than a fully variable interval.

### `Quick extract`

Use this fixed-interval option when you want to try a short test SfM run quickly. It prioritizes writing the requested `Interval` result sooner, so Step 2 does not receive extra quality-review flags from automatic scoring.

Enabling it turns `Motion` off. For final extraction, turn it off again when you want motion adjustment.

## Motion Adjustment

`Motion` is an additional option for fixed interval extraction. In the GUI, it uses pair analysis against the last kept frame. The fixed cadence remains the baseline, then analysis applies:

- Yaw-compensated residual: estimate the horizontal shift in the 360 equirectangular frame and measure the residual after removing changes explained by pure yaw.
- Redundant drops: fixed-cadence candidates with too little residual change from the last kept frame are marked `redundant_drop`.
- Novelty additions: frames before the next fixed-cadence point can be added as `novelty_added` when residual change is already sufficient.
- Safety keeps: if the kept-frame gap reaches `Max`, the candidate is kept as `gap_forced`.
- Candidate pair tracking: sparse feature tracking runs only at keep/drop decision points. Weak pairs are flagged as `weak_match` for Step 2.

### `Min`

- Unit: seconds
- Meaning: minimum spacing for inserted candidates
- Internal calculation: `min_gap_frames = round(min_gap_sec * fps)`
- Default: `0.25` seconds

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
- Meaning: horizontal decode width used for yaw-compensated residuals, candidate pair tracking, and lightweight blur checks
- Default: `1920`

Higher values can improve fine-feature detection but increase analysis time. Lower values are faster but can miss subtle motion and feature detail.

## Pair-Analysis Review Metadata

The GUI's normal extraction path no longer uses the old single-frame `SfM quality score` or nearby alternate-frame replacement. For carefully captured video, the useful question is less "which individual frame is prettier?" and more "does this pair still overlap while adding non-redundant viewpoint change?"

The CSV records these Step 2 review fields:

- `residual_score`: residual change after yaw compensation
- `raw_change_score`: luma difference before yaw compensation
- `yaw_shift_deg`: estimated horizontal yaw adjustment
- `track_count`: tracked feature count for the candidate pair
- `track_coverage`: screen coverage of tracked points
- `match_confidence`: review confidence from tracked count and coverage

## Image Format

### `JPEG Quality`

- Unit: ffmpeg `-q:v`
- UI range: `1-31`
- Control: horizontal drag on the numeric field
- Important: lower values mean higher quality and larger files
- Default: `2`

## Review Status

Step 2 reads `selected_frames.csv` and surfaces frames based on `status`.

- `ok`: normal kept candidate
- `novelty_added`: added before the next fixed cadence because residual change is sufficient
- `redundant_drop`: fixed-cadence candidate currently dropped because residual change is small
- `gap_forced`: kept by the `Max` gap guard
- `weak_match`: kept but flagged because candidate pair tracking is weak
- `smart_added` / `replaced` / `fallback_keep` / `thinned`: legacy statuses when `--analysis-pipeline legacy` is used

A practical starting point is `0.8-1.0` seconds with `Motion` enabled, then review how `novelty_added`, `redundant_drop`, `gap_forced`, and `weak_match` frames look in Step 2.
