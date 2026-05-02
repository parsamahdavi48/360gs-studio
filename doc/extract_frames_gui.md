# Frame Extraction GUI Parameters

This GUI extracts equirectangular still images from 360-degree video for Metashape SfM. The current extraction baseline is `Fixed Interval`. When `Motion` is enabled, the extractor keeps the fixed cadence but skips low-change candidates and inserts extra candidates in high-motion ranges.

## Assumptions

- Analysis and image export are separate phases.
- Analysis uses grayscale frames scaled to `Analysis Width`.
- Images written to `images/` keep the source video resolution.
- `Instant Estimate` is the fixed-interval baseline count. With `Motion` enabled, the final count can increase or decrease after analysis.

## Fixed Interval

### `Interval`

- Unit: seconds
- Meaning: baseline spacing between extracted frame candidates
- Internal calculation: `step = round(interval_sec * fps)`
- UI range: `0.05-60.0` seconds
- Control: horizontal drag on the numeric field
- Example: at 30fps, `0.8` seconds is about every 24 frames

Increasing the value reduces the baseline count. Decreasing it increases the count. For 3DGS-oriented SfM, a stable fixed cadence is easier to reason about than a fully variable interval.

## Motion Adjustment

`Motion` is an additional option for fixed interval extraction. The fixed cadence remains the baseline, then analysis applies:

- Low-change skipping: candidates with too little accumulated motion since the last kept frame are marked `thinned`
- High-motion insertion: frames inside a fixed interval with strong motion are added as `smart_added`
- Feature-motion scoring: sparse feature tracking is used alongside luma difference so the adjustment is closer to SfM-relevant parallax

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
- Meaning: horizontal decode width used for change scoring, feature-motion scoring, and SfM quality scoring
- Default: `1920`

Higher values can improve fine-feature detection but increase analysis time. Lower values are faster but can miss subtle motion and feature detail.

## SfM Quality Check

Advanced `SfM Quality Check` scores selected candidates for SfM and, when useful, chooses a nearby alternate representative frame. Low-change skipping belongs to `Motion` adjustment, not to this quality check.

### `Quality Review Score`

- Unit: normalized `0.0-1.0` score
- Calculation: combined feature count, feature spread, sharpness, contrast, and exposure penalty
- Meaning: if the final representative stays below this score, Step 2 records it as `fallback_keep`
- Default: `0.35`

### `Alternate Frame Criterion`

- Unit: normalized `0.0-1.0` score delta
- Calculation: `candidate quality score - original quality score`
- Meaning: minimum improvement required before using a nearby alternate frame
- Default: `0.08`

Lower values make alternate-frame selection more aggressive. Higher values only replace when the improvement is clearer.

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
- `smart_added`: added by motion adjustment
- `replaced`: replaced by a nearby SfM-ready representative
- `fallback_keep`: still below the quality review score after representative selection
- `thinned`: currently dropped because the interval had little motion

A practical starting point is `0.8-1.0` seconds with `Motion` enabled, then review how `smart_added` and `thinned` frames look in Step 2.
