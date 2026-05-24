# extract_frames.py - FFmpeg frame extraction with pair analysis

## Overview

`extract_frames.py` extracts SfM/3DGS-oriented still images from 360° video or normal video and writes a reviewable `_stechdrive/frames/selected_frames.csv`.

The goal is not to cut every video frame mechanically. The goal is to keep enough viewpoint change for SfM, reduce frames that are too similar, and surface blurry, low-texture, or weak-feature frames so the final image set is easier to review before training or reconstruction.

## Extraction Strategy

Video can contain a large number of frames even in a short clip. Sending every frame to SfM makes processing heavier, and too many near-duplicate images can make matching and reconstruction less stable. On the other hand, thinning too aggressively can remove the parallax and coverage that SfM needs.

This app starts from a fixed interval so the frame count and whole-video coverage stay predictable. It then compares each decision candidate with the last kept frame and, when useful, drops redundant candidates or inserts intermediate frames.

In equirectangular 360° images, a change in camera heading can create a large image difference by itself. That does not necessarily mean the frame adds useful SfM parallax. Pair analysis therefore estimates the horizontal roll, or yaw shift, between the frames and measures the remaining residual change after alignment. This makes it easier to focus on real viewpoint or scene-appearance changes instead of pure heading changes.

Automatic selection is not treated as the final decision. Frames that may cause SfM trouble, such as blur, possible blur, low texture, or weak tracked features, are written to `_stechdrive/frames/selected_frames.csv` with review flags. Step 2 lets you inspect those frames visually and adjust keep/drop decisions.

Frames marked `drop` are still extracted as images. This is intentional: you can inspect them later and restore them if needed. The design favors a reviewable workflow over irreversible automatic deletion.

## Current Analysis Path

The analyzed extraction path is fixed interval plus pair analysis. Pair analysis compares each decision candidate with the last kept frame: yaw-compensated residual change drives redundant drops and novelty additions, and sparse feature tracking plus candidate-only sharpness checks produce Step 2 review flags.

Sharpness drops are split into two tiers. Clear drops are marked `motion_blur`, planned as `drop`, and the analyzer searches the finite range from that point to the maximum allowed gap for a replacement. Milder drops are kept but marked `borderline_blur` so Step 2 can surface them for manual review. Replacement candidates are not accepted by sharpness alone: yaw-compensated residual change, sparse tracking, and low-texture checks are evaluated again against the last kept frame. Accepted replacements are marked `blur_replacement`, while the original blurred candidate remains as a `motion_blur` `drop` row for review.

Quick extraction is the only non-analyzed path. It skips pair analysis and motion adjustment, then extracts the requested fixed cadence directly. Use it when you want to skip analysis and quickly cut the video at the specified interval.

## Requirements

- FFmpeg (`ffmpeg`) and FFprobe (`ffprobe`) available in PATH
- Python modules: NumPy, OpenCV

## Usage

Basic pair-analyzed extraction:

```bash
python extract_frames.py input.mp4 ./scene01 --interval-sec 1.5
```

Pair analysis with motion adjustment:

```bash
python extract_frames.py input.mp4 ./scene01 \
  --interval-sec 1.5 \
  --fixed-smart \
  --min-gap-sec 0.8 \
  --max-gap-sec 4.0
```

Quick fixed-cadence extraction:

```bash
python extract_frames.py input.mp4 ./scene01 \
  --interval-sec 1.5 \
  --quick-extract
```

Specify a custom filename prefix:

```bash
python extract_frames.py input.mp4 ./scene01 --filename-prefix walk01
```

Estimate only:

```bash
python extract_frames.py input.mp4 ./scene01 --estimate-only --print-summary-json
```

## Key Options

| Option | Default | Description |
|---|---:|---|
| `--interval-sec` | `0.5` | Base fixed interval in seconds. GUI default is `1.5` |
| `--quick-extract` | off | Extract the fixed cadence without analysis or motion adjustment |
| `--fixed-smart` | off | Enable pair-analysis motion adjustment. It can drop redundant fixed candidates, add novelty candidates, and keep max-gap safety frames |
| `--min-gap-sec` | `0.25` | Minimum gap for pair-analysis additions |
| `--max-gap-sec` | `2.0` | Maximum safety gap before a frame is kept |
| `--fixed-smart-max-inserts-per-interval` | `2` | Maximum novelty anchors inserted inside one fixed interval |
| `--pair-motion-profile` | `walk` | Auto threshold profile. GUI default is `walk_standard`. See "Profiles and Automatic Thresholds" below |
| `--pair-drop-threshold` | `-1` | Pair residual below this drops fixed candidates. Negative means auto from interval/profile |
| `--pair-add-threshold` | `-1` | Pair residual at or above this adds novelty candidates. Negative means auto from interval/profile |
| `--pair-track-min-count` | `36` | Review threshold. Kept pairs below this tracked feature count are flagged `weak_match` |
| `--pair-track-min-confidence` | `0.25` | Review threshold from tracked count and coverage |
| `--analysis-width` | `1920` | Decode width for candidate tracking and candidate-only sharpness checks. Yaw/residual monitoring is internally capped to a 1280px gate width |
| `--image-ext` | `jpg` | Output image format |
| `--jpg-quality` | `2` | JPEG quality for ffmpeg `-q:v`; lower is higher quality |
| `--output-mode` | `overwrite` | `overwrite`, `append`, or `replace-video` for `_stechdrive/frames/selected_frames.csv` and `_stechdrive/frames/extract_sessions.json` |
| `--estimate-only` | off | Run selection and print counts without image extraction |

## Profiles and Automatic Thresholds

`--pair-motion-profile` chooses the assumption behind the pair-analysis `drop` / `add` automatic thresholds. It is not a hard workflow category; it is a coarse hint for subject distance and how much residual parallax appears in the image for the same camera movement.

- `walk_standard`: normal walking capture for facilities, streets, and common walking footage. The `1.5s` reference thresholds are `drop=0.035` and `add=0.095`.
- `walk_close`: close walking capture for nearby walls, exhibits, furniture, or narrow corridors. The `1.0s` reference thresholds are `drop=0.035` and `add=0.090`.
- `walk_wide`: wide walking capture for parks, plazas, exteriors, and more distant subjects. The `3.0s` reference thresholds are `drop=0.030` and `add=0.075`.
- `drone_distant`: aerial or distant-view capture where residual parallax tends to be weaker. The `3.0s` reference thresholds are `drop=0.025` and `add=0.065`.

When the interval changes, thresholds scale gently by `sqrt(interval_sec / reference_interval)`. Each profile clamps over its practical interval range. This keeps short intervals from becoming too sensitive and long intervals from becoming too insensitive.

`walk` and `drone` are compatibility profiles for existing CLI commands. `walk` keeps the previous near-walking thresholds, and `drone` keeps the previous aerial thresholds. The GUI uses the four profiles above.

If `--pair-drop-threshold` or `--pair-add-threshold` is set to a non-negative value, that threshold overrides the profile-derived automatic value. If only one is set, that side is manual and the other side still comes from the profile.

## Outputs

Under `output_dir`:

- `images/<prefix>_<source_frame_index>.jpg` or `.png`
- `_stechdrive/frames/selected_frames.csv`
- `_stechdrive/frames/extract_report.json`
- `_stechdrive/frames/extract_sessions.json`

`_stechdrive/frames/selected_frames.csv` fields include:

- `original_index`, `final_index`, `timestamp_sec`
- `status`: `ok`, `novelty_added`, `blur_replacement`, `redundant_drop`, `gap_forced`, `motion_blur`, `borderline_blur`, `low_texture`, `weak_match`
- `decision`: `keep` or `drop`, editable in Step 2
- `analysis_pipeline`: `pair` or `quick`
- `selection_reason`: `initial`, `fixed_interval`, `novelty_added`, `blur_replacement`, `redundant_drop`, `gap_forced`, `endpoint`, or `quick_extract`
- `residual_score`, `raw_change_score`, `yaw_shift_deg`, `track_count`, `track_coverage`, `match_confidence`
- `blur_score_final`, `sharpness_baseline`, `sharpness_ratio`
- `pair_motion_profile`, `pair_drop_threshold`, `pair_add_threshold`
- `output_file`

With `--quick-extract`, analysis score columns are blank because no pair analysis is run.

## Notes

- Drop rows are still extracted so Step 2 can preview them before finalizing.
- `blur_replacement` rows are frames selected near a dropped blur candidate. The original blur candidate remains as a `drop` row.
- `borderline_blur` rows are kept but excluded from the running sharpness baseline so later blur checks do not become too permissive.
- Default filename prefix is the input video filename stem; override with `--filename-prefix`.
- `--image-ext jpg` is recommended for speed during iteration.
- This script does not modify existing mask files.
