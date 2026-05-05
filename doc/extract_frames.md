# extract_frames.py — FFmpeg frame extraction with SfM-oriented representative selection

## Overview

`extract_frames.py` extracts still images from an equirectangular video and writes a reviewable selection table.

It supports two selection modes:

- `fixed`: fixed interval (`--interval-sec`)
- `change`: adaptive extraction based on inter-frame change (`--change-threshold` + min/max gap)

In the GUI workflow, fixed interval is the baseline. The default `--analysis-pipeline pair` uses last-kept-frame pair analysis: yaw-compensated residual change drives redundant drops and novelty additions, and sparse feature tracking is checked only at candidate decision points.

For quick test SfM runs, fixed mode also supports `--quick-extract`. This skips analysis, motion adjustment, and representative frame scoring, then extracts the requested fixed cadence directly.

The legacy `--analysis-pipeline legacy` path is still available for comparison. It uses whole-video luma/feature-motion scores plus the old bounded single-frame SfM quality score and nearby representative replacement.

Optional legacy **stationary thinning** (`--thin-motion-threshold`) drops selected frames whose cumulative luma change since the last kept frame is too low. `--fixed-smart` performs its own low-change skip with the same motion score used for high-motion insertion, so the legacy thinning pass is not applied when `--fixed-smart` is enabled.

The legacy quality / Laplacian / change / feature-motion scores are cached to `extract_cache.npz`. Pair analysis is streaming and parameter-dependent, so the comparison workflow should use `--no-cache` when measuring first-run speed.

## Requirements

- FFmpeg (`ffmpeg`) and FFprobe (`ffprobe`) available in PATH
- Python modules: NumPy, OpenCV

## Usage

Basic (change-based):

```bash
python extract_frames.py input.mp4 ./scene01
```

Fixed interval example:

```bash
python extract_frames.py input.mp4 ./scene01 --mode fixed --interval-sec 0.5
```

Quick fixed-interval test extraction:

```bash
python extract_frames.py input.mp4 ./scene01 \
  --mode fixed \
  --interval-sec 1.0 \
  --quick-extract
```

Fixed interval with motion adjustment:

```bash
python extract_frames.py input.mp4 ./scene01 \
  --mode fixed \
  --interval-sec 1.0 \
  --fixed-smart \
  --analysis-pipeline pair \
  --min-gap-sec 0.5 \
  --max-gap-sec 2.0
```

Compare legacy and pair analysis on one video:

```bash
python scripts/compare_extract_analysis.py input.mp4 ./compare_scene \
  --interval-sec 1.0 \
  --min-gap-sec 0.5 \
  --max-gap-sec 2.0
```

Change-based example with tuning:

```bash
python extract_frames.py input.mp4 ./scene01 \
  --mode change \
  --change-threshold 0.05 \
  --min-gap-sec 0.3 \
  --max-gap-sec 2.2
```

Specify custom filename prefix:

```bash
python extract_frames.py input.mp4 ./scene01 --filename-prefix walk01
```

Estimate only (no image extraction):

```bash
python extract_frames.py input.mp4 ./scene01 --estimate-only --print-summary-json
```

Sampled estimate only (faster):

```bash
python extract_frames.py input.mp4 ./scene01 \
  --estimate-only \
  --estimate-mode sampled \
  --sample-segments 5 \
  --sample-segment-sec 12 \
  --sample-fps 8 \
  --print-summary-json
```

Motion adjustment + cache example:

```bash
# 1st run: full analyze, write cache, extract
python extract_frames.py input.mp4 ./scene01 \
  --mode fixed --interval-sec 1.0 \
  --fixed-smart \
  --min-gap-sec 0.5 \
  --max-gap-sec 2.0

# 2nd run with different fixed-smart gaps: cache hits, skips analysis
python extract_frames.py input.mp4 ./scene01 \
  --mode fixed --interval-sec 1.0 \
  --fixed-smart \
  --min-gap-sec 0.35 \
  --max-gap-sec 2.5
```

## Key options

| Option | Default | Description |
|---|---|---|
| `--mode` | `change` | `fixed` or `change`. Fixed interval is recommended for SfM stability |
| `--interval-sec` | `0.5` | Fixed mode interval in seconds. GUI default is `1.0` |
| `--quick-extract` | (off) | Fixed-mode shortcut for test runs. Extracts the fixed cadence without analysis, motion adjustment, representative scoring, or legacy thinning |
| `--analysis-pipeline` | `pair` | `pair` uses last-kept-frame yaw-compensated residuals plus candidate-only tracking and blur/texture risk flags. `legacy` uses the older whole-video quality-score pipeline |
| `--fixed-smart` | (off) | Fixed-mode helper. With `pair`, marks redundant base candidates, adds novelty candidates, and keeps max-gap safety frames |
| `--fixed-smart-change-threshold` | `0.04` | Legacy normalized luma-difference threshold. Pair mode uses auto `--pair-drop-threshold` / `--pair-add-threshold` unless those are set manually |
| `--fixed-smart-feature-threshold` | `0.012` | Legacy sparse feature-motion threshold. Not used by `pair` |
| `--fixed-smart-max-inserts-per-interval` | `2` | Maximum extra anchors inserted inside one fixed interval |
| `--pair-motion-profile` | `walk` | Auto threshold profile. `walk` uses 1.0s as the handheld walking reference. `drone` uses lower residual thresholds for aerial 360 capture |
| `--pair-drop-threshold` | `-1` | Pair residual below this drops fixed candidates. Negative means auto from interval/profile |
| `--pair-add-threshold` | `-1` | Pair residual at or above this adds novelty candidates. Negative means auto from interval/profile |
| `--pair-track-min-count` | `36` | Pair pipeline review threshold. Kept pairs below this tracked feature count are flagged `weak_match` |
| `--pair-track-min-confidence` | `0.25` | Pair pipeline review threshold from tracked count and coverage |
| `--analysis-width` | `1920` | Decode width for candidate tracking and candidate-only sharpness checks. Yaw/residual monitoring is internally capped to a 1280px gate width. `0` or larger than source = full resolution |
| `--quality-min-score` | `0.35` | Legacy-only single-frame quality score threshold |
| `--quality-min-improvement` | `0.08` | Legacy-only representative replacement threshold |
| `--thin-motion-threshold` | `0.6` | Legacy low-change thinning for non-smart selection. `0` disables. Ignored when `--fixed-smart` is enabled |
| `--no-thin-keep-endpoints` | (off) | Allow the last frame to be dropped during thinning. By default first/last frames are always preserved |
| `--no-extract-thinned` | (off) | Skip image extraction for thinned frames. Default is to extract them so the review GUI can preview each thinned frame and flip back to keep if desired. Thinned rows always remain in CSV with `decision=drop` regardless |
| `--no-cache` | (off) | Force full re-analysis (ignore `extract_cache.npz`) |

## Outputs

Under `output_dir`:

- `images/<video_stem>_<source_frame_index>.jpg` (or `.png`) by default. The frame index is zero-padded to the shortest width that keeps the video's full frame range sortable.
- `selected_frames.csv`
- `extract_report.json`

`selected_frames.csv` fields:

- `original_index`: index from initial mode selection
- `final_index`: index after representative frame selection
- `status`: pair statuses such as `ok`, `novelty_added`, `redundant_drop`, `gap_forced`, `motion_blur`, `low_texture`, `weak_match`, or legacy statuses such as `smart_added`, `replaced`, `fallback_keep`, `thinned`
- `residual_score`, `raw_change_score`, `yaw_shift_deg`, `track_count`, `track_coverage`, `match_confidence`, `blur_score_final`, `sharpness_ratio`, `pair_gate_width`: pair-analysis review metadata
- `pair_motion_profile`, `pair_drop_threshold`, `pair_add_threshold`: resolved pair-analysis threshold metadata
- `quality_score_original`, `quality_score_final`: legacy-only bounded SfM-oriented quality scores used for representative selection
- `decision`: `keep` for extracted frames, `drop` for thinned frames (editable in review tool)
- `output_file`: image path for review and later processing (no file on disk for thinned rows)

With `--quick-extract`, `status` remains `ok` and the analysis score columns are left blank because no quality or change scoring is run.

`extract_cache.npz`: legacy cached SfM quality, Laplacian, change, and sparse feature-motion scores per analyzed frame, keyed by video file size+mtime, cache version, and `--analysis-width`. Auto-invalidated when any of these change. Add to `.gitignore` (already configured).

## Notes

- Representative frame candidates are chosen automatically from the midpoint between neighboring extraction anchors, so candidates do not cross into another anchor's region.
- Default filename prefix is input video filename stem; override with `--filename-prefix`.
- `--image-ext jpg` is recommended for speed during iteration.
- This script does not modify existing mask files.
- `--estimate-only --estimate-mode full` analyzes all frames for the estimate.
- `--estimate-only --estimate-mode sampled` analyzes temporal windows only and extrapolates.
