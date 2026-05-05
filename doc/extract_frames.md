# extract_frames.py - FFmpeg frame extraction with pair analysis

## Overview

`extract_frames.py` extracts still images from an equirectangular video and writes a reviewable `selected_frames.csv`.

The analyzed extraction path is fixed interval plus pair analysis. Pair analysis compares each decision candidate with the last kept frame: yaw-compensated residual change drives redundant drops and novelty additions, and sparse feature tracking plus candidate-only sharpness checks produce Step 2 review flags.

Quick extraction is the only non-analyzed path. It skips pair analysis and motion adjustment, then extracts the requested fixed cadence directly.

## Requirements

- FFmpeg (`ffmpeg`) and FFprobe (`ffprobe`) available in PATH
- Python modules: NumPy, OpenCV

## Usage

Basic pair-analyzed extraction:

```bash
python extract_frames.py input.mp4 ./scene01 --interval-sec 1.0
```

Pair analysis with motion adjustment:

```bash
python extract_frames.py input.mp4 ./scene01 \
  --interval-sec 1.0 \
  --fixed-smart \
  --min-gap-sec 0.5 \
  --max-gap-sec 2.0
```

Quick fixed-interval test extraction:

```bash
python extract_frames.py input.mp4 ./scene01 \
  --interval-sec 1.0 \
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
| `--interval-sec` | `0.5` | Base fixed interval in seconds. GUI default is `1.0` |
| `--quick-extract` | off | Extract the fixed cadence without analysis or motion adjustment |
| `--fixed-smart` | off | Enable pair-analysis motion adjustment. It can drop redundant fixed candidates, add novelty candidates, and keep max-gap safety frames |
| `--min-gap-sec` | `0.25` | Minimum gap for pair-analysis additions |
| `--max-gap-sec` | `2.0` | Maximum safety gap before a frame is kept |
| `--fixed-smart-max-inserts-per-interval` | `2` | Maximum novelty anchors inserted inside one fixed interval |
| `--pair-motion-profile` | `walk` | Auto threshold profile. `walk` uses a 1.0s nearby/walking reference. `drone` uses lower residual thresholds for distant/aerial 360 capture |
| `--pair-drop-threshold` | `-1` | Pair residual below this drops fixed candidates. Negative means auto from interval/profile |
| `--pair-add-threshold` | `-1` | Pair residual at or above this adds novelty candidates. Negative means auto from interval/profile |
| `--pair-track-min-count` | `36` | Review threshold. Kept pairs below this tracked feature count are flagged `weak_match` |
| `--pair-track-min-confidence` | `0.25` | Review threshold from tracked count and coverage |
| `--analysis-width` | `1920` | Decode width for candidate tracking and candidate-only sharpness checks. Yaw/residual monitoring is internally capped to a 1280px gate width |
| `--image-ext` | `jpg` | Output image format |
| `--jpg-quality` | `2` | JPEG quality for ffmpeg `-q:v`; lower is higher quality |
| `--output-mode` | `overwrite` | `overwrite`, `append`, or `replace-video` for `selected_frames.csv` and `extract_sessions.json` |
| `--estimate-only` | off | Run selection and print counts without image extraction |

## Outputs

Under `output_dir`:

- `images/<prefix>_<source_frame_index>.jpg` or `.png`
- `selected_frames.csv`
- `extract_report.json`
- `extract_sessions.json`

`selected_frames.csv` fields include:

- `original_index`, `final_index`, `timestamp_sec`
- `status`: `ok`, `novelty_added`, `redundant_drop`, `gap_forced`, `motion_blur`, `low_texture`, `weak_match`
- `decision`: `keep` or `drop`, editable in Step 2
- `analysis_pipeline`: `pair` or `quick`
- `selection_reason`: `initial`, `fixed_interval`, `novelty_added`, `redundant_drop`, `gap_forced`, `endpoint`, or `quick_extract`
- `residual_score`, `raw_change_score`, `yaw_shift_deg`, `track_count`, `track_coverage`, `match_confidence`
- `blur_score_final`, `sharpness_baseline`, `sharpness_ratio`
- `pair_motion_profile`, `pair_drop_threshold`, `pair_add_threshold`
- `output_file`

With `--quick-extract`, analysis score columns are blank because no pair analysis is run.

## Notes

- Drop rows are still extracted so Step 2 can preview them before finalizing.
- Default filename prefix is the input video filename stem; override with `--filename-prefix`.
- `--image-ext jpg` is recommended for speed during iteration.
- This script does not modify existing mask files.
