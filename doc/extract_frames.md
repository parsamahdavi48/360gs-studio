# extract_frames.py — FFmpeg frame extraction with blur-aware selection

## Overview

`extract_frames.py` extracts still images from an equirectangular video and writes a reviewable selection table.

It supports two selection modes:

- `fixed`: fixed interval (`--interval-sec`)
- `change`: adaptive extraction based on inter-frame change (`--change-threshold` + min/max gap)

After initial selection, frames below a blur percentile threshold are replaced with sharper nearby frames when possible.

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

Change-based example with tuning:

```bash
python extract_frames.py input.mp4 ./scene01 \
  --mode change \
  --change-threshold 0.05 \
  --min-gap-sec 0.3 \
  --max-gap-sec 2.2
```

Estimate only (no image extraction):

```bash
python extract_frames.py input.mp4 ./scene01 --estimate-only --print-summary-json
```

## Outputs

Under `output_dir`:

- `images/frame_XXXXXX.jpg` (or `.png`)
- `selected_frames.csv`
- `extract_report.json`

`selected_frames.csv` fields:

- `original_index`: index from initial mode selection
- `final_index`: index after blur replacement
- `status`: `ok` / `replaced` / `fallback_keep`
- `decision`: default `keep` (editable in review tool)
- `output_file`: image path for review and later processing

## Notes

- Auto blur replacement window uses a conservative default based on FPS and selected minimum gap.
- `--image-ext jpg` is recommended for speed during iteration.
- This script does not modify existing mask files.
- `--estimate-only` runs analysis and selection without creating `images/` or CSV/report files.
