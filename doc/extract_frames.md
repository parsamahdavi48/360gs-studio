# extract_frames.py — FFmpeg frame extraction with SfM-oriented representative selection

## Overview

`extract_frames.py` extracts still images from an equirectangular video and writes a reviewable selection table.

It supports two selection modes:

- `fixed`: fixed interval (`--interval-sec`)
- `change`: adaptive extraction based on inter-frame change (`--change-threshold` + min/max gap)

After initial selection, each selected frame is treated as an extraction anchor. The script searches a local candidate window and uses a bounded SfM-oriented quality score to replace the anchor only when a nearby frame is clearly better. No global percentile threshold is used.

Optional **stationary thinning** (`--thin-motion-threshold`) drops selected frames whose cumulative motion since the last kept frame is too low. This adapts to the recording style: standing still gets thinned automatically, walking is preserved.

The quality / Laplacian / change scores are cached to `extract_cache.npz` so re-running with different selection or thinning parameters skips the analysis pass entirely (full re-analysis only when the video file or `--analysis-width` changes).

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

Thinning + cache example (typical recommended fixed-interval workflow):

```bash
# 1st run: full analyze, write cache, extract
python extract_frames.py input.mp4 ./scene01 \
  --mode fixed --interval-sec 0.8 \
  --thin-motion-threshold 0.6

# 2nd run with different thinning threshold: cache hits, skips analysis
python extract_frames.py input.mp4 ./scene01 \
  --mode fixed --interval-sec 0.8 \
  --thin-motion-threshold 1.0
```

## Key options

| Option | Default | Description |
|---|---|---|
| `--mode` | `change` | `fixed` or `change`. Fixed interval is recommended for SfM stability |
| `--interval-sec` | `0.5` | Fixed mode interval in seconds |
| `--analysis-width` | `1920` | Decode width for change/quality analysis. Higher = more accurate, slower. `0` or larger than source = full resolution |
| `--quality-min-score` | `0.35` | Mark a frame for review if the final representative is below this 0.0-1.0 SfM-oriented quality score. The score combines feature count, feature spread, sharpness, contrast, and exposure penalty |
| `--quality-min-improvement` | `0.08` | Minimum 0.0-1.0 quality-score gain required before replacing an anchor. Computed as candidate quality minus original quality |
| `--thin-motion-threshold` | `0.6` | Low-change thinning. `0` disables. `0.3-1.0` is typical. Drops frames where cumulative change since last kept frame is below this value. Per-frame change is `mean(abs(gray_t - gray_(t-1))) / 255` |
| `--no-thin-keep-endpoints` | (off) | Allow the last frame to be dropped during thinning. By default first/last frames are always preserved |
| `--no-extract-thinned` | (off) | Skip image extraction for thinned frames. Default is to extract them so the review GUI can preview each thinned frame and flip back to keep if desired. Thinned rows always remain in CSV with `decision=drop` regardless |
| `--no-cache` | (off) | Force full re-analysis (ignore `extract_cache.npz`) |

## Outputs

Under `output_dir`:

- `images/<video_stem>_XXXXXX.jpg` (or `.png`) by default
- `selected_frames.csv`
- `extract_report.json`

`selected_frames.csv` fields:

- `original_index`: index from initial mode selection
- `final_index`: index after representative frame selection
- `status`: `ok` / `replaced` / `fallback_keep` / `thinned` / combinations like `replaced+thinned`
- `quality_score_original`, `quality_score_final`: bounded SfM-oriented quality scores used for representative selection
- `decision`: `keep` for extracted frames, `drop` for thinned frames (editable in review tool)
- `output_file`: image path for review and later processing (no file on disk for thinned rows)

`extract_cache.npz`: cached SfM quality, Laplacian, and change scores per analyzed frame, keyed by video file size+mtime and `--analysis-width`. Auto-invalidated when any of these change. Add to `.gitignore` (already configured).

## Notes

- Representative frame candidates are chosen automatically from the midpoint between neighboring extraction anchors, so candidates do not cross into another anchor's region.
- Default filename prefix is input video filename stem; override with `--filename-prefix`.
- `--image-ext jpg` is recommended for speed during iteration.
- This script does not modify existing mask files.
- `--estimate-only --estimate-mode full` analyzes all frames for the estimate.
- `--estimate-only --estimate-mode sampled` analyzes temporal windows only and extrapolates.
