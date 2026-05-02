# extract_frames_gui.py — Wrapper GUI for extraction workflow

## Overview

`extract_frames_gui.py` is a wrapper GUI around the extraction pipeline.

It provides three actions:

1. Run extraction (`extract_frames.py`)
2. Open visual review (`review_frames.py`)
3. Finalize reviewed frames (`apply_frame_decisions.py`)

It also provides analysis helpers:

- `Load Video Info`: shows resolution, FPS, duration, and total frames (via `ffprobe`)
- `Instant`: immediately shows a rough expected count from current parameters
- `Sampled`: runs a background sampled estimate and updates while editing
- `Refresh Sampled`: manually restart sampled estimate immediately
- `Filename Prefix`: optional override for output filename prefix (default: video filename)

Requirements:

- `PySide6`

## Usage

```bash
python extract_frames_gui.py --scene-dir ./scene01
```

## Notes

- The GUI runs each step via Python subprocess, so CLI and GUI behavior stay aligned.
- `Sampled` estimate uses `extract_frames.py --estimate-only --estimate-mode sampled`.
- Running extraction/export cancels any in-progress sampled estimate.
- Default export target is `images`; in this mode export performs in-place finalize (drop removal + keep renumber + CSV rewrite).
- The same `Filename Prefix` is used for extraction output names and in-place finalize names.
- `selected_frames.csv` is the shared state between extraction, review, and finalize.
- Recommended output extension during iteration is JPG for speed.

## Automatic Frame Selection

The advanced settings include an automatic selection section. It scores extracted candidates for SfM and, when needed, chooses alternate nearby frames, flags low-quality frames for Step 2 review, and skips low-change frames.

- `Quality Review Score`: 0.0-1.0 normalized SfM quality score. Frames below this after alternate-frame selection are flagged for review.
- `Alternate Frame Criterion`: 0.0-1.0 score delta. A nearby alternate is used only when its quality score exceeds the original by at least this value.
- `Skip Low-Change Frames`: cumulative normalized frame-change score. `0` disables; `0.3-1.0` is a typical range.
