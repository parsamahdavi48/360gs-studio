# extract_frames_gui.py — Wrapper GUI for extraction workflow

## Overview

`extract_frames_gui.py` is a wrapper GUI around the extraction pipeline.

It provides three actions:

1. Run extraction (`extract_frames.py`)
2. Open visual review (`review_frames.py`)
3. Export keep frames for Metashape (`apply_frame_decisions.py`)

Requirements:

- `PySide6`

## Usage

```bash
python extract_frames_gui.py --scene-dir ./scene01
```

## Notes

- The GUI runs each step via Python subprocess, so CLI and GUI behavior stay aligned.
- `selected_frames.csv` is the shared state between extraction, review, and keep-frame export.
- Recommended output extension during iteration is JPG for speed.
