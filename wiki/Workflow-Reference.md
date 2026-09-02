# Workflow Reference

| Workspace | Purpose | Main output |
| --- | --- | --- |
| Frame Extraction | Register sources and select useful frames | `images/` |
| Frame Review | Inspect quality and keep/drop decisions | review metadata |
| Mask Generation | Remove unwanted image regions | `masks/` |
| SfM | Solve or import cameras and sparse points | registered SfM artifact |
| Dataset | Convert SfM output for a trainer | `output/...` |
| Training | Launch a compatible external application | external training run |
| Perspective Export | Design and render rectilinear views | image/video sequence or COLMAP rig |

Jobs should write into temporary staging locations and only commit complete outputs. Interrupted v1 job records are marked for recovery instead of being treated as complete.
