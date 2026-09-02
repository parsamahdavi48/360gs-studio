# Perspective Export

The seventh workspace defines reusable `ViewSpec v1` views. Drag the globe to
orbit, select markers to edit them, and right-click a marker to toggle it.
Every enabled view carries yaw, pitch, roll, horizontal and optional vertical
FOV, dimensions, interpolation, and a stable identifier.

Video exports decode once per bounded view batch and split that decode across
multiple FFmpeg `v360` filters. Still-image exports load a source frame once,
reuse cached projection maps, then release it. Output is staged and atomically
committed so cancellation never presents partial output as complete.

COLMAP rig export currently requires identical dimensions and FOV across the
enabled cameras. Silent HEVC clips use `libx265` by default and NVENC only after
capability detection.
