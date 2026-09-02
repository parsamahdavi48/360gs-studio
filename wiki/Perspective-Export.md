# Perspective Export

Perspective Export converts one equirectangular source into reusable rectilinear views.

## View controls

Every view stores:

- stable ID and name;
- enabled state;
- yaw, pitch, and roll;
- horizontal and optional vertical FOV;
- output width and height;
- interpolation method.

Use the cubemap preset for six 90° faces or the eight-view ring as a starting point. The same view definitions drive preview, image/video export, and COLMAP rig metadata.

## Output modes

- PNG sequence
- JPEG sequence
- Silent H.265/HEVC video
- COLMAP rig metadata for image exports

Automatic batching limits memory while sharing source decoding across several views. Review the estimated output size and batch count before running a large export.
