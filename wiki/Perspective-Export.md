# Perspective Export

Perspective Export converts one equirectangular source into reusable rectilinear views.

When a project is open, the workspace automatically uses its `images/` folder after frame extraction. You can also choose a single panorama, a video, or an external image-sequence folder. The output defaults to `output/perspective/` inside the project.

## View controls

Every view stores:

- stable ID and name;
- enabled state;
- yaw, pitch, and roll;
- horizontal and optional vertical FOV;
- output width and height;
- interpolation method.

Use the cubemap preset for six 90° faces or the eight-view ring as a starting point. Edit interpolation directly in the view table. The same view definitions drive preview, image/video export, and COLMAP rig metadata.

The selected view is previewed from a reduced-resolution cached source frame. For image-sequence folders, the first supported image is used for preview while every supported image is included in the export.

## Output modes

- PNG sequence
- JPEG sequence
- Silent H.265/HEVC video
- COLMAP rig metadata for image exports

Automatic batching limits memory while sharing source decoding across several views. Review the estimated output size and batch count before running a large export.

## Cancellation and recovery

Progress is written to the project's job store while the export runs. Canceling removes temporary staging output, so an incomplete export is never presented as a finished dataset. If the application or computer stops during an export, reopen the project and use **Restore interrupted export** to reload its source, destination, output format, and view configuration. The restarted job keeps a link to the interrupted job for traceability.
