# 360GS Studio v0.1.0 Development Preview 1

This is the first public Windows development build of 360GS Studio. It is intended for testing the unified workstation, Persian interface, project migration, and perspective-export workflow before the stable release.

## Highlights

- Docked Midnight workstation interface with persistent layouts.
- English, Japanese, and Persian UI; Persian uses right-to-left layout.
- Non-destructive migration from `_stechdrive` project metadata.
- Automatic project-frame flow into Perspective Export.
- Arbitrary yaw, pitch, roll, FOV, resolution, and interpolation per view.
- PNG, JPEG, silent HEVC, and COLMAP-rig exports.
- Persisted progress, safe cancellation, and interrupted-job recovery.
- Packaged desktop application and CLI with no system Python requirement.

## Downloads

- Use the `setup.exe` asset for a normal Windows installation.
- Use the `windows-x64-portable.zip` asset for an installation-free copy. Extract the complete ZIP and keep `_internal` beside both executable files.
- Verify either download with `SHA256SUMS.txt`.

## Development-preview limitations

- Builds are unsigned unless the release workflow has access to an Authenticode certificate. Windows may show a SmartScreen warning for unsigned builds.
- FFmpeg, COLMAP, CUDA/PyTorch ML components, model weights, and external trainers are detected or installed separately.
- LichtFeld Studio, Postshot, Brush, gsplat, Metashape, and RealityScan are not bundled.
- Proprietary Insta360 `.insv` decoding is not included; export an equirectangular video with Insta360 software first.
- Perspective MP4 output is silent in this version.

Please report problems through the repository's structured bug, compatibility, performance, or external-tool issue forms.
