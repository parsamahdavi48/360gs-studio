# Your First Project

1. Create or choose a short ASCII-only scene path such as `D:\work\scene01`.
2. Add an equirectangular video, normal video, or still-image folder in **Frame Extraction**.
3. Extract frames and review keep/drop suggestions.
4. Open **Perspective Export** when you need rectilinear views. The project `images/` folder is selected automatically; design the views, check the preview, and export images, silent HEVC video, or a COLMAP rig.
5. Generate masks when people, the camera, tripod, sky, seams, or highlights should be excluded.
6. Run COLMAP or import a Metashape/RealityScan result.
7. Create the dataset required by LichtFeld Studio, Postshot, Brush, or gsplat.
8. Open the dataset in the external trainer or use the Training workspace launcher.

The **Project & Artifacts** dock shows project folders, registered SfM results, datasets, and persisted jobs. Double-click an item to open its location.

Long perspective exports can be canceled safely. If an export is interrupted by an application or computer restart, return to **Perspective Export** and choose **Restore interrupted export** before running it again.
