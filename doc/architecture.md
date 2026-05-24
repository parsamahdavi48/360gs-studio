# Maintenance Architecture

This project is maintained as a Windows-first GUI application. Reusable
implementation code lives in focused `core/` modules, and the GUI should call
typed core jobs directly whenever the work is internal to this app.

## Entry Points

Root-level compatibility `*.py` wrappers are intentionally not part of the
application structure. Do not add new GUI behavior by building shell command
strings for repository Python scripts. Add typed payloads and core runners
first. If a process boundary is needed for robustness, use a `core` module entry
point rather than reintroducing root wrappers.

Shared implementation lives in `core/`. Tests should cover the implementation
contract and the GUI job payloads. Public CLI compatibility is secondary to the
GUI workflow unless a release note explicitly promises a specific wrapper.

## Core Contracts

- `core/video_info.py` owns the shared video metadata dataclass used by frame
  extraction and downstream planning.
- `core/frame_pair_analysis.py` owns pair-frame metrics, blur/track risk
  thresholds, dependency checks, and selection analysis. Keep OpenCV/numpy
  imports guarded so command help and unrelated tests do not require optional
  runtime dependencies.
- `core/extract_frames.py` owns FFmpeg/FFprobe frame extraction, cache handling,
  stationary thinning, and decision CSV writing. Step 1 invokes it through the
  frame job runner.
- `core/frame_renumbering.py` owns the Step 2 kept-image renumbering contract:
  downstream-output blockers, collision-safe rename planning/application, and
  updates to frame/source image metadata when paths change.
- `core/scene_inventory.py` owns the shared read contract for scene images,
  masks, projection type, source IDs, and normal-camera defaults. Features that
  scope work to specific input sources should use `SceneInventory.source_groups()`
  instead of inferring source ownership from filenames.
- `core/scene_import*.py` owns external scene re-registration. Import should
  scan only the app-defined scene folders, replace external-import metadata as a
  full reimport, and avoid deleting real image/mask/output assets.
- `core/app_job.py`, `core/workflow_job_runner.py`, `core/sfm_job_runner.py`,
  `core/frame_job_runner.py`, and `core/dataset_job_runner.py` own internal GUI
  job execution. GUI steps should return `AppJob` for app-internal Python work
  and reserve raw command lists for external executables such as COLMAP, FFmpeg,
  SphereSfM binaries, or training application CLIs. GPU-heavy mask generation
  may run as `python -m core.<module>` to keep model memory and crashes isolated
  from the GUI process.
- `core/*_job_spec.py` modules own versioned job payload builders and
  validation. Keep required fields, value ranges, and view-set structure checked
  before a job file is written or executed. This applies to frame extraction and
  review finalization as well as workflow, SfM, and dataset jobs. Shared
  validation helpers live in `core/job_payload_validation.py`.
- `core/mask_job_spec.py` owns Step 3 mask command payloads. GPU-heavy mask
  workers still run in separate `python -m core.<module>` processes, but GUI
  command builders should create validated mask payloads first and then render
  those payloads to commands.
- `core/yolo_mask.py` normalizes YOLO/SAM runtime settings with
  `YoloMaskRuntimeSettings`; global runtime state updates are centralized in
  `apply_runtime_settings()`. Existing processing functions may still read the
  compatibility globals, but new settings must enter through this path.
- SphereSfM project preparation, GPU preflight, and sparse-model conversion live
  in `core/spheresfm_project.py`, `core/spheresfm_gpu_preflight.py`, and
  `core/spheresfm_to_transforms.py`. Matching files under `scripts/` are thin
  developer/CLI entry points and must not become the runtime implementation.
- Mask modules preserve the repository-wide mask polarity contract:
  white pixels are usable, black pixels are excluded. Mask merges should remain
  AND-style unless a tool explicitly documents a different operation.
- Metashape coordinate handling is centralized in `core/metashape_coordinates.py`.
  Any route that converts Metashape XML camera poses or PLY points must use this
  module instead of duplicating axis matrices. The Step 4 Metashape preprocess
  job uses `core/metashape_preprocess.py` to create the intermediate
  equirectangular `transforms.json`; no GUI route should depend on the old
  upstream Metashape converter.
- Cubemap view sets and remap request validation live in
  `core/cubemap_view_spec.py`. Default Cube6 views, custom view JSON parsing,
  and input-size/FOV/output-size checks should go through this module rather
  than adding ad hoc view parsing in image conversion code.
- Cubemap and COLMAP exports must preserve coordinate profile semantics:
  Postshot uses the default cubemap transform, Brush uses the Brush transform,
  and LichtFeld cubemap export writes final-orientation-corrected
  `transforms.json` plus `pointcloud.ply` from the Cubemap CLI. LichtFeld GUT
  ERP 360° output uses equirectangular inputs without cubemap conversion and
  applies the same final orientation correction while creating the direct
  dataset. RealityScan realignment output is handled by the Step 4 Metashape
  route: it cancels the Metashape import coordinate conversion, maps Metashape
  Y-up poses into RealityScan Z-up local Euclidean axes, then writes cubemap
  images and XMP sidecars under
  `output/realityscan/`; it does not require or pass Metashape PLY because
  RealityScan should regenerate its own point cloud after alignment.

## GUI Contracts

GUI steps should be thin orchestration surfaces. UI labels, hints, warnings, and
tooltips belong in `gui/i18n_ja.py` and `gui/i18n_en.py`, with Japanese and
English text kept semantically matched.

Step 3 mask generation is split by responsibility:

- `gui/steps/step3_mask.py` owns the page layout, controls, and state wiring.
- `gui/steps/step3_mask_actions.py` owns preview/regeneration actions and
  selected-image processing.
- `gui/steps/mask_commands.py` owns command construction.
- `gui/steps/mask_postprocess.py` owns saved-mask postprocessing.

SfM route selection, dataset conversion, and training launch are split by responsibility:

- `gui/steps/sfm_step.py` owns the Step 4 route cards.
- `gui/steps/dataset_step.py` owns the Step 5 dataset tool menu and delegates
  execution contracts to the selected tool.
- `gui/steps/step4_contracts.py` owns stable route/profile/output constants and
  diagnostic classifiers.
- `gui/steps/step4_command_plan.py` owns executable resolution and command
  planning.
- `gui/steps/step4_pipeline.py` owns route readiness and sub-stage navigation.
- `gui/steps/step4_paths.py` owns output path contracts, cleanup/reset, artifact
  validation, and sparse-model discovery.
- `gui/steps/step4_manifest.py` owns export settings and run manifest
  persistence.
- `gui/steps/step4_runtime.py` owns runtime completion handling, progress
  parsing, Metashape input discovery, and post-run actions.
- `gui/steps/step4_training.py` owns training backend UI, settings, and launch
  commands.
- `gui/steps/step4_widgets.py` owns small reusable widgets used by the step.
- `gui/steps/step4_cubemap.py` is the orchestration class that composes those
  mixins for Step 5 dataset creation tools. Avoid adding new workflow logic
  there when it clearly belongs to one of the focused modules above.

## Verification Rules

Use Ruff when changing Python code:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

For formatting, prefer formatting only files changed by the current work when
the repository contains existing untouched formatting differences.

Run the full test suite before committing broad refactors:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Before publishing a release, run the full release gate:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\create_release_zip.py
```

The release ZIP command must keep its setup preflight verification enabled
unless the owner explicitly asks to skip it.

When checking the PySide GUI offscreen in Codex or CI-like environments, set
`QT_QPA_PLATFORM=offscreen` and call `apply_theme(app)` before constructing
`MainWindow`. The theme applies a Windows Japanese-capable font such as Meiryo
UI, which prevents Japanese labels from rendering as square glyphs.
