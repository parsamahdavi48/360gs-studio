# Refactor Roadmap

This roadmap is the working checklist for the standalone GUI refactor. Update
the status before and after each focused refactor so the current point is clear.

## Goal

Keep the app GUI-first and robust for mixed 360 ERP, normal video, and still
image sequence sources. Internal app work should use typed `core` contracts and
job runners; raw command execution should be limited to external tools or
intentional isolated worker processes.

## Current Status

Last updated: 2026-05-25

Latest checkpoint:

- Large GUI split continued through the remaining Step 4 orchestration. Route
  state/path summaries, target profile/output-shape state, activation/scene
  preview behavior, and preview image-count updates now live in focused mixins.
  `step4_cubemap.py` is mostly widget construction plus scene setup.
- The cubemap legacy facade dependency audit was also tightened: non-CLI core
  imports now use the split cubemap modules directly; the facade is retained for
  CLI entry and compatibility tests.
- Validation at the latest checkpoint: Step 4 targeted `ruff` and
  `tests/test_colmap_mixed_project.py tests/test_apriltag_detection_pipeline.py
  tests/test_cubemap_view_spec.py tests/test_step4_output_dir.py
  tests/test_form_tooltips.py tests/test_scene_preview.py` pass locally; full
  `ruff check .` and `pytest -q` also pass locally.

### 1. Mask Job Contract

Status: mostly complete.

- `core/mask_job_spec.py` owns mask payload builders, validation, and command
  rendering.
- `gui/steps/mask_commands.py` builds validated payloads before converting to
  worker commands.
- GPU-heavy workers still run as separate `python -m core.<module>` processes
  by design.
- Remaining follow-up: decide whether source scope should become a first-class
  mask job field or stay resolved by the GUI into `image_list`.

### 2. Mask Core State

Status: first stage complete, not finished.

- `core/yolo_mask.py` has `YoloMaskRuntimeSettings` and
  `apply_runtime_settings()`.
- Compatibility globals remain inside the worker implementation.
- Remaining follow-up: move more processing entry points to options-driven
  function calls while preserving per-process model caches.

### 3. Cubemap Conversion Engine

Status: mostly complete; legacy facade remains for CLI compatibility.

- `core/cubemap_view_spec.py` owns view/remap request validation.
- `core/cubemap_remap.py` owns rotation matrices and remap table generation.
- `core/cubemap_image_io.py` owns load/save, output format selection, dtype
  conversion, and channel-safe remapping for image and mask assets.
- `core/cubemap_image_conversion.py` owns image/mask conversion worker state,
  remap table caches, progress-counting, and bounded process-pool execution.
- `core/cubemap_transform_export.py` owns `transforms.json` camera/output logic
  and final-orientation point-cloud handling.
- `core/cubemap_export_metadata.py` owns image-only and COLMAP-rig export
  manifests plus image-only input size discovery.
- `core/cubemap_worker_plan.py` owns worker count, remap cache sizing, and
  memory-budget decisions.
- `core/cubemap_transforms_json.py` is now a thin legacy facade for CLI entry
  and compatibility imports.
- Remaining follow-up:
  - Audit internal imports and remove any remaining low-level dependency on the
    legacy facade when it is not providing the CLI entry point or view helpers.
  - Keep RealityScan sidecars in `core/realityscan_xmp.py` and audit job-runner
    boundaries instead of creating another sidecar module.

### 4. Scene Asset Contract

Status: mostly complete; main Step 3/Step 4/preview routes audited.

- `core/scene_inventory.py` normalizes projection, source kind/id, image size,
  normal-camera defaults, mask presence, mask polarity, and mask size matching.
- `core.scene_inventory.build_scene_image_label_path_lookup()` and
  `resolve_scene_image_label()` own case-insensitive external camera label to
  scene image resolution for Metashape-style labels.
- Step 3 source scoping and mixed SfM/dataset routes use inventory for the main
  workflows.
- Step 3 mask fallback image lists, Step 4 Metashape XML image matching, and
  scene preview Metashape image lookup now use inventory.
- Remaining follow-up: audit lower-priority helper routes for direct filename
  inference or ad hoc scene scans that should use inventory.

### 5. Large GUI Step Split

Status: in progress; Step 3, Step 1, and Step 4 GUI slices complete.

- Step 4 still has a large widget-construction method in
  `gui/steps/step4_cubemap.py`, but the main behavior/state orchestration now
  lives in focused mixins.
- `gui/steps/step1_input_sources.py` owns Step 1 input-source queue state,
  file/folder add/remove actions, scene autoload, source-video registry lookup,
  prefix allocation, and queue labels.
- `gui/steps/step1_execution.py` owns Step 1 readiness checks, extraction/import
  `AppJob` construction, phase labels, progress parsing, and completion refresh.
- `gui/steps/step1_video_info.py` owns Step 1 FFprobe metadata loading, source
  video registry writes, video/image-sequence info labels, and instant estimate
  formatting.
- `gui/steps/step1_extract.py` now focuses on widget construction, scene label
  display, source-mode controls, extraction-mode controls, and interval/profile
  field synchronization.
- `gui/steps/step4_training.py` now only composes focused training mixins.
- `gui/steps/step4_training_ui.py` owns Step 4/5 training widget construction
  and training layout behavior.
- `gui/steps/step4_training_settings_restore.py` owns persisted training
  settings restore for LichtFeld, Postshot, and custom training.
- `gui/steps/step4_training_backend_state.py` owns training backend selection,
  backend-scoped executable state, and backend-dependent visibility hooks.
- `gui/steps/step4_training_dataset.py` owns training dataset requirements,
  default paths, sparse model selection, image counting, and auto step scaling.
- `gui/steps/step4_training_lfs_state.py` owns LichtFeld strategy/default state,
  advanced-row visibility, scaler application, and color preview state.
- `gui/steps/step4_training_commands.py` owns training option validation,
  output collision guards, and LichtFeld/Postshot/custom command construction.
- `gui/steps/step4_project_settings.py` owns Step 4 project settings restore,
  external-import defaults, settings path/text normalization, and persisted
  user preferences.
- `gui/steps/step4_output_shape_selector.py` owns the two-choice output-shape
  radio selector widget used by Metashape and SphereSfM output settings.
- `gui/steps/step4_activation.py` owns Step 4 activation refresh,
  primary-action state, and the scene preview launcher.
- `gui/steps/step4_route_state.py` owns SfM route state, settings-tab
  availability, path summaries, sparse-model input sync, Metashape input hints,
  and COLMAP/SphereSfM route controls.
- `gui/steps/step4_profile_output.py` owns target profile defaults,
  RealityScan option visibility, output-shape eligibility, and view-export
  enablement.
- `gui/steps/step4_preview_counts.py` owns Step 4 preview rendering debounce,
  source image counting, and output image count display.
- `gui/steps/step3_mask_progress.py` owns Step 3 worker-output progress
  parsing.
- `gui/steps/step3_mask_records.py` owns Step 3 mask metadata recording.
- `gui/steps/step3_mask_plan.py` owns batch command ordering, phase names,
  target manifest decisions, and mixed-projection planning.
- `gui/steps/step3_mask_manifests.py` owns Step 3 JSONL target/projection
  manifest writing.
- `gui/steps/step3_mask_settings.py` owns Step 3 SAM prompt parsing,
  SAM3.1 merge-mode normalization, and the control-state conversion to
  `MaskCommandContext` / refresh-plan settings snapshots.
- `gui/steps/step3_mask_preview_actions.py` owns Step 3 temporary preview
  rendering, preview worker `QProcess` sequencing, and preview temp-directory
  cleanup.
- `gui/steps/step3_mask_actions.py` now focuses on command-builder adapters,
  selected-image mask reprocessing, and per-image mask postprocessing.
- `gui/steps/step3_mask_scene.py` owns Step 3 scene folder labels,
  projection/source inventory state, readiness checks, source-scoped target
  selection, and Step 2 review guards.
- `gui/steps/step3_mask_batch.py` owns batch command orchestration, target
  manifest dispatch, phase labels, batch completion handling, and mask
  metadata recording.
- `gui/steps/step3_mask_license.py` owns model license notices and SAM3.1
  checkpoint availability checks.
- `gui/steps/step3_mask_controls.py` owns selected model/class/prompt state,
  model option visibility, custom-mask picker state, image-dir preview sync,
  and `MaskCommandContext` creation.
- `gui/steps/step3_mask.py` and `gui/steps/step1_extract.py` are now mostly
  widget construction/control glue.
- Split order after core contracts settle:
  1. Review the new Step 1 source modules for any reusable source contracts
     that should move to `core/`; keep GUI-only queue/display behavior in
     `gui/steps/`.
  2. Run pre-merge audit and fix any remaining blockers.
  3. If the audit is clean, merge the refactor branch.

### 6. Pre-Merge Audit

Status: ready to run; initial spot checks are clean.

Initial spot checks:

- No root-level Python compatibility wrappers were found.
- No root-level upstream `images/` or `masks/` asset folders were found.
- GUI/core source does not import `scripts/` implementations.
- `core.cubemap_transforms_json` imports remain limited to the CLI/facade pair;
  non-CLI core users now import split cubemap modules directly.

Audit gates:

- GUI runtime does not depend on root wrappers or `scripts/` implementations.
- Internal app work uses typed frame/workflow/SfM/dataset jobs or explicit
  worker payload contracts.
- Mixed source tests cover ERP multiple resolutions, normal video/stills, mask
  size mismatch, source-scoped mask regeneration, Metashape NeRF/COLMAP export,
  and RealityScan COLMAP export.
- Docs describe GUI-first behavior and release ZIP excludes old upstream assets.
