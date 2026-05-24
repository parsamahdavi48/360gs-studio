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

- Large GUI split continued into Step 1. Input source queue/autoload handling,
  extraction readiness/job construction/progress parsing, and video probe/source
  registry/estimate helpers now live in dedicated modules. `step1_extract.py`
  is now mostly widget layout plus mode-control glue.
- Validation at the latest checkpoint: Step 1 targeted `ruff` and
  `tests/test_step1_extract_ready.py tests/test_form_tooltips.py` pass locally;
  full `ruff check .` and `pytest -q` also pass locally.

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

Status: in progress; Step 3 first slices complete and Step 1 first slices complete.

- Step 4 has several mixins, but `gui/steps/step4_cubemap.py` and
  `gui/steps/step4_training.py` remain large.
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
  2. Split `step4_training.py`.
  3. Split remaining `step4_cubemap.py` orchestration.

### 6. Pre-Merge Audit

Status: not ready until large GUI split risks are reduced.

Audit gates:

- GUI runtime does not depend on root wrappers or `scripts/` implementations.
- Internal app work uses typed frame/workflow/SfM/dataset jobs or explicit
  worker payload contracts.
- Mixed source tests cover ERP multiple resolutions, normal video/stills, mask
  size mismatch, source-scoped mask regeneration, Metashape NeRF/COLMAP export,
  and RealityScan COLMAP export.
- Docs describe GUI-first behavior and release ZIP excludes old upstream assets.
