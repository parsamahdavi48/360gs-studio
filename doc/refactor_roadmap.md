# Refactor Roadmap

This roadmap is the working checklist for the standalone GUI refactor. Update
the status before and after each focused refactor so the current point is clear.

## Goal

Keep the app GUI-first and robust for mixed 360 ERP, normal video, and still
image sequence sources. Internal app work should use typed `core` contracts and
job runners; raw command execution should be limited to external tools or
intentional isolated worker processes.

## Current Status

Last updated: 2026-05-24

Latest checkpoint:

- Large GUI split continued for Step 3: mask command planning and JSONL target
  manifest writing now live in dedicated modules with focused tests.
- Validation at the latest checkpoint: `ruff check .` and `pytest -q` pass
  locally.

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

Status: in progress; Step 3 first slices complete.

- Step 4 has several mixins, but `gui/steps/step4_cubemap.py` and
  `gui/steps/step4_training.py` remain large.
- `gui/steps/step3_mask_progress.py` owns Step 3 worker-output progress
  parsing.
- `gui/steps/step3_mask_records.py` owns Step 3 mask metadata recording.
- `gui/steps/step3_mask_plan.py` owns batch command ordering, phase names,
  target manifest decisions, and mixed-projection planning.
- `gui/steps/step3_mask_manifests.py` owns Step 3 JSONL target/projection
  manifest writing.
- `gui/steps/step3_mask.py` and `gui/steps/step1_extract.py` still remain
  large.
- Split order after core contracts settle:
  1. Continue `step3_mask.py`: separate settings/state helpers from widget
     layout, then move preview action orchestration if useful.
  2. `step1_extract.py`
  3. `step4_training.py`
  4. remaining `step4_cubemap.py` orchestration

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
