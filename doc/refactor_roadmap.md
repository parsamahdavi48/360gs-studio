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

- Cubemap conversion split continued: remap math, image IO, `transforms.json`
  export, and worker planning now live in dedicated `core` modules.
- Validation: `ruff check .` and `pytest -q` pass locally.

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

Status: in progress; remap, image IO, transforms export, and worker planning slices complete.

- `core/cubemap_view_spec.py` owns view/remap request validation.
- `core/cubemap_remap.py` owns rotation matrices and remap table generation.
- `core/cubemap_image_io.py` owns load/save, output format selection, dtype
  conversion, and channel-safe remapping for image and mask assets.
- `core/cubemap_transform_export.py` owns `transforms.json` camera/output logic
  and final-orientation point-cloud handling.
- `core/cubemap_worker_plan.py` owns worker count, remap cache sizing, and
  memory-budget decisions.
- `core/cubemap_transforms_json.py` is still too broad and owns worker
  orchestration plus image-only/COLMAP-rig metadata.
- Next slices:
  - Split image conversion worker orchestration away from the legacy
    `cubemap_transforms_json` facade.
  - Keep RealityScan sidecars in `core/realityscan_xmp.py` and audit job-runner
    boundaries instead of creating another sidecar module.

### 4. Scene Asset Contract

Status: mostly complete, needs application audit.

- `core/scene_inventory.py` normalizes projection, source kind/id, image size,
  normal-camera defaults, mask presence, mask polarity, and mask size matching.
- Step 3 source scoping and mixed SfM/dataset routes use inventory for the main
  workflows.
- Remaining follow-up: audit preview and helper routes for direct filename
  inference or ad hoc scene scans that should use inventory.

### 5. Large GUI Step Split

Status: not complete.

- Step 4 has several mixins, but `gui/steps/step4_cubemap.py` and
  `gui/steps/step4_training.py` remain large.
- `gui/steps/step3_mask.py` and `gui/steps/step1_extract.py` also remain large.
- Split order after core contracts settle:
  1. `step3_mask.py`
  2. `step1_extract.py`
  3. `step4_training.py`
  4. remaining `step4_cubemap.py` orchestration

### 6. Pre-Merge Audit

Status: not ready until Cubemap engine and large GUI split risks are reduced.

Audit gates:

- GUI runtime does not depend on root wrappers or `scripts/` implementations.
- Internal app work uses typed frame/workflow/SfM/dataset jobs or explicit
  worker payload contracts.
- Mixed source tests cover ERP multiple resolutions, normal video/stills, mask
  size mismatch, source-scoped mask regeneration, Metashape NeRF/COLMAP export,
  and RealityScan COLMAP export.
- Docs describe GUI-first behavior and release ZIP excludes old upstream assets.
