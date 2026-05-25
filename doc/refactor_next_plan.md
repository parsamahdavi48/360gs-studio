# Next Refactor Plan

Last updated: 2026-05-25

This document tracks the remaining design refactors after the GUI-first
standalone refactor. It is intentionally code-grounded: update the status,
evidence, tests, and checkpoint notes as each item is implemented.

## Goal

Keep the GUI runtime independent from legacy upstream CLI wrappers while making
the remaining contracts stricter and easier to maintain. Required external
processes are still allowed for FFmpeg/FFprobe, COLMAP, SphereSfM binaries,
LichtFeld Studio CLI, Postshot CLI, Windows process/link operations, and
intentional isolated worker processes for GPU-heavy or cancelable internal
jobs.

## Current Audit Snapshot

Branch at audit: `codex/standalone-app-core`

Checkpoint at audit: `03b672a Complete pre-merge audit cleanup`

Code checks performed:

- No root-level Python compatibility wrappers are present.
- GUI/core source does not call `scripts/*.py` as runtime implementation.
- GUI-internal jobs use `AppJob` through frame/workflow/SfM/dataset runners for
  the main app-owned work.
- Remaining `python -m core...` worker commands are intentional internal worker
  boundaries for mask generation and AprilTag scale estimation.
- `scripts/*.py` still exists for setup, release, diagnostics, and developer
  CLI entry points; release packaging now includes only setup/check support
  scripts.

## Status Summary

| Area | Status | Merge Blocker |
| --- | --- | --- |
| Runner/build command type contract | Complete | No |
| YOLO/SAM mask runtime state | Complete | No |
| Scene inventory helper audit | Complete | No |
| Script/release surface cleanup | Complete | No |
| Large GUI construction files | Complete | No |
| Final verification pass | Passed for current checkpoint | No, unless new changes are made |

## 1. Runner And Build Command Type Contract

Status: complete at this checkpoint

### Code Evidence

`gui/common/runner_types.py` now owns the shared queue aliases:
`ExternalCommand`, `ExternalCommandQueue`, `StepCommand`, `StepCommandPhase`,
and `StepCommandQueue`.

The step-level `build_commands()` and `confirm_commands()` contracts now use
`StepCommandQueue`. External-only helper queues use `ExternalCommandQueue`.

Verification search:

```powershell
rg -n -g "*.py" -- "def build_commands\(self\) -> list\[tuple\[str, object\]\]|def build_commands\(self\) -> list\[tuple\[str, list\[str\]\]\]|list\[tuple\[str, object\]\]|list\[tuple\[str, list\[str\]\]\]" gui\steps gui\common
```

Result at completion: no matches.

### Plan

- Implemented in `gui/common/runner_types.py`.
- `ProcessRunner`, `BaseStepWidget`, route steps, dataset tools, Step 1, Step 2,
  Step 3, Step 4, and Step 5 training now share the same queue contract.
- Raw `list[str]` command lists remain represented through `ExternalCommand`
  and `ExternalCommandQueue` for external executables and intentional worker
  processes.

### Completion Criteria

- Complete. The loose step command annotations were removed from `gui/steps`
  and `gui/common`.
- Complete. Existing Step 1, Step 3, Step 4, and workflow reorg command tests
  pass.

### Suggested Tests

- Passed: `.\.venv\Scripts\python.exe -m ruff check gui\steps gui\common core\app_job.py`
- Passed: `.\.venv\Scripts\python.exe -m pytest tests\test_step1_extract_ready.py tests\test_step3_mask_guard.py tests\test_step4_output_dir.py tests\test_workflow_reorg.py -q`

## 2. YOLO/SAM Mask Runtime State

Status: complete at this checkpoint

### Code Evidence

`core/yolo_mask.py` now has `YoloMaskRuntimeContext`, which wraps
`YoloMaskRuntimeSettings`, the optional `ProfileRecorder`, and per-run mutable
state such as `proc_count`.

The normal processing path now passes the context through `profile_timer()`,
`profile_record_inference()`, `add_yolo_mask()`, `detect_yolo_bboxes()`,
`detect_yolo_bboxes_batch()`, `add_sam_mask()`, `detect_bottom_mask()`,
`process_image_path()`, and `process_file()`.

Compatibility globals such as `LEVEL`, `QUALITY`, `PROJECTION`, `EXPAND`, and
`PROFILE` remain for legacy direct calls and CLI adapter setup, but the image
processing path uses `context.settings` and `context.profile`.

### Plan

- Implemented `YoloMaskRuntimeContext` and `create_runtime_context()`.
- `apply_runtime_settings()` now returns and activates a context while still
  updating compatibility globals for old direct-call behavior.
- Processing functions accept an explicit context and fall back to a
  compatibility context only when no active context is supplied.
- Model caches remain process-local globals.
- The CLI adapter still owns argument parsing and worker process execution, but
  it now creates one runtime context and passes it through the run.
- Mask polarity remains unchanged: white means usable, black means excluded.

### Completion Criteria

- Complete. Main processing functions use `YoloMaskRuntimeContext` rather than
  reading settings globals directly.
- Complete. Compatibility globals remain only for adapter/fallback behavior and
  existing direct-call tests.
- Complete. Preview and saved-mask routes use the same runtime interpretation
  through the worker command settings and context.

### Suggested Tests

- Passed: `.\.venv\Scripts\python.exe -m ruff check core\yolo_mask.py tests\test_yolo_mask_bottom.py tests\test_yolo_mask_sam_merge.py tests\test_yolo_mask_profile.py`
- Passed: `.\.venv\Scripts\python.exe -m pytest tests\test_yolo_mask_bottom.py tests\test_yolo_mask_sam_merge.py tests\test_yolo_mask_profile.py -q` (`15 passed`)
- Passed: `.\.venv\Scripts\python.exe -m pytest tests\test_yolo_mask_profile.py tests\test_step3_mask_guard.py tests\test_mask_preview.py -q` (`86 passed`)
- Passed: `.\.venv\Scripts\python.exe -m pytest tests\test_custom_mask.py tests\test_stitch_mask.py tests\test_sky_mask.py -q` (`31 passed`)

## 3. Scene Inventory Helper Audit

Status: complete at this checkpoint

### Code Evidence

The main mixed-source workflows use `core/scene_inventory.py`. This checkpoint
removed the remaining Metashape preprocess image-label semantic scan and
classified the retained direct scans.

Implemented changes:

- `core/metashape_preprocess.py` now delegates camera-label lookup to
  `build_scene_image_label_path_lookup()` and `resolve_scene_image_label()`
  instead of maintaining a separate `images_dir.rglob("*")` lookup.
- `core/scene_inventory.py` now includes labels relative to explicit external
  image roots, so labels such as `wide/pano64.jpg` resolve even when the image
  root is not named `images`.
- `core/scene_inventory.py` now rejects ambiguous basename/stem labels and
  reports warnings to Metashape preprocess callers instead of silently choosing
  one matching image.
- `core/spheresfm_project.py` keeps a direct prepared-image-root scan with a
  comment documenting the SphereSfM same-resolution ERP preflight contract.
- `core/mask_targets.py` keeps its worker fallback scan with a comment
  documenting that source-scoped GUI runs pass an `image_list` manifest before
  the worker starts.
- `gui/steps/colmap_text_model_tool.py` and
  `gui/steps/realityscan_lfs_tool.py` still use globbing for XML/PLY/CSV default
  discovery only; these scans do not infer image projection or source ownership.

### Plan

- Complete. Direct scans were classified as asset discovery, output cleanup,
  external sparse discovery, worker target scan, or image semantic inference.
- Complete. The Metashape preprocess image semantic inference scan now uses
  `SceneInventory` label lookup.
- Complete. Intentionally retained SphereSfM and mask worker scans have
  comments documenting their contract.
- Complete. SphereSfM remains paired with the same-resolution ERP-only preflight
  contract.

### Completion Criteria

- Complete. No audited route derives projection/source ownership from filenames
  when `SceneInventory` can provide it.
- Complete. Retained direct scans have a narrow reason: asset discovery, output
  cleanup, sparse discovery, prepared SphereSfM input scanning, or worker
  fallback target scanning.
- Complete. Mixed ERP resolution, normal video, and still sequence workflows
  still pass through the targeted Step 3/4/5 tests listed below.

### Suggested Tests

- Passed: `.\.venv\Scripts\python.exe -m ruff check core\scene_inventory.py core\metashape_preprocess.py core\spheresfm_project.py core\mask_targets.py tests\test_scene_inventory.py tests\test_metashape_preprocess.py`
- Passed: `.\.venv\Scripts\python.exe -m pytest tests\test_scene_inventory.py tests\test_sfm_preflight.py tests\test_sfm_input_plan.py -q` (`18 passed`)
- Passed: `.\.venv\Scripts\python.exe -m pytest tests\test_metashape_preprocess.py tests\test_metashape_nerf_dataset.py tests\test_metashape_colmap_dataset.py -q` (`18 passed`)
- Passed: `.\.venv\Scripts\python.exe -m pytest tests\test_step3_mask_plan.py tests\test_step3_mask_guard.py -q` (`56 passed`)

## 4. Script And Release Surface Cleanup

Status: complete at this checkpoint

### Code Evidence

GUI runtime no longer depends on `scripts/*.py`. Release packaging now has an
explicit supported script surface:

- Included in release ZIPs:
  - `scripts/update_venv.py`
  - `scripts/check_venv.py`
- Excluded from release ZIPs:
  - `scripts/create_release_zip.py`
  - `scripts/estimate_apriltag_scale.py`
  - `scripts/benchmark_yolo_mask.py`
  - `scripts/dev_apriltag_scene_viewer.py`
  - `scripts/dev_scene_preview_viewer.py`
  - `scripts/export_metashape_colmap_dataset.py`
  - `scripts/export_metashape_nerf_dataset.py`
  - `scripts/import_image_sequence.py`
  - `scripts/prepare_colmap_mixed_project.py`
  - `scripts/prepare_spheresfm_project.py`
  - `scripts/run_workflow_job.py`
  - `scripts/spheresfm_gpu_preflight.py`
  - `scripts/spheresfm_to_transforms.py`

`scripts/create_release_zip.py` owns `RELEASE_SCRIPT_PATHS`, and
`include_in_release()` now excludes every `scripts/*.py` path except
`scripts/update_venv.py` and `scripts/check_venv.py`. `validate_release_member()`
also rejects developer-only script paths so accidental ZIP inclusion fails
early.

`doc/architecture.md` and `doc/architecture.ja.md` now document that developer
script wrappers may remain in the repository for local maintenance/tests, but
they are not part of the end-user release surface.

### Plan

- Complete. The end-user release script surface is limited to setup/check
  support: `scripts/update_venv.py` and `scripts/check_venv.py`.
- Complete. Developer-only helpers remain in the repository for tests and
  maintenance, but are excluded by `include_in_release()`.
- Complete. `tests/test_release_zip.py` asserts the complete `scripts/*.py`
  release policy, so newly added script wrappers require an explicit release
  decision.
- Complete. Architecture docs now state that developer wrappers are excluded
  from release ZIPs.

### Completion Criteria

- Complete. Release ZIPs contain only runtime GUI files, setup/check support,
  docs, and intentionally supported assets from this audit surface.
- Complete. Developer-only CLI wrappers are excluded by `include_in_release()`
  and rejected by `validate_release_member()`.
- Complete. `rg "scripts[\\\\/]" gui core` shows no GUI/core runtime dependency
  on script implementations.

### Suggested Tests

- Passed: `.\.venv\Scripts\python.exe -m ruff check scripts\create_release_zip.py tests\test_release_zip.py`
- Passed: `.\.venv\Scripts\python.exe -m pytest tests\test_release_zip.py tests\test_update_venv_script.py tests\test_check_venv.py -q` (`34 passed`)
- Passed: `rg "scripts[\\\\/]" gui core` (no matches)
- Passed: `.\.venv\Scripts\python.exe scripts\create_release_zip.py --skip-setup-verify`
  - ZIP script members: `scripts/check_venv.py`, `scripts/update_venv.py`

## 5. Large GUI Construction Files

Status: complete at this checkpoint

### Code Evidence

The behavioral split is much better, but several files remain large because
they still construct complex UI:

- `gui/steps/step3_mask.py`
- `gui/steps/step4_cubemap.py`
- `gui/steps/step4_training_ui.py`
- `gui/steps/sfm_step.py`
- `gui/steps/colmap_text_model_tool.py`

Large size alone is not a bug. This checkpoint extracted a repeated, stable UI
contract instead of splitting files only by line count.

Implemented changes:

- `gui/steps/step4_widgets.py` now owns `OutputImageControls` and
  `make_output_image_controls()`, the shared output image format/bit-depth,
  mask inversion, and JPG quality controls used by cubemap-style dataset
  exporters.
- `gui/steps/step4_cubemap.py` and
  `gui/steps/colmap_text_model_tool.py` now reuse this widget contract while
  preserving the existing attribute names consumed by command planning and
  tests.
- The remaining large GUI files keep route/page orchestration and complex
  layout construction locally. Further extraction should be triggered by a new
  stable state contract, repeated validation group, or reusable preview panel,
  not by file length alone.

### Plan

- Complete. No broad split was made purely by line count.
- Complete. The repeated output-image controls were extracted into
  `step4_widgets.py` with a tested state contract.
- Complete. Route/page orchestration remains in the step classes; only the
  repeated control group moved.

### Completion Criteria

- Complete. The extraction does not move domain logic into widget construction;
  it centralizes a reusable UI control contract.
- Complete. `tests/test_step4_widgets.py` exercises the extracted contract
  directly, while Step 4/5 workflow tests verify the previous attributes remain
  usable.

### Suggested Tests

- Passed: `.\.venv\Scripts\python.exe -m ruff check gui\steps\step4_widgets.py gui\steps\step4_cubemap.py gui\steps\colmap_text_model_tool.py tests\test_step4_widgets.py`
- Passed: `.\.venv\Scripts\python.exe -m pytest tests\test_step4_widgets.py tests\test_workflow_reorg.py tests\test_step4_output_dir.py -q` (`124 passed`)

## 6. Final Verification Before Next Merge

Status: passed for current checkpoint; rerun if new changes are made

Run this after implementing any of the above areas:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

Latest checkpoint result:

- Passed: `.\.venv\Scripts\python.exe -m ruff check .`
- Passed: `.\.venv\Scripts\python.exe -m pytest -q` (`985 passed`)

Before release packaging changes, also run:

```powershell
.\.venv\Scripts\python.exe scripts\create_release_zip.py
```

## Update Rules For Codex

When implementing a task from this plan:

1. Re-run the relevant code search from the task's "Code Evidence" section.
2. Update `Status`, `Code Evidence`, `Completion Criteria`, and test results in
   this file.
3. Keep completed items in the document with the commit hash and validation
   command results.
4. If a task is intentionally deferred, record why and what would trigger it.
5. Do not mark the final verification complete until full `ruff` and full
   `pytest` pass after the implementation.
