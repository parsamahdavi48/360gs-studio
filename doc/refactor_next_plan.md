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
  CLI entry points; release inclusion policy still needs a dedicated pass.

## Status Summary

| Area | Status | Merge Blocker |
| --- | --- | --- |
| Runner/build command type contract | Complete | No |
| YOLO/SAM mask runtime state | Pending | No |
| Scene inventory helper audit | Pending | No |
| Script/release surface cleanup | Pending | No |
| Large GUI construction files | Deferred | No |
| Final verification pass | Passed for current checkpoint | Yes, rerun before next merge |

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

Status: passed for current checkpoint; rerun before the next merge

### Code Evidence

`core/yolo_mask.py` has `YoloMaskRuntimeSettings` and
`apply_runtime_settings()`, but the processing path still reads compatibility
globals such as `LEVEL`, `QUALITY`, `PROJECTION`, `EXPAND`, and `PROFILE`.
Model cache globals (`yolo`, `yolo_models`, `sam`) are acceptable as per-process
worker caches, but settings globals make preview/save consistency harder to
reason about.

### Plan

- Introduce an explicit runtime context dataclass that wraps
  `YoloMaskRuntimeSettings`, active recipe, profile recorder, and process count.
- Move image processing entry points toward accepting the runtime context
  explicitly.
- Preserve model caches as process-local caches.
- Keep the current CLI adapter and worker process boundary, but make it a thin
  creator of runtime context rather than the owner of global settings.
- Keep mask polarity unchanged: white means usable, black means excluded.

### Completion Criteria

- Main processing functions no longer read `LEVEL`, `QUALITY`, `PROJECTION`,
  `EXPAND`, or `PROFILE` directly.
- Compatibility globals, if still present, are limited to CLI/backward adapter
  setup or deleted.
- Preview and saved-mask routes use the same runtime interpretation.

### Suggested Tests

- `.\.venv\Scripts\python.exe -m pytest tests\test_yolo_mask_profile.py tests\test_step3_mask_guard.py tests\test_mask_preview.py -q`
- `.\.venv\Scripts\python.exe -m pytest tests\test_custom_mask.py tests\test_stitch_mask.py tests\test_sky_mask.py -q`

## 3. Scene Inventory Helper Audit

Status: pending

### Code Evidence

The main mixed-source workflows use `core/scene_inventory.py`, but there are
still helper-level direct file scans. Some are appropriate because they scan
outputs, sparse model folders, or cleanup targets. Others may still infer image
semantics directly and should be audited before future mixed-source work.

Observed candidates:

- `gui/steps/colmap_text_model_tool.py`: `_guess_single_file()` uses scene-level
  globbing for XML/PLY default discovery. This is probably acceptable because it
  does not infer image projection/source ownership.
- `gui/steps/realityscan_lfs_tool.py`: `_first_existing()` uses output-folder
  globbing for RealityScan CSV/PLY defaults. This is probably acceptable.
- `core/metashape_preprocess.py`: scans `images_dir.rglob("*")`; verify whether
  it still needs inventory-style image metadata for mixed sources.
- `core/spheresfm_project.py`: scans image files directly; SphereSfM remains
  constrained to same-resolution ERP-only input, so keep this paired with
  `core/sfm_preflight.py`.
- `core/mask_targets.py`, `core/custom_mask.py`, `core/sky_mask.py`: scan target
  images inside worker processes; these should remain driven by manifest/image
  list when source scoping is required.

### Plan

- Classify each direct scan as one of:
  `asset discovery`, `output cleanup`, `external sparse discovery`,
  `worker target scan`, or `image semantic inference`.
- Replace only `image semantic inference` scans with `SceneInventory`.
- Add comments for intentionally retained scans where a future audit would
  otherwise misread them.
- Ensure SphereSfM keeps the documented constraint: same-resolution ERP-only
  input.

### Completion Criteria

- No route derives projection/source ownership from filenames when
  `SceneInventory` can provide it.
- Direct scans that remain have a narrow reason and tests.
- Mixed ERP resolution, normal video, and still sequence workflows still pass
  through Step 3/4/5 tests.

### Suggested Tests

- `.\.venv\Scripts\python.exe -m pytest tests\test_scene_inventory.py tests\test_sfm_preflight.py tests\test_sfm_input_plan.py -q`
- `.\.venv\Scripts\python.exe -m pytest tests\test_metashape_preprocess.py tests\test_metashape_nerf_dataset.py tests\test_metashape_colmap_dataset.py -q`
- `.\.venv\Scripts\python.exe -m pytest tests\test_step3_mask_plan.py tests\test_step3_mask_guard.py -q`

## 4. Script And Release Surface Cleanup

Status: pending

### Code Evidence

GUI runtime no longer depends on `scripts/*.py`, but these tracked files remain:

- `scripts/update_venv.py`
- `scripts/check_venv.py`
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

`scripts/create_release_zip.py` currently excludes tests/devtools and itself,
but it does not yet classify developer-only scripts separately.

### Plan

- Decide which scripts are part of the end-user release surface:
  likely `update_venv.py`, `check_venv.py`, and setup/release support only.
- Move developer-only helpers under `devtools/` or exclude them from release
  ZIPs while keeping tests for the supported setup scripts.
- Remove or keep thin CLI wrappers based on whether they are still useful for
  local diagnostics. They must not become GUI runtime dependencies.
- Update `tests/test_release_zip.py` to assert the intended release surface.
- Update architecture docs if the policy changes from "thin wrappers may be
  included" to "developer wrappers are excluded from release".

### Completion Criteria

- Release ZIP contains only runtime GUI files, setup/check support, docs, and
  intentionally supported assets.
- Developer-only CLI wrappers are either moved under `devtools/` or excluded by
  `include_in_release()`.
- `rg "scripts[\\\\/]" gui core` still shows no GUI/core runtime dependency on
  script implementations.

### Suggested Tests

- `.\.venv\Scripts\python.exe -m pytest tests\test_release_zip.py tests\test_update_venv_script.py tests\test_check_venv.py -q`
- `.\.venv\Scripts\python.exe scripts\create_release_zip.py --skip-setup-verify`

## 5. Large GUI Construction Files

Status: deferred

### Code Evidence

The behavioral split is much better, but several files remain large because
they still construct complex UI:

- `gui/steps/step3_mask.py`
- `gui/steps/step4_cubemap.py`
- `gui/steps/step4_training_ui.py`
- `gui/steps/sfm_step.py`
- `gui/steps/colmap_text_model_tool.py`

Large size alone is not a bug. These should only be split further when a split
creates a clearer ownership boundary, not just to reduce line count.

### Plan

- Do not split purely by line count.
- Extract reusable UI widgets only when they have a stable state contract.
- Keep route/page orchestration in the step class; move repeated controls,
  validation groups, and preview panels into focused helper modules.

### Completion Criteria

- New feature work does not add domain logic back into widget construction
  files.
- Any new split has tests that exercise the extracted contract rather than only
  the previous monolithic step.

## 6. Final Verification Before Next Merge

Status: pending

Run this after implementing any of the above areas:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

Latest checkpoint result:

- Passed: `.\.venv\Scripts\python.exe -m ruff check .`
- Passed: `.\.venv\Scripts\python.exe -m pytest -q` (`976 passed`)

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
