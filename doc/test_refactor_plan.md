# Test Refactor Plan

Last updated: 2026-05-25

This document tracks the next test-suite refactor after the GUI-first runtime
refactor. The purpose is not to make tests pass by weakening them. The purpose
is to make tests exercise the current ownership boundaries directly, so future
maintenance catches real regressions without carrying old implementation seams.

## Goal

Keep the test suite aligned with the current architecture:

- GUI-owned app work is expressed as typed core jobs and `AppJob` phases.
- External tools stay as raw process commands only where they are real external
  dependencies.
- Long-running internal workers may still run as isolated `python -m core...`
  processes when this is the runtime contract.
- Tests should patch explicit seams, not unrelated module globals or legacy
  wrapper modules.
- Compatibility behavior can still be tested, but compatibility tests should be
  small and named as such.

## Current Audit Snapshot

Branch at audit: `codex/standalone-app-core`

Checkpoint at audit: `f32ba43 Extract shared output image controls`

Code checks performed:

- `rg -n "sys\.modules|getattr\(sys\.modules|tests can monkeypatch|QFileDialog|monkeypatch\.setattr\(step4_cubemap|monkeypatch\.setattr\(step3_mask_module|from scripts|import scripts|legacy|compat|互換" gui core tests`
- `Get-ChildItem tests -File -Filter *.py | Sort-Object Length -Descending`
- `.\.venv\Scripts\python.exe -m ruff check .`
- `.\.venv\Scripts\python.exe -m pytest -q` (`985 passed`)

## Status Summary

| Area | Status | Merge Blocker |
| --- | --- | --- |
| Explicit process and dialog seams | Complete | No |
| Step 4 test responsibility split | Complete | No |
| YOLO/SAM runtime-context tests | Complete | No |
| Developer-only scripts test surface | Complete | No |
| Shared GUI test fixtures | Complete | No |
| Final verification pass | Passed at T6 checkpoint | No |

## Implementation Queue

Work through this queue in order unless a later task becomes a prerequisite for
the current edit. Each item should finish with a targeted test run and an update
to this document.

| ID | Task | Primary Files | Done When |
| --- | --- | --- | --- |
| T1 | Replace test-only process/dialog monkeypatch seams with explicit seams | `gui/steps/step4_runtime.py`, `gui/steps/step1_input_sources.py`, Step 1/3/scene-viewer tests | Complete: runtime code has no `sys.modules` test lookup and tests patch named seams/providers |
| T2 | Split Step 4 omnibus tests by responsibility | Step 4 focused tests and `tests/helpers/step4.py` | Complete: Step 4 tests are grouped by contract without reducing assertions |
| T3 | Move YOLO/SAM behavior tests onto explicit runtime contexts | `tests/test_yolo_mask_profile.py`, `tests/test_yolo_mask_bottom.py`, `core/yolo_mask.py` tests | Complete: main tests build `YoloMaskRuntimeContext`; global mutation is limited to compatibility tests |
| T4 | Normalize developer-only script tests | `tests/test_benchmark_yolo_mask.py`, benchmark/devtool modules | Complete: tests no longer import dev-only benchmark code through `scripts.*` |
| T5 | Add shared GUI test fixtures where they reduce duplication | GUI-heavy tests, `tests/helpers/` | Complete: new/touched GUI tests use common app/theme/dialog/message helpers |
| T6 | Run final audit and full verification | whole repo | Complete: audit searches reviewed, ruff and full pytest pass, and this document records the final checkpoint |

### Per-Task Update Template

When one task is implemented, append or update these fields inside that task's
section before committing:

```text
Status: complete at <commit-or-checkpoint>

Implementation Notes:
- ...

Verification:
- <command> -> <result>

Deferred:
- None / <explicit reason>
```

## 1. Explicit Process And Dialog Seams

Status: complete at T1 checkpoint

### Code Evidence

Current tests still patch implementation globals that exist mainly because the
old monolithic modules were hard to test:

- `gui/steps/step4_runtime.py` uses
  `getattr(sys.modules.get("gui.steps.step4_cubemap"), "QProcess", QProcess)`
  to allow tests to replace `step4_cubemap.QProcess`.
- `tests/test_step4_output_dir.py` patches `step4_cubemap.QProcess` for the
  SphereSfM GUI launch test.
- `gui/steps/step1_input_sources.py` has `_step1_file_dialog()` with a docstring
  saying it exists so tests can monkeypatch `step1_extract.QFileDialog`.
- `tests/test_step1_extract_ready.py` patches
  `gui.steps.step1_extract.QFileDialog.getOpenFileNames` and
  `getExistingDirectory`.
- `tests/test_step3_mask_guard.py` imports `gui.steps.step3_mask as
  step3_mask_module` to patch `step3_mask_module.QFileDialog`.
- `tests/test_apriltag_scene_viewer.py` patches `scene_viewer_module.QFileDialog`.

### Refactor Plan

1. Add explicit process seams:
   - Prefer a small method such as `Step4RuntimeMixin._create_spheresfm_gui_process()`
     or a small `gui/common/process_factory.py` wrapper.
   - Runtime code should call that seam directly instead of looking into
     `sys.modules`.
   - Tests should replace the explicit seam or factory, not `step4_cubemap.QProcess`.
2. Add explicit dialog seams:
   - Introduce a small dialog provider module, for example
     `gui/common/dialogs.py`, with functions for selecting files/folders.
   - `BrowseWidget`, Step 1 source selection, Step 3 custom mask selection, app
     scene import, and scene preview open should call those provider functions.
   - Tests should patch provider functions or inject a provider object where
     the widget already has an ownership boundary.
3. Remove test-only comments/exports after the replacement:
   - `_step1_file_dialog()` should disappear.
   - `__all__` exports that expose `QFileDialog` only for tests should be
     revisited.
   - `sys.modules` lookup in `step4_runtime.py` should disappear.

### Completion Criteria

- No production code contains a comment saying a function exists so tests can
  monkeypatch it.
- No test patches `gui.steps.step4_cubemap.QProcess`.
- No test patches `gui.steps.step1_extract.QFileDialog` or
  `gui.steps.step3_mask.QFileDialog` directly.
- Process and dialog tests still verify the same user-facing behavior.

### Suggested Tests

- `.\.venv\Scripts\python.exe -m pytest tests\test_step4_output_dir.py -q`
- `.\.venv\Scripts\python.exe -m pytest tests\test_step1_extract_ready.py tests\test_step3_mask_guard.py -q`
- `.\.venv\Scripts\python.exe -m pytest tests\test_apriltag_scene_viewer.py tests\test_scene_preview_view.py -q`

### Implementation Notes

- Added `gui/common/dialogs.py` as the explicit dialog provider.
- Step 1 source selection, Step 3 custom-mask selection, app scene import,
  embedded scene preview, generic browse widgets, and the AprilTag dev viewer
  now call the provider instead of importing `QFileDialog` directly.
- `Step4RuntimeMixin` now owns `_create_spheresfm_gui_process()` and no longer
  reaches into `sys.modules` to let tests replace `QProcess`.
- Step 1/3/4 and AprilTag viewer tests now patch provider/factory seams instead
  of module-level Qt classes.

### Verification

- `.\.venv\Scripts\python.exe -m pytest tests\test_step1_extract_ready.py tests\test_step3_mask_guard.py -q` -> `93 passed`
- `.\.venv\Scripts\python.exe -m pytest tests\test_apriltag_scene_viewer.py tests\test_scene_preview_view.py -q` -> `53 passed`
- `.\.venv\Scripts\python.exe -m pytest tests\test_step4_output_dir.py -q` -> `112 passed`
- `.\.venv\Scripts\python.exe -m ruff check <T1 touched files>` -> passed

### Deferred

- None.

## 2. Step 4 Test Responsibility Split

Status: complete at T2 checkpoint

### Code Evidence

Before this task, `tests/test_step4_output_dir.py` was about 140 KB and covered
multiple independent responsibilities:

- output directory and reset behavior
- Metashape preprocessing and dataset job command planning
- COLMAP/SphereSfM route behavior
- RealityScan output behavior
- training backend path/settings behavior
- preview behavior
- AprilTag scale interactions

This mirrored the old large Step 4 implementation rather than the current split
across `step4_command_plan.py`, `step4_paths.py`, `step4_training_*`,
`step4_runtime.py`, and related focused modules.

The omnibus file has been removed. The tests now live in:

- `tests/test_step4_colmap_route.py`
- `tests/test_step4_dataset_jobs.py`
- `tests/test_step4_output_paths.py`
- `tests/test_step4_preview_behavior.py`
- `tests/test_step4_realityscan_route.py`
- `tests/test_step4_spheresfm_route.py`
- `tests/test_step4_training_paths.py`
- shared setup helpers in `tests/helpers/step4.py`

### Refactor Plan

Move tests incrementally without changing assertions:

1. Create focused test files:
   - `tests/test_step4_output_paths.py`
   - `tests/test_step4_dataset_jobs.py`
   - `tests/test_step4_spheresfm_route.py`
   - `tests/test_step4_realityscan_route.py`
   - `tests/test_step4_training_paths.py`
   - `tests/test_step4_preview_behavior.py`
2. Extract shared fixtures/helpers only when they reduce duplication without
   hiding behavior:
   - `tests/helpers/step4.py` can own `make_cubemap_step()`, lightweight scene
     builders, fake XML/PLY writers, and message-box helpers.
   - Keep assertions near the test that describes the behavior.
3. Preserve test names or use names that map to the user-facing behavior.
4. After each move, run the old and new targeted subsets before deleting the
   moved block from `test_step4_output_dir.py`.

### Completion Criteria

- Complete. `tests/test_step4_output_dir.py` was removed.
- Complete. Each new file maps to one Step 4 responsibility boundary.
- Complete. Shared helpers contain setup/writer utilities only; assertions stay
  in focused test files.
- Complete. Full Step 4 targeted tests pass after the split.

### Suggested Tests

- `.\.venv\Scripts\python.exe -m pytest tests\test_step4_colmap_route.py tests\test_step4_dataset_jobs.py tests\test_step4_output_paths.py tests\test_step4_preview_behavior.py tests\test_step4_realityscan_route.py tests\test_step4_spheresfm_route.py tests\test_step4_training_paths.py tests\test_step4_widgets.py -q`
- `.\.venv\Scripts\python.exe -m pytest tests\test_workflow_reorg.py tests\test_form_tooltips.py -q`

### Implementation Notes

- Split the 112-test Step 4 omnibus suite into focused files for COLMAP route,
  dataset jobs, output paths/settings, preview behavior, RealityScan route,
  SphereSfM route, and training paths.
- Moved shared Step 4 test setup into `tests/helpers/step4.py`.
- Removed `tests/test_step4_output_dir.py` so future tests must choose a
  responsibility-specific home.

### Verification

- `.\.venv\Scripts\python.exe -m ruff check tests\helpers\step4.py tests\test_step4_colmap_route.py tests\test_step4_dataset_jobs.py tests\test_step4_output_paths.py tests\test_step4_preview_behavior.py tests\test_step4_realityscan_route.py tests\test_step4_spheresfm_route.py tests\test_step4_training_paths.py` -> passed
- `.\.venv\Scripts\python.exe -m pytest tests\test_step4_colmap_route.py tests\test_step4_dataset_jobs.py tests\test_step4_output_paths.py tests\test_step4_preview_behavior.py tests\test_step4_realityscan_route.py tests\test_step4_spheresfm_route.py tests\test_step4_training_paths.py tests\test_step4_widgets.py -q` -> `113 passed`
- `.\.venv\Scripts\python.exe -m pytest tests\test_workflow_reorg.py tests\test_form_tooltips.py -q` -> `34 passed`

### Deferred

- None.

## 3. YOLO/SAM Runtime-Context Tests

Status: complete at T3 checkpoint

### Code Evidence

Runtime settings are now represented by `YoloMaskRuntimeContext`, but some tests
still patch old module globals:

- `tests/test_yolo_mask_profile.py` patches `PROJECTION`, `LEVEL`, `QUALITY`,
  `PROFILE`, and `proc_count`.
- `tests/test_yolo_mask_bottom.py` patches `LEVEL`, `QUALITY`, `PROJECTION`,
  `EXPAND`, and bottom-redetection globals.
- `core/yolo_mask.py` intentionally keeps compatibility globals for direct-call
  fallback and CLI adapter setup.

### Refactor Plan

1. Keep a small compatibility test group:
   - Verify `active_runtime_context()` can still build from compatibility
     globals.
   - Verify `clear_runtime_context()` resets compatibility state.
2. Convert primary processing tests to explicit context tests:
   - Build `YoloMaskRuntimeSettings` and `YoloMaskRuntimeContext` directly.
   - Pass the context into processing helpers instead of patching global runtime
     fields.
3. Keep model-heavy functions patched where necessary, but patch algorithmic
   boundaries such as fake YOLO/SAM inference rather than runtime configuration
   globals.

### Completion Criteria

- Complete. YOLO/SAM behavior tests build explicit runtime contexts.
- Complete. Global compatibility-state mutation is limited to the compatibility
  test that verifies `apply_runtime_settings()` updates the legacy globals.
- Complete. Tests still verify bottom redetection, SAM merge behavior, profile
  recording, and preview/saved-mask consistency.

### Suggested Tests

- `.\.venv\Scripts\python.exe -m pytest tests\test_yolo_mask_profile.py tests\test_yolo_mask_bottom.py tests\test_yolo_mask_sam_merge.py -q`
- `.\.venv\Scripts\python.exe -m pytest tests\test_mask_preview.py tests\test_step3_mask_guard.py -q`

### Implementation Notes

- Added explicit runtime-context builders in the YOLO/SAM tests.
- `process_file()`, `detect_bottom_mask()`, and `add_sam_mask()` tests now pass
  context objects directly instead of patching `LEVEL`, `QUALITY`,
  `PROJECTION`, `EXPAND`, `BOTTOM_*`, or `PROFILE`.
- Renamed the remaining global-state assertion to make its compatibility role
  explicit.

### Verification

- `rg -n "monkeypatch\.setattr\(yolo_mask, \"(LEVEL|QUALITY|PROJECTION|EXPAND|BOTTOM_|PROFILE)" tests\test_yolo_mask_profile.py tests\test_yolo_mask_bottom.py tests\test_yolo_mask_sam_merge.py` -> no matches
- `.\.venv\Scripts\python.exe -m ruff check tests\test_yolo_mask_profile.py tests\test_yolo_mask_bottom.py tests\test_yolo_mask_sam_merge.py` -> passed
- `.\.venv\Scripts\python.exe -m pytest tests\test_yolo_mask_profile.py tests\test_yolo_mask_bottom.py tests\test_yolo_mask_sam_merge.py -q` -> `15 passed`
- `.\.venv\Scripts\python.exe -m pytest tests\test_mask_preview.py tests\test_step3_mask_guard.py -q` -> `82 passed`

### Deferred

- None.

## 4. Developer-Only Scripts Test Surface

Status: complete at T4 checkpoint

### Code Evidence

Release packaging now includes only:

- `scripts/update_venv.py`
- `scripts/check_venv.py`

Tests still import developer-only scripts directly:

- `tests/test_benchmark_yolo_mask.py` imports `scripts.benchmark_yolo_mask`.
- `tests/test_release_zip.py`, `tests/test_check_venv.py`, and
  `tests/test_update_venv_script.py` correctly test supported release/setup
  script behavior.

### Refactor Plan

1. Keep setup/release script tests as-is where they validate the supported
   release surface.
2. Move benchmark logic to a clearer developer-owned location if it remains
   valuable:
   - Option A: `devtools/yolo_mask_benchmark.py`, with tests importing devtools.
   - Option B: `core/yolo_mask_benchmark.py` if the logic is reusable beyond a
     developer script.
3. Keep `scripts/benchmark_yolo_mask.py` as a thin developer entry point if the
   script remains useful locally.
4. Update tests so direct `scripts.*` imports only refer to release/setup
   scripts.

### Completion Criteria

- Complete. `rg "from scripts|import scripts" tests` shows only release/setup
  and release packaging tests.
- Complete. Developer benchmark tests import `devtools.yolo_mask_benchmark`.
- Complete. Release ZIP tests still assert developer scripts are excluded.

### Suggested Tests

- `.\.venv\Scripts\python.exe -m pytest tests\test_benchmark_yolo_mask.py tests\test_release_zip.py -q`
- `.\.venv\Scripts\python.exe -m pytest tests\test_check_venv.py tests\test_update_venv_script.py -q`

### Implementation Notes

- Moved the benchmark implementation to `devtools/yolo_mask_benchmark.py`.
- Kept `scripts/benchmark_yolo_mask.py` as a thin local entry point.
- Updated benchmark tests to import the developer module directly.

### Verification

- `.\.venv\Scripts\python.exe -m ruff check devtools\yolo_mask_benchmark.py scripts\benchmark_yolo_mask.py tests\test_benchmark_yolo_mask.py` -> passed
- `.\.venv\Scripts\python.exe -m pytest tests\test_benchmark_yolo_mask.py tests\test_release_zip.py -q` -> `24 passed`
- `.\.venv\Scripts\python.exe -m pytest tests\test_check_venv.py tests\test_update_venv_script.py -q` -> `12 passed`
- `rg -n "from scripts|import scripts" tests` -> only setup/release tests remain

### Deferred

- None.

## 5. Shared GUI Test Fixtures

Status: complete at T5 checkpoint

### Code Evidence

Many GUI tests repeat the same local patterns:

- `QApplication.instance() or QApplication([])`
- offscreen setup
- optional `apply_theme(app)`
- manual scene folder setup
- repeated fake message box/dialog callbacks

This is not a correctness bug, but it increases maintenance cost as GUI
contracts evolve.

### Refactor Plan

1. Add a small helper module such as `tests/gui_helpers.py` or
   `tests/helpers/gui.py`.
2. Include only stable helpers:
   - `qt_app()` that sets `QT_QPA_PLATFORM=offscreen`.
   - `themed_qt_app()` for tests that inspect Japanese text/layout rendering.
   - optional `patch_messagebox_yes/no()` helpers.
   - optional scene builders for standard `images/`, `masks/`, and `output/`.
3. Avoid hiding important assertions inside helpers.
4. Convert tests gradually when they are touched by other test refactors.

### Completion Criteria

- Complete. New/touched GUI tests use shared helpers instead of reimplementing
  application setup.
- Complete. Existing tests touched by this refactor use the helper where it
  improves clarity.
- Complete. Offscreen theme behavior remains covered by
  `tests/test_form_tooltips.py::test_offscreen_theme_loads_windows_japanese_fonts`.

### Suggested Tests

- `.\.venv\Scripts\python.exe -m pytest tests\test_form_tooltips.py tests\test_workflow_reorg.py -q`
- Focused GUI tests touched by each implementation step.

### Implementation Notes

- Added `tests/helpers/gui.py` with `qt_app()`, `themed_qt_app()`, and small
  message-box patch helpers.
- Step 1, Step 3, and shared Step 4 test setup now use `qt_app()`.
- Kept conversion incremental: broader GUI suites can adopt the helper as they
  are touched by future behavior changes.

### Verification

- `.\.venv\Scripts\python.exe -m ruff check tests\helpers\gui.py tests\helpers\step4.py tests\test_step1_extract_ready.py tests\test_step3_mask_guard.py` -> passed
- `.\.venv\Scripts\python.exe -m pytest tests\test_step1_extract_ready.py tests\test_step3_mask_guard.py tests\test_step4_colmap_route.py tests\test_step4_dataset_jobs.py tests\test_step4_output_paths.py tests\test_step4_preview_behavior.py tests\test_step4_realityscan_route.py tests\test_step4_spheresfm_route.py tests\test_step4_training_paths.py -q` -> `205 passed`
- `.\.venv\Scripts\python.exe -m pytest tests\test_form_tooltips.py tests\test_workflow_reorg.py -q` -> `34 passed`

### Deferred

- None.

## 6. Final Verification

Status: passed at T6 checkpoint

Run this after implementing any area above:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

Before merging the whole test refactor, also rerun the audit searches:

```powershell
rg -n "tests can monkeypatch|sys\.modules|getattr\(sys\.modules|monkeypatch\.setattr\(step4_cubemap|monkeypatch\.setattr\(step3_mask_module" gui tests
rg -n "from scripts|import scripts" tests
```

### Implementation Notes

- Final audit searches were rerun after T1-T5.
- Remaining `sys.modules` match is `tests/test_sam31_download.py`, where a
  fake module is intentionally injected to test SAM3.1 download/import handling.
- Remaining `monkeypatch.setattr(step4_cubemap.os, "link", ...)` match is the
  intentional Windows hardlink-failure fallback test in
  `tests/test_step4_dataset_jobs.py`.
- Remaining `scripts.*` imports are setup/release support tests only:
  `check_venv`, `update_venv`, and `create_release_zip`.

### Verification

- `rg -n "tests can monkeypatch|sys\.modules|getattr\(sys\.modules|monkeypatch\.setattr\(step4_cubemap|monkeypatch\.setattr\(step3_mask_module" gui tests devtools` -> reviewed; only intentional matches remain.
- `rg -n "from scripts|import scripts" tests` -> reviewed; only setup/release tests remain.
- `.\.venv\Scripts\python.exe -m ruff check .` -> passed.
- `.\.venv\Scripts\python.exe -m pytest -q` -> `985 passed`.

### Deferred

- None.

## Update Rules For Codex

When implementing from this plan:

1. Re-run the relevant "Code Evidence" search before editing.
2. Change tests to match current contracts, not to weaken assertions.
3. Prefer one responsibility area per commit.
4. Update this file after each implementation:
   - status
   - code evidence
   - test results
   - intentionally deferred items
5. Do not mark final verification complete until full `ruff` and full `pytest`
   pass after the implementation.
