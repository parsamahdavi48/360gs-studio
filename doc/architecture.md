# Maintenance Architecture

This project keeps stable script-style CLI entry points while moving reusable
implementation code into focused modules. The GUI in `gui/` builds commands for
the same public scripts, so CLI behavior must remain compatible when GUI code is
changed.

## Entry Points

Root-level `*.py` files are public compatibility wrappers. Keep their names,
flags, exit behavior, and importable helper symbols stable unless a breaking
change is intentional and documented.

Shared implementation lives in `core/`. New CLI behavior should normally be
implemented in `core/` first, then exposed through the root wrapper. Tests should
cover both the implementation contract and any public command-line behavior that
the GUI depends on.

## Core Contracts

- `core/video_info.py` owns the shared video metadata dataclass used by frame
  extraction and downstream planning.
- `core/frame_pair_analysis.py` owns pair-frame metrics, blur/track risk
  thresholds, dependency checks, and selection analysis. Keep OpenCV/numpy
  imports guarded so command help and unrelated tests do not require optional
  runtime dependencies.
- `core/extract_frames.py` owns FFmpeg/FFprobe frame extraction, cache handling,
  stationary thinning, decision CSV writing, and CLI argument handling.
- `core/frame_renumbering.py` owns the Step 2 kept-image renumbering contract:
  downstream-output blockers, collision-safe rename planning/application, and
  updates to frame/source image metadata when paths change.
- `core/scene_import*.py` owns external scene re-registration. Import should
  scan only the app-defined scene folders, replace external-import metadata as a
  full reimport, and avoid deleting real image/mask/output assets.
- Mask modules preserve the repository-wide mask polarity contract:
  white pixels are usable, black pixels are excluded. Mask merges should remain
  AND-style unless a tool explicitly documents a different operation.
- Cubemap and COLMAP exports must preserve coordinate profile semantics:
  Postshot uses the default cubemap transform, Brush uses the Brush transform,
  and LichtFeld cubemap export writes final-orientation-corrected
  `transforms.json` plus `pointcloud.ply` from the Cubemap CLI. LichtFeld direct
  3DGUT uses equirectangular inputs without cubemap conversion and applies the
  same final orientation correction while creating the direct dataset.
  RealityScan export is a Metashape-output preset that cancels the Metashape
  import coordinate conversion, maps Metashape Y-up poses into RealityScan Z-up
  local Euclidean axes, then writes cubemap images and XMP sidecars under
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
- `gui/mask/mask_commands.py` owns command construction.
- `gui/mask/mask_postprocess.py` owns saved-mask postprocessing.

Step 4 conversion/training is split by responsibility:

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
  mixins. Avoid adding new workflow logic there when it clearly belongs to one
  of the focused modules above.

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

When checking the PySide GUI offscreen in Codex or CI-like environments, set
`QT_QPA_PLATFORM=offscreen` and call `apply_theme(app)` before constructing
`MainWindow`. The theme applies a Windows Japanese-capable font such as Meiryo
UI, which prevents Japanese labels from rendering as square glyphs.
