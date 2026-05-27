"""Mask output metadata recording for Step 3."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from core.scene_asset_metadata import update_scene_asset_mask_metadata
from core.scene_project import append_mask_run, scene_relative, utc_now_iso, write_mask_item
from gui.steps.mask_postprocess import mask_stats

MaskPathResolver = Callable[[Path], Path]
RunIdFactory = Callable[[str], str]


def record_mask_outputs(
    scene_dir: str | Path,
    image_paths: Sequence[Path],
    *,
    mode: str,
    settings: dict | None,
    phases: Sequence[str],
    mask_path_for_image: MaskPathResolver,
    run_id: str | None = None,
    run_id_factory: RunIdFactory | None = None,
) -> None:
    if not image_paths:
        return
    scene = Path(scene_dir)
    settings = settings or {}
    run_id = run_id or (run_id_factory("mask") if run_id_factory is not None else _default_run_id("mask"))

    generated: list[dict] = []
    for image_path in image_paths:
        mask_path = mask_path_for_image(image_path)
        if not mask_path.is_file():
            continue
        stats = mask_stats(mask_path)
        write_mask_item(
            scene,
            image_path=image_path,
            mask_path=mask_path,
            settings=settings,
            run_id=run_id,
            stats=stats,
        )
        update_scene_asset_mask_metadata(scene, image_path=image_path, mask_path=mask_path)
        generated.append(
            {
                "image": scene_relative(scene, image_path),
                "mask": scene_relative(scene, mask_path),
                "stats": stats,
            }
        )

    if not generated:
        return
    append_mask_run(
        scene,
        {
            "id": run_id,
            "created_at": utc_now_iso(),
            "mode": mode,
            "phases": list(phases),
            "settings": settings,
            "image_count": len(image_paths),
            "mask_count": len(generated),
            "generated": generated,
        },
    )


def _default_run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '')}"
