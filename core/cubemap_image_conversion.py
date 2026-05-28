from __future__ import annotations

import os
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.cancellation import CancellationToken, raise_if_cancelled
from core.colmap_rig_export import (
    colmap_camera_image_dir,
    colmap_camera_mask_dir,
    frame_filename,
)
from core.cubemap_image_io import (
    load_equirect,
    remap_with_channels,
    resolve_output_ext,
    save_image,
    split_filename_for_output,
)
from core.cubemap_image_io import (
    max_value_for_dtype as _max_value_for_dtype,
)
from core.cubemap_remap import (
    build_remap,
)
from core.cubemap_remap import (
    remap_cache_key as _remap_cache_key,
)
from core.cubemap_worker_plan import (
    parse_positive_int_or_auto,
    resolve_remap_cache_limit,
    resolve_worker_count,
)
from core.image_io import imread_unicode
from core.path_safety import is_path_inside

_WORKER_REMAP_TABLES: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
_WORKER_VIEWS: list[dict] | None = None
_WORKER_IMAGE_DIR = ""
_WORKER_MASK_DIR = ""
_WORKER_OUTPUT_IMAGE_DIR = ""
_WORKER_OUTPUT_MASK_DIR = ""
_WORKER_MASK_FROM_ALPHA = False
_WORKER_INVERT_MASKS = False
_WORKER_OUTPUT_FORMAT: str | None = None
_WORKER_OUTPUT_BIT_DEPTH = "8"
_WORKER_JPG_QUALITY = 95
_WORKER_EXPORT_IMAGES = True
_WORKER_EXPORT_MASKS = True
_WORKER_COLMAP_RIG_IMAGE_DIRS: dict[str, str] = {}
_WORKER_COLMAP_RIG_MASK_DIRS: dict[str, str] = {}
# 入力解像度 + 出力サイズ + yaw オフセット別キャッシュ: key = (W, H, output_size, round(yaw_offset, 3))
_WORKER_REMAP_CACHE: OrderedDict[tuple[int, int, int, float], dict[str, tuple[np.ndarray, np.ndarray]]] = OrderedDict()
_WORKER_REMAP_CACHE_LIMIT = 12
_WORKER_INPUT_SIZE: tuple[int, int] = (0, 0)
_WORKER_FOV: float = 90.0
_WORKER_OUTPUT_SIZE: int = 0
_WORKER_OUTPUT_SIZE_BY_FRAME: dict[str, int] = {}


def _frame_key(frame_file: str) -> str:
    return str(frame_file).replace("\\", "/").casefold()


def _resolve_source_path(root: str | Path, frame_file: str | Path, *, kind: str) -> Path:
    root_path = Path(root)
    raw = Path(frame_file)
    candidate = raw if raw.is_absolute() else root_path / raw
    if not is_path_inside(candidate, root_path, allow_equal=False):
        raise ValueError(f"{kind} path escapes input root: {frame_file}")
    return candidate


def _candidate_under_root(root: str | Path, candidate: Path) -> Path | None:
    return candidate if is_path_inside(candidate, root, allow_equal=False) else None


def _output_size_for_frame(frame_file: str) -> int:
    return int(_WORKER_OUTPUT_SIZE_BY_FRAME.get(_frame_key(frame_file), _WORKER_OUTPUT_SIZE))


def _worker_remap_cache_key(
    input_size: tuple[int, int],
    yaw_offset: float,
    output_size: int,
) -> tuple[int, int, int, float]:
    width, height, offset = _remap_cache_key(input_size, yaw_offset)
    return width, height, int(output_size), offset


def get_remap_tables_for_offset(yaw_offset: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """ワーカー側で yaw_offset に対応するリマップテーブル群を取得（無ければ生成してキャッシュ）。"""
    return get_remap_tables_for_input_size(_WORKER_INPUT_SIZE, yaw_offset, _WORKER_OUTPUT_SIZE)


def get_remap_tables_for_input_size(
    input_size: tuple[int, int],
    yaw_offset: float,
    output_size: int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Get remap tables for the actual source image size and yaw offset."""
    global _WORKER_REMAP_CACHE
    resolved_output_size = int(output_size if output_size is not None else _WORKER_OUTPUT_SIZE)
    key = _worker_remap_cache_key(input_size, yaw_offset, resolved_output_size)

    cached = _WORKER_REMAP_CACHE.get(key)
    if cached is not None:
        _WORKER_REMAP_CACHE.move_to_end(key)
        return cached

    assert _WORKER_VIEWS is not None
    input_size = (int(input_size[0]), int(input_size[1]))
    tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for view in _WORKER_VIEWS:
        eff_yaw = float(view["yaw"]) + key[3]
        tables[view["name"]] = build_remap(
            input_size,
            _WORKER_FOV,
            eff_yaw,
            float(view["pitch"]),
            resolved_output_size,
        )
    _WORKER_REMAP_CACHE[key] = tables
    _WORKER_REMAP_CACHE.move_to_end(key)
    limit = max(1, int(_WORKER_REMAP_CACHE_LIMIT))
    while len(_WORKER_REMAP_CACHE) > limit:
        _WORKER_REMAP_CACHE.popitem(last=False)
    return tables


def remap_input_size(path: str) -> tuple[int, int]:
    """Read image dimensions for remap table selection without assuming all sources match."""
    try:
        with Image.open(path) as image:
            return int(image.size[0]), int(image.size[1])
    except Exception:
        arr = load_equirect(path)
        return int(arr.shape[1]), int(arr.shape[0])


def get_remap_tables_for_file(
    path: str,
    yaw_offset: float,
    output_size: int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return get_remap_tables_for_input_size(remap_input_size(path), yaw_offset, output_size)


def remap_image(
    input_file: str,
    output_dir: str,
    remap_tables: dict[str, tuple[np.ndarray, np.ndarray]],
    views: list[dict],
    mask_from_alpha: bool,
    output_mask_dir: str,
    invert_masks: bool,
    output_format: str | None = None,
    output_bit_depth: str = "8",
    jpg_quality: int = 95,
    write_output: bool = True,
    write_alpha_mask: bool = True,
) -> int:
    basename, ext2, in_ext = split_filename_for_output(input_file)
    out_ext = resolve_output_ext(in_ext, output_format)

    print(f"Processing: {input_file}", flush=True)
    equi = load_equirect(input_file)

    is_grayscale = equi.ndim == 2
    has_alpha = equi.ndim == 3 and equi.shape[2] == 4
    max_val = _max_value_for_dtype(equi.dtype)
    written = 0

    for view in views:
        view_name = view["name"]
        map_x, map_y = remap_tables[view_name]

        converted = remap_with_channels(equi, map_x, map_y)

        if is_grayscale:
            # 2 値マスクとして閾値化
            _, converted = cv2.threshold(converted, max_val // 2, max_val, cv2.THRESH_BINARY)
            if invert_masks:
                converted = max_val - converted

        out_path = os.path.join(output_dir, f"{basename}_{view_name}{ext2}{out_ext}")

        if mask_from_alpha and has_alpha:
            color = converted[..., :3]
            alpha = converted[..., 3]
            if write_output:
                save_image(color, out_path, jpg_quality, force_8bit=output_bit_depth == "8")
                written += 1

            mask_thresh = max_val // 2
            _, mask = cv2.threshold(alpha, mask_thresh, max_val, cv2.THRESH_BINARY)
            if invert_masks:
                mask = max_val - mask
            mask_out_path = os.path.join(output_mask_dir, f"{basename}_{view_name}{ext2}.png")
            if write_alpha_mask:
                save_image(mask, mask_out_path, jpg_quality, force_8bit=True)
                written += 1
        else:
            if write_output:
                save_image(
                    converted,
                    out_path,
                    jpg_quality,
                    force_8bit=is_grayscale or output_bit_depth == "8",
                )
                written += 1

    return written


def mask_candidates(mask_dir: str, frame_file: str) -> list[str]:
    frame_path = Path(frame_file)
    candidates: list[Path] = []

    variants: list[Path] = [frame_path, Path(frame_path.name)]
    if frame_path.parts and frame_path.parts[0].lower() == "images" and len(frame_path.parts) > 1:
        variants.append(Path(*frame_path.parts[1:]))

    for rel in variants:
        for candidate in (
            Path(mask_dir) / rel,
            Path(mask_dir) / f"{rel.name}.png",
            Path(mask_dir) / f"{rel.stem}.png",
        ):
            safe_candidate = _candidate_under_root(mask_dir, candidate)
            if safe_candidate is not None:
                candidates.append(safe_candidate)

    uniq: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        key = str(c).lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(str(c))
    return uniq


def worker_init(
    input_size: tuple[int, int],
    fov: float,
    output_size: int,
    views: list[dict],
    image_dir: str,
    mask_dir: str,
    output_image_dir: str,
    output_mask_dir: str,
    mask_from_alpha: bool,
    invert_masks: bool,
    output_format: str | None,
    output_bit_depth: str,
    jpg_quality: int,
    export_images: bool = True,
    export_masks: bool = True,
    remap_cache_limit: int = 12,
    output_sizes_by_frame: dict[str, int] | None = None,
) -> None:
    global _WORKER_REMAP_TABLES
    global _WORKER_VIEWS
    global _WORKER_IMAGE_DIR
    global _WORKER_MASK_DIR
    global _WORKER_OUTPUT_IMAGE_DIR
    global _WORKER_OUTPUT_MASK_DIR
    global _WORKER_MASK_FROM_ALPHA
    global _WORKER_INVERT_MASKS
    global _WORKER_OUTPUT_FORMAT
    global _WORKER_OUTPUT_BIT_DEPTH
    global _WORKER_JPG_QUALITY
    global _WORKER_EXPORT_IMAGES
    global _WORKER_EXPORT_MASKS
    global _WORKER_REMAP_CACHE
    global _WORKER_REMAP_CACHE_LIMIT
    global _WORKER_INPUT_SIZE
    global _WORKER_FOV
    global _WORKER_OUTPUT_SIZE
    global _WORKER_OUTPUT_SIZE_BY_FRAME

    _WORKER_VIEWS = views
    _WORKER_INPUT_SIZE = input_size
    _WORKER_FOV = fov
    _WORKER_OUTPUT_SIZE = output_size
    _WORKER_IMAGE_DIR = image_dir
    _WORKER_MASK_DIR = mask_dir
    _WORKER_OUTPUT_IMAGE_DIR = output_image_dir
    _WORKER_OUTPUT_MASK_DIR = output_mask_dir
    _WORKER_MASK_FROM_ALPHA = mask_from_alpha
    _WORKER_INVERT_MASKS = invert_masks
    _WORKER_OUTPUT_FORMAT = output_format
    _WORKER_OUTPUT_BIT_DEPTH = output_bit_depth
    _WORKER_JPG_QUALITY = jpg_quality
    _WORKER_EXPORT_IMAGES = export_images
    _WORKER_EXPORT_MASKS = export_masks
    _WORKER_REMAP_CACHE_LIMIT = max(1, int(remap_cache_limit))
    _WORKER_OUTPUT_SIZE_BY_FRAME = {
        _frame_key(frame_file): int(size)
        for frame_file, size in (output_sizes_by_frame or {}).items()
        if int(size) > 0
    }

    # offset=0 のテーブルを事前構築（per-frame yaw を使わない場合の通常パス）
    _WORKER_REMAP_CACHE = OrderedDict()
    _WORKER_REMAP_TABLES = get_remap_tables_for_offset(0.0)


def worker_init_colmap_rig(
    input_size: tuple[int, int],
    fov: float,
    output_size: int,
    views: list[dict],
    image_dir: str,
    mask_dir: str,
    image_dirs_by_view: dict[str, str],
    mask_dirs_by_view: dict[str, str],
    mask_from_alpha: bool,
    invert_masks: bool,
    output_format: str | None,
    output_bit_depth: str,
    jpg_quality: int,
    export_images: bool = True,
    export_masks: bool = True,
    remap_cache_limit: int = 12,
) -> None:
    global _WORKER_COLMAP_RIG_IMAGE_DIRS
    global _WORKER_COLMAP_RIG_MASK_DIRS

    worker_init(
        input_size,
        fov,
        output_size,
        views,
        image_dir,
        mask_dir,
        "",
        "",
        mask_from_alpha,
        invert_masks,
        output_format,
        output_bit_depth,
        jpg_quality,
        export_images,
        export_masks,
        remap_cache_limit,
        None,
    )
    _WORKER_COLMAP_RIG_IMAGE_DIRS = image_dirs_by_view
    _WORKER_COLMAP_RIG_MASK_DIRS = mask_dirs_by_view


def _image_has_alpha(path: str) -> bool:
    try:
        with Image.open(path) as img:
            bands = img.getbands()
            return "A" in bands or "transparency" in img.info
    except Exception:
        pass
    img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    return img is not None and img.ndim == 3 and img.shape[2] == 4


def count_planned_outputs(
    image_files: list[str],
    views: list[dict],
    image_dir: str,
    mask_dir: str,
    mask_from_alpha: bool,
    export_images: bool = True,
    export_masks: bool = True,
) -> int:
    view_count = len(views)
    total = 0
    mask_dir_exists = bool(mask_dir) and os.path.isdir(mask_dir)

    for frame_file in image_files:
        image = _resolve_source_path(image_dir, frame_file, kind="image")
        image_exists = os.path.exists(image)
        if image_exists:
            if export_images:
                total += view_count
            if export_masks and mask_from_alpha and _image_has_alpha(str(image)):
                total += view_count

        if not export_masks or mask_from_alpha or not mask_dir_exists:
            continue

        for mask in mask_candidates(mask_dir, frame_file):
            if os.path.exists(mask):
                total += view_count
                break

    return total


def proc_convert_images(frame_file: str, yaw_offset: float = 0.0) -> int:
    if _WORKER_VIEWS is None:
        raise RuntimeError("worker views are not initialized")

    written = 0

    image = str(_resolve_source_path(_WORKER_IMAGE_DIR, frame_file, kind="image"))
    output_size = _output_size_for_frame(frame_file)
    if os.path.exists(image) and (_WORKER_EXPORT_IMAGES or (_WORKER_EXPORT_MASKS and _WORKER_MASK_FROM_ALPHA)):
        tables = get_remap_tables_for_file(image, yaw_offset, output_size)
        written += remap_image(
            image,
            _WORKER_OUTPUT_IMAGE_DIR,
            tables,
            _WORKER_VIEWS,
            _WORKER_MASK_FROM_ALPHA,
            _WORKER_OUTPUT_MASK_DIR,
            _WORKER_INVERT_MASKS,
            output_format=_WORKER_OUTPUT_FORMAT,
            output_bit_depth=_WORKER_OUTPUT_BIT_DEPTH,
            jpg_quality=_WORKER_JPG_QUALITY,
            write_output=_WORKER_EXPORT_IMAGES,
            write_alpha_mask=_WORKER_EXPORT_MASKS,
        )

    if (
        not _WORKER_EXPORT_MASKS
        or _WORKER_MASK_FROM_ALPHA
        or not _WORKER_MASK_DIR
        or not os.path.isdir(_WORKER_MASK_DIR)
    ):
        return written

    for mask in mask_candidates(_WORKER_MASK_DIR, frame_file):
        if os.path.exists(mask):
            tables = get_remap_tables_for_file(mask, yaw_offset, output_size)
            # マスクは PNG 出力固定（α 不要、ロスレス必須）
            written += remap_image(
                mask,
                _WORKER_OUTPUT_MASK_DIR,
                tables,
                _WORKER_VIEWS,
                False,
                _WORKER_OUTPUT_MASK_DIR,
                _WORKER_INVERT_MASKS,
                output_format="png",
                output_bit_depth="8",
                jpg_quality=_WORKER_JPG_QUALITY,
                write_output=True,
                write_alpha_mask=False,
            )
            break

    return written


def _binary_mask_from_remapped(arr: np.ndarray, invert_masks: bool) -> np.ndarray:
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[..., 3]
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    max_val = _max_value_for_dtype(arr.dtype)
    _, out = cv2.threshold(arr, max_val // 2, max_val, cv2.THRESH_BINARY)
    if invert_masks:
        out = max_val - out
    return out


def proc_convert_images_colmap_rig(job: tuple[str, str]) -> int:
    if _WORKER_VIEWS is None:
        raise RuntimeError("worker views are not initialized")

    frame_file, output_filename = job
    written = 0

    image = str(_resolve_source_path(_WORKER_IMAGE_DIR, frame_file, kind="image"))
    if os.path.exists(image) and (_WORKER_EXPORT_IMAGES or (_WORKER_EXPORT_MASKS and _WORKER_MASK_FROM_ALPHA)):
        print(f"Processing: {image}", flush=True)
        equi = load_equirect(image)
        tables = get_remap_tables_for_input_size((int(equi.shape[1]), int(equi.shape[0])), 0.0)
        has_alpha = equi.ndim == 3 and equi.shape[2] == 4
        max_val = _max_value_for_dtype(equi.dtype)

        for view in _WORKER_VIEWS:
            view_name = view["name"]
            map_x, map_y = tables[view_name]
            converted = remap_with_channels(equi, map_x, map_y)

            if _WORKER_EXPORT_IMAGES:
                image_dir = _WORKER_COLMAP_RIG_IMAGE_DIRS[view_name]
                out_path = os.path.join(image_dir, output_filename)
                if has_alpha:
                    converted_image = converted[..., :3]
                else:
                    converted_image = converted
                save_image(
                    converted_image,
                    out_path,
                    _WORKER_JPG_QUALITY,
                    force_8bit=_WORKER_OUTPUT_BIT_DEPTH == "8",
                )
                written += 1

            if _WORKER_EXPORT_MASKS and _WORKER_MASK_FROM_ALPHA and has_alpha:
                alpha = converted[..., 3]
                _, mask = cv2.threshold(alpha, max_val // 2, max_val, cv2.THRESH_BINARY)
                if _WORKER_INVERT_MASKS:
                    mask = max_val - mask
                mask_dir = _WORKER_COLMAP_RIG_MASK_DIRS[view_name]
                save_image(
                    mask,
                    os.path.join(mask_dir, f"{output_filename}.png"),
                    _WORKER_JPG_QUALITY,
                    force_8bit=True,
                )
                written += 1

    if (
        not _WORKER_EXPORT_MASKS
        or _WORKER_MASK_FROM_ALPHA
        or not _WORKER_MASK_DIR
        or not os.path.isdir(_WORKER_MASK_DIR)
    ):
        return written

    for mask_path in mask_candidates(_WORKER_MASK_DIR, frame_file):
        if not os.path.exists(mask_path):
            continue

        print(f"Processing: {mask_path}", flush=True)
        equi_mask = load_equirect(mask_path)
        tables = get_remap_tables_for_input_size((int(equi_mask.shape[1]), int(equi_mask.shape[0])), 0.0)
        for view in _WORKER_VIEWS:
            view_name = view["name"]
            map_x, map_y = tables[view_name]
            converted = remap_with_channels(equi_mask, map_x, map_y)
            mask = _binary_mask_from_remapped(converted, _WORKER_INVERT_MASKS)
            mask_dir = _WORKER_COLMAP_RIG_MASK_DIRS[view_name]
            save_image(
                mask,
                os.path.join(mask_dir, f"{output_filename}.png"),
                _WORKER_JPG_QUALITY,
                force_8bit=True,
            )
            written += 1
        break

    return written


def _raise_worker_failures(failures: list[tuple[str, BaseException]], context: str) -> None:
    if not failures:
        return
    shown = "; ".join(f"{label}: {error}" for label, error in failures[:3])
    if len(failures) > 3:
        shown += f"; ... {len(failures) - 3} more"
    raise RuntimeError(f"{context}: {len(failures)} worker(s) failed; {shown}")


def _run_bounded_conversion_jobs(
    jobs,
    submit_job,
    label_job,
    *,
    total_outputs: int,
    max_workers: int,
    failure_context: str,
    cancel_event: CancellationToken | None = None,
) -> None:
    """Run conversion jobs without materializing a Future for every input frame."""
    pending_limit = max(1, int(max_workers) * 2)
    job_iter = iter(jobs)
    pending = {}
    failures: list[tuple[str, BaseException]] = []

    def submit_until_limit() -> None:
        while len(pending) < pending_limit:
            raise_if_cancelled(cancel_event)
            try:
                job = next(job_iter)
            except StopIteration:
                return
            label = label_job(job)
            try:
                pending[submit_job(job)] = label
            except Exception as e:
                failures.append((label, e))

    raise_if_cancelled(cancel_event)
    submit_until_limit()
    done = 0
    try:
        while pending:
            raise_if_cancelled(cancel_event)
            finished, _not_done = wait(tuple(pending), timeout=0.1, return_when=FIRST_COMPLETED)
            if not finished:
                continue
            future = next(iter(finished))
            frame_file = pending.pop(future)
            try:
                done += future.result()
                print(f"[progress] {done}/{total_outputs}", flush=True)
            except Exception as e:
                failures.append((frame_file, e))
                print(f"Worker failed: {frame_file}: {e}", flush=True)
            submit_until_limit()
    except Exception:
        for future in pending:
            future.cancel()
        raise

    _raise_worker_failures(failures, failure_context)


def convert_images(
    image_files: list[str],
    input_size: tuple[int, int],
    output_size: int,
    views: list[dict],
    fov: float,
    image_dir: str,
    mask_dir: str,
    output_image_dir: str,
    output_mask_dir: str,
    mask_from_alpha: bool,
    invert_masks: bool,
    output_format: str | None = None,
    output_bit_depth: str = "8",
    jpg_quality: int = 95,
    frame_yaw_offsets: list[float] | None = None,
    frame_output_sizes: list[int] | None = None,
    export_images: bool = True,
    export_masks: bool = True,
    workers: str | int | None = "auto",
    remap_cache_limit: str | int | None = "auto",
    cancel_event: CancellationToken | None = None,
) -> None:
    raise_if_cancelled(cancel_event)
    if frame_yaw_offsets is None:
        frame_yaw_offsets = [0.0] * len(image_files)
    if len(frame_yaw_offsets) != len(image_files):
        raise ValueError(
            f"frame_yaw_offsets length ({len(frame_yaw_offsets)}) must match image_files length ({len(image_files)})"
        )
    if frame_output_sizes is None:
        frame_output_sizes = [int(output_size)] * len(image_files)
    if len(frame_output_sizes) != len(image_files):
        raise ValueError(
            f"frame_output_sizes length ({len(frame_output_sizes)}) must match image_files length ({len(image_files)})"
        )
    frame_output_sizes = [max(1, int(size)) for size in frame_output_sizes]
    output_sizes_by_frame = {
        _frame_key(frame_file): int(size)
        for frame_file, size in zip(image_files, frame_output_sizes, strict=True)
    }
    max_output_size = max(frame_output_sizes, default=int(output_size))

    tentative_workers = parse_positive_int_or_auto(workers, "--workers")
    if tentative_workers is None:
        tentative_workers = min(16, os.cpu_count() or 1)
    resolved_cache_limit = resolve_remap_cache_limit(
        remap_cache_limit,
        frame_yaw_offsets,
        max_output_size,
        len(views),
        tentative_workers,
    )
    max_workers = resolve_worker_count(
        workers,
        input_size,
        max_output_size,
        len(views),
        resolved_cache_limit,
    )
    total_outputs = count_planned_outputs(
        image_files=image_files,
        views=views,
        image_dir=image_dir,
        mask_dir=mask_dir,
        mask_from_alpha=mask_from_alpha,
        export_images=export_images,
        export_masks=export_masks,
    )
    print(f"Converting {total_outputs} files...")
    print(f"Workers: {max_workers} (remap cache limit={resolved_cache_limit})")
    print(f"[progress] 0/{total_outputs}", flush=True)

    if total_outputs <= 0:
        return

    raise_if_cancelled(cancel_event)
    if export_images:
        os.makedirs(output_image_dir, exist_ok=True)
    if export_masks and (mask_from_alpha or os.path.isdir(mask_dir)):
        os.makedirs(output_mask_dir, exist_ok=True)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=worker_init,
        initargs=(
            input_size,
            fov,
            output_size,
            views,
            image_dir,
            mask_dir,
            output_image_dir,
            output_mask_dir,
            mask_from_alpha,
            invert_masks,
            output_format,
            output_bit_depth,
            jpg_quality,
            export_images,
            export_masks,
            resolved_cache_limit,
            output_sizes_by_frame,
        ),
    ) as executor:
        _run_bounded_conversion_jobs(
            zip(image_files, frame_yaw_offsets, strict=True),
            lambda job: executor.submit(proc_convert_images, job[0], job[1]),
            lambda job: job[0],
            total_outputs=total_outputs,
            max_workers=max_workers,
            failure_context="Cubemap conversion failed",
            cancel_event=cancel_event,
        )


def make_colmap_rig_jobs(
    image_files: list[str],
    output_format: str | None,
) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    total = len(image_files)
    for idx, frame_file in enumerate(image_files, start=1):
        _basename, _ext2, in_ext = split_filename_for_output(frame_file)
        out_ext = resolve_output_ext(in_ext, output_format)
        jobs.append((frame_file, frame_filename(idx, total, out_ext)))
    return jobs


def convert_images_colmap_rig(
    image_files: list[str],
    input_size: tuple[int, int],
    output_size: int,
    views: list[dict],
    fov: float,
    image_dir: str,
    mask_dir: str,
    output_dir: str,
    rig_name: str,
    mask_from_alpha: bool,
    invert_masks: bool,
    output_format: str | None = None,
    output_bit_depth: str = "8",
    jpg_quality: int = 95,
    export_images: bool = True,
    export_masks: bool = True,
    workers: str | int | None = "auto",
    remap_cache_limit: str | int | None = "auto",
    cancel_event: CancellationToken | None = None,
) -> None:
    raise_if_cancelled(cancel_event)
    image_dirs_by_view = {
        view["name"]: str(colmap_camera_image_dir(output_dir, rig_name, view["camera_name"])) for view in views
    }
    mask_dirs_by_view = {
        view["name"]: str(colmap_camera_mask_dir(output_dir, rig_name, view["camera_name"])) for view in views
    }
    if export_images:
        for path in image_dirs_by_view.values():
            os.makedirs(path, exist_ok=True)
    if export_masks and (mask_from_alpha or os.path.isdir(mask_dir)):
        for path in mask_dirs_by_view.values():
            os.makedirs(path, exist_ok=True)

    total_outputs = count_planned_outputs(
        image_files=image_files,
        views=views,
        image_dir=image_dir,
        mask_dir=mask_dir,
        mask_from_alpha=mask_from_alpha,
        export_images=export_images,
        export_masks=export_masks,
    )
    print(f"Converting {total_outputs} files...")
    tentative_workers = parse_positive_int_or_auto(workers, "--workers")
    if tentative_workers is None:
        tentative_workers = min(16, os.cpu_count() or 1)
    resolved_cache_limit = resolve_remap_cache_limit(
        remap_cache_limit,
        [0.0 for _ in image_files],
        output_size,
        len(views),
        tentative_workers,
    )
    max_workers = resolve_worker_count(
        workers,
        input_size,
        output_size,
        len(views),
        resolved_cache_limit,
    )
    print(f"Workers: {max_workers} (remap cache limit={resolved_cache_limit})")
    print(f"[progress] 0/{total_outputs}", flush=True)
    if total_outputs <= 0:
        return

    raise_if_cancelled(cancel_event)
    jobs = make_colmap_rig_jobs(image_files, output_format)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=worker_init_colmap_rig,
        initargs=(
            input_size,
            fov,
            output_size,
            views,
            image_dir,
            mask_dir,
            image_dirs_by_view,
            mask_dirs_by_view,
            mask_from_alpha,
            invert_masks,
            output_format,
            output_bit_depth,
            jpg_quality,
            export_images,
            export_masks,
            resolved_cache_limit,
        ),
    ) as executor:
        _run_bounded_conversion_jobs(
            jobs,
            lambda job: executor.submit(proc_convert_images_colmap_rig, job),
            lambda job: job[0],
            total_outputs=total_outputs,
            max_workers=max_workers,
            failure_context="COLMAP rig image conversion failed",
            cancel_event=cancel_event,
        )
