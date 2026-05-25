from __future__ import annotations

import os

from core.cubemap_remap import quantize_yaw_offset


def parse_positive_int_or_auto(value: str | int | None, name: str) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "auto"}:
        return None
    try:
        parsed = int(text)
    except ValueError as e:
        raise ValueError(f"{name} must be 'auto' or a positive integer") from e
    if parsed <= 0:
        raise ValueError(f"{name} must be 'auto' or a positive integer")
    return parsed


def available_memory_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        except Exception:
            return None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * avail_pages
    except (AttributeError, ValueError, OSError):
        return None


def estimate_remap_offset_bytes(output_size: int, view_count: int) -> int:
    output_pixels = max(1, int(output_size)) ** 2
    # map_x + map_y, both float32.
    return output_pixels * max(1, int(view_count)) * 2 * 4


def estimate_worker_memory_bytes(
    input_size: tuple[int, int],
    output_size: int,
    view_count: int,
    remap_cache_limit: int,
) -> int:
    src_w, src_h = input_size
    input_bytes = max(1, int(src_w)) * max(1, int(src_h)) * 8
    remap_bytes = estimate_remap_offset_bytes(output_size, view_count) * max(1, int(remap_cache_limit))
    scratch_bytes = max(1, int(output_size)) ** 2 * 16
    return (256 * 1024 * 1024) + input_bytes + remap_bytes + scratch_bytes


def resolve_worker_count(
    value: str | int | None,
    input_size: tuple[int, int],
    output_size: int,
    view_count: int,
    remap_cache_limit: int,
) -> int:
    requested = parse_positive_int_or_auto(value, "--workers")
    if requested is not None:
        return requested

    cpu_cap = min(16, os.cpu_count() or 1)
    available = available_memory_bytes()
    if not available:
        return cpu_cap

    per_worker = estimate_worker_memory_bytes(input_size, output_size, view_count, remap_cache_limit)
    if per_worker <= 0:
        return cpu_cap
    memory_cap = int((available * 0.55) // per_worker)
    return max(1, min(cpu_cap, memory_cap))


def resolve_remap_cache_limit(
    value: str | int | None,
    frame_yaw_offsets: list[float] | None,
    output_size: int,
    view_count: int,
    worker_count: int,
) -> int:
    requested = parse_positive_int_or_auto(value, "--remap-cache-limit")
    if requested is not None:
        return requested

    if frame_yaw_offsets:
        desired = len({quantize_yaw_offset(offset) for offset in frame_yaw_offsets})
    else:
        desired = 1
    desired = max(1, min(desired, 12))

    available = available_memory_bytes()
    if not available:
        return desired

    per_offset = estimate_remap_offset_bytes(output_size, view_count)
    if per_offset <= 0:
        return desired
    per_worker_budget = int((available * 0.35) // max(1, int(worker_count)))
    memory_limit = max(1, per_worker_budget // per_offset)
    return max(1, min(desired, memory_limit))
