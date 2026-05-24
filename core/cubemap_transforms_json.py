from core.cubemap_export_metadata import (
    collect_image_files as collect_image_files,
)
from core.cubemap_export_metadata import (
    infer_image_only_sizes as infer_image_only_sizes,
)
from core.cubemap_export_metadata import (
    write_colmap_rig_metadata as write_colmap_rig_metadata,
)
from core.cubemap_export_metadata import (
    write_image_only_metadata as write_image_only_metadata,
)
from core.cubemap_image_conversion import (
    convert_images as convert_images,
)
from core.cubemap_image_conversion import (
    convert_images_colmap_rig as convert_images_colmap_rig,
)
from core.cubemap_image_conversion import (
    count_planned_outputs as count_planned_outputs,
)
from core.cubemap_image_conversion import (
    get_remap_tables_for_file as get_remap_tables_for_file,
)
from core.cubemap_image_conversion import (
    get_remap_tables_for_input_size as get_remap_tables_for_input_size,
)
from core.cubemap_image_conversion import (
    get_remap_tables_for_offset as get_remap_tables_for_offset,
)
from core.cubemap_image_conversion import (
    make_colmap_rig_jobs as make_colmap_rig_jobs,
)
from core.cubemap_image_conversion import (
    mask_candidates as mask_candidates,
)
from core.cubemap_image_conversion import (
    proc_convert_images as proc_convert_images,
)
from core.cubemap_image_conversion import (
    proc_convert_images_colmap_rig as proc_convert_images_colmap_rig,
)
from core.cubemap_image_conversion import (
    remap_image as remap_image,
)
from core.cubemap_image_conversion import (
    remap_input_size as remap_input_size,
)
from core.cubemap_image_conversion import (
    worker_init as worker_init,
)
from core.cubemap_image_conversion import (
    worker_init_colmap_rig as worker_init_colmap_rig,
)
from core.cubemap_view_spec import (
    load_views_json,
    make_default_cube6_views,
    views_to_dicts,
)


def parse_args():
    from core.cubemap_transforms_json_cli import parse_args as _parse_args

    return _parse_args()


def make_default_views(yaw: float, stitch: float, no_top: bool, no_bottom: bool) -> list[dict]:
    return views_to_dicts(make_default_cube6_views(yaw, stitch, no_top=no_top, no_bottom=no_bottom))


def load_custom_views(path: str) -> list[dict]:
    return views_to_dicts(load_views_json(path))


def main() -> None:
    from core.cubemap_transforms_json_cli import main as _main

    _main()


if __name__ == "__main__":
    main()
