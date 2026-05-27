"""Step 4 SfM/route backend registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SfmRouteKind = Literal["external_input", "in_app"]

SFM_ROUTE_METASHAPE = "metashape"
SFM_ROUTE_COLMAP = "colmap"
SFM_ROUTE_SPHERESFM = "spheresfm"
DEFAULT_SFM_ROUTE = SFM_ROUTE_METASHAPE

OUTPUT_SHAPE_PROJECTED = "projected"
OUTPUT_SHAPE_EQUIRECT_3DGUT = "equirect_3dgut"


@dataclass(frozen=True, slots=True)
class SfmRouteSpec:
    route_id: str
    label_key: str
    tooltip_key: str
    stack_order: int
    kind: SfmRouteKind
    runs_sfm_in_app: bool
    supports_projected_output: bool
    supports_3dgut_output: bool
    default_output_shape: str = OUTPUT_SHAPE_PROJECTED
    official_url: str = ""
    official_link_key: str = ""

    def supports_output_shape(self, output_shape: str) -> bool:
        if output_shape == OUTPUT_SHAPE_EQUIRECT_3DGUT:
            return self.supports_3dgut_output
        return output_shape == OUTPUT_SHAPE_PROJECTED and self.supports_projected_output


_SPECS: tuple[SfmRouteSpec, ...] = (
    SfmRouteSpec(
        route_id=SFM_ROUTE_METASHAPE,
        label_key="METHOD_METASHAPE_IMPORT",
        tooltip_key="METHOD_METASHAPE_IMPORT",
        stack_order=0,
        kind="external_input",
        runs_sfm_in_app=False,
        supports_projected_output=True,
        supports_3dgut_output=True,
    ),
    SfmRouteSpec(
        route_id=SFM_ROUTE_COLMAP,
        label_key="METHOD_COLMAP_EXPORT",
        tooltip_key="METHOD_COLMAP_EXPORT",
        stack_order=1,
        kind="in_app",
        runs_sfm_in_app=True,
        supports_projected_output=True,
        supports_3dgut_output=False,
        official_url="https://github.com/colmap/colmap",
        official_link_key="COLMAP_REPOSITORY_LINK",
    ),
    SfmRouteSpec(
        route_id=SFM_ROUTE_SPHERESFM,
        label_key="METHOD_SPHERESFM",
        tooltip_key="METHOD_SPHERESFM",
        stack_order=2,
        kind="in_app",
        runs_sfm_in_app=True,
        supports_projected_output=True,
        supports_3dgut_output=True,
        official_url="https://github.com/json87/SphereSfM",
        official_link_key="SPHERESFM_REPOSITORY_LINK",
    ),
)

SFM_ROUTE_SPECS: dict[str, SfmRouteSpec] = {spec.route_id: spec for spec in _SPECS}
SFM_ROUTE_IDS: tuple[str, ...] = tuple(spec.route_id for spec in _SPECS)


def sfm_route_specs() -> tuple[SfmRouteSpec, ...]:
    return _SPECS


def normalize_sfm_route(route_id: str | None) -> str:
    normalized = (route_id or "").strip().lower()
    if normalized in SFM_ROUTE_SPECS:
        return normalized
    return DEFAULT_SFM_ROUTE


def get_sfm_route_spec(route_id: str | None) -> SfmRouteSpec:
    return SFM_ROUTE_SPECS[normalize_sfm_route(route_id)]
