"""Route-specific intent behavior for Step 4 SfM workflows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gui.steps.sfm_route_specs import (
    SFM_ROUTE_COLMAP,
    SFM_ROUTE_METASHAPE,
    SFM_ROUTE_SPHERESFM,
    SfmRouteSpec,
    get_sfm_route_spec,
    normalize_sfm_route,
)


@dataclass(frozen=True, slots=True)
class SfmRouteBackend:
    """Small route adapter for the Step 4 pipeline intent model."""

    spec: SfmRouteSpec

    def sfm_intent(self, step: Any) -> bool:
        raise NotImplementedError

    def conversion_intent(self, step: Any) -> bool:
        raise NotImplementedError

    def set_sfm_intent(self, step: Any, enabled: bool) -> None:
        raise NotImplementedError

    def set_conversion_intent(self, step: Any, enabled: bool) -> None:
        raise NotImplementedError

    def sfm_intent_toggle_enabled(self, _step: Any) -> bool:
        return self.spec.runs_sfm_in_app

    def sfm_runs_in_app(self, step: Any) -> bool:
        return self.spec.runs_sfm_in_app and self.sfm_intent(step)


class MetashapeRouteBackend(SfmRouteBackend):
    def sfm_intent(self, step: Any) -> bool:
        return self.conversion_intent(step)

    def conversion_intent(self, step: Any) -> bool:
        return bool(step._conversion_intent)

    def set_sfm_intent(self, _step: Any, _enabled: bool) -> None:
        return

    def set_conversion_intent(self, step: Any, enabled: bool) -> None:
        step._conversion_intent = bool(enabled)

    def sfm_intent_toggle_enabled(self, _step: Any) -> bool:
        return False


class ColmapRouteBackend(SfmRouteBackend):
    def sfm_intent(self, step: Any) -> bool:
        return step.run_colmap_cb.isChecked()

    def conversion_intent(self, step: Any) -> bool:
        return bool(step._conversion_intent)

    def set_sfm_intent(self, step: Any, enabled: bool) -> None:
        step.run_colmap_cb.setChecked(bool(enabled))

    def set_conversion_intent(self, step: Any, enabled: bool) -> None:
        step._conversion_intent = bool(enabled)


class SphereSfmRouteBackend(SfmRouteBackend):
    def sfm_intent(self, step: Any) -> bool:
        return step._spheresfm_runs_sfm()

    def conversion_intent(self, step: Any) -> bool:
        return step._spheresfm_runs_conversion()

    def set_sfm_intent(self, step: Any, enabled: bool) -> None:
        step._set_spheresfm_stage_intents(
            run_sfm=bool(enabled),
            run_conversion=self.conversion_intent(step),
        )

    def set_conversion_intent(self, step: Any, enabled: bool) -> None:
        step._set_spheresfm_stage_intents(
            run_sfm=self.sfm_intent(step),
            run_conversion=bool(enabled),
        )


_BACKENDS: dict[str, SfmRouteBackend] = {
    SFM_ROUTE_METASHAPE: MetashapeRouteBackend(get_sfm_route_spec(SFM_ROUTE_METASHAPE)),
    SFM_ROUTE_COLMAP: ColmapRouteBackend(get_sfm_route_spec(SFM_ROUTE_COLMAP)),
    SFM_ROUTE_SPHERESFM: SphereSfmRouteBackend(get_sfm_route_spec(SFM_ROUTE_SPHERESFM)),
}


def get_sfm_route_backend(route_id: str | None) -> SfmRouteBackend:
    return _BACKENDS[normalize_sfm_route(route_id)]
