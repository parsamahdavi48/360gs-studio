"""Shared command queue type aliases for GUI workflow steps."""

from __future__ import annotations

from core.app_job import AppJob

ExternalCommand = list[str]
ExternalCommandQueue = list[tuple[str, ExternalCommand]]
StepCommand = ExternalCommand | AppJob
StepCommandPhase = tuple[str, StepCommand]
StepCommandQueue = list[StepCommandPhase]
