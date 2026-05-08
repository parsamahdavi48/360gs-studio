"""Step 5: 学習アプリの起動."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QVBoxLayout

from gui import i18n
from gui.steps.base_step import BaseStepWidget
from gui.steps.step4_cubemap import CubemapStep


class TrainingStep(BaseStepWidget):
    """Training UI and command entrypoint backed by the Step 4 dataset state."""

    def __init__(self, base_dir: Path, dataset_step: CubemapStep, parent=None) -> None:
        super().__init__(base_dir, parent)
        self.dataset_step = dataset_step

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.dataset_step.apply_training_wide_layout())

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        if self.dataset_step.scene_dir != path:
            self.dataset_step.set_scene_dir(path)

    def on_activated(self) -> None:
        self.dataset_step.prepare_training_step()

    def primary_action_text(self) -> str:
        return i18n.t("LAUNCH")

    def primary_action_tooltip(self) -> str:
        return i18n.tip("LAUNCH_TRAINING")

    def primary_action_enabled(self) -> bool:
        return self.dataset_step.training_primary_action_enabled()

    def build_commands(self) -> list[tuple[str, list[str]]]:
        return self.dataset_step.build_training_launch_commands()

    def process_log_dir(self) -> Path | None:
        return self.dataset_step.process_log_dir()

    def phase_display_name(self, phase: str) -> str:
        return self.dataset_step.phase_display_name(phase)

    def on_line(self, line: str) -> tuple[int, int] | None:
        return self.dataset_step.on_line(line)

    def on_phase_started(self, phase: str) -> tuple[int, int] | None:
        return self.dataset_step.on_phase_started(phase)

    def on_phase_log_started(self, phase: str, path: str) -> None:
        self.dataset_step.on_phase_log_started(phase, path)

    def on_phase_finished(self, phase: str, exit_code: int, canceled: bool) -> None:
        self.dataset_step.on_phase_finished(phase, exit_code, canceled)

    def on_queue_finished(self, success: bool) -> None:
        self.dataset_step.on_training_queue_finished(success)
