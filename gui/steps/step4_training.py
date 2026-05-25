"""Step 4 training backend UI, settings, and launch command helpers."""

from __future__ import annotations

from gui.steps.step4_training_backend_state import Step4TrainingBackendStateMixin
from gui.steps.step4_training_commands import Step4TrainingCommandsMixin
from gui.steps.step4_training_dataset import Step4TrainingDatasetMixin
from gui.steps.step4_training_lfs_state import Step4TrainingLfsStateMixin
from gui.steps.step4_training_settings_restore import Step4TrainingSettingsRestoreMixin
from gui.steps.step4_training_ui import Step4TrainingUiMixin


class Step4TrainingMixin(
    Step4TrainingUiMixin,
    Step4TrainingSettingsRestoreMixin,
    Step4TrainingCommandsMixin,
    Step4TrainingDatasetMixin,
    Step4TrainingLfsStateMixin,
    Step4TrainingBackendStateMixin,
):
    pass
