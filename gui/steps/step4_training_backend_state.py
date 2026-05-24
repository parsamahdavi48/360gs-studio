"""Step 4 training backend selection and executable state helpers."""

from __future__ import annotations

from gui import i18n
from gui.steps.training_backend_specs import (
    TRAINING_BACKEND_LICHTFELD as _TRAINING_BACKEND_LICHTFELD,
)
from gui.steps.training_backend_specs import (
    TRAINING_BACKEND_POSTSHOT as _TRAINING_BACKEND_POSTSHOT,
)
from gui.steps.training_backend_specs import (
    get_training_backend_spec,
    normalize_training_backend,
    training_backend_visible_in_selector,
)


class Step4TrainingBackendStateMixin:
    def _training_backend(self) -> str:
        return self._training_backend_value

    def _training_backend_display_name(self, backend: str) -> str:
        if hasattr(self, "training_backend_selector"):
            return self.training_backend_selector.display_name(backend)
        return i18n.t(get_training_backend_spec(backend).short_label_key)

    def _set_training_backend(self, backend: str) -> None:
        backend = normalize_training_backend(backend)
        if not training_backend_visible_in_selector(backend):
            backend = _TRAINING_BACKEND_LICHTFELD
        previous_backend = getattr(self, "_training_backend_value", "")
        if (
            previous_backend
            and previous_backend != backend
            and hasattr(self, "training_executable_browse")
            and not getattr(self, "_syncing_training_executable", False)
        ):
            current_executable = self.training_executable_browse.text()
            if current_executable or previous_backend not in self._training_executable_by_backend:
                self._training_executable_by_backend[previous_backend] = current_executable
        spec = get_training_backend_spec(backend)
        self._training_backend_value = backend
        if hasattr(self, "training_backend_selector"):
            self.training_backend_selector.set_backend(backend)
        self.training_backend_label.setToolTip(i18n.tip(spec.tooltip_key))
        stack_index = self.training_options_stack_indices[backend]
        self.training_options_stack.setCurrentIndex(stack_index)
        self.training_executable_browse.line_edit.setPlaceholderText(self._default_training_executable(backend))
        self._apply_training_executable_for_backend(backend)
        self.training_headless_cb.setVisible(spec.supports_headless)
        self._refresh_training_settings_layout()
        self._update_training_paths()
        if backend == _TRAINING_BACKEND_POSTSHOT:
            self._update_postshot_project_name()
        if backend == _TRAINING_BACKEND_LICHTFELD:
            self._update_lfs_auto_steps_scaler()
        if getattr(self, "_user_preferences_enabled", False):
            self._save_user_preferences()

    def _restore_training_executables(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        for backend, executable in payload.items():
            normalized = normalize_training_backend(str(backend))
            if not training_backend_visible_in_selector(normalized):
                continue
            text = self._settings_text(executable)
            if text:
                self._training_executable_by_backend[normalized] = text

    def _training_executables_for_settings(self) -> dict[str, str]:
        executables = dict(getattr(self, "_training_executable_by_backend", {}))
        if hasattr(self, "training_executable_browse"):
            executables[self._training_backend()] = self.training_executable_browse.text()
        return {backend: path for backend, path in executables.items() if path}

    def _apply_training_executable_for_backend(self, backend: str) -> None:
        if not hasattr(self, "training_executable_browse"):
            return
        executable = self._training_executable_by_backend.get(backend, "")
        self._syncing_training_executable = True
        try:
            if self.training_executable_browse.text() != executable:
                self.training_executable_browse.set_text(executable)
        finally:
            self._syncing_training_executable = False

    def _on_training_executable_changed(self, path: str) -> None:
        if getattr(self, "_syncing_training_executable", False):
            return
        self._training_executable_by_backend[self._training_backend()] = path.strip()
        self._save_user_preferences()

    def _on_training_dataset_edited(self, _path: str) -> None:
        if self._syncing_training_paths:
            return
        self._training_dataset_user_edited = True
        self._update_path_labels()
        self._update_lfs_auto_steps_scaler()

    def _on_training_output_edited(self, _path: str) -> None:
        if self._syncing_training_paths:
            return
        self._training_output_user_edited = True
        self._update_path_labels()
        self._save_user_preferences()

    def _on_lfs_output_name_edited(self, _text: str) -> None:
        if self._syncing_lfs_output_name:
            return
        self._lfs_output_name_user_edited = True
        self._on_training_settings_changed()

    def _on_postshot_project_name_edited(self, _text: str) -> None:
        if self._syncing_postshot_project_name:
            return
        self._postshot_project_name_user_edited = True
        self._on_training_settings_changed()

    def _on_training_settings_changed(self, *_args) -> None:
        self._update_path_labels()
        self.primary_action_state_changed.emit()

    def _on_lfs_iterations_edited(self, _text: str) -> None:
        if self._syncing_lfs_auto_fields or getattr(self, "_syncing_lfs_strategy_state", False):
            return
        self._on_training_settings_changed()

    def _on_lfs_auto_steps_scaler_changed(self, checked: bool) -> None:
        self.lfs_steps_scaler_edit.setEnabled(not checked)
        if checked:
            self._update_lfs_auto_steps_scaler()
        else:
            self._save_lfs_active_state()
        self._on_training_settings_changed()

    def _on_lfs_gut_changed(self, _checked: bool) -> None:
        self._update_lfs_conditional_visibility()
        self._on_training_settings_changed()

