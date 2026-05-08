"""Compact Step 4 training backend selector."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from gui import i18n
from gui.steps.training_backend_specs import (
    DEFAULT_TRAINING_BACKEND,
    OTHER_TRAINING_BACKEND_IDS,
    TrainingBackendSpec,
    get_training_backend_spec,
    normalize_training_backend,
    training_backend_specs,
)


class TrainingBackendSelector(QWidget):
    """Shows primary backends first and keeps secondary backends behind Other."""

    backend_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._backend = DEFAULT_TRAINING_BACKEND
        self._last_other_backend = (
            OTHER_TRAINING_BACKEND_IDS[0] if OTHER_TRAINING_BACKEND_IDS else DEFAULT_TRAINING_BACKEND
        )

        self.backend_buttons: dict[str, QRadioButton] = {}
        self.primary_backend_buttons: dict[str, QRadioButton] = {}
        self.other_backend_buttons: dict[str, QRadioButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.primary_row = QWidget()
        self.primary_row.setObjectName("trainingBackendPrimaryRow")
        primary_layout = QHBoxLayout(self.primary_row)
        primary_layout.setContentsMargins(0, 0, 0, 0)
        primary_layout.setSpacing(10)

        self.label = QLabel(i18n.t("TRAINING_BACKEND_LABEL"))
        primary_layout.addWidget(self.label)

        self.primary_group = QButtonGroup(self)
        self.primary_group.setExclusive(True)
        for spec in training_backend_specs(category="primary"):
            btn = self._make_backend_button(spec)
            btn.clicked.connect(
                lambda _checked=False, backend_id=spec.backend_id: self.set_backend(backend_id, emit=True)
            )
            self.primary_group.addButton(btn)
            self.primary_backend_buttons[spec.backend_id] = btn
            self.backend_buttons[spec.backend_id] = btn
            primary_layout.addWidget(btn)

        self.other_button = QRadioButton(i18n.t("TRAINING_BACKEND_OTHER"))
        self.other_button.setObjectName("optionRadio")
        self.other_button.setToolTip(i18n.tip("TRAINING_BACKEND_OTHER"))
        self.other_button.clicked.connect(lambda _checked=False: self.set_backend(self._last_other_backend, emit=True))
        self.primary_group.addButton(self.other_button)
        primary_layout.addWidget(self.other_button)
        primary_layout.addStretch()
        layout.addWidget(self.primary_row)

        self.other_row = QWidget()
        self.other_row.setObjectName("trainingBackendOtherRow")
        other_layout = QHBoxLayout(self.other_row)
        other_layout.setContentsMargins(0, 0, 0, 0)
        other_layout.setSpacing(10)
        self.other_label = QLabel(i18n.t("TRAINING_BACKEND_OTHER_LABEL"))
        self.other_label.setObjectName("subOptionLabel")
        self.other_label.setToolTip(i18n.tip("TRAINING_BACKEND_OTHER"))
        other_layout.addSpacing(12)
        other_layout.addWidget(self.other_label)

        self.other_group = QButtonGroup(self)
        self.other_group.setExclusive(True)
        for spec in training_backend_specs(category="other"):
            btn = self._make_backend_button(spec)
            btn.clicked.connect(
                lambda _checked=False, backend_id=spec.backend_id: self.set_backend(backend_id, emit=True)
            )
            self.other_group.addButton(btn)
            self.other_backend_buttons[spec.backend_id] = btn
            self.backend_buttons[spec.backend_id] = btn
            other_layout.addWidget(btn)
        other_layout.addStretch()
        layout.addWidget(self.other_row)

        self.set_backend(DEFAULT_TRAINING_BACKEND)

    @staticmethod
    def _make_backend_button(spec: TrainingBackendSpec) -> QRadioButton:
        btn = QRadioButton(i18n.t(spec.short_label_key))
        btn.setObjectName("optionRadio")
        btn.setToolTip(i18n.tip(spec.tooltip_key))
        return btn

    def current_backend(self) -> str:
        return self._backend

    def display_name(self, backend_id: str | None = None) -> str:
        spec = get_training_backend_spec(backend_id or self._backend)
        return i18n.t(spec.short_label_key)

    def set_backend(self, backend_id: str | None, *, emit: bool = False) -> None:
        backend = normalize_training_backend(backend_id)
        spec = get_training_backend_spec(backend)
        changed = backend != self._backend
        self._backend = backend

        if spec.category == "other":
            self._last_other_backend = backend
            if not self.other_button.isChecked():
                self.other_button.setChecked(True)
            btn = self.other_backend_buttons.get(backend)
            if btn is not None and not btn.isChecked():
                btn.setChecked(True)
            self.other_row.setVisible(True)
        else:
            btn = self.primary_backend_buttons.get(backend)
            if btn is not None and not btn.isChecked():
                btn.setChecked(True)
            self.other_row.setVisible(False)

        self.label.setToolTip(i18n.tip(spec.tooltip_key))
        if changed and emit:
            self.backend_changed.emit(backend)
