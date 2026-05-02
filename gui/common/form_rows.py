"""Small helpers for consistent form rows."""
from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QLayout, QWidget


FormField = QWidget | QLayout


def add_tooltip_row(
    form: QFormLayout,
    label_text: str,
    field: FormField,
    tooltip: str | None = None,
) -> None:
    """Add a QFormLayout row and mirror the field tooltip on its label."""
    form.addRow(label_text, field)
    if not tooltip:
        return

    label = form.labelForField(field)
    if label is not None:
        label.setToolTip(tooltip)

    if isinstance(field, QWidget) and not field.toolTip():
        field.setToolTip(tooltip)
