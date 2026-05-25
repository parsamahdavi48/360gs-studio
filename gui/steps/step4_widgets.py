"""Small Step 4 widgets shared by the main panel and mixins."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QStackedWidget, QVBoxLayout, QWidget

from gui import i18n


class CurrentPageStack(QStackedWidget):
    """Stacked widget whose size hint follows only the visible page."""

    def sizeHint(self) -> QSize:
        widget = self.currentWidget()
        return widget.sizeHint() if widget is not None else super().sizeHint()

    def minimumSizeHint(self) -> QSize:
        widget = self.currentWidget()
        return widget.minimumSizeHint() if widget is not None else super().minimumSizeHint()


class ElidedPathLabel(QLabel):
    """Keep long paths on one line while preserving the full path internally."""

    def __init__(self, text: str = "-", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = text
        self.setWordWrap(False)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self._full_text = text or "-"
        self._apply_elide()

    def full_text(self) -> str:
        return self._full_text

    def sizeHint(self) -> QSize:
        base = super().sizeHint()
        return QSize(0, base.height())

    def minimumSizeHint(self) -> QSize:
        base = super().minimumSizeHint()
        return QSize(0, base.height())

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_elide()

    def _apply_elide(self) -> None:
        width = max(0, self.contentsRect().width())
        text = self._full_text
        if width > 0:
            text = QFontMetrics(self.font()).elidedText(text, Qt.ElideMiddle, width)
        QLabel.setText(self, text)


@dataclass(frozen=True, slots=True)
class OutputImageControls:
    """Shared output-image option controls used by dataset export pages."""

    widget: QWidget
    format_row: QWidget
    quality_row: QWidget
    output_format_combo: QComboBox
    output_bit_depth_combo: QComboBox
    output_format_label: QLabel
    output_bit_depth_label: QLabel
    invert_masks_cb: QCheckBox
    jpg_quality_edit: QLineEdit
    jpg_quality_label: QLabel


def make_output_image_controls(parent: QWidget | None = None) -> OutputImageControls:
    """Create the shared image format/quality controls for cubemap-style exports."""
    container = QWidget(parent)
    container_layout = QVBoxLayout(container)
    container_layout.setContentsMargins(0, 0, 0, 0)
    container_layout.setSpacing(8)

    format_row = QWidget(container)
    format_layout = QHBoxLayout(format_row)
    format_layout.setContentsMargins(0, 0, 0, 0)
    format_layout.setSpacing(8)

    output_format_combo = QComboBox(format_row)
    output_format_combo.setToolTip(i18n.tip("OUTPUT_FORMAT"))
    output_format_combo.addItem(i18n.t("OUTPUT_FORMAT_AUTO"), "auto")
    for fmt in ("jpg", "png", "tiff", "webp"):
        output_format_combo.addItem(fmt, fmt)
    output_format_combo.setFixedWidth(96)

    output_bit_depth_combo = QComboBox(format_row)
    output_bit_depth_combo.setToolTip(i18n.tip("OUTPUT_BIT_DEPTH"))
    output_bit_depth_combo.addItem(i18n.t("OUTPUT_BIT_DEPTH_8"), "8")
    output_bit_depth_combo.addItem(i18n.t("OUTPUT_BIT_DEPTH_SOURCE"), "source")
    output_bit_depth_combo.setFixedWidth(86)

    output_format_label = QLabel(i18n.t("OUTPUT_FORMAT_COMPACT"), format_row)
    output_format_label.setToolTip(i18n.tip("OUTPUT_FORMAT"))
    format_layout.addWidget(output_format_label)
    format_layout.addWidget(output_format_combo)

    output_bit_depth_label = QLabel(i18n.t("OUTPUT_BIT_DEPTH_COMPACT"), format_row)
    output_bit_depth_label.setToolTip(i18n.tip("OUTPUT_BIT_DEPTH"))
    format_layout.addWidget(output_bit_depth_label)
    format_layout.addWidget(output_bit_depth_combo)
    format_layout.addStretch()
    container_layout.addWidget(format_row)

    quality_row = QWidget(container)
    quality_layout = QHBoxLayout(quality_row)
    quality_layout.setContentsMargins(0, 0, 0, 0)
    quality_layout.setSpacing(8)

    invert_masks_cb = QCheckBox(i18n.INVERT_MASKS, quality_row)
    invert_masks_cb.setToolTip(i18n.tip("INVERT_MASKS"))
    quality_layout.addWidget(invert_masks_cb)
    quality_layout.addSpacing(8)

    jpg_quality_label = QLabel(i18n.t("JPG_QUALITY_COMPACT"), quality_row)
    jpg_quality_label.setToolTip(i18n.tip("JPG_QUALITY"))
    quality_layout.addWidget(jpg_quality_label)

    jpg_quality_edit = QLineEdit("95", quality_row)
    jpg_quality_edit.setToolTip(i18n.tip("JPG_QUALITY"))
    jpg_quality_edit.setFixedWidth(64)
    quality_layout.addWidget(jpg_quality_edit)
    quality_layout.addStretch()
    container_layout.addWidget(quality_row)

    return OutputImageControls(
        widget=container,
        format_row=format_row,
        quality_row=quality_row,
        output_format_combo=output_format_combo,
        output_bit_depth_combo=output_bit_depth_combo,
        output_format_label=output_format_label,
        output_bit_depth_label=output_bit_depth_label,
        invert_masks_cb=invert_masks_cb,
        jpg_quality_edit=jpg_quality_edit,
        jpg_quality_label=jpg_quality_label,
    )
