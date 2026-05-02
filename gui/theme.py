"""ダークモダンテーマ (DaVinci Resolve / Blender 風)"""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

# カラーパレット
BG_DARK = "#15171a"
BG_MID = "#1e2227"
BG_PANEL = "#252a31"
BG_INPUT = "#101316"
BORDER = "#3a424d"
BORDER_FOCUS = "#2dd4bf"
TEXT = "#e5e7eb"
TEXT_DIM = "#9ca3af"
TEXT_BRIGHT = "#ffffff"
ACCENT = "#2dd4bf"
ACCENT_HOVER = "#5eead4"
ACCENT_PRESSED = "#14b8a6"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
DANGER = "#ef4444"
TAB_ACTIVE = "#2dd4bf"
TAB_INACTIVE = "#252a31"
SCROLL_HANDLE = "#4b5563"

FONT_FAMILY = "Meiryo UI"
FONT_SIZE = 10

QSS = f"""
/* ========== Global ========== */
QWidget {{
    background-color: {BG_DARK};
    color: {TEXT};
    font-family: "{FONT_FAMILY}";
    font-size: {FONT_SIZE}pt;
}}

/* ========== App Shell ========== */
QWidget#appHeader {{
    background-color: {BG_MID};
    border-bottom: 1px solid {BORDER};
}}
QLabel#appTitle {{
    color: {TEXT_BRIGHT};
    font-size: 15pt;
    font-weight: 700;
}}
QLabel#appSubtitle {{
    color: {TEXT_DIM};
    font-size: 9pt;
}}
QWidget#sidebar {{
    background-color: {BG_MID};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QPushButton#navStep {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    color: {TEXT};
    padding: 6px 4px;
    font-weight: 600;
    font-size: 9pt;
}}
QPushButton#navStep:hover {{
    background-color: {BG_PANEL};
    border-color: {BORDER};
}}
QPushButton#navStep:checked {{
    background-color: {BG_PANEL};
    border-color: {ACCENT};
    color: {TEXT_BRIGHT};
}}
QWidget#contentPanel {{
    background-color: {BG_MID};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QLabel#stepHeader {{
    color: {TEXT_BRIGHT};
    font-size: 13pt;
    font-weight: 700;
    padding: 2px 4px 0 4px;
}}
QLabel#stepSubheader {{
    color: {TEXT_DIM};
    font-size: 9pt;
    padding: 0 4px 8px 4px;
}}
QWidget#jobPanel {{
    background-color: {BG_MID};
    border-top: 1px solid {BORDER};
}}
QWidget#settingsPane {{
    background-color: transparent;
}}
QWidget#workPane {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QLabel#paneTitle {{
    color: {TEXT_BRIGHT};
    font-size: 10pt;
    font-weight: 700;
    padding-bottom: 4px;
}}
QLabel#workflowNote {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER_FOCUS};
    border-radius: 6px;
    color: {ACCENT_HOVER};
    padding: 12px;
    font-size: 9pt;
}}
QLabel#emptyPaneMessage {{
    color: {TEXT_DIM};
    font-size: 10pt;
}}

/* ========== QLineEdit / QComboBox / QSpinBox ========== */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    color: {TEXT};
    min-height: 20px;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {BORDER_FOCUS};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {TEXT_DIM};
    background-color: {BG_MID};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {TEXT_DIM};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    selection-color: {TEXT_BRIGHT};
    padding: 4px;
}}

/* ========== QPushButton ========== */
QPushButton {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px 16px;
    color: {TEXT};
    min-height: 22px;
}}
QPushButton:hover {{
    background-color: {BORDER};
    border-color: {TEXT_DIM};
}}
QPushButton:pressed {{
    background-color: {BG_MID};
}}
QPushButton:disabled {{
    color: {TEXT_DIM};
    background-color: {BG_MID};
    border-color: {BG_MID};
}}

/* Primary buttons (objectName = "primary") */
QPushButton#primary {{
    background-color: {ACCENT};
    border: none;
    color: {TEXT_BRIGHT};
    font-weight: 600;
    padding: 8px 24px;
}}
QPushButton#primary:hover {{
    background-color: {ACCENT_HOVER};
}}
QPushButton#primary:pressed {{
    background-color: {ACCENT_PRESSED};
}}
QPushButton#primary:disabled {{
    background-color: {BG_PANEL};
    color: {TEXT_DIM};
}}

/* Danger buttons */
QPushButton#danger {{
    background-color: {DANGER};
    border: none;
    color: {TEXT_BRIGHT};
    font-weight: 600;
}}
QPushButton#danger:hover {{
    background-color: #dc2626;
}}
QPushButton#danger:disabled {{
    background-color: {BG_PANEL};
    color: {TEXT_DIM};
}}
QToolButton#iconToolButton {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px;
}}
QToolButton#iconToolButton:hover {{
    background-color: {BORDER};
    border-color: {TEXT_DIM};
}}
QToolButton#iconToolButton:pressed {{
    background-color: {BG_MID};
}}
QToolButton#iconToolButton:disabled {{
    background-color: {BG_MID};
    border-color: {BG_MID};
}}

/* ========== QTabWidget ========== */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background-color: {BG_MID};
    top: -1px;
}}
QTabBar::tab {{
    background-color: {TAB_INACTIVE};
    border: 1px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 20px;
    margin-right: 2px;
    color: {TEXT_DIM};
    font-weight: 500;
    min-width: 120px;
}}
QTabBar::tab:selected {{
    background-color: {BG_MID};
    color: {TEXT_BRIGHT};
    border-bottom: 2px solid {TAB_ACTIVE};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{
    background-color: {BG_PANEL};
    color: {TEXT};
}}

/* ========== QProgressBar ========== */
QProgressBar {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    text-align: center;
    color: {TEXT};
    min-height: 20px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT}, stop:1 {ACCENT_HOVER});
    border-radius: 3px;
}}

/* ========== QPlainTextEdit (log) ========== */
QPlainTextEdit {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT};
    font-family: "Cascadia Code";
    font-size: 9pt;
    padding: 4px;
}}

/* ========== QCheckBox / QRadioButton ========== */
QCheckBox, QRadioButton {{
    spacing: 6px;
    color: {TEXT};
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER};
    background-color: {BG_INPUT};
}}
QCheckBox::indicator {{
    border-radius: 3px;
}}
QRadioButton::indicator {{
    border-radius: 8px;
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background-color: {BG_MID};
    border-color: {BG_MID};
}}

/* ========== QScrollBar ========== */
QScrollBar:vertical {{
    background-color: {BG_DARK};
    width: 10px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background-color: {SCROLL_HANDLE};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {TEXT_DIM};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background-color: {BG_DARK};
    height: 10px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background-color: {SCROLL_HANDLE};
    border-radius: 5px;
    min-width: 30px;
}}

/* ========== QSlider ========== */
QSlider::groove:horizontal {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    height: 6px;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background-color: {ACCENT};
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background-color: {ACCENT_HOVER};
}}

/* ========== QToolButton (collapsible) ========== */
QToolButton {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 12px;
    color: {TEXT};
    font-weight: 500;
}}
QToolButton:hover {{
    background-color: {BORDER};
}}
QToolButton:checked {{
    background-color: {BG_MID};
    border-color: {ACCENT};
}}

/* ========== QSplitter ========== */
QSplitter::handle {{
    background-color: {BORDER};
}}
QSplitter::handle:horizontal {{
    width: 3px;
}}
QSplitter::handle:vertical {{
    height: 3px;
}}

/* ========== QScrollArea ========== */
QScrollArea {{
    border: none;
    background-color: transparent;
}}

/* ========== QFormLayout labels ========== */
QLabel {{
    color: {TEXT};
    background-color: transparent;
}}

/* ========== QGroupBox ========== */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 16px;
    color: {TEXT};
    font-weight: 500;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    color: {ACCENT_HOVER};
}}

/* ========== QMessageBox ========== */
QMessageBox {{
    background-color: {BG_PANEL};
}}

/* ========== Tooltip ========== */
QToolTip {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    color: {TEXT};
    padding: 4px 8px;
    border-radius: 4px;
}}
"""


def apply_theme(app: QApplication) -> None:
    """QApplicationにダークテーマを適用する。"""
    app.setStyleSheet(QSS)
    font = QFont()
    font.setFamily(FONT_FAMILY)
    font.setPointSize(FONT_SIZE)
    app.setFont(font)
