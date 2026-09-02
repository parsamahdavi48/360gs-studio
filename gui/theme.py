"""Original Midnight workstation theme for 360GS Studio."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

# カラーパレット
BG_DARK = "#10131a"
BG_MID = "#171b24"
BG_PANEL = "#202632"
BG_INPUT = "#0c0f15"
BORDER = "#353d4d"
BORDER_FOCUS = "#5aa7ff"
TEXT = "#e8edf6"
TEXT_DIM = "#9ba7b9"
TEXT_BRIGHT = "#f8fbff"
ACCENT = "#5aa7ff"
ACCENT_HOVER = "#83bdff"
ACCENT_PRESSED = "#3d8ce6"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
DANGER = "#ef4444"
TAB_ACTIVE = "#a66fe0"
TAB_INACTIVE = "#202632"
SCROLL_HANDLE = "#4b5563"

FONT_FAMILY = "Inter"
FONT_SIZE = 10
_EXTRA_FONT_PATHS = (
    Path(__file__).resolve().parent / "assets" / "fonts" / "Inter-Regular.ttf",
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/meiryo.ttc"),
    Path("C:/Windows/Fonts/YuGothR.ttc"),
    Path("C:/Windows/Fonts/msgothic.ttc"),
)
_EXTRA_FONTS_LOADED = False


def _ensure_font_available() -> None:
    """Load Windows Japanese fonts explicitly when Qt offscreen skips system fonts."""
    global _EXTRA_FONTS_LOADED
    if _EXTRA_FONTS_LOADED:
        return

    families = set(QFontDatabase.families())
    if FONT_FAMILY in families:
        _EXTRA_FONTS_LOADED = True
        return

    for path in _EXTRA_FONT_PATHS:
        if path.exists():
            QFontDatabase.addApplicationFont(str(path))
    _EXTRA_FONTS_LOADED = True

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
QLabel#statusPill {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 9px;
    color: {TEXT_DIM};
    padding: 3px 8px;
    font-size: 8pt;
}}
QLabel#dockSectionTitle {{
    color: {TEXT_DIM};
    font-size: 8pt;
    font-weight: 700;
    letter-spacing: 1px;
}}
QLabel#projectName, QLabel#inspectorTitle {{
    color: {TEXT_BRIGHT};
    font-size: 12pt;
    font-weight: 700;
}}
QLabel#mutedText {{
    color: {TEXT_DIM};
    font-size: 8pt;
}}
QLabel#inspectorValue {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    color: {TEXT};
    padding: 8px;
}}
QWidget#inspectorDivider {{
    background-color: {BORDER};
}}
QTreeWidget#artifactTree {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 3px;
}}
QPushButton#compactButton {{
    min-height: 22px;
    padding: 2px 8px;
    font-size: 8pt;
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
    padding: 4px 4px 0 0;
}}
QWidget#jobPanel {{
    background-color: {BG_MID};
    border-top: 1px solid {BORDER};
}}
QWidget#settingsPane {{
    background-color: transparent;
}}
QWidget#stickySummaryBar {{
    border-top: 1px solid {BORDER};
    background-color: transparent;
}}
QLabel#stickySummaryLabel {{
    color: {TEXT_DIM};
    font-size: 9pt;
    padding-top: 2px;
}}
QWidget#tabPathSummary {{
    background-color: transparent;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 3px;
}}
QLabel#tabPathSummaryKind {{
    color: {TEXT_DIM};
    font-size: 8pt;
    font-weight: 600;
}}
QLabel#tabPathSummaryValue {{
    color: {TEXT};
    font-size: 8pt;
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
QLabel#assetFolderHint {{
    color: {TEXT_DIM};
    font-size: 8pt;
    padding-left: 2px;
}}
QLabel#emptyPaneMessage {{
    color: {TEXT_DIM};
    font-size: 10pt;
}}
QWidget#workflowCardGrid {{
    background-color: transparent;
}}
QPushButton#workflowCard {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 0px;
    text-align: left;
    min-height: 184px;
    max-height: 184px;
}}
QPushButton#workflowCard:hover {{
    background-color: {BORDER};
    border-color: {BORDER_FOCUS};
}}
QPushButton#workflowCard:pressed {{
    background-color: {BG_MID};
}}
QLabel#workflowCardTitle {{
    color: {TEXT_BRIGHT};
    font-size: 12pt;
    font-weight: 700;
}}
QLabel#workflowCardBody {{
    color: {TEXT};
    font-size: 9pt;
}}
QLabel#workflowCardFooter {{
    color: {ACCENT_HOVER};
    font-size: 8pt;
    font-weight: 600;
}}
QWidget#toolDetailHeader {{
    background-color: transparent;
}}
QWidget#detailActionRow {{
    background-color: transparent;
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
QWidget#segmentedControl {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QWidget#radioOptionRow {{
    background-color: transparent;
}}
QWidget#trainingBackendPrimaryRow {{
    background-color: transparent;
}}
QWidget#trainingBackendOtherPicker {{
    background-color: transparent;
}}
QRadioButton#optionRadio {{
    color: {TEXT_DIM};
    font-weight: 600;
    padding: 2px 0px;
}}
QRadioButton#optionRadio:checked {{
    color: {TEXT_BRIGHT};
}}
QToolButton#optionMenuArrow {{
    background-color: transparent;
    border: 1px solid transparent;
    color: {TEXT_DIM};
    min-width: 13px;
    max-width: 13px;
    min-height: 20px;
    padding: 0px;
}}
QToolButton#optionMenuArrow:hover {{
    background-color: transparent;
    border-color: transparent;
    color: {TEXT};
}}
QToolButton#optionMenuArrow::menu-indicator {{
    image: none;
    width: 0px;
    height: 0px;
}}
QPushButton#segmentedOption {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    color: {TEXT_DIM};
    font-weight: 600;
    min-height: 22px;
    padding: 5px 6px;
}}
QPushButton#segmentedOption:checked {{
    background-color: {BG_PANEL};
    border-color: {ACCENT};
    color: {TEXT_BRIGHT};
}}
QPushButton#segmentedOption:hover:!checked {{
    background-color: {BG_PANEL};
    border-color: {BORDER_FOCUS};
    color: {TEXT};
}}
QPushButton#segmentedOption:pressed {{
    background-color: {BG_MID};
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
QToolButton#iconToolButton:checked {{
    background-color: {BG_MID};
    border-color: {ACCENT};
}}
QToolButton#iconToolButton:checked:hover {{
    background-color: {BG_PANEL};
    border-color: {ACCENT_HOVER};
}}
QToolButton#iconToolButton:disabled {{
    background-color: {BG_MID};
    border-color: {BG_MID};
}}
QToolButton#iconToolButton[hideMenuIndicator="true"]::menu-indicator {{
    image: none;
    width: 0px;
    height: 0px;
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
QTabWidget#maskSettingsTabs QTabBar::tab,
QTabWidget#step4SettingsTabs QTabBar::tab {{
    min-width: 86px;
    padding: 7px 10px;
}}
QTabWidget#step4SettingsTabs QTabBar::tab {{
    min-width: 54px;
    padding: 7px 6px;
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
QCheckBox:disabled, QRadioButton:disabled {{
    color: {TEXT_DIM};
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
QCheckBox::indicator:checked:disabled, QRadioButton::indicator:checked:disabled {{
    background-color: {ACCENT_PRESSED};
    border-color: {ACCENT_PRESSED};
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
    _ensure_font_available()
    app.setStyleSheet(QSS)
    font = QFont()
    font.setFamily(FONT_FAMILY)
    font.setPointSize(FONT_SIZE)
    app.setFont(font)
