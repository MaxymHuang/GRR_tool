"""Windows 95/98 classic theme, QSS, and shared UI building blocks."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Classic Windows 95/98 palette
BG_APP = "#c0c0c0"
BG_FACE = "#c0c0c0"
BG_FIELD = "#ffffff"
BG_TITLE = "#000080"
BG_TITLE_INACTIVE = "#808080"
BG_SELECTION = "#000080"
TEXT_PRIMARY = "#000000"
TEXT_ON_TITLE = "#ffffff"
TEXT_DISABLED = "#808080"
HIGHLIGHT = "#ffffff"
SHADOW = "#808080"
SHADOW_DARK = "#404040"

# 3D border snippets (raised / sunken)
BORDER_RAISED = (
    f"border-top: 2px solid {HIGHLIGHT};"
    f"border-left: 2px solid {HIGHLIGHT};"
    f"border-right: 2px solid {SHADOW};"
    f"border-bottom: 2px solid {SHADOW};"
)
BORDER_SUNKEN = (
    f"border-top: 2px solid {SHADOW};"
    f"border-left: 2px solid {SHADOW};"
    f"border-right: 2px solid {HIGHLIGHT};"
    f"border-bottom: 2px solid {HIGHLIGHT};"
)
BORDER_PRESSED = (
    f"border-top: 2px solid {SHADOW};"
    f"border-left: 2px solid {SHADOW};"
    f"border-right: 2px solid {HIGHLIGHT};"
    f"border-bottom: 2px solid {HIGHLIGHT};"
)

FONT_UI = "'MS Shell Dlg 2', 'Tahoma', 'MS Sans Serif', sans-serif"
FONT_MONO = "'Fixedsys', 'Courier New', monospace"

VERDICT_COLORS = {
    "acceptable": "#000000",
    "marginal": "#000000",
    "unacceptable": "#000000",
}

VERDICT_BADGE_BG = {
    "acceptable": BG_FACE,
    "marginal": BG_FACE,
    "unacceptable": BG_FACE,
}

VERDICT_BORDER = {
    "acceptable": "#008000",
    "marginal": "#808000",
    "unacceptable": "#800000",
    "": SHADOW,
}

VERDICT_DISPLAY = {
    "acceptable": "Acceptable",
    "marginal": "Marginal",
    "unacceptable": "Unacceptable",
}

STYLESHEET = f"""
/* ===== Base ===== */
QMainWindow, QWidget {{
    background-color: {BG_APP};
    color: {TEXT_PRIMARY};
    font-family: {FONT_UI};
    font-size: 11px;
}}

/* ===== App header (title bar) ===== */
QWidget#appHeader {{
    background-color: {BG_TITLE};
    border: none;
}}
QLabel#appTitle {{
    color: {TEXT_ON_TITLE};
    font-size: 13px;
    font-weight: bold;
    background: transparent;
}}
QLabel#appSubtitle {{
    color: {TEXT_ON_TITLE};
    font-size: 11px;
    background: transparent;
}}

/* ===== Nav sidebar ===== */
QWidget#navSidebar {{
    background-color: {BG_FACE};
    border: none;
    border-right: 1px solid {SHADOW};
}}
QLabel#navSectionLabel {{
    color: {TEXT_PRIMARY};
    font-size: 11px;
    font-weight: bold;
    background: transparent;
}}
QListWidget#navSidebarList {{
    background-color: {BG_FACE};
    border: none;
    outline: none;
    padding: 2px;
}}
QListWidget#navSidebarList::item {{
    padding: 4px 8px;
    color: {TEXT_PRIMARY};
}}
QListWidget#navSidebarList::item:selected {{
    background-color: {BG_SELECTION};
    color: {TEXT_ON_TITLE};
}}
QListWidget#navSidebarList::item:hover:!selected {{
    background-color: #d4d0c8;
}}

/* ===== Page header ===== */
QWidget#pageHeader {{
    background-color: {BG_FACE};
    border: none;
    border-bottom: 1px solid {SHADOW};
}}
QLabel#pageTitle {{
    color: {TEXT_PRIMARY};
    font-size: 13px;
    font-weight: bold;
    background: transparent;
}}
QLabel#pageDescription {{
    color: {TEXT_PRIMARY};
    font-size: 11px;
    background: transparent;
}}

/* ===== Action bar ===== */
QFrame#actionBar {{
    background-color: {BG_FACE};
    border: none;
    border-top: 2px solid {SHADOW};
}}

/* ===== Chart card ===== */
QFrame#chartCard {{
    background-color: {BG_FACE};
    {BORDER_RAISED}
}}
QLabel#chartCardTitle {{
    color: {TEXT_PRIMARY};
    font-weight: bold;
    font-size: 11px;
    background: transparent;
}}
QLabel#chartImageArea {{
    background-color: {BG_FIELD};
    color: {TEXT_DISABLED};
    padding: 4px;
    {BORDER_SUNKEN}
}}

/* ===== Verdict badge ===== */
QLabel#verdictBadge {{
    padding: 2px 8px;
    font-size: 11px;
    font-weight: bold;
    background-color: {BG_FACE};
}}

/* ===== Group Box ===== */
QGroupBox {{
    background-color: {BG_FACE};
    border: 1px solid {SHADOW};
    border-radius: 0px;
    margin-top: 14px;
    padding: 14px 8px 8px 8px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    color: {TEXT_PRIMARY};
    background-color: {BG_FACE};
}}

/* ===== Inputs (sunken) ===== */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {BG_FIELD};
    color: {TEXT_PRIMARY};
    padding: 2px 4px;
    min-height: 18px;
    border-radius: 0px;
    {BORDER_SUNKEN}
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: {BG_FACE};
    color: {TEXT_DISABLED};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 16px;
    {BORDER_RAISED}
    background-color: {BG_FACE};
}}
QComboBox QAbstractItemView {{
    background-color: {BG_FIELD};
    color: {TEXT_PRIMARY};
    border: 1px solid {SHADOW};
    selection-background-color: {BG_SELECTION};
    selection-color: {TEXT_ON_TITLE};
}}

/* ===== Buttons (raised) ===== */
QPushButton {{
    background-color: {BG_FACE};
    color: {TEXT_PRIMARY};
    padding: 4px 12px;
    min-height: 18px;
    min-width: 64px;
    border-radius: 0px;
    {BORDER_RAISED}
}}
QPushButton:hover {{
    background-color: #d4d0c8;
}}
QPushButton:pressed {{
    background-color: {BG_FACE};
    {BORDER_PRESSED}
}}
QPushButton:disabled {{
    color: {TEXT_DISABLED};
    background-color: {BG_FACE};
}}
QPushButton[cssClass="primary"],
QPushButton[cssClass="success"] {{
    background-color: {BG_FACE};
    color: {TEXT_PRIMARY};
    font-weight: bold;
    {BORDER_RAISED}
}}
QPushButton[cssClass="primary"]:pressed,
QPushButton[cssClass="success"]:pressed {{
    {BORDER_PRESSED}
}}
QPushButton[cssClass="compact"] {{
    padding: 2px 8px;
    min-width: 48px;
    min-height: 16px;
}}

/* ===== Checkbox / Radio ===== */
QCheckBox, QRadioButton {{
    spacing: 6px;
    color: {TEXT_PRIMARY};
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 13px;
    height: 13px;
    background-color: {BG_FACE};
    {BORDER_RAISED}
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {BG_FACE};
    {BORDER_SUNKEN}
}}

/* ===== Tables ===== */
QTableWidget {{
    background-color: {BG_FIELD};
    alternate-background-color: #f0f0f0;
    color: {TEXT_PRIMARY};
    gridline-color: {SHADOW};
    border-radius: 0px;
    {BORDER_SUNKEN}
    selection-background-color: {BG_SELECTION};
    selection-color: {TEXT_ON_TITLE};
}}
QTableWidget::item {{
    padding: 2px 4px;
}}
QHeaderView::section {{
    background-color: {BG_FACE};
    color: {TEXT_PRIMARY};
    padding: 4px 6px;
    border: none;
    border-right: 1px solid {SHADOW};
    border-bottom: 1px solid {SHADOW};
    font-weight: bold;
}}

/* ===== Scroll ===== */
QScrollArea {{
    border: none;
    background-color: {BG_FACE};
}}
QScrollBar:vertical {{
    background-color: {BG_FACE};
    width: 16px;
    border: none;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background-color: {BG_FACE};
    min-height: 20px;
    {BORDER_RAISED}
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    background-color: {BG_FACE};
    height: 16px;
    {BORDER_RAISED}
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background-color: {BG_FACE};
}}
QScrollBar:horizontal {{
    background-color: {BG_FACE};
    height: 16px;
}}
QScrollBar::handle:horizontal {{
    background-color: {BG_FACE};
    min-width: 20px;
    {BORDER_RAISED}
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    background-color: {BG_FACE};
    width: 16px;
    {BORDER_RAISED}
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background-color: {BG_FACE};
}}

/* ===== Text Edit ===== */
QTextEdit {{
    background-color: {BG_FIELD};
    color: {TEXT_PRIMARY};
    font-family: {FONT_MONO};
    font-size: 11px;
    padding: 4px;
    border-radius: 0px;
    {BORDER_SUNKEN}
}}

/* ===== Labels ===== */
QLabel {{
    color: {TEXT_PRIMARY};
    background-color: transparent;
}}
QLabel[cssClass="heading"] {{
    font-size: 11px;
    font-weight: bold;
    color: {TEXT_PRIMARY};
}}
QLabel[cssClass="status"] {{
    background-color: {BG_FIELD};
    color: {TEXT_PRIMARY};
    padding: 4px 8px;
    {BORDER_SUNKEN}
}}
QLabel[cssClass="hint"] {{
    color: {TEXT_PRIMARY};
    font-size: 11px;
}}
QLabel[cssClass="fieldLabel"] {{
    color: {TEXT_PRIMARY};
    min-width: 110px;
}}
QLabel[cssClass="metricLabel"] {{
    color: {TEXT_PRIMARY};
}}
QLabel[cssClass="metricValue"] {{
    color: {TEXT_PRIMARY};
    font-weight: bold;
}}

/* ===== List Widget ===== */
QListWidget {{
    background-color: {BG_FIELD};
    color: {TEXT_PRIMARY};
    border-radius: 0px;
    {BORDER_SUNKEN}
}}
QListWidget::item {{
    padding: 2px 4px;
}}
QListWidget::item:selected {{
    background-color: {BG_SELECTION};
    color: {TEXT_ON_TITLE};
}}

/* ===== Status bar ===== */
QStatusBar {{
    background-color: {BG_FACE};
    color: {TEXT_PRIMARY};
    border-top: 1px solid {SHADOW};
    font-size: 11px;
}}
QStatusBar::item {{
    border: none;
}}

/* ===== Dialog ===== */
QDialog, QMessageBox {{
    background-color: {BG_FACE};
    color: {TEXT_PRIMARY};
}}
QMessageBox QLabel {{
    color: {TEXT_PRIMARY};
}}

/* ===== Separator (etched) ===== */
QFrame[cssClass="separator"] {{
    background-color: {SHADOW};
    max-height: 2px;
    border: none;
    border-top: 1px solid {SHADOW};
    border-bottom: 1px solid {HIGHLIGHT};
}}

/* ===== Splitter ===== */
QSplitter::handle {{
    background-color: {BG_FACE};
    {BORDER_RAISED}
}}
QSplitter::handle:horizontal {{
    width: 4px;
}}
QSplitter::handle:vertical {{
    height: 4px;
}}
"""


def polish_widget(widget: QWidget) -> None:
    """Re-apply QSS after dynamic property changes."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def make_btn(text: str, css_class: str = "") -> QPushButton:
    btn = QPushButton(text)
    if css_class:
        btn.setProperty("cssClass", css_class)
        polish_widget(btn)
    return btn


def make_heading(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("cssClass", "heading")
    polish_widget(lbl)
    return lbl


def make_status_label(text: str = "") -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("cssClass", "status")
    polish_widget(lbl)
    return lbl


def make_hint_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setProperty("cssClass", "hint")
    polish_widget(lbl)
    return lbl


def make_field_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("cssClass", "fieldLabel")
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    polish_widget(lbl)
    return lbl


def make_separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setProperty("cssClass", "separator")
    line.setFixedHeight(2)
    polish_widget(line)
    return line


class AppHeader(QWidget):
    def __init__(self, title: str = "Gage R&R Desktop", subtitle: str = "MSA analysis & data preparation"):
        super().__init__()
        self.setObjectName("appHeader")
        self.setFixedHeight(44)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(0)
        title_lbl = QLabel(title)
        title_lbl.setObjectName("appTitle")
        sub_lbl = QLabel(subtitle)
        sub_lbl.setObjectName("appSubtitle")
        layout.addWidget(title_lbl)
        layout.addWidget(sub_lbl)


class PageHeader(QWidget):
    def __init__(self, title: str, description: str):
        super().__init__()
        self.setObjectName("pageHeader")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)
        title_lbl = QLabel(title)
        title_lbl.setObjectName("pageTitle")
        desc_lbl = QLabel(description)
        desc_lbl.setObjectName("pageDescription")
        desc_lbl.setWordWrap(True)
        layout.addWidget(title_lbl)
        layout.addWidget(desc_lbl)


class StickyActionBar(QFrame):
    """Fixed bottom bar for primary workflow actions."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("actionBar")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 6, 8, 6)
        self._layout.setSpacing(6)

    def add_row(self, *widgets: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        for w in widgets:
            row.addWidget(w)
        row.addStretch()
        self._layout.addLayout(row)
        return row

    def add_widget(self, widget: QWidget) -> None:
        self._layout.addWidget(widget)


def verdict_badge_style(verdict: str) -> str:
    key = (verdict or "").lower()
    border = VERDICT_BORDER.get(key, SHADOW)
    return (
        f"background-color: {BG_FACE}; color: {TEXT_PRIMARY}; "
        f"border: 2px solid {border}; border-radius: 0px;"
    )
