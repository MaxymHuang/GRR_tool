"""PySide6 UI for the standalone data parser (aligned with GRRTool Parse tab)."""

STYLESHEET = """
/* ===== Base ===== */
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 13px;
}

/* ===== Tab Widget ===== */
QTabWidget::pane {
    border: 1px solid #45475a;
    border-radius: 6px;
    background-color: #1e1e2e;
    top: -1px;
}
QTabBar::tab {
    background-color: #313244;
    color: #a6adc8;
    padding: 10px 28px;
    margin-right: 2px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 500;
    font-size: 13px;
}
QTabBar::tab:selected {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-bottom: 2px solid #7c8aff;
}
QTabBar::tab:hover:!selected {
    background-color: #3b3b52;
    color: #cdd6f4;
}

/* ===== Group Box (Cards) ===== */
QGroupBox {
    background-color: #2a2a3c;
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 16px;
    padding: 18px 14px 14px 14px;
    font-weight: 600;
    font-size: 13px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 12px;
    color: #7c8aff;
    font-size: 13px;
}

/* ===== Inputs ===== */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 10px;
    color: #cdd6f4;
    min-height: 22px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #7c8aff;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    background-color: #252536;
    color: #585b70;
}
QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #313244;
    border: 1px solid #45475a;
    color: #cdd6f4;
    selection-background-color: #7c8aff;
    selection-color: #1e1e2e;
}

/* ===== Buttons ===== */
QPushButton {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px 20px;
    color: #cdd6f4;
    font-weight: 500;
    min-height: 18px;
}
QPushButton:hover {
    background-color: #3b3b52;
    border-color: #585b70;
}
QPushButton:pressed {
    background-color: #45475a;
}
QPushButton:disabled {
    background-color: #252536;
    color: #45475a;
    border-color: #3b3b52;
}
QPushButton[cssClass="primary"] {
    background-color: #7c8aff;
    color: #1e1e2e;
    border: none;
    font-weight: 600;
}
QPushButton[cssClass="primary"]:hover {
    background-color: #9099ff;
}
QPushButton[cssClass="primary"]:pressed {
    background-color: #6c7aef;
}
QPushButton[cssClass="success"] {
    background-color: #a6e3a1;
    color: #1e1e2e;
    border: none;
    font-weight: 600;
}
QPushButton[cssClass="success"]:hover {
    background-color: #b6f3b1;
}
QPushButton[cssClass="success"]:pressed {
    background-color: #96d391;
}

/* ===== Checkbox ===== */
QCheckBox {
    spacing: 8px;
    color: #cdd6f4;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #45475a;
    background-color: #313244;
}
QCheckBox::indicator:checked {
    background-color: #7c8aff;
    border-color: #7c8aff;
}
QCheckBox::indicator:hover {
    border-color: #7c8aff;
}

/* ===== Tables ===== */
QTableWidget {
    background-color: #2a2a3c;
    alternate-background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    gridline-color: #3b3b52;
    color: #cdd6f4;
    selection-background-color: #45475a;
    selection-color: #cdd6f4;
}
QTableWidget::item {
    padding: 4px 8px;
}
QHeaderView::section {
    background-color: #313244;
    color: #7c8aff;
    padding: 8px;
    border: none;
    border-bottom: 2px solid #45475a;
    font-weight: 600;
    font-size: 12px;
}

/* ===== Scroll ===== */
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollBar:vertical {
    background-color: #1e1e2e;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: #45475a;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background-color: #585b70;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    height: 0; background: none;
}
QScrollBar:horizontal {
    background-color: #1e1e2e;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background-color: #45475a;
    border-radius: 5px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #585b70;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    width: 0; background: none;
}

/* ===== Text Edit ===== */
QTextEdit {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px;
    color: #cdd6f4;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}

/* ===== Labels ===== */
QLabel {
    color: #bac2de;
    background-color: transparent;
}
QLabel[cssClass="heading"] {
    font-size: 15px;
    font-weight: 700;
    color: #cdd6f4;
    padding: 4px 0;
}
QLabel[cssClass="status"] {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px 12px;
    color: #a6adc8;
    font-size: 12px;
}

/* ===== List Widget ===== */
QListWidget {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    color: #cdd6f4;
    outline: none;
}
QListWidget::item {
    padding: 6px 10px;
    border-radius: 4px;
}
QListWidget::item:hover {
    background-color: #3b3b52;
}

/* ===== Dialog ===== */
QDialog {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
QMessageBox {
    background-color: #1e1e2e;
}
QMessageBox QLabel {
    color: #cdd6f4;
}

/* ===== Separator ===== */
QFrame[cssClass="separator"] {
    background-color: #45475a;
    max-height: 1px;
    margin: 4px 0;
}

/* ===== Splitter ===== */
QSplitter::handle {
    background-color: #45475a;
    width: 2px;
}
"""


import os
import sys

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from data_parser_app.parser import (
    assign_operators_sequential,
    get_components_preview,
    get_measurement_columns,
    load_and_clean_data,
    remove_outliers_iqr,
)


def select_file(parent: QWidget, caption: str = "Select Data File") -> str:
    path, _ = QFileDialog.getOpenFileName(
        parent, caption, os.getcwd(), "Data Files (*.txt *.csv);;All Files (*)"
    )
    return path or ""


def _make_btn(text: str, css_class: str = "") -> QPushButton:
    btn = QPushButton(text)
    if css_class:
        btn.setProperty("cssClass", css_class)
    return btn


def _make_status_label(text: str = "") -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("cssClass", "status")
    return lbl


def _make_vsep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setStyleSheet("color: #45475a;")
    sep.setFixedWidth(2)
    return sep


class TablePanel(QTableWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)

    def load_dataframe(self, df) -> None:
        if df is None:
            self.setRowCount(0)
            self.setColumnCount(0)
            return
        self.setRowCount(len(df.index))
        self.setColumnCount(len(df.columns))
        self.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for i, row in enumerate(df.itertuples(index=False)):
            for j, val in enumerate(row):
                self.setItem(i, j, QTableWidgetItem(str(val)))
        self.resizeColumnsToContents()


class ParseTabWidget(QWidget):
    """Parse workflow: load, filter components/algorithms, preview, IQR, save."""

    def __init__(self) -> None:
        super().__init__()
        self.df_base = None
        self.df_preview = None
        self._selected_components = []
        self._selected_algorithms = []
        self._init_ui()

    def _init_ui(self) -> None:
        main = QVBoxLayout()
        main.setSpacing(10)
        main.setContentsMargins(16, 12, 16, 12)

        file_grp = QGroupBox("File && Settings")
        fg = QGridLayout()
        fg.setSpacing(8)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select a data file (.txt or .csv)")
        browse_btn = QPushButton("Browse?")
        browse_btn.clicked.connect(self._browse)
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("Optional output filename prefix")
        self.op_spin = QSpinBox()
        self.op_spin.setRange(1, 10)
        self.op_spin.setValue(3)
        self.keep_raw_chk = QCheckBox("Keep all columns (raw)")
        self.algo_combo = QComboBox()
        self.algo_combo.setPlaceholderText("Load a file to populate")

        fg.addWidget(QLabel("Data File"), 0, 0)
        fg.addWidget(self.file_edit, 0, 1, 1, 2)
        fg.addWidget(browse_btn, 0, 3)
        fg.addWidget(QLabel("Output Prefix"), 1, 0)
        fg.addWidget(self.prefix_edit, 1, 1, 1, 2)
        fg.addWidget(QLabel("Operators"), 2, 0)
        fg.addWidget(self.op_spin, 2, 1)
        fg.addWidget(self.keep_raw_chk, 2, 2)
        fg.addWidget(QLabel("Measurement"), 3, 0)
        fg.addWidget(self.algo_combo, 3, 1, 1, 2)
        fg.setColumnStretch(1, 1)
        file_grp.setLayout(fg)
        main.addWidget(file_grp)

        actions = QHBoxLayout()
        actions.setSpacing(8)

        load_btn = _make_btn("Load File", "primary")
        load_btn.clicked.connect(self._load_file)
        select_comp_btn = QPushButton("Select Components?")
        select_comp_btn.clicked.connect(self._select_components_dialog)
        select_algo_btn = QPushButton("Select Algorithms?")
        select_algo_btn.clicked.connect(self._select_algorithms_dialog)
        preview_btn = _make_btn("Preview", "primary")
        preview_btn.clicked.connect(self._preview)
        preview_iqr_btn = QPushButton("Preview with IQR")
        preview_iqr_btn.clicked.connect(self._preview_with_iqr)
        disp_comp_btn = QPushButton("Display Components")
        disp_comp_btn.clicked.connect(self._display_components)
        save_btn = _make_btn("Save?", "success")
        save_btn.clicked.connect(self._save_dialog)

        actions.addWidget(load_btn)
        actions.addWidget(_make_vsep())
        actions.addWidget(select_comp_btn)
        actions.addWidget(select_algo_btn)
        actions.addWidget(_make_vsep())
        actions.addWidget(preview_btn)
        actions.addWidget(preview_iqr_btn)
        actions.addWidget(disp_comp_btn)
        actions.addStretch()
        actions.addWidget(save_btn)
        main.addLayout(actions)

        self.status_lbl = _make_status_label("Ready ? load a file to begin")
        main.addWidget(self.status_lbl)

        self.comp_text = QTextEdit()
        self.comp_text.setReadOnly(True)
        self.comp_text.setMaximumHeight(80)
        self.comp_text.setPlaceholderText(
            "Component names will appear here after clicking Display Components"
        )
        main.addWidget(self.comp_text)

        self.table = TablePanel()
        main.addWidget(self.table, 1)

        self.setLayout(main)

    def _browse(self) -> None:
        path = select_file(self, "Select Input File")
        if path:
            self.file_edit.setText(path)

    def _load_file(self) -> None:
        try:
            path = self.file_edit.text().strip()
            if not path:
                raise ValueError("No file selected")
            df = load_and_clean_data(path, keep_all_columns=self.keep_raw_chk.isChecked())
            self.df_base = df.reset_index(drop=True)
            meas_cols = get_measurement_columns(self.df_base)
            self.algo_combo.clear()
            self.algo_combo.addItems(meas_cols)
            self._set_status(
                f"Loaded: {len(self.df_base)} rows, {len(self.df_base.columns)} cols ? "
                f"{len(meas_cols)} numeric measurements"
            )
            if self.keep_raw_chk.isChecked():
                self.df_preview = self.df_base.copy()
            else:
                self.df_preview = assign_operators_sequential(
                    self.df_base.copy(), n_operators=self.op_spin.value()
                )
            self._render(self.df_preview)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _display_components(self) -> None:
        if self.df_base is None or "Comp_Name" not in self.df_base.columns:
            self.comp_text.setPlainText("(No components to display)")
            return
        preview = get_components_preview(self.df_base)
        self.comp_text.setPlainText(preview)

    def _apply_filters(self, base_df):
        df = base_df.copy()
        if "Comp_Name" in df.columns and self._selected_components:
            df = df[df["Comp_Name"].isin(set(self._selected_components))].reset_index(
                drop=True
            )
        if self._selected_algorithms:
            keep_cols = []
            for base_col in ["Comp_Name", "Box_Name", "Component"]:
                if base_col in df.columns:
                    keep_cols.append(base_col)
            keep_cols += [c for c in self._selected_algorithms if c in df.columns]
            if keep_cols:
                df = df[keep_cols]
        if not self.keep_raw_chk.isChecked():
            df = assign_operators_sequential(df, n_operators=self.op_spin.value())
        return df

    def _select_algorithms_dialog(self) -> None:
        if self.df_base is None:
            QMessageBox.information(self, "Select Algorithms", "Load a file first.")
            return
        meas_cols = get_measurement_columns(self.df_base)
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Algorithms")
        dlg.setMinimumWidth(360)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Check the algorithms to keep:"))
        lst = QListWidget()
        lst.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        preselected = set(self._selected_algorithms or meas_cols)
        for col in meas_cols:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if col in preselected else Qt.CheckState.Unchecked
            )
            lst.addItem(item)
        layout.addWidget(lst)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            selected = [
                lst.item(i).text()
                for i in range(lst.count())
                if lst.item(i).checkState() == Qt.CheckState.Checked
            ]
            self._selected_algorithms = selected
            QMessageBox.information(
                self, "Algorithms Selected", f"Selected {len(selected)} algorithms."
            )

    def _select_components_dialog(self) -> None:
        if self.df_base is None or "Comp_Name" not in self.df_base.columns:
            QMessageBox.information(self, "Select Components", "Load a file first.")
            return
        components = sorted(
            self.df_base["Comp_Name"].dropna().astype(str).unique().tolist()
        )
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Components")
        dlg.setMinimumWidth(360)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Check the components to keep:"))
        lst = QListWidget()
        lst.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        preselected = set(self._selected_components or components)
        for comp in components:
            item = QListWidgetItem(comp)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if comp in preselected else Qt.CheckState.Unchecked
            )
            lst.addItem(item)
        layout.addWidget(lst)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            selected = [
                lst.item(i).text()
                for i in range(lst.count())
                if lst.item(i).checkState() == Qt.CheckState.Checked
            ]
            self._selected_components = selected
            QMessageBox.information(
                self, "Components Selected", f"Selected {len(selected)} components."
            )

    def _preview(self) -> None:
        try:
            if self.df_base is None:
                raise ValueError("Load a file first")
            df = self._apply_filters(self.df_base)
            if df.empty:
                raise ValueError("No data after filters")
            self.df_preview = df
            self._render(self.df_preview)
            self._set_status(f"Preview: {len(df)} rows ? {len(df.columns)} cols")
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", str(e))

    def _preview_with_iqr(self) -> None:
        try:
            if self.df_base is None:
                raise ValueError("Load a file first")
            meas = self.algo_combo.currentText()
            if not meas:
                raise ValueError("Select a measurement column first")
            df = self._apply_filters(self.df_base)
            if meas not in df.columns:
                raise ValueError(f"Column not in preview: {meas}")
            before = len(df)
            df2, removed = remove_outliers_iqr(df, meas)
            self.df_preview = df2
            self._render(self.df_preview)
            self._set_status(
                f"IQR on {meas}: removed {removed} (from {before} to {len(df2)})"
            )
        except Exception as e:
            QMessageBox.critical(self, "IQR Preview Error", str(e))

    def _save_dialog(self) -> None:
        try:
            if self.df_preview is None or self.df_preview.empty:
                raise ValueError("Nothing to save. Run Preview first.")
            dlg = QDialog(self)
            dlg.setWindowTitle("Save Parsed Data")
            dlg.setMinimumWidth(280)
            lay = QVBoxLayout(dlg)
            lay.addWidget(QLabel("Choose output format:"))
            btns = QDialogButtonBox()
            btn_csv = btns.addButton("CSV", QDialogButtonBox.ButtonRole.AcceptRole)
            btn_txt = btns.addButton("TXT", QDialogButtonBox.ButtonRole.AcceptRole)
            btns.addButton(QDialogButtonBox.StandardButton.Cancel)
            lay.addWidget(btns)

            chosen = {"fmt": None}

            def choose_csv() -> None:
                chosen["fmt"] = "csv"
                dlg.accept()

            def choose_txt() -> None:
                chosen["fmt"] = "txt"
                dlg.accept()

            btn_csv.clicked.connect(choose_csv)
            btn_txt.clicked.connect(choose_txt)
            btns.rejected.connect(dlg.reject)

            if dlg.exec() != QDialog.DialogCode.Accepted or not chosen["fmt"]:
                return

            prefix = self.prefix_edit.text().strip() or ""
            if chosen["fmt"] == "csv":
                suggested = prefix + "parsed_data.csv"
                dst, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save CSV As",
                    suggested,
                    "CSV Files (*.csv);;All Files (*)",
                )
                if not dst:
                    return
                self.df_preview.to_csv(dst, index=False)
                df_check = pd.read_csv(dst)
                self._render(df_check)
                self._set_status(
                    f"Saved: {dst} | {len(df_check)} rows ? {len(df_check.columns)} cols"
                )
                QMessageBox.information(self, "Saved", f"Saved CSV to: {dst}")
            else:
                suggested = prefix + "parsed_data.txt"
                dst, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save TXT As",
                    suggested,
                    "Text Files (*.txt);;All Files (*)",
                )
                if not dst:
                    return
                self.df_preview.to_csv(dst, index=False, sep="\t")
                self._set_status(
                    f"Saved: {dst} | {len(self.df_preview)} rows ? {len(self.df_preview.columns)} cols"
                )
                QMessageBox.information(self, "Saved", f"Saved TXT to: {dst}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _render(self, df) -> None:
        self.table.load_dataframe(df)

    def _set_status(self, msg: str) -> None:
        self.status_lbl.setText(msg)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Data Parser")
        self.resize(1200, 800)
        self.setCentralWidget(ParseTabWidget())


def run_gui() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    return app.exec()
