import os
import sys
import tempfile
from typing import List
import shutil

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QFileDialog, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QGridLayout, QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QMessageBox, QScrollArea, QListWidget, QListWidgetItem, QSplitter, QDialog,
    QDialogButtonBox, QSizePolicy, QFrame
)
from PySide6.QtGui import QPixmap

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_parser import (
    load_and_clean_data,
    apply_component_filters,
    remove_outliers_iqr,
    remove_outliers_iqr_series,
    get_measurement_columns,
    get_components_preview,
    assign_operators_sequential
)

from gage_rr_analysis import perform_anova_grr, create_anova_table
from gage_rr_analysis import plot_components_of_variation, plot_algorithm_by_component, plot_s_chart_by_operator, plot_algo_by_operator, plot_merged_charts

from gage_rr_type1 import compute_type1_metrics, create_type1_summary_df
from gage_rr_type1 import plot_distribution_vs_tolerance, plot_individuals_chart, plot_moving_range_chart, plot_merged


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def select_file(parent: QWidget, caption: str = "Select Data File") -> str:
    path, _ = QFileDialog.getOpenFileName(parent, caption, os.getcwd(), "Data Files (*.txt *.csv);;All Files (*)")
    return path or ""


def to_list_from_tokens(tokens: List[str]) -> List[str]:
    out: List[str] = []
    for tok in tokens:
        for t in str(tok).split(','):
            t2 = t.strip()
            if t2:
                out.append(t2)
    return out


def _make_btn(text: str, css_class: str = "") -> QPushButton:
    btn = QPushButton(text)
    if css_class:
        btn.setProperty("cssClass", css_class)
    return btn


def _make_heading(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("cssClass", "heading")
    return lbl


def _make_status_label(text: str = "") -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("cssClass", "status")
    return lbl


def _make_separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setProperty("cssClass", "separator")
    line.setFixedHeight(1)
    return line


# ---------------------------------------------------------------------------
# Image Panel
# ---------------------------------------------------------------------------

class ImagePanel(QWidget):
    def __init__(self, title: str):
        super().__init__()
        self.current_path = ""
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(
            "font-weight: 600; color: #cdd6f4; font-size: 12px; padding: 4px;"
        )

        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setStyleSheet(
            "background-color: #313244; border: 1px solid #45475a; "
            "border-radius: 6px; padding: 8px; color: #585b70;"
        )
        self.image.setMinimumHeight(200)
        self.image.setText("No chart generated")

        self.save_btn = QPushButton("Save Chart…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_image)

        layout.addWidget(self.title_label)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.save_btn)
        self.setLayout(layout)
        self.setMinimumWidth(380)

    def set_image(self, path: str):
        self.current_path = path if path and os.path.exists(path) else ""
        if not self.current_path:
            self.image.setText("No chart generated" if not path else f"Missing: {path}")
            self.image.setPixmap(QPixmap())
            self.save_btn.setEnabled(False)
            return

        pix = QPixmap(self.current_path)
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        max_w, max_h = 800, 500
        if avail:
            max_w = min(max_w, max(400, avail.width() // 2))
            max_h = min(max_h, max(300, avail.height() // 3))
        scaled = pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image.setPixmap(scaled)
        self.save_btn.setEnabled(True)

    def _save_image(self):
        if not self.current_path:
            return
        base = os.path.basename(self.current_path)
        dst, _ = QFileDialog.getSaveFileName(self, "Save Chart As", base, "PNG Image (*.png);;All Files (*)")
        if not dst:
            return
        try:
            shutil.copyfile(self.current_path, dst)
        except Exception:
            pix = self.image.pixmap()
            if pix is not None:
                pix.save(dst)


# ---------------------------------------------------------------------------
# Table Panel
# ---------------------------------------------------------------------------

class TablePanel(QTableWidget):
    def __init__(self):
        super().__init__()
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSelectionMode(QTableWidget.SingleSelection)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)

    def load_dataframe(self, df):
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


# ---------------------------------------------------------------------------
# ANOVA Tab
# ---------------------------------------------------------------------------

class AnovaTab(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.tmpdir = tempfile.mkdtemp(prefix="grr_anova_")
        self._last_chart_paths = []
        self._init_ui()

    def _init_ui(self):
        main = QVBoxLayout()
        main.setSpacing(10)
        main.setContentsMargins(16, 12, 16, 12)

        # -- File & Output card --
        file_grp = QGroupBox("File && Output")
        fg = QGridLayout()
        fg.setSpacing(8)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select a data file (.txt or .csv)")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("Optional output filename prefix")
        self.merge_chk = QCheckBox("Merge all charts into one image")
        fg.addWidget(QLabel("Data File"), 0, 0)
        fg.addWidget(self.file_edit, 0, 1)
        fg.addWidget(browse_btn, 0, 2)
        fg.addWidget(QLabel("Output Prefix"), 1, 0)
        fg.addWidget(self.prefix_edit, 1, 1)
        fg.addWidget(self.merge_chk, 1, 2)
        fg.setColumnStretch(1, 1)
        file_grp.setLayout(fg)

        # -- Parameters card --
        param_grp = QGroupBox("Parameters")
        pg = QGridLayout()
        pg.setSpacing(8)
        self.algo_combo = QComboBox()
        self.sv_spin = QDoubleSpinBox()
        self.sv_spin.setRange(0.001, 9999)
        self.sv_spin.setValue(6.0)
        self.av_spin = QDoubleSpinBox()
        self.av_spin.setRange(0.0001, 1.0)
        self.av_spin.setSingleStep(0.005)
        self.av_spin.setValue(0.025)
        self.op_spin = QSpinBox()
        self.op_spin.setRange(1, 10)
        self.op_spin.setValue(3)
        pg.addWidget(QLabel("Algorithm"), 0, 0)
        pg.addWidget(self.algo_combo, 0, 1)
        pg.addWidget(QLabel("Study Var (sv)"), 1, 0)
        pg.addWidget(self.sv_spin, 1, 1)
        pg.addWidget(QLabel("Alpha (av)"), 2, 0)
        pg.addWidget(self.av_spin, 2, 1)
        pg.addWidget(QLabel("Operators"), 3, 0)
        pg.addWidget(self.op_spin, 3, 1)
        pg.setColumnStretch(1, 1)
        param_grp.setLayout(pg)

        # -- Filters card --
        filt_grp = QGroupBox("Filters")
        flg = QGridLayout()
        flg.setSpacing(8)
        self.exclude_edit = QLineEdit()
        self.exclude_edit.setPlaceholderText("Space or comma separated component names")
        self.rm_chk = QCheckBox("Remove outliers (IQR)")
        flg.addWidget(QLabel("Exclude"), 0, 0)
        flg.addWidget(self.exclude_edit, 0, 1)
        flg.addWidget(self.rm_chk, 1, 0, 1, 2)
        flg.setColumnStretch(1, 1)
        filt_grp.setLayout(flg)

        # Layout: file card full-width, params + filters side-by-side
        cards = QHBoxLayout()
        cards.setSpacing(10)
        cards.addWidget(param_grp, 1)
        cards.addWidget(filt_grp, 1)

        main.addWidget(file_grp)
        main.addLayout(cards)

        # -- Action buttons --
        actions = QHBoxLayout()
        actions.setSpacing(10)
        load_btn = _make_btn("Load File", "primary")
        load_btn.clicked.connect(self._load_file)
        run_btn = _make_btn("Run ANOVA", "success")
        run_btn.clicked.connect(self._run)
        actions.addStretch()
        actions.addWidget(load_btn)
        actions.addWidget(run_btn)
        actions.addStretch()
        main.addLayout(actions)

        # -- Results (scrollable) --
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        results = QWidget()
        rl = QVBoxLayout()
        rl.setSpacing(10)
        rl.setContentsMargins(4, 8, 4, 4)

        rl.addWidget(_make_heading("ANOVA Results"))
        self.anova_table = TablePanel()
        self.anova_table.setMinimumHeight(140)
        rl.addWidget(self.anova_table)

        rl.addWidget(_make_separator())
        rl.addWidget(_make_heading("Charts"))

        self.img_cov = ImagePanel("Components of Variation")
        self.img_alg = ImagePanel("Algorithm by Component")
        self.img_s = ImagePanel("S Chart by Operator")
        self.img_op = ImagePanel("Algorithm Type by Operator")
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(self.img_cov)
        row1.addWidget(self.img_alg)
        rl.addLayout(row1)
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(self.img_s)
        row2.addWidget(self.img_op)
        rl.addLayout(row2)

        dl_btn = QPushButton("Download All Charts…")
        dl_btn.clicked.connect(self._download_all)
        rl.addWidget(dl_btn, alignment=Qt.AlignRight)

        results.setLayout(rl)
        scroll.setWidget(results)
        main.addWidget(scroll, 1)
        self.setLayout(main)

    # -- Slots (logic unchanged) --

    def _browse(self):
        path = select_file(self, "Select Input File")
        if path:
            self.file_edit.setText(path)

    def _load_file(self):
        try:
            path = self.file_edit.text().strip()
            if not path:
                raise ValueError("No file selected")
            df = load_and_clean_data(path)
            excl = to_list_from_tokens(self.exclude_edit.text().split())
            if excl:
                df = apply_component_filters(df, exclude=excl)
            measurement_cols = get_measurement_columns(df)
            self.df = df
            self.algo_combo.clear()
            self.algo_combo.addItems(measurement_cols)
            QMessageBox.information(self, "Loaded", f"Loaded file. {len(measurement_cols)} measurement columns found.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _run(self):
        try:
            if self.df is None:
                raise ValueError("Load a file first")
            algo = self.algo_combo.currentText()
            sv = self.sv_spin.value()
            prefix = self.prefix_edit.text().strip()
            merge = self.merge_chk.isChecked()
            remove_outliers = self.rm_chk.isChecked()

            run_df = assign_operators_sequential(self.df.copy(), n_operators=self.op_spin.value())
            if remove_outliers:
                before = len(run_df)
                run_df, removed = remove_outliers_iqr(run_df, algo)
                QMessageBox.information(self, "Outlier Removal", f"Removed {removed} rows (from {before} to {len(run_df)}).")

            results = perform_anova_grr(run_df, algo, study_var=sv)
            table = create_anova_table(results)
            self.anova_table.load_dataframe(table)

            if merge:
                merged_path = os.path.join(self.tmpdir, f"{prefix}merged_analysis.png")
                plot_merged_charts(run_df, results, algo, merged_path)
                self.img_cov.set_image(merged_path)
                self.img_alg.set_image("")
                self.img_s.set_image("")
                self.img_op.set_image("")
                self._last_chart_paths = [merged_path]
            else:
                cov = os.path.join(self.tmpdir, f"{prefix}components_of_variation.png")
                alg = os.path.join(self.tmpdir, f"{prefix}algorithm_by_component.png")
                sch = os.path.join(self.tmpdir, f"{prefix}s_chart_by_operator.png")
                aop = os.path.join(self.tmpdir, f"{prefix}algo_by_operator.png")
                plot_components_of_variation(results, cov)
                plot_algorithm_by_component(run_df, algo, alg)
                plot_s_chart_by_operator(run_df, algo, sch)
                plot_algo_by_operator(run_df, algo, aop)
                self.img_cov.set_image(cov)
                self.img_alg.set_image(alg)
                self.img_s.set_image(sch)
                self.img_op.set_image(aop)
                self._last_chart_paths = [cov, alg, sch, aop]
        except Exception as e:
            QMessageBox.critical(self, "Run Error", str(e))

    def _download_all(self):
        if not self._last_chart_paths:
            QMessageBox.information(self, "No Charts", "Run analysis to generate charts first.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select Folder to Save Charts", os.getcwd())
        if not out_dir:
            return
        copied = 0
        for p in self._last_chart_paths:
            if p and os.path.exists(p):
                try:
                    shutil.copy(p, os.path.join(out_dir, os.path.basename(p)))
                    copied += 1
                except Exception:
                    pass
        QMessageBox.information(self, "Saved", f"Saved {copied} chart(s) to {out_dir}.")


# ---------------------------------------------------------------------------
# Type 1 Tab
# ---------------------------------------------------------------------------

class Type1Tab(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.tmpdir = tempfile.mkdtemp(prefix="grr_type1_")
        self._last_chart_paths = []
        self._init_ui()

    def _init_ui(self):
        main = QVBoxLayout()
        main.setSpacing(10)
        main.setContentsMargins(16, 12, 16, 12)

        # -- File & Output card --
        file_grp = QGroupBox("File && Output")
        fg = QGridLayout()
        fg.setSpacing(8)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select a data file (.txt or .csv)")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("Optional output filename prefix")
        self.merge_chk = QCheckBox("Merge all charts into one image")
        fg.addWidget(QLabel("Data File"), 0, 0)
        fg.addWidget(self.file_edit, 0, 1)
        fg.addWidget(browse_btn, 0, 2)
        fg.addWidget(QLabel("Output Prefix"), 1, 0)
        fg.addWidget(self.prefix_edit, 1, 1)
        fg.addWidget(self.merge_chk, 1, 2)
        fg.setColumnStretch(1, 1)
        file_grp.setLayout(fg)

        # -- Parameters card --
        param_grp = QGroupBox("Parameters")
        pg = QGridLayout()
        pg.setSpacing(8)
        self.algo_combo = QComboBox()
        self.comp_combo = QComboBox()
        self.n_spin = QSpinBox()
        self.n_spin.setRange(0, 100000)
        self.n_spin.setValue(30)
        self.sv_spin = QDoubleSpinBox()
        self.sv_spin.setRange(0.001, 9999)
        self.sv_spin.setValue(6.0)
        self.av_spin = QDoubleSpinBox()
        self.av_spin.setRange(0.0001, 1.0)
        self.av_spin.setSingleStep(0.005)
        self.av_spin.setValue(0.25)
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(0.0, 1e9)
        self.tol_spin.setDecimals(6)
        self.tol_spin.setSingleStep(0.1)
        self.tol_spin.setValue(0.0)
        self.tol_spin.setSpecialValueText("Auto")
        self.tf_spin = QDoubleSpinBox()
        self.tf_spin.setRange(0.0, 1e6)
        self.tf_spin.setDecimals(6)
        self.tf_spin.setSingleStep(0.1)
        self.tf_spin.setValue(1.0)

        r = 0
        for label, widget in [
            ("Algorithm", self.algo_combo),
            ("Component", self.comp_combo),
            ("Measurements (n)", self.n_spin),
            ("Study Var (sv)", self.sv_spin),
            ("Alpha (av)", self.av_spin),
            ("Tolerance (tol)", self.tol_spin),
            ("Tol. Factor (tf)", self.tf_spin),
        ]:
            pg.addWidget(QLabel(label), r, 0)
            pg.addWidget(widget, r, 1)
            r += 1
        pg.setColumnStretch(1, 1)
        param_grp.setLayout(pg)

        # -- Filters card --
        filt_grp = QGroupBox("Filters")
        flg = QGridLayout()
        flg.setSpacing(8)
        self.include_edit = QLineEdit()
        self.include_edit.setPlaceholderText("Space or comma separated")
        self.exclude_edit = QLineEdit()
        self.exclude_edit.setPlaceholderText("Space or comma separated")
        self.rm_chk = QCheckBox("Remove outliers (IQR)")
        flg.addWidget(QLabel("Include"), 0, 0)
        flg.addWidget(self.include_edit, 0, 1)
        flg.addWidget(QLabel("Exclude"), 1, 0)
        flg.addWidget(self.exclude_edit, 1, 1)
        flg.addWidget(self.rm_chk, 2, 0, 1, 2)
        flg.setColumnStretch(1, 1)
        filt_grp.setLayout(flg)

        cards = QHBoxLayout()
        cards.setSpacing(10)
        cards.addWidget(param_grp, 2)
        cards.addWidget(filt_grp, 1)

        main.addWidget(file_grp)
        main.addLayout(cards)

        # -- Action buttons --
        actions = QHBoxLayout()
        actions.setSpacing(10)
        load_btn = _make_btn("Load File", "primary")
        load_btn.clicked.connect(self._load_file)
        run_btn = _make_btn("Run Type 1", "success")
        run_btn.clicked.connect(self._run)
        actions.addStretch()
        actions.addWidget(load_btn)
        actions.addWidget(run_btn)
        actions.addStretch()
        main.addLayout(actions)

        # -- Results (scrollable) --
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        results = QWidget()
        rl = QVBoxLayout()
        rl.setSpacing(10)
        rl.setContentsMargins(4, 8, 4, 4)

        rl.addWidget(_make_heading("Type 1 Summary"))
        self.summary_table = TablePanel()
        self.summary_table.setMinimumHeight(140)
        rl.addWidget(self.summary_table)

        rl.addWidget(_make_separator())
        rl.addWidget(_make_heading("Charts"))

        self.img_dist = ImagePanel("Distribution vs Tolerance")
        self.img_ind = ImagePanel("Individuals Chart")
        self.img_mr = ImagePanel("Moving Range Chart")
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(self.img_dist)
        row1.addWidget(self.img_ind)
        rl.addLayout(row1)
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(self.img_mr)
        row2.addStretch(1)
        rl.addLayout(row2)

        dl_btn = QPushButton("Download All Charts…")
        dl_btn.clicked.connect(self._download_all)
        rl.addWidget(dl_btn, alignment=Qt.AlignRight)

        results.setLayout(rl)
        scroll.setWidget(results)
        main.addWidget(scroll, 1)
        self.setLayout(main)

    # -- Slots (logic unchanged) --

    def _browse(self):
        path = select_file(self, "Select Input File")
        if path:
            self.file_edit.setText(path)

    def _load_file(self):
        try:
            path = self.file_edit.text().strip()
            if not path:
                raise ValueError("No file selected")
            df = load_and_clean_data(path)
            inc = to_list_from_tokens(self.include_edit.text().split())
            exc = to_list_from_tokens(self.exclude_edit.text().split())
            df = apply_component_filters(df, include=inc, exclude=exc)
            self.df = df
            measurement_cols = get_measurement_columns(df)
            comps = sorted(df['Comp_Name'].unique().tolist())
            self.algo_combo.clear()
            self.algo_combo.addItems(measurement_cols)
            self.comp_combo.clear()
            self.comp_combo.addItems(comps)
            QMessageBox.information(self, "Loaded", f"Loaded file. {len(measurement_cols)} measurement columns; {len(comps)} components.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _run(self):
        try:
            if self.df is None:
                raise ValueError("Load a file first")
            algo = self.algo_combo.currentText()
            comp = self.comp_combo.currentText()
            sv = self.sv_spin.value()
            av = self.av_spin.value()
            tol_in = self.tol_spin.value()
            tf = self.tf_spin.value()
            n = self.n_spin.value()
            prefix = self.prefix_edit.text().strip()
            merge = self.merge_chk.isChecked()
            remove_outliers = self.rm_chk.isChecked()

            subset = self.df[self.df['Comp_Name'] == comp].copy()
            values = subset[algo].dropna()
            if n and n > 0:
                values = values.iloc[:n]
            if remove_outliers:
                before_n = len(values)
                values, removed = remove_outliers_iqr_series(values)
                QMessageBox.information(self, "Outlier Removal", f"Removed {removed} readings (from {before_n} to {len(values)}).")
            if len(values) < 2:
                raise ValueError("Not enough readings after filtering/limit")

            tol_val = None if tol_in <= 0 else float(tol_in)
            metrics = compute_type1_metrics(values, sv=sv, tol=tol_val, target=None, alpha=av, tolerance_factor=tf)
            summary = create_type1_summary_df(metrics)
            self.summary_table.load_dataframe(summary)

            if merge:
                merged = os.path.join(self.tmpdir, f"{prefix}type1_merged.png")
                plot_merged(values, metrics, algo, merged)
                self.img_dist.set_image(merged)
                self.img_ind.set_image("")
                self.img_mr.set_image("")
                self._last_chart_paths = [merged]
            else:
                dist = os.path.join(self.tmpdir, f"{prefix}distribution_vs_tolerance.png")
                ind = os.path.join(self.tmpdir, f"{prefix}individuals_chart.png")
                mr = os.path.join(self.tmpdir, f"{prefix}moving_range_chart.png")
                plot_distribution_vs_tolerance(values, metrics, dist)
                plot_individuals_chart(values, metrics, ind)
                plot_moving_range_chart(values, mr)
                self.img_dist.set_image(dist)
                self.img_ind.set_image(ind)
                self.img_mr.set_image(mr)
                self._last_chart_paths = [dist, ind, mr]
        except Exception as e:
            QMessageBox.critical(self, "Run Error", str(e))

    def _download_all(self):
        if not self._last_chart_paths:
            QMessageBox.information(self, "No Charts", "Run analysis to generate charts first.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select Folder to Save Charts", os.getcwd())
        if not out_dir:
            return
        copied = 0
        for p in self._last_chart_paths:
            if p and os.path.exists(p):
                try:
                    shutil.copy(p, os.path.join(out_dir, os.path.basename(p)))
                    copied += 1
                except Exception:
                    pass
        QMessageBox.information(self, "Saved", f"Saved {copied} chart(s) to {out_dir}.")


# ---------------------------------------------------------------------------
# Parse Tab
# ---------------------------------------------------------------------------

class ParseTab(QWidget):
    def __init__(self):
        super().__init__()
        self.df_base = None
        self.df_preview = None
        self._selected_components = []
        self._selected_algorithms = []
        self._init_ui()

    def _init_ui(self):
        main = QVBoxLayout()
        main.setSpacing(10)
        main.setContentsMargins(16, 12, 16, 12)

        # -- File & Settings card --
        file_grp = QGroupBox("File && Settings")
        fg = QGridLayout()
        fg.setSpacing(8)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select a data file (.txt or .csv)")
        browse_btn = QPushButton("Browse…")
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

        # -- Action buttons in two groups --
        actions = QHBoxLayout()
        actions.setSpacing(8)

        load_btn = _make_btn("Load File", "primary")
        load_btn.clicked.connect(self._load_file)
        select_comp_btn = QPushButton("Select Components…")
        select_comp_btn.clicked.connect(self._select_components_dialog)
        select_algo_btn = QPushButton("Select Algorithms…")
        select_algo_btn.clicked.connect(self._select_algorithms_dialog)
        preview_btn = _make_btn("Preview", "primary")
        preview_btn.clicked.connect(self._preview)
        preview_iqr_btn = QPushButton("Preview with IQR")
        preview_iqr_btn.clicked.connect(self._preview_with_iqr)
        disp_comp_btn = QPushButton("Display Components")
        disp_comp_btn.clicked.connect(self._display_components)
        save_btn = _make_btn("Save…", "success")
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

        # -- Status --
        self.status_lbl = _make_status_label("Ready — load a file to begin")
        main.addWidget(self.status_lbl)

        # -- Components preview --
        self.comp_text = QTextEdit()
        self.comp_text.setReadOnly(True)
        self.comp_text.setMaximumHeight(80)
        self.comp_text.setPlaceholderText("Component names will appear here after clicking Display Components")
        main.addWidget(self.comp_text)

        # -- Data table --
        self.table = TablePanel()
        main.addWidget(self.table, 1)

        self.setLayout(main)

    # -- Slots --

    def _browse(self):
        path = select_file(self, "Select Input File")
        if path:
            self.file_edit.setText(path)

    def _load_file(self):
        try:
            path = self.file_edit.text().strip()
            if not path:
                raise ValueError("No file selected")
            df = load_and_clean_data(path, keep_all_columns=self.keep_raw_chk.isChecked())
            self.df_base = df.reset_index(drop=True)
            meas_cols = get_measurement_columns(self.df_base)
            self.algo_combo.clear()
            self.algo_combo.addItems(meas_cols)
            self._set_status(f"Loaded: {len(self.df_base)} rows, {len(self.df_base.columns)} cols — {len(meas_cols)} numeric measurements")
            if self.keep_raw_chk.isChecked():
                self.df_preview = self.df_base.copy()
            else:
                self.df_preview = assign_operators_sequential(self.df_base.copy(), n_operators=self.op_spin.value())
            self._render(self.df_preview)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _display_components(self):
        if self.df_base is None or 'Comp_Name' not in self.df_base.columns:
            self.comp_text.setPlainText("(No components to display)")
            return
        preview = get_components_preview(self.df_base)
        self.comp_text.setPlainText(preview)

    def _apply_filters(self, base_df):
        df = base_df.copy()
        if 'Comp_Name' in df.columns and self._selected_components:
            df = df[df['Comp_Name'].isin(set(self._selected_components))].reset_index(drop=True)
        if self._selected_algorithms:
            keep_cols = []
            for base_col in ['Comp_Name', 'Box_Name', 'Component']:
                if base_col in df.columns:
                    keep_cols.append(base_col)
            keep_cols += [c for c in self._selected_algorithms if c in df.columns]
            if keep_cols:
                df = df[keep_cols]
        if not self.keep_raw_chk.isChecked():
            df = assign_operators_sequential(df, n_operators=self.op_spin.value())
        return df

    def _select_algorithms_dialog(self):
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
        lst.setSelectionMode(QListWidget.NoSelection)
        preselected = set(self._selected_algorithms or meas_cols)
        for col in meas_cols:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if col in preselected else Qt.Unchecked)
            lst.addItem(item)
        layout.addWidget(lst)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec() == QDialog.Accepted:
            selected = [lst.item(i).text() for i in range(lst.count()) if lst.item(i).checkState() == Qt.Checked]
            self._selected_algorithms = selected
            QMessageBox.information(self, "Algorithms Selected", f"Selected {len(selected)} algorithms.")

    def _select_components_dialog(self):
        if self.df_base is None or 'Comp_Name' not in self.df_base.columns:
            QMessageBox.information(self, "Select Components", "Load a file first.")
            return
        components = sorted(self.df_base['Comp_Name'].dropna().astype(str).unique().tolist())
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Components")
        dlg.setMinimumWidth(360)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Check the components to keep:"))
        lst = QListWidget()
        lst.setSelectionMode(QListWidget.NoSelection)
        preselected = set(self._selected_components or components)
        for comp in components:
            item = QListWidgetItem(comp)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if comp in preselected else Qt.Unchecked)
            lst.addItem(item)
        layout.addWidget(lst)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec() == QDialog.Accepted:
            selected = [lst.item(i).text() for i in range(lst.count()) if lst.item(i).checkState() == Qt.Checked]
            self._selected_components = selected
            QMessageBox.information(self, "Components Selected", f"Selected {len(selected)} components.")

    def _preview(self):
        try:
            if self.df_base is None:
                raise ValueError("Load a file first")
            df = self._apply_filters(self.df_base)
            if df.empty:
                raise ValueError("No data after filters")
            self.df_preview = df
            self._render(self.df_preview)
            self._set_status(f"Preview: {len(df)} rows × {len(df.columns)} cols")
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", str(e))

    def _preview_with_iqr(self):
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
            self._set_status(f"IQR on {meas}: removed {removed} (from {before} to {len(df2)})")
        except Exception as e:
            QMessageBox.critical(self, "IQR Preview Error", str(e))

    def _save_dialog(self):
        try:
            if self.df_preview is None or self.df_preview.empty:
                raise ValueError("Nothing to save. Run Preview first.")
            dlg = QDialog(self)
            dlg.setWindowTitle("Save Parsed Data")
            dlg.setMinimumWidth(280)
            lay = QVBoxLayout(dlg)
            lay.addWidget(QLabel("Choose output format:"))
            btns = QDialogButtonBox()
            btn_csv = btns.addButton("CSV", QDialogButtonBox.AcceptRole)
            btn_txt = btns.addButton("TXT", QDialogButtonBox.AcceptRole)
            btns.addButton(QDialogButtonBox.Cancel)
            lay.addWidget(btns)

            chosen = {"fmt": None}

            def choose_csv():
                chosen["fmt"] = "csv"
                dlg.accept()

            def choose_txt():
                chosen["fmt"] = "txt"
                dlg.accept()

            btn_csv.clicked.connect(choose_csv)
            btn_txt.clicked.connect(choose_txt)
            btns.rejected.connect(dlg.reject)

            if dlg.exec() != QDialog.Accepted or not chosen["fmt"]:
                return

            prefix = self.prefix_edit.text().strip() or ""
            if chosen["fmt"] == "csv":
                suggested = prefix + "parsed_data.csv"
                dst, _ = QFileDialog.getSaveFileName(self, "Save CSV As", suggested, "CSV Files (*.csv);;All Files (*)")
                if not dst:
                    return
                self.df_preview.to_csv(dst, index=False)
                import pandas as pd
                df_check = pd.read_csv(dst)
                self._render(df_check)
                self._set_status(f"Saved: {dst} | {len(df_check)} rows × {len(df_check.columns)} cols")
                QMessageBox.information(self, "Saved", f"Saved CSV to: {dst}")
            else:
                suggested = prefix + "parsed_data.txt"
                dst, _ = QFileDialog.getSaveFileName(self, "Save TXT As", suggested, "Text Files (*.txt);;All Files (*)")
                if not dst:
                    return
                self.df_preview.to_csv(dst, index=False, sep='\t')
                self._set_status(f"Saved: {dst} | {len(self.df_preview)} rows × {len(self.df_preview.columns)} cols")
                QMessageBox.information(self, "Saved", f"Saved TXT to: {dst}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _render(self, df):
        self.table.load_dataframe(df)

    def _set_status(self, msg: str):
        self.status_lbl.setText(msg)


def _make_vsep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.VLine)
    sep.setStyleSheet("color: #45475a;")
    sep.setFixedWidth(2)
    return sep


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gage R&R Desktop")
        self.resize(1400, 900)

        tabs = QTabWidget()
        tabs.addTab(AnovaTab(), "  ANOVA  ")
        tabs.addTab(Type1Tab(), "  Type 1 Gage  ")
        tabs.addTab(ParseTab(), "  Parsing  ")
        self.setCentralWidget(tabs)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
