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
    QDialogButtonBox
)
from PySide6.QtGui import QPixmap

# Reuse backend functions from existing scripts
# Ensure project root is on sys.path so we can import sibling modules
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


class ImagePanel(QWidget):
    def __init__(self, title: str):
        super().__init__()
        self.current_path = ""
        self.setLayout(QVBoxLayout())
        self.label = QLabel(title)
        self.label.setAlignment(Qt.AlignCenter)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.save_btn = QPushButton("Save Chart…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_image)
        self.layout().addWidget(self.label)
        self.layout().addWidget(self.image)
        self.layout().addWidget(self.save_btn)

        # Reasonable minimum size for preview
        self.setMinimumWidth(400)

    def set_image(self, path: str):
        self.current_path = path if os.path.exists(path) else ""
        if not self.current_path:
            self.image.setText(f"Missing: {path}")
            self.save_btn.setEnabled(False)
            return

        pix = QPixmap(self.current_path)
        # Scale down to fit screen nicely
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        max_w = 800
        max_h = 500
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
            # Try copying the original file for fidelity
            shutil.copyfile(self.current_path, dst)
        except Exception:
            # Fallback: save from pixmap
            pix = self.image.pixmap()
            if pix is not None:
                pix.save(dst)


class TablePanel(QTableWidget):
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


class AnovaTab(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.tmpdir = tempfile.mkdtemp(prefix="grr_anova_")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # Controls
        controls = QGridLayout()
        self.file_edit = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse)

        self.algo_combo = QComboBox()
        self.sv_spin = QDoubleSpinBox(); self.sv_spin.setRange(0.001, 9999); self.sv_spin.setValue(6.0)
        self.av_spin = QDoubleSpinBox(); self.av_spin.setRange(0.0001, 1.0); self.av_spin.setSingleStep(0.005); self.av_spin.setValue(0.025)
        self.op_spin = QSpinBox(); self.op_spin.setRange(1, 10); self.op_spin.setValue(3)
        self.prefix_edit = QLineEdit()
        self.merge_chk = QCheckBox("Merge charts")

        self.exclude_edit = QLineEdit(); self.exclude_edit.setPlaceholderText("Exclude components (space/comma)")
        self.rm_chk = QCheckBox("Remove outliers (IQR)")

        load_btn = QPushButton("Load File")
        load_btn.clicked.connect(self._load_file)
        run_btn = QPushButton("Run ANOVA")
        run_btn.clicked.connect(self._run)

        r = 0
        controls.addWidget(QLabel("File"), r, 0); controls.addWidget(self.file_edit, r, 1); controls.addWidget(browse_btn, r, 2); r += 1
        controls.addWidget(QLabel("Algorithm"), r, 0); controls.addWidget(self.algo_combo, r, 1); r += 1
        controls.addWidget(QLabel("Study Var (sv)"), r, 0); controls.addWidget(self.sv_spin, r, 1); r += 1
        controls.addWidget(QLabel("Alpha (av)"), r, 0); controls.addWidget(self.av_spin, r, 1); r += 1
        controls.addWidget(QLabel("Operators"), r, 0); controls.addWidget(self.op_spin, r, 1); r += 1
        controls.addWidget(QLabel("Output Prefix"), r, 0); controls.addWidget(self.prefix_edit, r, 1); controls.addWidget(self.merge_chk, r, 2); r += 1
        controls.addWidget(QLabel("Exclude"), r, 0); controls.addWidget(self.exclude_edit, r, 1); r += 1
        controls.addWidget(load_btn, r, 0); controls.addWidget(run_btn, r, 1); r += 1
        controls.addWidget(self.rm_chk, r, 1); r += 1

        layout.addLayout(controls)

        # Outputs (scrollable container)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        out_container = QWidget(); out_layout = QVBoxLayout()

        self.anova_table = TablePanel()
        out_layout.addWidget(QLabel("ANOVA Table"))
        out_layout.addWidget(self.anova_table)

        self.img_cov = ImagePanel("Components of Variation")
        self.img_alg = ImagePanel("Algorithm by Component")
        self.img_s = ImagePanel("S Chart by Operator")
        self.img_op = ImagePanel("Algorithm Type by Operator")

        img_row = QHBoxLayout(); img_row.addWidget(self.img_cov); img_row.addWidget(self.img_alg)
        out_layout.addLayout(img_row)
        img_row2 = QHBoxLayout(); img_row2.addWidget(self.img_s); img_row2.addWidget(self.img_op)
        out_layout.addLayout(img_row2)

        dl_all = QPushButton("Download All Charts…")
        dl_all.clicked.connect(self._download_all)
        out_layout.addWidget(dl_all)

        out_container.setLayout(out_layout)
        scroll.setWidget(out_container)
        layout.addWidget(scroll)

        self.setLayout(layout)
        self._last_chart_paths = []

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
            # apply exclude
            excl = to_list_from_tokens(self.exclude_edit.text().split())
            if excl:
                df = apply_component_filters(df, exclude=excl)
            # Identify measurement columns
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

            # Optionally remove outliers for the chosen measurement
            run_df = assign_operators_sequential(self.df.copy(), n_operators=self.op_spin.value())
            if remove_outliers:
                before = len(run_df)
                run_df, removed = remove_outliers_iqr(run_df, algo)
                QMessageBox.information(self, "Outlier Removal", f"Removed {removed} rows (from {before} to {len(run_df)}).")

            results = perform_anova_grr(run_df, algo, study_var=sv)
            table = create_anova_table(results)
            self.anova_table.load_dataframe(table)

            # Save plots to tmp then display
            if merge:
                merged_path = os.path.join(self.tmpdir, f"{prefix}merged_analysis.png")
                plot_merged_charts(run_df, results, algo, merged_path)
                # Load into the first image slot
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


class Type1Tab(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.tmpdir = tempfile.mkdtemp(prefix="grr_type1_")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        controls = QGridLayout()
        self.file_edit = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse)

        self.algo_combo = QComboBox()
        self.comp_combo = QComboBox()
        self.n_spin = QSpinBox(); self.n_spin.setRange(0, 100000); self.n_spin.setValue(30)
        self.sv_spin = QDoubleSpinBox(); self.sv_spin.setRange(0.001, 9999); self.sv_spin.setValue(6.0)
        self.av_spin = QDoubleSpinBox(); self.av_spin.setRange(0.0001, 1.0); self.av_spin.setSingleStep(0.005); self.av_spin.setValue(0.25)
        self.tol_spin = QDoubleSpinBox(); self.tol_spin.setRange(0.0, 1e9); self.tol_spin.setDecimals(6); self.tol_spin.setSingleStep(0.1); self.tol_spin.setValue(0.0); self.tol_spin.setSpecialValueText("Auto")
        self.tf_spin = QDoubleSpinBox(); self.tf_spin.setRange(0.0, 1e6); self.tf_spin.setDecimals(6); self.tf_spin.setSingleStep(0.1); self.tf_spin.setValue(1.0)
        self.prefix_edit = QLineEdit()
        self.merge_chk = QCheckBox("Merge charts")
        self.rm_chk = QCheckBox("Remove outliers (IQR)")

        self.include_edit = QLineEdit(); self.include_edit.setPlaceholderText("Include components (space/comma)")
        self.exclude_edit = QLineEdit(); self.exclude_edit.setPlaceholderText("Exclude components (space/comma)")

        load_btn = QPushButton("Load File")
        load_btn.clicked.connect(self._load_file)
        run_btn = QPushButton("Run Type 1")
        run_btn.clicked.connect(self._run)

        r = 0
        controls.addWidget(QLabel("File"), r, 0); controls.addWidget(self.file_edit, r, 1); controls.addWidget(browse_btn, r, 2); r += 1
        controls.addWidget(QLabel("Algorithm"), r, 0); controls.addWidget(self.algo_combo, r, 1); r += 1
        controls.addWidget(QLabel("Component"), r, 0); controls.addWidget(self.comp_combo, r, 1); r += 1
        controls.addWidget(QLabel("Measurements (n)"), r, 0); controls.addWidget(self.n_spin, r, 1); r += 1
        controls.addWidget(QLabel("Study Var (sv)"), r, 0); controls.addWidget(self.sv_spin, r, 1); r += 1
        controls.addWidget(QLabel("Alpha (av)"), r, 0); controls.addWidget(self.av_spin, r, 1); r += 1
        controls.addWidget(QLabel("Tolerance (tol)"), r, 0); controls.addWidget(self.tol_spin, r, 1); r += 1
        controls.addWidget(QLabel("Tol. Factor (tf)"), r, 0); controls.addWidget(self.tf_spin, r, 1); r += 1
        controls.addWidget(QLabel("Output Prefix"), r, 0); controls.addWidget(self.prefix_edit, r, 1); controls.addWidget(self.merge_chk, r, 2); r += 1
        controls.addWidget(QLabel("Include"), r, 0); controls.addWidget(self.include_edit, r, 1); r += 1
        controls.addWidget(QLabel("Exclude"), r, 0); controls.addWidget(self.exclude_edit, r, 1); r += 1
        controls.addWidget(load_btn, r, 0); controls.addWidget(run_btn, r, 1); r += 1
        controls.addWidget(self.rm_chk, r, 1); r += 1

        layout.addLayout(controls)

        # Scrollable outputs
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        out_container = QWidget(); out_layout = QVBoxLayout()

        self.summary_table = TablePanel()
        out_layout.addWidget(QLabel("Type 1 Summary"))
        out_layout.addWidget(self.summary_table)

        self.img_dist = ImagePanel("Distribution vs Tolerance")
        self.img_ind = ImagePanel("Individuals Chart")
        self.img_mr = ImagePanel("Moving Range Chart")

        img_row = QHBoxLayout(); img_row.addWidget(self.img_dist); img_row.addWidget(self.img_ind)
        out_layout.addLayout(img_row)
        img_row2 = QHBoxLayout(); img_row2.addWidget(self.img_mr)
        out_layout.addLayout(img_row2)

        dl_all = QPushButton("Download All Charts…")
        dl_all.clicked.connect(self._download_all)
        out_layout.addWidget(dl_all)

        out_container.setLayout(out_layout)
        scroll.setWidget(out_container)
        layout.addWidget(scroll)

        self.setLayout(layout)
        self._last_chart_paths = []

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
            # include/exclude
            inc = to_list_from_tokens(self.include_edit.text().split())
            exc = to_list_from_tokens(self.exclude_edit.text().split())
            df = apply_component_filters(df, include=inc, exclude=exc)
            self.df = df
            # Populate algorithms and components
            measurement_cols = get_measurement_columns(df)
            comps = sorted(df['Comp_Name'].unique().tolist())
            self.algo_combo.clear(); self.algo_combo.addItems(measurement_cols)
            self.comp_combo.clear(); self.comp_combo.addItems(comps)
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

            # Plot to temp then display
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


class ParseTab(QWidget):
    def __init__(self):
        super().__init__()
        self.df_base = None
        self.df_preview = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        controls = QGridLayout()

        # File and prefix
        self.file_edit = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse)
        self.prefix_edit = QLineEdit()
        self.op_spin = QSpinBox(); self.op_spin.setRange(1, 10); self.op_spin.setValue(3)
        self.keep_raw_chk = QCheckBox("Keep all columns (raw)")

        # Component selection via dialog (no manual include/exclude text inputs)

        # Measurement dropdown removed from main UI; selection handled in IQR dialog
        # self.algo_combo = QComboBox()

        # Actions (organized by flow)
        load_btn = QPushButton("Load File")
        load_btn.clicked.connect(self._load_file)
        select_comp_btn = QPushButton("Select Components…")
        select_comp_btn.clicked.connect(self._select_components_dialog)
        select_algo_btn = QPushButton("Select Algorithms…")
        select_algo_btn.clicked.connect(self._select_algorithms_dialog)
        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self._preview)
        disp_comp_btn = QPushButton("Display Components")
        disp_comp_btn.clicked.connect(self._display_components)
        preview_iqr_btn = QPushButton("Preview with IQR")
        preview_iqr_btn.clicked.connect(self._preview_with_iqr)
        save_btn = QPushButton("Save…")
        save_btn.clicked.connect(self._save_dialog)

        r = 0
        controls.addWidget(QLabel("File"), r, 0); controls.addWidget(self.file_edit, r, 1); controls.addWidget(browse_btn, r, 2); r += 1
        controls.addWidget(QLabel("Output Prefix"), r, 0); controls.addWidget(self.prefix_edit, r, 1); r += 1
        controls.addWidget(QLabel("Operators"), r, 0); controls.addWidget(self.op_spin, r, 1); r += 1
        controls.addWidget(self.keep_raw_chk, r, 1); r += 1
        # Component include/exclude inputs removed; use Select Components… instead
        # Only file/prefix rows in controls grid
        layout.addLayout(controls)

        # Components preview area
        self.comp_text = QTextEdit()
        self.comp_text.setReadOnly(True)
        self.comp_text.setMaximumHeight(90)

        # Status line
        self.status_lbl = QLabel("")

        # Table preview
        self.table = TablePanel()

        # Build a row: left (controls already added above + preview widgets) and right (buttons column)
        top_row = QHBoxLayout()

        left_container = QWidget()
        left_v = QVBoxLayout()
        # controls already added to main layout; keep left content focused on preview/status
        left_v.addWidget(self.comp_text)
        left_v.addWidget(self.status_lbl)
        left_v.addWidget(self.table)
        left_container.setLayout(left_v)
        top_row.addWidget(left_container, 1)

        # Right buttons column with uniform size, Save at bottom
        buttons_col = QVBoxLayout()
        btns = [load_btn, select_comp_btn, select_algo_btn, preview_btn, preview_iqr_btn, disp_comp_btn]
        try:
            uniform_w = max(b.sizeHint().width() for b in btns)
            uniform_h = max(b.sizeHint().height() for b in btns)
        except Exception:
            uniform_w, uniform_h = 140, 28
        for b in btns:
            b.setFixedWidth(uniform_w)
            b.setFixedHeight(uniform_h)
            buttons_col.addWidget(b)
        buttons_col.addStretch(1)
        # Save button at bottom-right
        save_btn.setFixedWidth(uniform_w)
        save_btn.setFixedHeight(uniform_h)
        buttons_col.addWidget(save_btn)
        top_row.addLayout(buttons_col)

        layout.addLayout(top_row)

        self.setLayout(layout)

    def _browse(self):
        path = select_file(self, "Select Input File")
        if path:
            self.file_edit.setText(path)

    def _load_file(self):
        try:
            path = self.file_edit.text().strip()
            if not path:
                raise ValueError("No file selected")
            # Reuse type1 loader (supports .txt and .csv)
            df = load_and_clean_data(path, keep_all_columns=self.keep_raw_chk.isChecked())
            self.df_base = df.reset_index(drop=True)
            # Populate measurement list (numeric columns except component labels)
            meas_cols = get_measurement_columns(self.df_base)
            # algo selection handled in dialog; keep count in status only
            self._set_status(f"Loaded: {len(self.df_base)} rows, {len(self.df_base.columns)} cols; {len(meas_cols)} numeric measurements")
            # Initial preview without filters
            # Assign operators unless raw mode is enabled
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
        # Apply selected components filter (if any)
        if 'Comp_Name' in df.columns and hasattr(self, '_selected_components') and self._selected_components:
            df = df[df['Comp_Name'].isin(set(self._selected_components))].reset_index(drop=True)
        # Apply algorithm selection (selected measurement columns)
        if hasattr(self, '_selected_algorithms') and self._selected_algorithms:
            keep_cols = []
            # Always keep identifier columns if present
            for base_col in ['Comp_Name', 'Component']:
                if base_col in df.columns:
                    keep_cols.append(base_col)
            keep_cols += [c for c in self._selected_algorithms if c in df.columns]
            if keep_cols:
                df = df[keep_cols]
        # Assign operators after filtering, unless raw mode is enabled
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
        layout = QVBoxLayout(dlg)
        lst = QListWidget()
        lst.setSelectionMode(QListWidget.NoSelection)
        # Pre-check prior selection or default to all
        preselected = set(getattr(self, '_selected_algorithms', meas_cols))
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
            selected = []
            for i in range(lst.count()):
                it = lst.item(i)
                if it.checkState() == Qt.Checked:
                    selected.append(it.text())
            self._selected_algorithms = selected
            QMessageBox.information(self, "Algorithms Selected", f"Selected {len(selected)} algorithms.")

    def _select_components_dialog(self):
        if self.df_base is None or 'Comp_Name' not in self.df_base.columns:
            QMessageBox.information(self, "Select Components", "Load a file first.")
            return
        components = sorted(self.df_base['Comp_Name'].dropna().astype(str).unique().tolist())

        dlg = QDialog(self)
        dlg.setWindowTitle("Select Components")
        layout = QVBoxLayout(dlg)
        lst = QListWidget()
        lst.setSelectionMode(QListWidget.NoSelection)
        preselected = set(getattr(self, '_selected_components', components))
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
            selected = []
            for i in range(lst.count()):
                it = lst.item(i)
                if it.checkState() == Qt.Checked:
                    selected.append(it.text())
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
                raise ValueError("Select a measurement column for outlier preview")
            df = self._apply_filters(self.df_base)
            if meas not in df.columns:
                raise ValueError(f"Column not in preview: {meas}")
            before = len(df)
            df2, removed = remove_outliers_iqr(df, meas)
            self.df_preview = df2
            self._render(self.df_preview)
            self._set_status(f"IQR preview on {meas}: removed {removed} (from {before} to {len(df2)})")
        except Exception as e:
            QMessageBox.critical(self, "IQR Preview Error", str(e))

    def _save_csv(self):
        try:
            if self.df_preview is None or self.df_preview.empty:
                raise ValueError("Nothing to save. Run Preview first.")
            suggested = (self.prefix_edit.text().strip() or "") + "parsed_data.csv"
            dst, _ = QFileDialog.getSaveFileName(self, "Save CSV As", suggested, "CSV Files (*.csv);;All Files (*)")
            if not dst:
                return
            self.df_preview.to_csv(dst, index=False)
            # Re-read to confirm and render
            import pandas as pd
            df_check = pd.read_csv(dst)
            self._render(df_check)
            self._set_status(f"Saved: {dst} | {len(df_check)} rows × {len(df_check.columns)} cols")
            QMessageBox.information(self, "Saved", f"Saved CSV to: {dst}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _save_dialog(self):
        try:
            if self.df_preview is None or self.df_preview.empty:
                raise ValueError("Nothing to save. Run Preview first.")
            # Ask user for format
            dlg = QDialog(self)
            dlg.setWindowTitle("Save Parsed Data")
            lay = QVBoxLayout(dlg)
            lay.addWidget(QLabel("Choose format:"))
            btns = QDialogButtonBox()
            btn_csv = btns.addButton("CSV", QDialogButtonBox.AcceptRole)
            btn_txt = btns.addButton("TXT", QDialogButtonBox.AcceptRole)
            btn_cancel = btns.addButton(QDialogButtonBox.Cancel)
            lay.addWidget(btns)

            chosen = {"fmt": None}
            def choose_csv():
                chosen["fmt"] = "csv"; dlg.accept()
            def choose_txt():
                chosen["fmt"] = "txt"; dlg.accept()
            btn_csv.clicked.connect(choose_csv)
            btn_txt.clicked.connect(choose_txt)
            btn_cancel.clicked.connect(dlg.reject)

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
                # TXT as tab-separated
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gage R&R Desktop (PySide6)")
        self.resize(1400, 900)
        tabs = QTabWidget()
        tabs.addTab(AnovaTab(), "ANOVA (Type I)")
        tabs.addTab(Type1Tab(), "Type 1 Gage")
        tabs.addTab(ParseTab(), "Parsing")
        self.setCentralWidget(tabs)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()


