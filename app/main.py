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
    QMessageBox, QScrollArea
)
from PySide6.QtGui import QPixmap

# Reuse backend functions from existing scripts
# Ensure project root is on sys.path so we can import sibling modules
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gage_rr_analysis import load_and_clean_data as load_anova_data, perform_anova_grr, create_anova_table
from gage_rr_analysis import plot_components_of_variation, plot_algorithm_by_component, plot_s_chart_by_operator, plot_algo_by_operator, plot_merged_charts

from gage_rr_type1 import load_and_clean_data as load_type1_data, compute_type1_metrics, create_type1_summary_df
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
            df = load_anova_data(path)
            # apply exclude
            excl = to_list_from_tokens(self.exclude_edit.text().split())
            if excl:
                df = df[~df['Comp_Name'].isin(set(excl))].reset_index(drop=True)
            # Identify measurement columns
            measurement_cols = [c for c in df.columns if c not in ['Comp_Name','Operator','Part','Component','Part_ID','Measurement_Order'] and str(df[c].dtype) in ['float64','int64']]
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
            run_df = self.df
            if remove_outliers:
                from gage_rr_analysis import remove_outliers_iqr
                before = len(run_df)
                run_df, removed = remove_outliers_iqr(run_df, algo)
                QMessageBox.information(self, "Outlier Removal", f"Removed {removed} rows (from {before} to {len(run_df)}).")

            results = perform_anova_grr(run_df, algo, study_var=sv)
            table = create_anova_table(results)
            self.anova_table.load_dataframe(table)

            # Save plots to tmp then display
            if merge:
                merged_path = os.path.join(self.tmpdir, f"{prefix}merged_analysis.png")
                plot_merged_charts(self.df, results, algo, merged_path)
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
        self.limit_spin = QSpinBox(); self.limit_spin.setRange(0, 100000); self.limit_spin.setValue(30)
        self.sv_spin = QDoubleSpinBox(); self.sv_spin.setRange(0.001, 9999); self.sv_spin.setValue(6.0)
        self.av_spin = QDoubleSpinBox(); self.av_spin.setRange(0.0001, 1.0); self.av_spin.setSingleStep(0.005); self.av_spin.setValue(0.25)
        self.tol_spin = QDoubleSpinBox(); self.tol_spin.setRange(0.0, 1e9); self.tol_spin.setDecimals(6); self.tol_spin.setSingleStep(0.1); self.tol_spin.setValue(0.0); self.tol_spin.setSpecialValueText("Auto")
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
        controls.addWidget(QLabel("Limit"), r, 0); controls.addWidget(self.limit_spin, r, 1); r += 1
        controls.addWidget(QLabel("Study Var (sv)"), r, 0); controls.addWidget(self.sv_spin, r, 1); r += 1
        controls.addWidget(QLabel("Alpha (av)"), r, 0); controls.addWidget(self.av_spin, r, 1); r += 1
        controls.addWidget(QLabel("Tolerance (tol)"), r, 0); controls.addWidget(self.tol_spin, r, 1); r += 1
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
            df = load_type1_data(path)
            # include/exclude
            inc = set(to_list_from_tokens(self.include_edit.text().split()))
            exc = set(to_list_from_tokens(self.exclude_edit.text().split()))
            if inc:
                df = df[df['Comp_Name'].isin(inc)].reset_index(drop=True)
            if exc:
                df = df[~df['Comp_Name'].isin(exc)].reset_index(drop=True)
            self.df = df
            # Populate algorithms and components
            measurement_cols = [c for c in df.columns if c not in ['Comp_Name','Component'] and str(df[c].dtype) in ['float64','int64']]
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
            limit = self.limit_spin.value()
            prefix = self.prefix_edit.text().strip()
            merge = self.merge_chk.isChecked()
            remove_outliers = self.rm_chk.isChecked()

            subset = self.df[self.df['Comp_Name'] == comp].copy()
            values = subset[algo].dropna()
            if limit and limit > 0:
                values = values.iloc[:limit]
            if remove_outliers:
                from gage_rr_type1 import remove_outliers_iqr_series
                before_n = len(values)
                values, removed = remove_outliers_iqr_series(values)
                QMessageBox.information(self, "Outlier Removal", f"Removed {removed} readings (from {before_n} to {len(values)}).")
            if len(values) < 2:
                raise ValueError("Not enough readings after filtering/limit")

            tol_val = None if tol_in <= 0 else float(tol_in)
            metrics = compute_type1_metrics(values, sv=sv, tol=tol_val, target=None, alpha=av)
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gage R&R Desktop (PySide6)")
        self.resize(1400, 900)
        tabs = QTabWidget()
        tabs.addTab(AnovaTab(), "ANOVA (Type I)")
        tabs.addTab(Type1Tab(), "Type 1 Gage")
        self.setCentralWidget(tabs)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()


