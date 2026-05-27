import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional
import shutil

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget, QFileDialog, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QGridLayout, QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QMessageBox, QScrollArea, QListWidget, QListWidgetItem, QSplitter, QDialog,
    QDialogButtonBox, QFrame, QRadioButton, QButtonGroup, QStatusBar, QStyleFactory,
)
from PySide6.QtGui import QPixmap

try:
    from app.theme import (
        STYLESHEET,
        VERDICT_DISPLAY,
        AppHeader,
        PageHeader,
        StickyActionBar,
        make_btn,
        make_field_label,
        make_heading,
        make_hint_label,
        make_separator,
        make_status_label,
        polish_widget,
        verdict_badge_style,
    )
except ImportError:
    from theme import (
        STYLESHEET,
        VERDICT_DISPLAY,
        AppHeader,
        PageHeader,
        StickyActionBar,
        make_btn,
        make_field_label,
        make_heading,
        make_hint_label,
        make_separator,
        make_status_label,
        polish_widget,
        verdict_badge_style,
    )

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
    apply_study_design,
)

from grr_tool.msa import (
    perform_anova_grr,
    create_anova_table,
    create_variance_summary_df,
    DesignMode,
)
from grr_tool.msa.gage_rr import ReproMode
from grr_tool.msa.xbar_r import perform_xbar_r
from grr_tool.msa.nested import perform_nested_grr
from grr_tool.msa.acceptance import build_gage_rr_acceptance
from gage_rr_analysis import (
    plot_components_of_variation,
    plot_algorithm_by_component,
    plot_s_chart_by_operator,
    plot_algo_by_operator,
    plot_merged_charts,
)

from grr_tool.msa.type1 import compute_type1_metrics
from grr_tool.msa.tables import create_type1_summary_df
from gage_rr_type1 import (
    plot_distribution_vs_tolerance,
    plot_individuals_chart,
    plot_moving_range_chart,
    plot_merged,
)


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


def set_global_status(widget: QWidget, msg: str) -> None:
    win = widget.window()
    if isinstance(win, MainWindow):
        win.set_status(msg)


def _add_form_row(grid: QGridLayout, row: int, label: str, widget: QWidget) -> int:
    grid.addWidget(make_field_label(label), row, 0)
    grid.addWidget(widget, row, 1)
    return row + 1


def _style_selection_dialog(dlg: QDialog, layout: Optional[QVBoxLayout] = None) -> None:
    dlg.setMinimumWidth(400)
    if layout is not None:
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)


# ---------------------------------------------------------------------------
# Image Panel
# ---------------------------------------------------------------------------

class ImagePanel(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("chartCard")
        self.current_path = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setObjectName("chartCardTitle")
        self.save_btn = make_btn("Save…", "compact")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_image)
        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(self.save_btn)

        self.image = QLabel()
        self.image.setObjectName("chartImageArea")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumHeight(200)
        self.image.setText("Chart will appear after analysis")

        layout.addLayout(header)
        layout.addWidget(self.image, 1)
        self.setMinimumWidth(340)

    def set_image(self, path: str):
        self.current_path = path if path and os.path.exists(path) else ""
        if not self.current_path:
            self.image.setText(
                "Chart will appear after analysis" if not path else f"Missing: {path}"
            )
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


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class VerdictPanel(QGroupBox):
    """Persistent AIAG-style acceptance summary."""

    def __init__(self, title: str = "Acceptance"):
        super().__init__(title)
        outer = QVBoxLayout(self)
        outer.setSpacing(10)

        header = QHBoxLayout()
        header.addStretch()
        self.badge = QLabel("—")
        self.badge.setObjectName("verdictBadge")
        self._set_badge("")
        header.addWidget(self.badge)
        outer.addLayout(header)

        self.metrics_grid = QGridLayout()
        self.metrics_grid.setSpacing(6)
        self.metrics_grid.setColumnStretch(1, 1)
        outer.addLayout(self.metrics_grid)

        self.warning_label = make_hint_label("")
        self.warning_label.hide()
        outer.addWidget(self.warning_label)
        self.clear()

    def _set_badge(self, verdict: str) -> None:
        key = (verdict or "").lower()
        display = VERDICT_DISPLAY.get(key, "—" if not verdict else verdict.title())
        self.badge.setText(display)
        self.badge.setStyleSheet(verdict_badge_style(key))

    def _clear_metrics(self) -> None:
        while self.metrics_grid.count():
            item = self.metrics_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _add_metric(self, row: int, label: str, value: str) -> None:
        lbl = QLabel(label)
        lbl.setProperty("cssClass", "metricLabel")
        polish_widget(lbl)
        val = QLabel(value)
        val.setProperty("cssClass", "metricValue")
        polish_widget(val)
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.metrics_grid.addWidget(lbl, row, 0)
        self.metrics_grid.addWidget(val, row, 1)

    def clear(self) -> None:
        self._set_badge("")
        self._clear_metrics()
        self._add_metric(0, "Status", "Run analysis to see verdicts.")
        self.warning_label.hide()

    def set_type2(self, results: Dict[str, Any]) -> None:
        acc = results.get("acceptance") or {}
        worst = acc.get("pct_tol_verdict") or acc.get("pct_sv_verdict", "")
        self._set_badge(worst)
        self._clear_metrics()
        row = 0
        self._add_metric(row, "NDC", f"{results.get('ndc', '—')} ({acc.get('ndc_verdict', '—')})")
        row += 1
        self._add_metric(
            row,
            "%SV GRR",
            f"{results['pct_study_var']['grr']:.2f}% ({acc.get('pct_sv_verdict', '—')})",
        )
        row += 1
        if results.get("tolerance") and results.get("pct_tolerance"):
            self._add_metric(
                row,
                "%GRR (Tol)",
                f"{results['pct_tolerance']['grr']:.2f}% ({acc.get('pct_tol_verdict', '—')})",
            )
        self.warning_label.hide()

    def set_type1(self, metrics: Dict[str, Any]) -> None:
        acc = metrics.get("acceptance") or {}
        overall = acc.get("overall", "")
        self._set_badge(overall)
        self._clear_metrics()
        cg = metrics.get("cg", float("nan"))
        cgk = metrics.get("cgk", float("nan"))
        rows = [
            ("Cg", f"{cg:.4g} ({acc.get('cg_verdict', '—')})"),
            ("Cgk", f"{cgk:.4g} ({acc.get('cgk_verdict', '—')})"),
            ("%Var (repeat)", f"{metrics.get('pct_var_repeatability', 0):.2f}%"),
            ("%Var (repeat+bias)", f"{metrics.get('pct_var_repeatability_bias', 0):.2f}%"),
            ("Bias significant", "yes" if acc.get("bias_significant") else "no"),
        ]
        for i, (label, value) in enumerate(rows):
            self._add_metric(i, label, value)
        if metrics.get("exploratory"):
            self.warning_label.setText(
                "Exploratory mode: set explicit tolerance and target for production acceptance."
            )
            self.warning_label.show()
        else:
            self.warning_label.hide()


def export_msa_type2(
    out_dir: str,
    prefix: str,
    results: Dict[str, Any],
    tables: Dict[str, Any],
) -> List[str]:
    """Write Type 2 CSV/JSON artifacts; returns paths written."""
    p = prefix or ""
    written: List[str] = []
    if tables.get("anova_table") is not None:
        path = os.path.join(out_dir, f"{p}anova_table.csv")
        tables["anova_table"].to_csv(path, index=False)
        written.append(path)
    if tables.get("variance_summary") is not None:
        path = os.path.join(out_dir, f"{p}variance_components.csv")
        tables["variance_summary"].to_csv(path, index=False)
        written.append(path)
    full = tables.get("full_anova")
    if full is not None:
        path = os.path.join(out_dir, f"{p}full_anova_table.csv")
        full.to_csv(path, index=False)
        written.append(path)
    payload = {
        "study_type": results.get("study_type", "gage_rr_type2"),
        "measurement": results.get("measurement"),
        "statistics": {
            "n_parts": results.get("n_parts"),
            "n_operators": results.get("n_operators"),
            "n_measurements": results.get("n_measurements"),
            "ndc": results.get("ndc"),
        },
        "variance_components": results.get("variance_components"),
        "std_dev": results.get("std_dev"),
        "study_var": results.get("study_var"),
        "pct_contribution": results.get("pct_contribution"),
        "pct_study_var": results.get("pct_study_var"),
        "pct_tolerance": results.get("pct_tolerance"),
        "acceptance": results.get("acceptance"),
        "tolerance": results.get("tolerance"),
    }
    path = os.path.join(out_dir, f"{p}grr_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    written.append(path)
    return written


def export_msa_type1(
    out_dir: str,
    prefix: str,
    measurement: str,
    component: str,
    metrics: Dict[str, Any],
    summary_df,
) -> List[str]:
    p = prefix or ""
    written: List[str] = []
    path = os.path.join(out_dir, f"{p}type1_summary.csv")
    summary_df.to_csv(path, index=False)
    written.append(path)
    path = os.path.join(out_dir, f"{p}type1_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "study_type": "type1",
                "measurement": measurement,
                "component": component,
                "metrics": metrics,
                "acceptance": metrics.get("acceptance"),
            },
            f,
            indent=2,
            default=_json_default,
        )
    written.append(path)
    return written


def _populate_column_combos(combo: QComboBox, columns: List[str], allow_empty: bool = True):
    combo.clear()
    if allow_empty:
        combo.addItem("")
    combo.addItems([str(c) for c in columns])


# ---------------------------------------------------------------------------
# Gage R&R (Type 2) Tab
# ---------------------------------------------------------------------------

class GageRRTab(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.tmpdir = tempfile.mkdtemp(prefix="grr_anova_")
        self._last_chart_paths: List[str] = []
        self._last_results: Optional[Dict[str, Any]] = None
        self._last_tables: Optional[Dict[str, Any]] = None
        self._last_run_df = None
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(PageHeader(
            "Gage R&R (Type 2)",
            "Configure study, run analysis, review acceptance and charts",
        ))

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_panel = QWidget()
        left_l = QVBoxLayout(left_panel)
        left_l.setSpacing(10)
        left_l.setContentsMargins(8, 8, 8, 8)

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
        fg.addWidget(make_field_label("Data File"), 0, 0)
        fg.addWidget(self.file_edit, 0, 1)
        fg.addWidget(browse_btn, 0, 2)
        fg.addWidget(make_field_label("Output Prefix"), 1, 0)
        fg.addWidget(self.prefix_edit, 1, 1, 1, 2)
        fg.addWidget(self.merge_chk, 2, 0, 1, 3)
        fg.setColumnStretch(1, 1)
        file_grp.setLayout(fg)

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
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(0.0, 1e9)
        self.tol_spin.setDecimals(6)
        self.tol_spin.setValue(0.0)
        self.tol_spin.setSpecialValueText("Off")
        self.design_combo = QComboBox()
        self.design_combo.addItems(["sequential", "comp_name", "columns"])
        self.design_combo.currentTextChanged.connect(self._on_design_changed)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["anova", "xbar_r", "nested"])
        self.method_combo.currentTextChanged.connect(
            lambda m: self.repro_combo.setEnabled(m == "anova")
        )
        self.repro_combo = QComboBox()
        self.repro_combo.addItems(["operator_only", "operator_plus_interaction"])
        self.part_col_combo = QComboBox()
        self.operator_col_combo = QComboBox()
        self.part_col_label = make_field_label("Part column")
        self.operator_col_label = make_field_label("Operator column")
        r = 0
        for label, widget in [
            ("Algorithm", self.algo_combo),
            ("Study Var (sv)", self.sv_spin),
            ("Alpha (av)", self.av_spin),
            ("Method", self.method_combo),
            ("Repro mode", self.repro_combo),
            ("Operators", self.op_spin),
            ("Tolerance", self.tol_spin),
            ("Design", self.design_combo),
        ]:
            r = _add_form_row(pg, r, label, widget)
        pg.addWidget(self.part_col_label, r, 0)
        pg.addWidget(self.part_col_combo, r, 1)
        r += 1
        pg.addWidget(self.operator_col_label, r, 0)
        pg.addWidget(self.operator_col_combo, r, 1)
        pg.setColumnStretch(1, 1)
        param_grp.setLayout(pg)
        self._on_design_changed(self.design_combo.currentText())

        filt_grp = QGroupBox("Filters")
        flg = QGridLayout()
        flg.setSpacing(8)
        self.include_edit = QLineEdit()
        self.include_edit.setPlaceholderText("Space or comma separated")
        self.exclude_edit = QLineEdit()
        self.exclude_edit.setPlaceholderText("Space or comma separated component names")
        self.rm_chk = QCheckBox("Remove outliers (IQR)")
        _add_form_row(flg, 0, "Include", self.include_edit)
        _add_form_row(flg, 1, "Exclude", self.exclude_edit)
        flg.addWidget(self.rm_chk, 2, 0, 1, 2)
        flg.setColumnStretch(1, 1)
        filt_grp.setLayout(flg)

        left_l.addWidget(file_grp)
        left_l.addWidget(param_grp)
        left_l.addWidget(filt_grp)
        left_l.addStretch()
        left_scroll.setWidget(left_panel)

        left_col = QWidget()
        left_col.setMinimumWidth(320)
        left_col.setMaximumWidth(480)
        left_col_l = QVBoxLayout(left_col)
        left_col_l.setContentsMargins(0, 0, 0, 0)
        left_col_l.setSpacing(0)
        left_col_l.addWidget(left_scroll, 1)

        action_bar = StickyActionBar()
        load_btn = make_btn("Load File", "primary")
        load_btn.clicked.connect(self._load_file)
        run_btn = make_btn("Run Gage R&R", "success")
        run_btn.clicked.connect(self._run)
        export_btn = QPushButton("Export Results…")
        export_btn.clicked.connect(self._export_results)
        action_bar.add_row(load_btn, run_btn)
        action_bar.add_widget(export_btn)
        left_col_l.addWidget(action_bar)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        results = QWidget()
        rl = QVBoxLayout(results)
        rl.setSpacing(10)
        rl.setContentsMargins(12, 8, 12, 12)

        charts_toolbar = QHBoxLayout()
        charts_toolbar.addWidget(make_heading("Charts"))
        charts_toolbar.addStretch()
        dl_btn = QPushButton("Download All Charts…")
        dl_btn.clicked.connect(self._download_all)
        charts_toolbar.addWidget(dl_btn)

        self.verdict_panel = VerdictPanel("Acceptance")
        rl.addWidget(self.verdict_panel)
        rl.addWidget(make_heading("Summary"))
        self.anova_table = TablePanel()
        self.anova_table.setMinimumHeight(120)
        rl.addWidget(self.anova_table)
        self.variance_table = TablePanel()
        self.variance_table.setMinimumHeight(100)
        rl.addWidget(make_heading("Variance Components"))
        rl.addWidget(self.variance_table)
        rl.addWidget(make_heading("Full ANOVA"))
        self.full_anova_table = TablePanel()
        self.full_anova_table.setMinimumHeight(120)
        rl.addWidget(self.full_anova_table)
        rl.addWidget(make_separator())
        rl.addLayout(charts_toolbar)

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

        results.setLayout(rl)
        right_scroll.setWidget(results)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_col)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1000])
        root.addWidget(splitter, 1)

    def _on_design_changed(self, design: str):
        columns_mode = design == "columns"
        self.op_spin.setEnabled(not columns_mode)
        self.part_col_label.setVisible(columns_mode)
        self.part_col_combo.setVisible(columns_mode)
        self.operator_col_label.setVisible(columns_mode)
        self.operator_col_combo.setVisible(columns_mode)
        self.repro_combo.setEnabled(self.method_combo.currentText() == "anova")

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
            excl = to_list_from_tokens(self.exclude_edit.text().split())
            df = apply_component_filters(df, include=inc, exclude=excl)
            measurement_cols = get_measurement_columns(df)
            self.df = df
            self.algo_combo.clear()
            self.algo_combo.addItems(measurement_cols)
            cols = list(df.columns)
            _populate_column_combos(self.part_col_combo, cols)
            _populate_column_combos(self.operator_col_combo, cols)
            msg = f"Loaded {len(measurement_cols)} measurement column(s)"
            set_global_status(self, msg)
            QMessageBox.information(self, "Loaded", f"Loaded file. {len(measurement_cols)} measurement columns found.")
        except Exception as e:
            set_global_status(self, "Load failed")
            QMessageBox.critical(self, "Load Error", str(e))

    def _build_run_df(self):
        design = DesignMode(self.design_combo.currentText())
        part_col = self.part_col_combo.currentText().strip() or None
        op_col = self.operator_col_combo.currentText().strip() or None
        if design == DesignMode.COLUMNS:
            if not part_col or not op_col:
                raise ValueError("Select Part column and Operator column for columns design mode.")
        return apply_study_design(
            self.df.copy(),
            mode=design,
            n_operators=self.op_spin.value(),
            part_col=part_col,
            operator_col=op_col,
        )

    def _adapt_alt_results(self, alt: Dict[str, Any], algo: str, tol: Optional[float], sv: float) -> Dict[str, Any]:
        vc = alt["variance_components"]
        var_total = vc["total"]
        pct_contrib = {
            k: (vc[k] / var_total * 100) if var_total > 0 else 0
            for k in ("repeatability", "reproducibility", "grr", "part")
        }
        results = {
            "study_type": alt.get("study_type", alt.get("method", "gage_rr")),
            "measurement": algo,
            "n_parts": 0,
            "n_operators": 0,
            "n_measurements": 0,
            "study_var_multiplier": sv,
            "tolerance": tol,
            "variance_components": vc,
            "std_dev": alt["std_dev"],
            "study_var": alt["study_var"],
            "pct_contribution": pct_contrib,
            "pct_study_var": alt["pct_study_var"],
            "pct_tolerance": alt.get("pct_tolerance"),
            "ndc": 0,
            "full_anova_table": None,
        }
        results["acceptance"] = build_gage_rr_acceptance(results)
        return results

    def _export_results(self):
        if not self._last_results or not self._last_tables:
            QMessageBox.information(self, "Export", "Run analysis first.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Export Results To", os.getcwd())
        if not out_dir:
            return
        prefix = self.prefix_edit.text().strip()
        paths = export_msa_type2(out_dir, prefix, self._last_results, self._last_tables)
        QMessageBox.information(self, "Exported", f"Saved {len(paths)} file(s) to:\n{out_dir}")

    def _run(self):
        try:
            if self.df is None:
                raise ValueError("Load a file first")
            algo = self.algo_combo.currentText()
            sv = self.sv_spin.value()
            prefix = self.prefix_edit.text().strip()
            merge = self.merge_chk.isChecked()
            remove_outliers = self.rm_chk.isChecked()
            method = self.method_combo.currentText()

            run_df = self._build_run_df()
            if remove_outliers:
                before = len(run_df)
                run_df, removed = remove_outliers_iqr(run_df, algo)
                QMessageBox.information(self, "Outlier Removal", f"Removed {removed} rows (from {before} to {len(run_df)}).")

            tol_in = self.tol_spin.value()
            tol = None if tol_in <= 0 else float(tol_in)
            repro_mode: ReproMode = self.repro_combo.currentText()  # type: ignore[assignment]

            if method == "xbar_r":
                alt = perform_xbar_r(run_df, algo, study_var=sv, tolerance=tol)
                if alt is None:
                    raise ValueError("Xbar-R analysis failed.")
                results = self._adapt_alt_results(alt, algo, tol, sv)
            elif method == "nested":
                alt = perform_nested_grr(run_df, algo, study_var=sv, tolerance=tol)
                if alt is None:
                    raise ValueError("Nested GRR analysis failed.")
                results = self._adapt_alt_results(alt, algo, tol, sv)
            else:
                results = perform_anova_grr(
                    run_df,
                    algo,
                    study_var=sv,
                    tolerance=tol,
                    alpha=self.av_spin.value(),
                    repro_mode=repro_mode,
                )
                if results is None:
                    raise ValueError("ANOVA analysis failed (no valid data).")

            table = create_anova_table(results)
            var_table = create_variance_summary_df(results)
            full_anova = results.get("full_anova_table")
            self.anova_table.load_dataframe(table)
            self.variance_table.load_dataframe(var_table)
            if full_anova is not None:
                self.full_anova_table.load_dataframe(full_anova)
            else:
                self.full_anova_table.setRowCount(0)
                self.full_anova_table.setColumnCount(0)
            self.verdict_panel.set_type2(results)
            self._last_results = results
            self._last_run_df = run_df
            self._last_tables = {
                "anova_table": table,
                "variance_summary": var_table,
                "full_anova": full_anova,
            }

            if method != "anova":
                QMessageBox.information(
                    self,
                    "Gage R&R Complete",
                    f"{method} analysis complete. Charts are available for ANOVA method only.",
                )
                self._last_chart_paths = []
            elif merge:
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
            set_global_status(self, f"Analysis complete — {algo}")
        except Exception as e:
            set_global_status(self, "Analysis failed")
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
        self._last_chart_paths: List[str] = []
        self._last_metrics: Optional[Dict[str, Any]] = None
        self._last_summary = None
        self._last_measurement = ""
        self._last_component = ""
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(PageHeader(
            "Type 1 Gage Study",
            "Single-operator repeatability and bias vs tolerance",
        ))

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_panel = QWidget()
        left_l = QVBoxLayout(left_panel)
        left_l.setSpacing(10)
        left_l.setContentsMargins(8, 8, 8, 8)

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
        fg.addWidget(make_field_label("Data File"), 0, 0)
        fg.addWidget(self.file_edit, 0, 1)
        fg.addWidget(browse_btn, 0, 2)
        fg.addWidget(make_field_label("Output Prefix"), 1, 0)
        fg.addWidget(self.prefix_edit, 1, 1, 1, 2)
        fg.addWidget(self.merge_chk, 2, 0, 1, 3)
        fg.setColumnStretch(1, 1)
        file_grp.setLayout(fg)

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
        self.target_spin = QDoubleSpinBox()
        self.target_spin.setRange(-1e9, 1e9)
        self.target_spin.setDecimals(6)
        self.target_spin.setValue(0.0)
        self.target_spin.setSpecialValueText("Auto")
        self.tf_spin = QDoubleSpinBox()
        self.tf_spin.setRange(0.0, 1e6)
        self.tf_spin.setDecimals(6)
        self.tf_spin.setSingleStep(0.1)
        self.tf_spin.setValue(1.0)
        self.require_ref_chk = QCheckBox("Require explicit tolerance & target")
        self.require_ref_chk.setToolTip(
            "When checked, tolerance and target must be set (not Auto) before running."
        )

        r = 0
        for label, widget in [
            ("Algorithm", self.algo_combo),
            ("Component", self.comp_combo),
            ("Measurements (n)", self.n_spin),
            ("Study Var (sv)", self.sv_spin),
            ("Alpha (av)", self.av_spin),
            ("Tolerance (tol)", self.tol_spin),
            ("Target (ref)", self.target_spin),
            ("Tol. Factor (tf)", self.tf_spin),
        ]:
            r = _add_form_row(pg, r, label, widget)
        pg.addWidget(self.require_ref_chk, r, 0, 1, 2)
        pg.setColumnStretch(1, 1)
        param_grp.setLayout(pg)

        filt_grp = QGroupBox("Filters")
        flg = QGridLayout()
        flg.setSpacing(8)
        self.include_edit = QLineEdit()
        self.include_edit.setPlaceholderText("Space or comma separated")
        self.exclude_edit = QLineEdit()
        self.exclude_edit.setPlaceholderText("Space or comma separated")
        self.rm_chk = QCheckBox("Remove outliers (IQR)")
        _add_form_row(flg, 0, "Include", self.include_edit)
        _add_form_row(flg, 1, "Exclude", self.exclude_edit)
        flg.addWidget(self.rm_chk, 2, 0, 1, 2)
        flg.setColumnStretch(1, 1)
        filt_grp.setLayout(flg)

        left_l.addWidget(file_grp)
        left_l.addWidget(param_grp)
        left_l.addWidget(filt_grp)
        left_l.addStretch()
        left_scroll.setWidget(left_panel)

        left_col = QWidget()
        left_col.setMinimumWidth(320)
        left_col.setMaximumWidth(480)
        left_col_l = QVBoxLayout(left_col)
        left_col_l.setContentsMargins(0, 0, 0, 0)
        left_col_l.setSpacing(0)
        left_col_l.addWidget(left_scroll, 1)

        action_bar = StickyActionBar()
        load_btn = make_btn("Load File", "primary")
        load_btn.clicked.connect(self._load_file)
        run_btn = make_btn("Run Type 1", "success")
        run_btn.clicked.connect(self._run)
        export_btn = QPushButton("Export Results…")
        export_btn.clicked.connect(self._export_results)
        action_bar.add_row(load_btn, run_btn)
        action_bar.add_widget(export_btn)
        left_col_l.addWidget(action_bar)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        results = QWidget()
        rl = QVBoxLayout(results)
        rl.setSpacing(10)
        rl.setContentsMargins(12, 8, 12, 12)

        charts_toolbar = QHBoxLayout()
        charts_toolbar.addWidget(make_heading("Charts"))
        charts_toolbar.addStretch()
        dl_btn = QPushButton("Download All Charts…")
        dl_btn.clicked.connect(self._download_all)
        charts_toolbar.addWidget(dl_btn)

        self.verdict_panel = VerdictPanel("Acceptance")
        rl.addWidget(self.verdict_panel)
        rl.addWidget(make_heading("Summary"))
        self.summary_table = TablePanel()
        self.summary_table.setMinimumHeight(140)
        rl.addWidget(self.summary_table)
        rl.addWidget(make_separator())
        rl.addLayout(charts_toolbar)

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

        results.setLayout(rl)
        right_scroll.setWidget(results)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_col)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1000])
        root.addWidget(splitter, 1)

    def _browse(self):
        path = select_file(self, "Select Input File")
        if path:
            self.file_edit.setText(path)

    def _export_results(self):
        if not self._last_metrics or self._last_summary is None:
            QMessageBox.information(self, "Export", "Run analysis first.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Export Results To", os.getcwd())
        if not out_dir:
            return
        prefix = self.prefix_edit.text().strip()
        paths = export_msa_type1(
            out_dir,
            prefix,
            self._last_measurement,
            self._last_component,
            self._last_metrics,
            self._last_summary,
        )
        QMessageBox.information(self, "Exported", f"Saved {len(paths)} file(s) to:\n{out_dir}")

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
            set_global_status(
                self,
                f"Loaded {len(measurement_cols)} measurement(s), {len(comps)} component(s)",
            )
            QMessageBox.information(self, "Loaded", f"Loaded file. {len(measurement_cols)} measurement columns; {len(comps)} components.")
        except Exception as e:
            set_global_status(self, "Load failed")
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
            require_ref = self.require_ref_chk.isChecked()

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
            target_in = self.target_spin.value()
            target_is_auto = target_in == 0
            target_val = None if target_is_auto else float(target_in)
            if require_ref:
                if tol_val is None:
                    raise ValueError("Require reference: set Tolerance (tol) to a value > 0.")
                if target_val is None:
                    raise ValueError("Require reference: set Target (ref) to a value (not Auto).")

            metrics = compute_type1_metrics(
                values,
                sv=sv,
                tol=tol_val,
                target=target_val,
                alpha=av,
                tolerance_factor=tf,
                require_reference=require_ref,
            )
            summary = create_type1_summary_df(metrics)
            self.summary_table.load_dataframe(summary)
            self.verdict_panel.set_type1(metrics)
            self._last_metrics = metrics
            self._last_summary = summary
            self._last_measurement = algo
            self._last_component = comp

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
            set_global_status(self, f"Type 1 complete — {comp} / {algo}")
        except Exception as e:
            set_global_status(self, "Analysis failed")
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

    def _preserve_source_columns(self) -> bool:
        return self.radio_original.isChecked()

    def _on_mode_changed(self):
        prepared = self.radio_prepared.isChecked()
        self.op_spin.setEnabled(prepared)
        self.op_spin.setToolTip(
            ""
            if prepared
            else "Not used in original-columns mode (no Operator / Part columns are added)."
        )
        # Do not load from disk here — only "Load File" should read the file.
        if self.df_base is not None:
            self.df_base = None
            self.df_preview = None
            self.algo_combo.clear()
            self.comp_text.clear()
            self._render(None)
            self._set_status("Loading mode changed — click Load File to reload data.")

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(PageHeader(
            "Data Preparation",
            "Load, filter, preview, and export study-ready data",
        ))

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_panel = QWidget()
        left_l = QVBoxLayout(left_panel)
        left_l.setSpacing(10)
        left_l.setContentsMargins(8, 8, 8, 8)

        src_grp = QGroupBox("Data source")
        sg = QGridLayout()
        sg.setSpacing(8)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select a data file (.txt or .csv)")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("Optional output filename prefix")
        sg.addWidget(make_field_label("Data File"), 0, 0)
        sg.addWidget(self.file_edit, 0, 1)
        sg.addWidget(browse_btn, 0, 2)
        sg.addWidget(make_field_label("Output Prefix"), 1, 0)
        sg.addWidget(self.prefix_edit, 1, 1, 1, 2)
        sg.setColumnStretch(1, 1)
        src_grp.setLayout(sg)

        mode_grp = QGroupBox("Loading mode")
        mv = QVBoxLayout()
        mv.setSpacing(6)
        self.radio_prepared = QRadioButton("Prepared for Gage R&R")
        self.radio_original = QRadioButton("Original columns only (no derived columns)")
        self.radio_prepared.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.radio_prepared, 0)
        self.mode_group.addButton(self.radio_original, 1)
        self.mode_group.buttonClicked.connect(lambda _: self._on_mode_changed())
        mode_hint = make_hint_label(
            "Prepared: drop auxiliary fields, add Component, and you can assign operators for export. "
            "Original: keep file columns; no Component, Operator, or Part columns."
        )
        mv.addWidget(self.radio_prepared)
        mv.addWidget(self.radio_original)
        mv.addWidget(mode_hint)
        mode_grp.setLayout(mv)

        meas_grp = QGroupBox("Measurement setup")
        mg = QGridLayout()
        mg.setSpacing(8)
        self.op_spin = QSpinBox()
        self.op_spin.setRange(1, 10)
        self.op_spin.setValue(3)
        self.design_combo = QComboBox()
        self.design_combo.addItems(["sequential", "comp_name"])
        self.algo_combo = QComboBox()
        self.algo_combo.setPlaceholderText("Load a file to populate")
        _add_form_row(mg, 0, "Operators", self.op_spin)
        _add_form_row(mg, 1, "Design", self.design_combo)
        _add_form_row(mg, 2, "Measurement", self.algo_combo)
        design_hint = make_hint_label(
            "Use comp_name when each Comp_Name is a distinct part in the Gage R&R study."
        )
        mg.addWidget(design_hint, 3, 0, 1, 2)
        mg.setColumnStretch(1, 1)
        meas_grp.setLayout(mg)

        sel_grp = QGroupBox("Selection")
        sv = QVBoxLayout()
        sv.setSpacing(8)
        r1 = QHBoxLayout()
        select_comp_btn = QPushButton("Components…")
        select_comp_btn.clicked.connect(self._select_components_dialog)
        select_algo_btn = QPushButton("Algorithms…")
        select_algo_btn.clicked.connect(self._select_algorithms_dialog)
        r1.addWidget(select_comp_btn)
        r1.addWidget(select_algo_btn)
        disp_comp_btn = QPushButton("Display component list")
        disp_comp_btn.clicked.connect(self._display_components)
        sv.addLayout(r1)
        sv.addWidget(disp_comp_btn)
        sel_grp.setLayout(sv)

        left_l.addWidget(src_grp)
        left_l.addWidget(mode_grp)
        left_l.addWidget(meas_grp)
        left_l.addWidget(sel_grp)
        left_l.addStretch()
        left_scroll.setWidget(left_panel)

        left_col = QWidget()
        left_col.setMinimumWidth(340)
        left_col.setMaximumWidth(440)
        left_col_l = QVBoxLayout(left_col)
        left_col_l.setContentsMargins(0, 0, 0, 0)
        left_col_l.setSpacing(0)
        left_col_l.addWidget(left_scroll, 1)

        action_bar = StickyActionBar()
        load_btn = make_btn("Load File", "primary")
        load_btn.clicked.connect(self._load_file)
        preview_btn = make_btn("Preview", "primary")
        preview_btn.clicked.connect(self._preview)
        preview_iqr_btn = QPushButton("Preview + IQR")
        preview_iqr_btn.clicked.connect(self._preview_with_iqr)
        save_btn = make_btn("Save…", "success")
        save_btn.clicked.connect(self._save_dialog)
        action_bar.add_row(load_btn, preview_btn, preview_iqr_btn)
        action_bar.add_widget(save_btn)
        left_col_l.addWidget(action_bar)

        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setSpacing(8)
        right_l.setContentsMargins(12, 8, 12, 12)
        self.status_lbl = make_status_label("Ready — load a file to begin")
        right_l.addWidget(self.status_lbl)
        right_l.addWidget(make_heading("Components"))
        self.comp_text = QTextEdit()
        self.comp_text.setReadOnly(True)
        self.comp_text.setMinimumHeight(120)
        self.comp_text.setPlaceholderText("Component names appear here after “Display component list”.")
        right_l.addWidget(self.comp_text)
        right_l.addWidget(make_heading("Data preview"))
        self.table = TablePanel()
        right_l.addWidget(self.table, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_col)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 980])
        root.addWidget(splitter, 1)

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
            preserve = self._preserve_source_columns()
            df = load_and_clean_data(path, preserve_source_columns=preserve)
            self.df_base = df.reset_index(drop=True)
            meas_cols = get_measurement_columns(self.df_base)
            self.algo_combo.clear()
            self.algo_combo.addItems(meas_cols)
            self._set_status(
                f"Loaded: {len(self.df_base)} rows, {len(self.df_base.columns)} cols — "
                f"{len(meas_cols)} numeric measurements"
            )
            if preserve:
                self.df_preview = self.df_base.copy()
            else:
                self.df_preview = apply_study_design(
                    self.df_base.copy(),
                    mode=DesignMode(self.design_combo.currentText()),
                    n_operators=self.op_spin.value(),
                )
            self._render(self.df_preview)
        except Exception as e:
            self.df_base = None
            self.df_preview = None
            self._render(None)
            QMessageBox.critical(self, "Load Error", str(e))
            self._set_status("Load failed — fix the file or mode and try again.")

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
        if not self._preserve_source_columns():
            df = apply_study_design(
                df,
                mode=DesignMode(self.design_combo.currentText()),
                n_operators=self.op_spin.value(),
            )
        return df

    def _select_algorithms_dialog(self):
        if self.df_base is None:
            QMessageBox.information(self, "Select Algorithms", "Load a file first.")
            return
        meas_cols = get_measurement_columns(self.df_base)
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Algorithms")
        layout = QVBoxLayout(dlg)
        _style_selection_dialog(dlg, layout)
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
        layout = QVBoxLayout(dlg)
        _style_selection_dialog(dlg, layout)
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
            lay = QVBoxLayout(dlg)
            _style_selection_dialog(dlg, lay)
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
        set_global_status(self, msg)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

_NAV_ITEMS = (
    ("Gage R&R (Type 2)", "Type 2 study with ANOVA, Xbar-R, or nested methods"),
    ("Type 1 Gage", "Single-operator repeatability and bias analysis"),
    ("Parsing", "Load, filter, and export study-ready measurement data"),
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gage R&R Desktop")
        self.resize(1400, 900)

        central = QWidget()
        central_l = QVBoxLayout(central)
        central_l.setContentsMargins(0, 0, 0, 0)
        central_l.setSpacing(0)
        central_l.addWidget(AppHeader())

        body = QWidget()
        body_l = QHBoxLayout(body)
        body_l.setContentsMargins(0, 0, 0, 0)
        body_l.setSpacing(0)

        nav_col = QWidget()
        nav_col.setFixedWidth(232)
        nav_col.setObjectName("navSidebar")
        nav_l = QVBoxLayout(nav_col)
        nav_l.setContentsMargins(12, 16, 8, 12)
        nav_l.setSpacing(8)
        nav_title = QLabel("WORKFLOW")
        nav_title.setObjectName("navSectionLabel")
        nav_l.addWidget(nav_title)
        nav = QListWidget()
        nav.setObjectName("navSidebarList")
        for text, tip in _NAV_ITEMS:
            item = QListWidgetItem(text, nav)
            item.setToolTip(tip)
        nav.setCurrentRow(0)
        nav_l.addWidget(nav, 1)

        self.stack = QStackedWidget()
        self.stack.addWidget(GageRRTab())
        self.stack.addWidget(Type1Tab())
        self.stack.addWidget(ParseTab())
        nav.currentRowChanged.connect(self.stack.setCurrentIndex)

        body_l.addWidget(nav_col)
        body_l.addWidget(self.stack, 1)
        central_l.addWidget(body, 1)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Ready")

    def set_status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)


def main():
    app = QApplication(sys.argv)
    for style_name in ("Windows", "Fusion"):
        if style_name in QStyleFactory.keys():
            app.setStyle(style_name)
            break
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
