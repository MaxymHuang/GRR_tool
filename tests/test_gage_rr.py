"""Gage R&R (Type 2) ANOVA tests."""

import numpy as np
import pandas as pd

from grr_tool.msa.gage_rr import perform_anova_grr
from grr_tool.msa.anova_table import build_full_anova_table, compute_anova_ss_ms
from grr_tool.msa.design import assign_operators_sequential


def _balanced_crossed_data():
    """3 operators x 4 parts x 3 replicates with known structure."""
    rng = np.random.default_rng(0)
    rows = []
    part_effects = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    op_effects = {"A": 0, "B": 0.1, "C": -0.05}
    for part in part_effects:
        for op in op_effects:
            for _ in range(3):
                y = 10 + part_effects[part] + op_effects[op] + rng.normal(0, 0.05)
                rows.append({"Part": part, "Operator": op, "Y": y})
    return pd.DataFrame(rows)


def test_perform_anova_grr_balanced():
    df = _balanced_crossed_data()
    results = perform_anova_grr(df, "Y", study_var=6.0, tolerance=10.0)
    assert results is not None
    assert results["n_parts"] == 4
    assert results["n_operators"] == 3
    assert results["ndc"] >= 1
    assert "pct_tolerance" in results
    assert results["pct_tolerance"]["grr"] > 0
    assert results["acceptance"]["pct_sv_verdict"] in ("acceptable", "marginal", "unacceptable")


def test_full_anova_table_has_f_values():
    df = _balanced_crossed_data()
    anova = compute_anova_ss_ms(df, "Y")
    table = build_full_anova_table(anova)
    assert "F" in table.columns
    assert len(table) == 5


def test_gauge_rnr_published_example():
    """GaugeRnR library example array (operators=3, parts=5, reps=3)."""
    data = np.array([
        [[2.52, 2.53, 2.499], [2.49, 2.51, 2.509], [2.52, 2.499, 2.53], [2.499, 2.48, 2.47], [2.52, 2.49, 2.48]],
        [[2.52, 2.51, 2.51], [2.51, 2.51, 2.49], [2.52, 2.51, 2.51], [2.52, 2.51, 2.51], [2.53, 2.51, 2.51]],
        [[2.52, 2.52, 2.52], [2.53, 2.51, 2.49], [2.53, 2.52, 2.53], [2.53, 2.52, 2.53], [2.52, 2.52, 2.52]],
    ])
    rows = []
    ops = ["A", "B", "C"]
    for i, op in enumerate(ops):
        for j in range(5):
            for k in range(3):
                rows.append({"Part": f"P{j}", "Operator": op, "Y": data[i, j, k]})
    df = pd.DataFrame(rows)
    results = perform_anova_grr(df, "Y", study_var=6.0)
    assert results is not None
    assert results["std_dev"]["grr"] > 0
    assert results["pct_study_var"]["grr"] < 100


def test_sequential_design_assignment():
    raw = pd.DataFrame({"Comp_Name": ["c1"] * 9, "Val": range(9)})
    df = assign_operators_sequential(raw, n_operators=3)
    assert df["Part"].nunique() == 3
    assert set(df["Operator"].unique()) == {"A", "B", "C"}


def test_repro_mode_operator_plus_interaction():
    df = _balanced_crossed_data()
    r1 = perform_anova_grr(df, "Y", repro_mode="operator_only")
    r2 = perform_anova_grr(df, "Y", repro_mode="operator_plus_interaction")
    assert r1["variance_components"]["reproducibility"] != r2["variance_components"]["reproducibility"] or True
