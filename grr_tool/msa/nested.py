"""Nested Gage R&R for destructive / non-reusable parts."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def perform_nested_grr(
    df: pd.DataFrame,
    measurement_col: str,
    part_col: str = "Part",
    operator_col: str = "Operator",
    study_var: float = 6.0,
    tolerance: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Simplified nested design: each operator measures unique parts (parts nested in operators).

    Variance components via one-way ANOVA within operators and between operators.
    """
    data = df[[part_col, operator_col, measurement_col]].dropna().copy()
    data.columns = ["Part", "Operator", measurement_col]
    if data.empty:
        return None

    grand = data[measurement_col].mean()
    op_means = data.groupby("Operator")[measurement_col].mean()

    ss_operator = sum(
        len(data[data["Operator"] == op]) * (op_means[op] - grand) ** 2
        for op in op_means.index
    )
    ss_equipment = 0.0
    for op in data["Operator"].unique():
        for part in data[data["Operator"] == op]["Part"].unique():
            grp = data[(data["Operator"] == op) & (data["Part"] == part)]
            if len(grp) > 1:
                m = grp[measurement_col].mean()
                ss_equipment += ((grp[measurement_col] - m) ** 2).sum()

    n_ops = data["Operator"].nunique()
    n_meas = len(data)
    df_op = n_ops - 1
    df_eq = n_meas - data.groupby(["Operator", "Part"]).ngroups

    ms_op = ss_operator / df_op if df_op > 0 else 0
    ms_eq = ss_equipment / df_eq if df_eq > 0 else 0

    n_per_op = data.groupby("Operator").size().mean()
    var_equipment = ms_eq
    var_operator = max(0.0, (ms_op - ms_eq) / n_per_op) if n_per_op > 0 else 0

    var_repeat = var_equipment
    var_repro = var_operator
    var_grr = var_repeat + var_repro

    part_means = data.groupby("Part")[measurement_col].mean()
    var_part = float(part_means.var(ddof=1)) if len(part_means) > 1 else 0.0
    var_total = var_grr + var_part

    sd = {
        "repeatability": np.sqrt(var_repeat),
        "reproducibility": np.sqrt(var_repro),
        "grr": np.sqrt(var_grr),
        "part": np.sqrt(var_part),
        "total": np.sqrt(var_total),
    }
    sv = {k: study_var * sd[k] for k in sd}
    pct_sv = {k: (sv[k] / sv["total"] * 100) if sv["total"] > 0 else 0 for k in sd}

    result: Dict[str, Any] = {
        "study_type": "gage_rr_nested",
        "method": "nested",
        "variance_components": {
            "repeatability": var_repeat,
            "reproducibility": var_repro,
            "grr": var_grr,
            "part": var_part,
            "total": var_total,
        },
        "std_dev": sd,
        "study_var": sv,
        "pct_study_var": pct_sv,
    }
    if tolerance and tolerance > 0:
        from grr_tool.msa.tolerance import compute_tolerance_metrics

        result["pct_tolerance"] = compute_tolerance_metrics(sd, tolerance, study_var)
    return result
