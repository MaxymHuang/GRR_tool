"""ANOVA sum-of-squares and F-tests for crossed Gage R&R."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _compute_ss(
    data: pd.DataFrame,
    measurement_col: str,
    parts: np.ndarray,
    operators: np.ndarray,
) -> Dict[str, float]:
    grand_mean = data[measurement_col].mean()
    part_means = data.groupby("Part")[measurement_col].mean()
    operator_means = data.groupby("Operator")[measurement_col].mean()

    ss_part = 0.0
    for part in parts:
        part_data = data[data["Part"] == part]
        n_part = len(part_data)
        ss_part += n_part * (part_means[part] - grand_mean) ** 2

    ss_operator = 0.0
    for operator in operators:
        op_data = data[data["Operator"] == operator]
        n_op = len(op_data)
        ss_operator += n_op * (operator_means[operator] - grand_mean) ** 2

    ss_interaction = 0.0
    for (part, operator), group in data.groupby(["Part", "Operator"]):
        if len(group) > 0:
            group_mean = group[measurement_col].mean()
            n_group = len(group)
            expected = part_means[part] + operator_means[operator] - grand_mean
            ss_interaction += n_group * (group_mean - expected) ** 2

    ss_equipment = 0.0
    for (_, _), group in data.groupby(["Part", "Operator"]):
        if len(group) > 1:
            group_mean = group[measurement_col].mean()
            ss_equipment += np.sum((group[measurement_col] - group_mean) ** 2)

    ss_total = np.sum((data[measurement_col] - grand_mean) ** 2)
    return {
        "total": ss_total,
        "part": ss_part,
        "operator": ss_operator,
        "interaction": ss_interaction,
        "equipment": ss_equipment,
    }


def compute_anova_ss_ms(
    data: pd.DataFrame,
    measurement_col: str,
) -> Dict[str, Any]:
    """Compute SS, DF, MS for crossed Part x Operator design."""
    parts = data["Part"].unique()
    operators = data["Operator"].unique()
    n_parts = len(parts)
    n_operators = len(operators)
    n_measurements = len(data)
    part_op_counts = data.groupby(["Part", "Operator"]).size()
    n_groups = len(part_op_counts)

    ss = _compute_ss(data, measurement_col, parts, operators)

    df_total = n_measurements - 1
    df_part = n_parts - 1
    df_operator = n_operators - 1
    df_interaction = (n_parts - 1) * (n_operators - 1)
    df_equipment = n_measurements - n_groups

    ms_part = ss["part"] / df_part if df_part > 0 else 0.0
    ms_operator = ss["operator"] / df_operator if df_operator > 0 else 0.0
    ms_interaction = ss["interaction"] / df_interaction if df_interaction > 0 else 0.0
    ms_equipment = ss["equipment"] / df_equipment if df_equipment > 0 else 0.0

    # Harmonic mean replicate count for EMS (handles mild imbalance)
    counts = part_op_counts.values.astype(float)
    counts = counts[counts > 0]
    if len(counts) > 0:
        n_replicates = float(len(counts) / np.sum(1.0 / counts))  # harmonic mean
    else:
        n_replicates = 1.0
    n_replicates_int = max(1, int(round(n_replicates)))

    return {
        "ss": ss,
        "df": {
            "total": df_total,
            "part": df_part,
            "operator": df_operator,
            "interaction": df_interaction,
            "equipment": df_equipment,
        },
        "ms": {
            "part": ms_part,
            "operator": ms_operator,
            "interaction": ms_interaction,
            "equipment": ms_equipment,
        },
        "n_parts": n_parts,
        "n_operators": n_operators,
        "n_measurements": n_measurements,
        "n_replicates": n_replicates,
        "n_replicates_int": n_replicates_int,
        "n_groups": n_groups,
        "part_op_counts": part_op_counts,
    }


def _f_test(ms_num: float, ms_den: float, df_num: int, df_den: int) -> Tuple[float, float]:
    if df_num <= 0 or df_den <= 0 or ms_den <= 0:
        return np.nan, np.nan
    f_val = ms_num / ms_den
    p_val = 1.0 - stats.f.cdf(f_val, df_num, df_den)
    return float(f_val), float(p_val)


def build_full_anova_table(anova: Dict[str, Any]) -> pd.DataFrame:
    """Full ANOVA table with F and p-values."""
    ss = anova["ss"]
    df = anova["df"]
    ms = anova["ms"]

    f_op, p_op = _f_test(ms["operator"], ms["interaction"], df["operator"], df["interaction"])
    f_part, p_part = _f_test(ms["part"], ms["interaction"], df["part"], df["interaction"])
    f_int, p_int = _f_test(ms["interaction"], ms["equipment"], df["interaction"], df["equipment"])

    rows = [
        ("Part", df["part"], ss["part"], ms["part"], f_part, p_part),
        ("Operator", df["operator"], ss["operator"], ms["operator"], f_op, p_op),
        ("Part x Operator", df["interaction"], ss["interaction"], ms["interaction"], f_int, p_int),
        ("Equipment (Repeatability)", df["equipment"], ss["equipment"], ms["equipment"], np.nan, np.nan),
        ("Total", df["total"], ss["total"], np.nan, np.nan, np.nan),
    ]
    return pd.DataFrame(
        rows,
        columns=["Source", "DF", "SS", "MS", "F", "p-value"],
    )


def variance_component_ci(
    var_estimate: float,
    df: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Approximate chi-square CI for a variance component from MS with df degrees of freedom.
    Uses: (df * MS / chi2_{1-alpha/2}, df * MS / chi2_{alpha/2}).
    """
    if df <= 0 or var_estimate < 0:
        return (np.nan, np.nan)
    ms_equiv = var_estimate  # treating var as MS when df=1 per component (simplified)
    lo = df * ms_equiv / stats.chi2.ppf(1 - alpha / 2, df) if df > 0 else np.nan
    hi = df * ms_equiv / stats.chi2.ppf(alpha / 2, df) if df > 0 else np.nan
    return (max(0.0, float(lo)), max(0.0, float(hi)))
