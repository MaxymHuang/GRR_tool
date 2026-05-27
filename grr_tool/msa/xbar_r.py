"""Average and Range (Xbar-R) method for crossed Gage R&R."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# AIAG d2 for subgroup size 2-10 (for R-bar / sigma_equipment)
_D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}


def perform_xbar_r(
    df: pd.DataFrame,
    measurement_col: str,
    study_var: float = 6.0,
    tolerance: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    AIAG Average & Range shortcut for balanced crossed designs.

    Uses range across replicates within each part-operator cell.
    """
    data = df[["Part", "Operator", measurement_col]].dropna()
    if data.empty:
        return None

    cell_stats = data.groupby(["Part", "Operator"])[measurement_col].agg(["mean", "count", "std"])
    counts = cell_stats["count"].unique()
    if len(counts) != 1 or counts[0] < 2:
        raise ValueError("Xbar-R requires balanced replicates (same r>=2 per part-operator cell)")

    r = int(counts[0])
    d2 = _D2.get(r, 1.128)

    ranges = []
    for (_, _), grp in data.groupby(["Part", "Operator"]):
        vals = grp[measurement_col].values
        if len(vals) >= 2:
            ranges.append(np.max(vals) - np.min(vals))
    r_bar = float(np.mean(ranges)) if ranges else 0.0
    sigma_equipment = r_bar / d2 if d2 > 0 else 0.0

    part_means = data.groupby("Part")[measurement_col].mean()
    op_means = data.groupby("Operator")[measurement_col].mean()
    grand = data[measurement_col].mean()

    sigma_part = float(part_means.std(ddof=1)) if len(part_means) > 1 else 0.0
    sigma_operator = float(op_means.std(ddof=1)) if len(op_means) > 1 else 0.0

    var_repeat = sigma_equipment ** 2
    var_repro = sigma_operator ** 2
    var_part = sigma_part ** 2
    var_grr = var_repeat + var_repro
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

    out: Dict[str, Any] = {
        "method": "xbar_r",
        "r_bar": r_bar,
        "subgroup_size": r,
        "std_dev": sd,
        "study_var": sv,
        "pct_study_var": pct_sv,
        "variance_components": {
            "repeatability": var_repeat,
            "reproducibility": var_repro,
            "grr": var_grr,
            "part": var_part,
            "total": var_total,
        },
    }
    if tolerance and tolerance > 0:
        from grr_tool.msa.tolerance import compute_tolerance_metrics

        out["pct_tolerance"] = compute_tolerance_metrics(sd, tolerance, study_var)
    return out
