"""Crossed Gage R&R (Type 2) — ANOVA variance components."""

from typing import Any, Dict, Literal, Optional

import numpy as np

from grr_tool.msa.anova_table import (
    build_full_anova_table,
    compute_anova_ss_ms,
    variance_component_ci,
)
from grr_tool.msa.acceptance import build_gage_rr_acceptance
from grr_tool.msa.tolerance import compute_tolerance_metrics

ReproMode = Literal["operator_only", "operator_plus_interaction"]


def _variance_components(
    anova: Dict[str, Any],
    repro_mode: ReproMode = "operator_only",
) -> Dict[str, float]:
    ms = anova["ms"]
    n_parts = anova["n_parts"]
    n_operators = anova["n_operators"]
    r = anova["n_replicates_int"]

    if r > 1:
        var_equipment = ms["equipment"]
        var_interaction = max(0.0, (ms["interaction"] - ms["equipment"]) / r)
        var_operator = max(0.0, (ms["operator"] - ms["interaction"]) / (n_parts * r))
        var_part = max(0.0, (ms["part"] - ms["interaction"]) / (n_operators * r))
    else:
        var_equipment = 0.0
        var_interaction = ms["interaction"]
        var_operator = max(0.0, (ms["operator"] - ms["interaction"]) / n_parts)
        var_part = max(0.0, (ms["part"] - ms["interaction"]) / n_operators)

    var_repeatability = var_equipment + var_interaction

    if repro_mode == "operator_plus_interaction":
        var_reproducibility = var_operator + var_interaction
        var_repeatability = var_equipment
    else:
        var_reproducibility = var_operator

    var_grr = var_repeatability + var_reproducibility
    var_total = var_grr + var_part

    return {
        "equipment": var_equipment,
        "interaction": var_interaction,
        "operator": var_operator,
        "repeatability": var_repeatability,
        "reproducibility": var_reproducibility,
        "grr": var_grr,
        "part": var_part,
        "total": var_total,
    }


def perform_anova_grr(
    df,
    measurement_col: str,
    study_var: float = 6.0,
    tolerance: Optional[float] = None,
    alpha: float = 0.05,
    repro_mode: ReproMode = "operator_only",
) -> Optional[Dict[str, Any]]:
    """
    Perform ANOVA-based crossed Gage R&R (Type 2) for one measurement column.

    Requires columns Part, Operator, and measurement_col.
    """
    import pandas as pd

    data = df[["Part", "Operator", measurement_col]].copy()
    data = data.dropna()
    if len(data) == 0:
        return None

    anova = compute_anova_ss_ms(data, measurement_col)
    vc = _variance_components(anova, repro_mode=repro_mode)

    sd = {k: float(np.sqrt(v)) for k, v in vc.items() if k in (
        "repeatability", "reproducibility", "grr", "part", "total"
    )}

    sv = {k: study_var * sd[k] for k in sd}
    var_total = vc["total"]

    pct_contrib = {
        "repeatability": (vc["repeatability"] / var_total * 100) if var_total > 0 else 0,
        "reproducibility": (vc["reproducibility"] / var_total * 100) if var_total > 0 else 0,
        "grr": (vc["grr"] / var_total * 100) if var_total > 0 else 0,
        "part": (vc["part"] / var_total * 100) if var_total > 0 else 0,
    }
    pct_sv = {
        k: (sv[k] / sv["total"] * 100) if sv["total"] > 0 else 0
        for k in ("repeatability", "reproducibility", "grr", "part")
    }

    ndc = int(np.floor(1.41 * (sd["part"] / sd["grr"]))) if sd["grr"] > 0 else 0

    df_eq = anova["df"]["equipment"]
    var_ci = {}
    if alpha is not None and alpha > 0:
        for key, df_k in (
            ("repeatability", df_eq),
            ("grr", df_eq),
        ):
            lo, hi = variance_component_ci(vc.get(key, vc["grr"]), df_k, alpha)
            var_ci[key] = {"low": lo, "high": hi}

    full_anova_df = build_full_anova_table(anova)

    results: Dict[str, Any] = {
        "study_type": "gage_rr_type2",
        "measurement": measurement_col,
        "n_parts": anova["n_parts"],
        "n_operators": anova["n_operators"],
        "n_measurements": anova["n_measurements"],
        "n_replicates": anova["n_replicates"],
        "repro_mode": repro_mode,
        "study_var_multiplier": study_var,
        "tolerance": tolerance,
        "alpha": alpha,
        "anova": anova,
        "full_anova_table": full_anova_df,
        "variance_components": {
            "repeatability": vc["repeatability"],
            "reproducibility": vc["reproducibility"],
            "grr": vc["grr"],
            "part": vc["part"],
            "total": vc["total"],
            "equipment": vc["equipment"],
            "interaction": vc["interaction"],
            "operator": vc["operator"],
        },
        "std_dev": sd,
        "study_var": sv,
        "pct_contribution": pct_contrib,
        "pct_study_var": pct_sv,
        "variance_ci": var_ci,
        "ndc": ndc,
    }

    if tolerance is not None and tolerance > 0:
        results["pct_tolerance"] = compute_tolerance_metrics(sd, tolerance, study_var)

    results["acceptance"] = build_gage_rr_acceptance(results)
    return results
