"""Type 1 Gage Study — single part, repeated measurements."""

import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from grr_tool.msa.acceptance import build_type1_acceptance
from grr_tool.msa.tolerance import type1_pct_var, type1_pct_var_repeat_bias


def compute_type1_metrics(
    values: pd.Series,
    sv: float = 6.0,
    tol: Optional[float] = None,
    target: Optional[float] = None,
    alpha: float = 0.25,
    tolerance_factor: float = 1.0,
    require_reference: bool = False,
) -> Dict[str, Any]:
    """
    Compute Type 1 gage study metrics.

    When require_reference is True, tol and target must be provided.
    Exploratory defaults emit a warning when tol/target are inferred.
    """
    values = values.dropna().astype(float)
    n = int(values.shape[0])
    if n < 2:
        raise ValueError("Insufficient data for Type 1 study (need at least 2 readings)")

    exploratory = False
    if require_reference and (tol is None or target is None):
        raise ValueError("Type 1 production study requires explicit --tol and --target")

    mean_val = float(values.mean())
    sd = float(values.std(ddof=1))
    study_var = sv * sd

    if tol is None:
        tol = 6.0 * sd * 1.33 / 0.2
        exploratory = True
    if target is None:
        target = float(np.median(values.values))
        exploratory = True

    if exploratory:
        warnings.warn(
            "Type 1: tolerance and/or target were inferred (exploratory mode). "
            "Provide explicit --tol and --target for production acceptance.",
            UserWarning,
            stacklevel=2,
        )

    lsl = target - tol / 2.0
    usl = target + tol / 2.0
    ucl = target + 0.5 * tolerance_factor * tol
    lcl = target - 0.5 * tolerance_factor * tol

    cg = tol / (sv * sd) if (sv > 0 and sd > 0) else np.nan
    cgk = min(usl - mean_val, mean_val - lsl) / (3.0 * sd) if sd > 0 else np.nan

    bias = mean_val - target
    se_mean = sd / np.sqrt(n)
    t_stat = bias / se_mean if se_mean > 0 else np.nan
    df = n - 1
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df)) if not np.isnan(t_stat) else np.nan

    t_crit = stats.t.ppf(1 - alpha / 2.0, df)
    ci_low = mean_val - t_crit * se_mean
    ci_high = mean_val + t_crit * se_mean
    bias_pct_tol = (abs(bias) / tol * 100.0) if tol > 0 else np.nan
    pct_var_repeat = type1_pct_var(study_var, tol)
    pct_var_repeat_bias = type1_pct_var_repeat_bias(study_var, bias, tol)

    metrics: Dict[str, Any] = {
        "study_type": "type1",
        "n": n,
        "mean": mean_val,
        "sd": sd,
        "study_var": study_var,
        "tol": float(tol),
        "tf": float(tolerance_factor),
        "target": float(target),
        "lsl": float(lsl),
        "usl": float(usl),
        "lcl": float(lcl),
        "ucl": float(ucl),
        "cg": float(cg) if not np.isnan(cg) else np.nan,
        "cgk": float(cgk) if not np.isnan(cgk) else np.nan,
        "bias": float(bias),
        "bias_pct_tol": float(bias_pct_tol) if not np.isnan(bias_pct_tol) else np.nan,
        "pct_var_repeatability": float(pct_var_repeat),
        "pct_var_repeatability_bias": float(pct_var_repeat_bias),
        "t": float(t_stat) if not np.isnan(t_stat) else np.nan,
        "p": float(p_val) if not np.isnan(p_val) else np.nan,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "alpha": float(alpha),
        "sv": float(sv),
        "exploratory": exploratory,
    }
    metrics["acceptance"] = build_type1_acceptance(metrics)
    return metrics
