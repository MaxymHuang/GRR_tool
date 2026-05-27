"""Optional statsmodels REML cross-check for variance components."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def reml_variance_components(
    df: pd.DataFrame,
    measurement_col: str,
) -> Optional[Dict[str, float]]:
    """
    Fit crossed random effects via statsmodels MixedLM (validation only).

    Returns None if statsmodels is not installed or fit fails.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None

    data = df[["Part", "Operator", measurement_col]].dropna().copy()
    if len(data) < 10:
        return None

    data["Part"] = data["Part"].astype(str)
    data["Operator"] = data["Operator"].astype(str)
    groups = np.ones(len(data))

    try:
        model = smf.mixedlm(
            f"{measurement_col} ~ 1",
            data,
            groups=groups,
            vc_formula={
                "part": "0 + C(Part)",
                "operator": "0 + C(Operator)",
            },
        )
        result = model.fit(reml=True, method="lbfgs", maxiter=200, disp=False)
    except Exception:
        return None

    vc = {}
    for name in ("part", "operator"):
        key = f"{name} Var"
        if key in result.cov_re:
            vc[name] = float(result.cov_re[key])
    scale = float(result.scale) if hasattr(result, "scale") else np.nan
    vc["residual"] = scale
    return vc
