"""Type 1 gage study tests."""

import numpy as np
import pandas as pd
import pytest

from grr_tool.msa.type1 import compute_type1_metrics


def test_type1_spc_example():
    """SPC for Excel style: mean=12.3097, sd~0.00124, tol=0.05, target=12.31."""
    rng = np.random.default_rng(42)
    values = pd.Series(rng.normal(12.3097, 0.00124, 25))
    metrics = compute_type1_metrics(
        values, sv=6.0, tol=0.05, target=12.31, alpha=0.05, require_reference=True
    )
    assert metrics["cg"] >= 1.33
    assert metrics["cgk"] >= 1.0
    assert metrics["pct_var_repeatability"] < 20.0
    assert not metrics["exploratory"]


def test_type1_exploratory_warning():
    values = pd.Series([1.0, 1.1, 1.05, 0.98, 1.02])
    with pytest.warns(UserWarning, match="exploratory"):
        metrics = compute_type1_metrics(values)
    assert metrics["exploratory"] is True


def test_type1_require_reference_raises():
    values = pd.Series([1.0, 1.1, 1.05])
    with pytest.raises(ValueError, match="requires explicit"):
        compute_type1_metrics(values, require_reference=True)
