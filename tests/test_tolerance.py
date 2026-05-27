"""Tests for tolerance metrics."""

from grr_tool.msa.tolerance import (
    pct_grr_vs_tolerance,
    type1_pct_var,
    type1_pct_var_repeat_bias,
    compute_tolerance_metrics,
)


def test_pct_grr_vs_tolerance():
    assert abs(pct_grr_vs_tolerance(0.1, 1.0, 6.0) - 60.0) < 1e-9


def test_type1_pct_var():
    assert abs(type1_pct_var(0.6, 3.0) - 20.0) < 1e-9


def test_type1_pct_var_repeat_bias():
    assert abs(type1_pct_var_repeat_bias(0.6, 0.05, 3.0) - 23.333333) < 0.01


def test_compute_tolerance_metrics():
    sd = {"repeatability": 0.1, "reproducibility": 0.05, "grr": 0.12, "part": 0.5}
    m = compute_tolerance_metrics(sd, tolerance=2.0, nsigma=6.0)
    assert abs(m["grr"] - 36.0) < 1e-9
