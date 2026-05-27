"""Tolerance-based MSA metrics (%GRR vs tolerance, Type 1 %Var)."""

from typing import Dict, Optional


def pct_grr_vs_tolerance(sigma: float, tolerance: float, nsigma: float = 6.0) -> float:
    """%GRR(Tol) = 100 * (nsigma * sigma) / tolerance."""
    if tolerance <= 0:
        return 0.0
    return 100.0 * (nsigma * sigma) / tolerance


def compute_tolerance_metrics(
    std_dev: Dict[str, float],
    tolerance: float,
    nsigma: float = 6.0,
) -> Dict[str, float]:
    """Precision-to-tolerance ratios for GRR components."""
    return {
        "repeatability": pct_grr_vs_tolerance(std_dev["repeatability"], tolerance, nsigma),
        "reproducibility": pct_grr_vs_tolerance(std_dev["reproducibility"], tolerance, nsigma),
        "grr": pct_grr_vs_tolerance(std_dev["grr"], tolerance, nsigma),
        "part": pct_grr_vs_tolerance(std_dev["part"], tolerance, nsigma),
    }


def type1_pct_var(study_var: float, tolerance: float) -> float:
    """%Var(repeatability) = 100 * study_var / tolerance."""
    if tolerance <= 0:
        return 0.0
    return 100.0 * study_var / tolerance


def type1_pct_var_repeat_bias(study_var: float, bias: float, tolerance: float) -> float:
    """%Var(repeatability + bias) per SPC for Excel."""
    if tolerance <= 0:
        return 0.0
    return 100.0 * (study_var + 2.0 * abs(bias)) / tolerance
