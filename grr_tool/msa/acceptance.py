"""AIAG-style acceptance verdicts for MSA studies."""

from typing import Any, Dict, Optional


def verdict_pct_sv(pct_grr: float) -> str:
    if pct_grr < 10:
        return "acceptable"
    if pct_grr <= 30:
        return "marginal"
    return "unacceptable"


def verdict_pct_tol(pct_grr_tol: float) -> str:
    return verdict_pct_sv(pct_grr_tol)


def verdict_ndc(ndc: int) -> str:
    return "acceptable" if ndc >= 5 else "unacceptable"


def verdict_cg(cg: float, target: float = 1.33) -> str:
    if cg >= target:
        return "acceptable"
    return "unacceptable"


def build_gage_rr_acceptance(results: Dict[str, Any]) -> Dict[str, Any]:
    pct_sv = results["pct_study_var"]["grr"]
    out: Dict[str, Any] = {
        "pct_sv_grr": pct_sv,
        "pct_sv_verdict": verdict_pct_sv(pct_sv),
        "ndc": results["ndc"],
        "ndc_verdict": verdict_ndc(results["ndc"]),
    }
    if results.get("tolerance") is not None:
        pct_tol = results["pct_tolerance"]["grr"]
        out["pct_tol_grr"] = pct_tol
        out["pct_tol_verdict"] = verdict_pct_tol(pct_tol)
    return out


def build_type1_acceptance(metrics: Dict[str, Any]) -> Dict[str, Any]:
    cg_ok = verdict_cg(metrics["cg"])
    cgk_ok = verdict_cg(metrics["cgk"])
    bias_sig = metrics["p"] < metrics.get("alpha", 0.05) if metrics.get("p") == metrics.get("p") else False
    overall = "acceptable" if cg_ok == "acceptable" and cgk_ok == "acceptable" else "unacceptable"
    return {
        "cg": metrics["cg"],
        "cgk": metrics["cgk"],
        "cg_verdict": cg_ok,
        "cgk_verdict": cgk_ok,
        "bias_significant": bool(bias_sig),
        "overall": overall,
        "exploratory": metrics.get("exploratory", False),
    }
