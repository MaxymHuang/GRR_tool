"""Summary tables for MSA output."""

from typing import Any, Dict, Optional

import pandas as pd


def format4(x: float) -> str:
    try:
        return f"{x:.4g}"
    except Exception:
        return str(x)


def create_anova_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Compact summary table with correct %SV and optional %GRR(Tol) labels."""
    if results is None:
        return None

    pct_sv_grr = results["pct_study_var"]["grr"]
    cols = ["Source", "StdDev(SD)", f"StdVar({results.get('study_var_multiplier', 6)}*Std)", "%Study Var (%SV)"]
    pct_tol_col = "%GRR (Tol)" if results.get("tolerance") else None
    if pct_tol_col:
        cols.append(pct_tol_col)

    rows = [
        ("Repeatability", results["std_dev"]["repeatability"], results["study_var"]["repeatability"],
         results["pct_study_var"]["repeatability"]),
        ("Reproducibility", results["std_dev"]["reproducibility"], results["study_var"]["reproducibility"],
         results["pct_study_var"]["reproducibility"]),
        ("Gage R&R", results["std_dev"]["grr"], results["study_var"]["grr"], pct_sv_grr),
        ("Part-to-Part", results["std_dev"]["part"], results["study_var"]["part"],
         results["pct_study_var"]["part"]),
        ("Total Variation", results["std_dev"]["total"], results["study_var"]["total"], 100.0),
    ]

    data = {
        "Source": [r[0] for r in rows],
        "StdDev(SD)": [format4(r[1]) for r in rows],
        cols[2]: [format4(r[2]) for r in rows],
        "%Study Var (%SV)": [f"{r[3]:.4g}%" if r[3] != 100.0 else "100.0%" for r in rows],
    }
    if pct_tol_col and "pct_tolerance" in results:
        pt = results["pct_tolerance"]
        data[pct_tol_col] = [
            f"{pt['repeatability']:.4g}%",
            f"{pt['reproducibility']:.4g}%",
            f"{pt['grr']:.4g}%",
            f"{pt['part']:.4g}%",
            "",
        ]

    return pd.DataFrame(data)


def create_variance_summary_df(results: Dict[str, Any]) -> pd.DataFrame:
    """Full variance breakdown for export."""
    vc = results["variance_components"]
    sd = results["std_dev"]
    pc = results["pct_contribution"]
    psv = results["pct_study_var"]
    rows = {
        "Component": ["Repeatability", "Reproducibility", "Gage R&R", "Part-to-Part", "Total"],
        "Variance": [vc["repeatability"], vc["reproducibility"], vc["grr"], vc["part"], vc["total"]],
        "Std Dev": [sd["repeatability"], sd["reproducibility"], sd["grr"], sd["part"], sd["total"]],
        "%Contribution": [pc["repeatability"], pc["reproducibility"], pc["grr"], pc["part"], 100.0],
        "%Study Var": [psv["repeatability"], psv["reproducibility"], psv["grr"], psv["part"], 100.0],
    }
    if results.get("pct_tolerance"):
        pt = results["pct_tolerance"]
        rows["%GRR (Tol)"] = [
            pt["repeatability"], pt["reproducibility"], pt["grr"], pt["part"], "",
        ]
    df = pd.DataFrame(rows)
    for col in ("Variance", "Std Dev", "%Contribution", "%Study Var"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: format4(x) if isinstance(x, (int, float)) and x != "" else x)
    return df


def create_type1_summary_df(metrics: Dict[str, Any]) -> pd.DataFrame:
    row = {
        "n": metrics["n"],
        "mean": format4(metrics["mean"]),
        "sd": format4(metrics["sd"]),
        "6sigma": format4(metrics["study_var"]),
        "tol": format4(metrics["tol"]),
        "tf": format4(metrics.get("tf", 1.0)),
        "target": format4(metrics["target"]),
        "LSL": format4(metrics["lsl"]),
        "USL": format4(metrics["usl"]),
        "LCL": format4(metrics.get("lcl", float("nan"))),
        "UCL": format4(metrics.get("ucl", float("nan"))),
        "Cg": format4(metrics["cg"]),
        "Cgk": format4(metrics["cgk"]),
        "%Var_repeat": format4(metrics.get("pct_var_repeatability", float("nan"))),
        "%Var_repeat_bias": format4(metrics.get("pct_var_repeatability_bias", float("nan"))),
        "bias": format4(metrics["bias"]),
        "bias_%tol": format4(metrics["bias_pct_tol"]),
        "t": format4(metrics["t"]),
        "p": format4(metrics["p"]),
        "CI_low": format4(metrics["ci_low"]),
        "CI_high": format4(metrics["ci_high"]),
    }
    if metrics.get("acceptance"):
        acc = metrics["acceptance"]
        row["verdict"] = acc.get("overall", "")
    return pd.DataFrame([row])
