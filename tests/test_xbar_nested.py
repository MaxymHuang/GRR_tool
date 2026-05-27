"""Xbar-R and nested GRR tests."""

import pandas as pd
import pytest

from grr_tool.msa.xbar_r import perform_xbar_r
from grr_tool.msa.nested import perform_nested_grr


def _balanced():
    rows = []
    for p in range(3):
        for o in ("A", "B"):
            for r in range(2):
                rows.append({"Part": f"P{p}", "Operator": o, "Y": p + (0.1 if o == "B" else 0) + r * 0.01})
    return pd.DataFrame(rows)


def test_xbar_r_balanced():
    result = perform_xbar_r(_balanced(), "Y", tolerance=5.0)
    assert result is not None
    assert result["method"] == "xbar_r"
    assert "pct_tolerance" in result


def test_xbar_r_unbalanced_raises():
    df = pd.DataFrame([
        {"Part": "P0", "Operator": "A", "Y": 1.0},
        {"Part": "P0", "Operator": "A", "Y": 1.1},
        {"Part": "P0", "Operator": "B", "Y": 1.0},
    ])
    with pytest.raises(ValueError):
        perform_xbar_r(df, "Y")


def test_nested_grr():
    rows = []
    for op in ("A", "B"):
        for p in range(3):
            for _ in range(2):
                rows.append({"Part": f"{op}_{p}", "Operator": op, "Y": 1.0 + hash(op) % 3 * 0.1})
    df = pd.DataFrame(rows)
    result = perform_nested_grr(df, "Y", tolerance=2.0)
    assert result is not None
    assert result["study_type"] == "gage_rr_nested"
