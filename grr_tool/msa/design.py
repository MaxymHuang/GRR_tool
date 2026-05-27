"""Study design: operator/part assignment for Gage R&R."""

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class DesignMode(str, Enum):
    SEQUENTIAL = "sequential"
    COLUMNS = "columns"
    COMP_NAME = "comp_name"


def assign_operators_sequential(df: pd.DataFrame, n_operators: int = 3) -> pd.DataFrame:
    """Assign Part/Operator from row order (legacy behavior)."""
    df = df.copy()
    try:
        n = int(n_operators)
    except Exception:
        n = 3
    n = max(1, min(n, 10))
    if len(df) == 0:
        return df
    operators = [chr(ord("A") + i) for i in range(n)]
    df["Operator"] = [operators[i % n] for i in range(len(df))]
    df["Part_ID"] = np.arange(len(df)) // n
    df["Part"] = "Part_" + df["Part_ID"].astype(str)
    return df


def assign_from_comp_name(df: pd.DataFrame, n_operators: int = 3) -> pd.DataFrame:
    """Use Comp_Name as Part; assign operators sequentially within each part."""
    df = df.copy()
    if "Comp_Name" not in df.columns:
        raise ValueError("Comp_Name column required for comp_name design mode")
    n = max(1, min(int(n_operators), 10))
    operators = [chr(ord("A") + i) for i in range(n)]
    within = df.groupby("Comp_Name", sort=False).cumcount()
    df["Part"] = df["Comp_Name"].astype(str)
    df["Operator"] = within.apply(lambda i: operators[int(i) % n])
    df["Part_ID"] = pd.factorize(df["Part"])[0]
    return df


def assign_from_columns(
    df: pd.DataFrame,
    part_col: str,
    operator_col: str,
    replicate_col: Optional[str] = None,
) -> pd.DataFrame:
    """Map existing columns to Part, Operator, optional Replicate."""
    df = df.copy()
    for col, name in ((part_col, "Part"), (operator_col, "Operator")):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found for {name}")
    df["Part"] = df[part_col].astype(str)
    df["Operator"] = df[operator_col].astype(str)
    if replicate_col and replicate_col in df.columns:
        df["Replicate"] = df[replicate_col]
    df["Part_ID"] = pd.factorize(df["Part"])[0]
    return df


def apply_study_design(
    df: pd.DataFrame,
    mode: DesignMode = DesignMode.SEQUENTIAL,
    n_operators: int = 3,
    part_col: Optional[str] = None,
    operator_col: Optional[str] = None,
    replicate_col: Optional[str] = None,
) -> pd.DataFrame:
    """Apply Part/Operator assignment according to design mode."""
    if mode == DesignMode.SEQUENTIAL:
        return assign_operators_sequential(df, n_operators=n_operators)
    if mode == DesignMode.COMP_NAME:
        return assign_from_comp_name(df, n_operators=n_operators)
    if mode == DesignMode.COLUMNS:
        if not part_col or not operator_col:
            raise ValueError("part_col and operator_col required for columns design mode")
        return assign_from_columns(df, part_col, operator_col, replicate_col)
    raise ValueError(f"Unknown design mode: {mode}")
