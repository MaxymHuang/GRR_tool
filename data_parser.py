"""
Data Parser Module for Gage R&R Analysis

Consolidated parsing and data manipulation utilities for both ANOVA and Type 1 analysis scripts.
Handles both .txt and .csv formats with robust error handling and flexible filtering options.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import warnings

warnings.filterwarnings('ignore')


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean the measurement data file.
    
    Supports both .txt (tab-separated multi-section format) and .csv formats.
    
    Args:
        file_path: Path to the input data file
        
    Returns:
        Cleaned DataFrame with standardized column names
    """
    print(f"Loading data from {file_path}...")
    
    lower_path = str(file_path).lower()
    if lower_path.endswith('.csv'):
        # Robust CSV read for parsed datasets
        df = pd.read_csv(file_path, engine='python')
        print(f"Found 0 header lines at positions: []")
        print(f"Initial data shape: {df.shape}")
    else:
        # Parse custom tab-separated multi-section format
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Identify header lines that begin sections
        header_lines = []
        for i, line in enumerate(lines):
            if line.startswith('Comp_Name\t'):
                header_lines.append(i)
        
        print(f"Found {len(header_lines)} header lines at positions: {header_lines}")
        
        all_data_rows = []
        first_header = None
        
        for section_idx in range(len(header_lines)):
            start_line = header_lines[section_idx]
            end_line = header_lines[section_idx + 1] if section_idx + 1 < len(header_lines) else len(lines)
            
            header = lines[start_line].strip().split('\t')
            if first_header is None:
                first_header = header
            
            section_lines = lines[start_line + 1:end_line]
            for line in section_lines:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) > 0 and parts[0] and parts[0] != 'Comp_Name':
                        row_dict = {}
                        for i, col in enumerate(header):
                            row_dict[col] = parts[i] if i < len(parts) else ''
                        all_data_rows.append(row_dict)
        
        df = pd.DataFrame(all_data_rows)
        print(f"Initial data shape: {df.shape}")
    
    # Remove specified columns if present
    columns_to_remove = [
        'Subtype_NO', 'BoardIn_NO', 'Scan_NO',
        'CAD_X', 'CAD_Y', 'CAD_Width', 'CAD_Height', 'Board_Side'
    ]
    existing_cols_to_remove = [c for c in columns_to_remove if c in df.columns]
    if existing_cols_to_remove:
        df = df.drop(columns=existing_cols_to_remove)
    
    # Normalize empties and drop all-empty columns
    df = df.replace('', np.nan)
    df = df.dropna(axis=1, how='all')
    
    # Convert numeric columns (all except Comp_Name)
    numeric_cols = [c for c in df.columns if c != 'Comp_Name']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with all-NaN numeric data
    if numeric_cols:
        df = df.dropna(subset=numeric_cols, how='all')
    
    # For CSVs produced by our parser, Comp_Name should exist; if not, try fallbacks
    if 'Comp_Name' not in df.columns:
        # Try common alternative naming
        candidate_cols = [c for c in df.columns if c.lower() in ['comp_name', 'component']]
        if candidate_cols:
            df['Comp_Name'] = df[candidate_cols[0]]
    
    # Keep valid Comp_Name; set Component as Comp_Name or Comp_Name_Box when Box present
    if 'Comp_Name' in df.columns:
        df = df[df['Comp_Name'].notna() & (df['Comp_Name'] != '')]
        df['Comp_Name'] = df['Comp_Name'].astype(str).str.strip()
        if 'Box_Name' in df.columns:
            df['Box_Name'] = df['Box_Name'].astype(str).str.strip()
            has_box = df['Box_Name'].notna() & (df['Box_Name'] != '')
            df['Component'] = np.where(
                has_box,
                df['Comp_Name'] + '_' + df['Box_Name'],
                df['Comp_Name']
            )
        else:
            df['Component'] = df['Comp_Name']
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Columns retained: {list(df.columns)}")
    
    return df


def normalize_component_list(components: List[str]) -> List[str]:
    """
    Normalize component list by splitting on commas and stripping whitespace.
    
    Args:
        components: List of component strings (may contain comma-separated values)
        
    Returns:
        Flattened list of normalized component names
    """
    normalized = []
    for comp in components:
        for item in str(comp).split(','):
            item = item.strip()
            if item:
                normalized.append(item)
    return normalized


def apply_component_filters(df: pd.DataFrame, include: Optional[List[str]] = None, 
                          exclude: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Apply component include/exclude filters to DataFrame.
    
    Args:
        df: Input DataFrame
        include: List of components to include (None for all)
        exclude: List of components to exclude (None for none)
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    if include:
        include_normalized = normalize_component_list(include)
        include_set = set(include_normalized)
        if include_set and 'Comp_Name' in filtered_df.columns:
            before_rows = len(filtered_df)
            filtered_df = filtered_df[filtered_df['Comp_Name'].isin(include_set)].reset_index(drop=True)
            after_rows = len(filtered_df)
            print(f"Including only components: {sorted(list(include_set))}")
            print(f"Rows before: {before_rows}, after: {after_rows}")
    
    if exclude:
        exclude_normalized = normalize_component_list(exclude)
        exclude_set = set(exclude_normalized)
        if exclude_set and 'Comp_Name' in filtered_df.columns:
            before_rows = len(filtered_df)
            filtered_df = filtered_df[~filtered_df['Comp_Name'].isin(exclude_set)].reset_index(drop=True)
            after_rows = len(filtered_df)
            print(f"Excluding components: {sorted(list(exclude_set))}")
            print(f"Rows before: {before_rows}, after: {after_rows}")
    
    return filtered_df


def apply_column_drops(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Drop specified columns from DataFrame.
    
    Args:
        df: Input DataFrame
        columns_to_drop: List of column names to drop
        
    Returns:
        DataFrame with specified columns removed
    """
    if not columns_to_drop:
        return df
    
    existing_cols_to_drop = [c for c in columns_to_drop if c in df.columns]
    if existing_cols_to_drop:
        print(f"Dropping columns: {existing_cols_to_drop}")
        return df.drop(columns=existing_cols_to_drop)
    
    return df


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of numeric columns from DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of column names with numeric data types
    """
    numeric_kinds = set(['i', 'u', 'f'])  # int, unsigned, float
    return [
        c for c in df.columns
        if hasattr(df[c].dtype, 'kind') and df[c].dtype.kind in numeric_kinds
    ]


def get_measurement_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of measurement columns (numeric columns excluding metadata).
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of measurement column names
    """
    metadata_cols = {'Comp_Name', 'Component', 'Operator', 'Part', 'Part_ID', 'Measurement_Order'}
    numeric_cols = get_numeric_columns(df)
    return [c for c in numeric_cols if c not in metadata_cols]


def assign_operators_sequential(df: pd.DataFrame, n_operators: int = 3) -> pd.DataFrame:
    """
    Assign operators sequentially to measurements for ANOVA analysis.
    
    Args:
        df: Input DataFrame
        n_operators: Number of operators (default: 3)
        
    Returns:
        DataFrame with Operator and Part columns added
    """
    operators = [chr(ord('A') + i) for i in range(n_operators)]
    
    print(f"\nAssigning operators sequentially (every {n_operators} consecutive measurements = 1 part)...")
    
    # Assign operator based on position mod n_operators
    df['Operator'] = [operators[i % n_operators] for i in range(len(df))]
    
    # Create Part ID: every n_operators consecutive measurements belong to the same part
    df['Part_ID'] = np.arange(len(df)) // n_operators
    df['Part'] = 'Part_' + df['Part_ID'].astype(str)
    
    print(f"Operator distribution:")
    print(df['Operator'].value_counts().sort_index())
    print(f"\nTotal unique parts (measurement groups): {df['Part'].nunique()}")
    
    # Show sample for verification
    print(f"\nSample assignment (first 12 rows):")
    sample_data = df[['Component', 'Part', 'Operator']].head(12)
    print(sample_data.to_string(index=False))
    
    return df


def remove_outliers_iqr(df: pd.DataFrame, measurement_col: str) -> Tuple[pd.DataFrame, int]:
    """
    Remove outliers from DataFrame for the given measurement column using the IQR rule (1.5*IQR).
    
    Args:
        df: Input DataFrame
        measurement_col: Name of the measurement column
        
    Returns:
        Tuple of (filtered DataFrame, number of rows removed)
    """
    if measurement_col not in df.columns:
        return df, 0
    
    series = df[measurement_col].astype(float)
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    if pd.isna(iqr) or iqr == 0:
        return df, 0
    
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = series.between(lower, upper, inclusive='both')
    removed = int((~mask).sum())
    
    return df[mask].reset_index(drop=True), removed


def remove_outliers_iqr_series(values: pd.Series) -> Tuple[pd.Series, int]:
    """
    Remove outliers from a numeric series using IQR (1.5*IQR) rule.
    
    Args:
        values: Input pandas Series
        
    Returns:
        Tuple of (filtered series, number of points removed)
    """
    series = values.dropna().astype(float)
    if series.empty:
        return series, 0
    
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    if pd.isna(iqr) or iqr == 0:
        return series, 0
    
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = series.between(lower, upper, inclusive='both')
    removed = int((~mask).sum())
    
    return series[mask], removed


def get_components_preview(df: pd.DataFrame, max_components: int = 60) -> str:
    """
    Get a preview string of component names for display.
    
    Args:
        df: Input DataFrame
        max_components: Maximum number of components to show
        
    Returns:
        String with component names separated by " | "
    """
    if 'Comp_Name' not in df.columns:
        return "(No components to display)"
    
    comps = sorted(df['Comp_Name'].dropna().astype(str).unique().tolist())
    preview = comps[:max_components]
    return " | ".join(preview) if preview else "(none)"


def validate_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate data quality and return summary statistics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with data quality metrics
    """
    metrics = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(get_numeric_columns(df)),
        'measurement_columns': len(get_measurement_columns(df))
    }
    
    if 'Comp_Name' in df.columns:
        metrics['unique_components'] = df['Comp_Name'].nunique()
    
    return metrics
