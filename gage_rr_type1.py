"""
Type 1 Gage Study (Single part, single operator, repeated measurements)

This script mirrors the UX of the Type I ANOVA app but implements a Type 1
gage study with:
- CLI flags consistent with the main script
- 4-significant-figure outputs for CSV summaries
- Optional merged plot output
- Optional parse-only mode to export cleaned data

Key metrics:
- n, mean, sd
- Study variation = sv * sd (default sv = 6)
- Tolerance: if --tol provided, use directly; otherwise default to sv * sd
- Target: provided via --target; defaults to sample mean
- LSL/USL = target ± tol/2
- Cg = tol / (sv * sd)
- Cgk = min(USL-mean, mean-LSL) / (3 * sd)
- Bias = mean - target; t-test for bias; mean CI at alpha
"""

import argparse
import json
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean the measurement data file (tab-separated) while preserving
    the data structure. Unlike the Type I ANOVA script, this loader does NOT
    assign operators or parts; it only cleans and standardizes the table.
    """
    print(f"Loading data from {file_path}...")

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
        'Box_Name', 'Subtype_NO', 'BoardIn_NO', 'Scan_NO',
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

    # Keep valid Comp_Name
    df = df[df['Comp_Name'].notna() & (df['Comp_Name'] != '')]

    # Keep original component name for convenience
    df['Component'] = df['Comp_Name']

    print(f"Cleaned data shape: {df.shape}")
    print(f"Columns retained: {list(df.columns)}")

    return df


def compute_type1_metrics(
    values: pd.Series,
    sv: float = 6.0,
    tol: Optional[float] = None,
    target: Optional[float] = None,
    alpha: float = 0.25,
) -> Dict:
    """
    Compute Type 1 gage study metrics given a series of repeated measurements.
    """
    values = values.dropna().astype(float)
    n = int(values.shape[0])
    if n < 2:
        raise ValueError("Insufficient data for Type 1 study (need at least 2 readings)")

    mean_val = float(values.mean())
    sd = float(values.std(ddof=1))

    # Study variation
    study_var = sv * sd

    # Tolerance
    if tol is None:
        tol = study_var  # Default to 6σ per requirement

    # Target
    if target is None:
        target = mean_val

    # Specs
    lsl = target - tol / 2.0
    usl = target + tol / 2.0

    # Capability indices for gage
    cg = tol / (sv * sd) if (sv > 0 and sd > 0) else np.nan
    cgk = min(usl - mean_val, mean_val - lsl) / (3.0 * sd) if sd > 0 else np.nan

    # Bias and bias inference
    bias = mean_val - target
    se_mean = sd / np.sqrt(n)
    t_stat = bias / se_mean if se_mean > 0 else np.nan
    df = n - 1
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df)) if not np.isnan(t_stat) else np.nan

    # Mean confidence interval
    t_crit = stats.t.ppf(1 - alpha / 2.0, df)
    ci_low = mean_val - t_crit * se_mean
    ci_high = mean_val + t_crit * se_mean

    # Bias % of tolerance
    bias_pct_tol = (abs(bias) / tol * 100.0) if tol > 0 else np.nan

    return {
        'n': n,
        'mean': mean_val,
        'sd': sd,
        'study_var': study_var,
        'tol': float(tol),
        'target': float(target),
        'lsl': float(lsl),
        'usl': float(usl),
        'cg': float(cg) if not np.isnan(cg) else np.nan,
        'cgk': float(cgk) if not np.isnan(cgk) else np.nan,
        'bias': float(bias),
        'bias_pct_tol': float(bias_pct_tol) if not np.isnan(bias_pct_tol) else np.nan,
        't': float(t_stat) if not np.isnan(t_stat) else np.nan,
        'p': float(p_val) if not np.isnan(p_val) else np.nan,
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'alpha': float(alpha),
        'sv': float(sv),
    }


def format4(x: float) -> str:
    try:
        return f"{x:.4g}"
    except Exception:
        return str(x)


def create_type1_summary_df(metrics: Dict) -> pd.DataFrame:
    """Create a one-row summary DataFrame with 4 significant figures."""
    row = {
        'n': metrics['n'],
        'mean': format4(metrics['mean']),
        'sd': format4(metrics['sd']),
        '6sigma': format4(metrics['study_var']),
        'tol': format4(metrics['tol']),
        'target': format4(metrics['target']),
        'LSL': format4(metrics['lsl']),
        'USL': format4(metrics['usl']),
        'Cg': format4(metrics['cg']),
        'Cgk': format4(metrics['cgk']),
        'bias': format4(metrics['bias']),
        'bias_%tol': format4(metrics['bias_pct_tol']),
        't': format4(metrics['t']),
        'p': format4(metrics['p']),
        'CI_low': format4(metrics['ci_low']),
        'CI_high': format4(metrics['ci_high']),
    }
    return pd.DataFrame([row])


def plot_distribution_vs_tolerance(values: pd.Series, metrics: Dict, output_path: str):
    data = values.dropna().astype(float).values
    mean_val = metrics['mean']
    sd = metrics['sd']
    lsl = metrics['lsl']
    usl = metrics['usl']
    target = metrics['target']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=min(30, max(8, int(np.sqrt(len(data))))), color='#1f77b4', alpha=0.6, density=True, label='Data')

    # Normal fit
    xs = np.linspace(min(data.min(), lsl), max(data.max(), usl), 200)
    if sd > 0:
        pdf = stats.norm.pdf(xs, loc=mean_val, scale=sd)
        ax.plot(xs, pdf, color='black', linewidth=2, label='Normal fit')

    # Spec and target lines
    ax.axvline(target, color='green', linestyle='-', linewidth=2, label=f'Target={format4(target)}')
    ax.axvline(lsl, color='red', linestyle='--', linewidth=2, label=f'LSL={format4(lsl)}')
    ax.axvline(usl, color='red', linestyle='--', linewidth=2, label=f'USL={format4(usl)}')
    ax.axvline(mean_val, color='orange', linestyle=':', linewidth=2, label=f'Mean={format4(mean_val)}')

    ax.set_title('Distribution vs Tolerance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Measurement')
    ax.set_ylabel('Density')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Annotate Cg/Cgk
    ax.text(0.99, 0.97, f"Cg={format4(metrics['cg'])}\nCgk={format4(metrics['cgk'])}",
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_individuals_chart(values: pd.Series, metrics: Dict, output_path: str):
    data = values.dropna().astype(float).values
    mean_val = metrics['mean']
    sd = metrics['sd']

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(1, len(data) + 1)
    ax.plot(x, data, marker='o', color='#1f77b4', linewidth=1.5)
    ax.axhline(mean_val, color='green', linewidth=2, label=f'Mean={format4(mean_val)}')
    if sd > 0:
        ax.axhline(mean_val + 3 * sd, color='red', linestyle='--', label=f'+3σ={format4(mean_val + 3*sd)}')
        ax.axhline(mean_val - 3 * sd, color='red', linestyle='--', label=f'-3σ={format4(mean_val - 3*sd)}')
    ax.set_title('Individuals Chart (X)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Reading #')
    ax.set_ylabel('Measurement')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_moving_range_chart(values: pd.Series, output_path: str):
    data = values.dropna().astype(float).values
    if len(data) < 2:
        print(f"Skipping MR chart - insufficient data")
        return

    mr = np.abs(np.diff(data))
    mr_bar = float(mr.mean()) if mr.size > 0 else np.nan
    # MR chart constants for n=2
    d4 = 3.267  # For MR with subgroup size=2
    ucl = d4 * mr_bar
    lcl = 0.0

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(1, len(mr) + 1)
    ax.plot(x, mr, marker='o', color='#ff7f0e', linewidth=1.5, label='MR')
    ax.axhline(mr_bar, color='green', linewidth=2, label=f'MR̄={format4(mr_bar)}')
    ax.axhline(ucl, color='red', linestyle='--', label=f'UCL={format4(ucl)}')
    ax.axhline(lcl, color='red', linestyle='--', label=f'LCL={format4(lcl)}')
    ax.set_title('Moving Range Chart (n=2)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Interval #')
    ax.set_ylabel('MR')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_bias_chart(metrics: Dict, output_path: str):
    mean_val = metrics['mean']
    ci_low = metrics['ci_low']
    ci_high = metrics['ci_high']
    target = metrics['target']
    lsl = metrics['lsl']
    usl = metrics['usl']

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar([0], [mean_val], yerr=[[mean_val - ci_low], [ci_high - mean_val]], fmt='o', color='#1f77b4', capsize=6, label='Mean ± CI')
    ax.axhline(target, color='green', linewidth=2, label=f'Target={format4(target)}')
    ax.axhline(lsl, color='red', linestyle='--', linewidth=2, label=f'LSL={format4(lsl)}')
    ax.axhline(usl, color='red', linestyle='--', linewidth=2, label=f'USL={format4(usl)}')
    ax.set_xlim(-1, 1)
    ax.set_xticks([])
    ax.set_title('Bias vs Target (with CI)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Measurement')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_merged(values: pd.Series, metrics: Dict, measurement: str, output_path: str):
    data = values.dropna().astype(float).values
    mean_val = metrics['mean']
    sd = metrics['sd']
    lsl = metrics['lsl']
    usl = metrics['usl']
    target = metrics['target']

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Distribution (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(data, bins=min(30, max(8, int(np.sqrt(len(data))))), color='#1f77b4', alpha=0.6, density=True)
    xs = np.linspace(min(data.min(), lsl), max(data.max(), usl), 200)
    if sd > 0:
        ax1.plot(xs, stats.norm.pdf(xs, loc=mean_val, scale=sd), color='black', linewidth=2)
    ax1.axvline(target, color='green', linewidth=2)
    ax1.axvline(lsl, color='red', linestyle='--', linewidth=2)
    ax1.axvline(usl, color='red', linestyle='--', linewidth=2)
    ax1.axvline(mean_val, color='orange', linestyle=':', linewidth=2)
    ax1.set_title('Distribution vs Tolerance', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Individuals (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(1, len(data) + 1)
    ax2.plot(x, data, marker='o', color='#1f77b4', linewidth=1.5)
    ax2.axhline(mean_val, color='green', linewidth=2)
    if sd > 0:
        ax2.axhline(mean_val + 3 * sd, color='red', linestyle='--')
        ax2.axhline(mean_val - 3 * sd, color='red', linestyle='--')
    ax2.set_title('Individuals Chart (X)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # MR (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    if len(data) > 1:
        mr = np.abs(np.diff(data))
        mr_bar = float(mr.mean()) if mr.size > 0 else np.nan
        d4 = 3.267
        ucl = d4 * mr_bar
        lcl = 0.0
        x_mr = np.arange(1, len(mr) + 1)
        ax3.plot(x_mr, mr, marker='o', color='#ff7f0e', linewidth=1.5)
        ax3.axhline(mr_bar, color='green', linewidth=2)
        ax3.axhline(ucl, color='red', linestyle='--')
        ax3.axhline(lcl, color='red', linestyle='--')
    ax3.set_title('Moving Range Chart (n=2)', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # Bias (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.errorbar([0], [mean_val], yerr=[[mean_val - metrics['ci_low']], [metrics['ci_high'] - mean_val]], fmt='o', color='#1f77b4', capsize=6)
    ax4.axhline(target, color='green', linewidth=2)
    ax4.axhline(metrics['lsl'], color='red', linestyle='--', linewidth=2)
    ax4.axhline(metrics['usl'], color='red', linestyle='--', linewidth=2)
    ax4.set_xlim(-1, 1)
    ax4.set_xticks([])
    ax4.set_title('Bias vs Target (with CI)', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Type 1 Gage Study - {measurement}', fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved merged chart: {output_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Type 1 Gage Study - single part, single operator repeated readings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse and save cleaned data only
  python gage_rr_type1.py -p -f EMI_20um_SV.txt -o T1_

  # List available algorithms and components
  python gage_rr_type1.py --list -f EMI_20um_SV.txt

  # Run Type 1 with default tol=6σ and target=mean
  python gage_rr_type1.py -f EMI_20um_SV.txt --algo Solder_ThicknessN1_Layer3 --component C100_1

  # Run with explicit tolerance and target
  python gage_rr_type1.py -f EMI_20um_SV.txt --algo Solder_ThicknessN1_Layer3 --component C100_1 --tol 3.0 --target 40.0 -o T1_
        """
    )

    parser.add_argument('-f', '--file',
                        default='/Users/maxymhuang/GRR_plots/EMI_20um_SV.txt',
                        help='Input data file path (default: EMI_20um_SV.txt)')

    parser.add_argument('--algo', '--algorithm',
                        default='Solder_ThicknessN1_Layer3',
                        help='Measurement column to analyze')

    parser.add_argument('--component',
                        default=None,
                        help='Component (Comp_Name) to filter (default: first available)')

    parser.add_argument('--operator',
                        default=None,
                        help='Optional operator to filter IF an Operator column exists; otherwise ignored')

    parser.add_argument('--tol', type=float, default=None,
                        help='Total tolerance width. If omitted, defaults to sv*sd (6σ by default).')

    parser.add_argument('--target', type=float, default=None,
                        help='Target value. If omitted, defaults to sample mean of selected data.')

    parser.add_argument('--sv', '--study-var', type=float, default=6.0,
                        help='Study variation multiplier (default: 6.0)')

    parser.add_argument('--av', '--alpha', type=float, default=0.25,
                        help='Alpha value for confidence intervals (default: 0.25)')

    parser.add_argument('--list', action='store_true',
                        help='List available measurement columns and components and exit')

    parser.add_argument('-o', '--output-prefix', default='',
                        help='Prefix for output files (default: none)')

    parser.add_argument('-m', '--merge', action='store_true',
                        help='Merge all 4 charts into a single image (only outputs merged chart)')

    parser.add_argument('-p', '--parse', action='store_true',
                        help='Parse and save cleaned data to CSV file, then exit (no analysis)')

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 70)
    print("TYPE 1 GAGE STUDY")
    print("=" * 70)

    print(f"\nInput file: {args.file}")
    df = load_and_clean_data(args.file)

    # Parse-only mode
    if args.parse:
        prefix = args.output_prefix
        output_file = f"{prefix}parsed_data.csv"
        df.to_csv(output_file, index=False)
        print(f"\n{'='*70}")
        print("PARSED DATA SAVED")
        print(f"{'='*70}")
        print(f"\nCleaned data saved to: {output_file}")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nColumns: {list(df.columns)}")
        return

    # List mode
    # Identify measurement and components
    measurement_cols = [c for c in df.columns if c not in ['Comp_Name', 'Component'] and df[c].dtype in ['float64', 'int64']]
    components = sorted(df['Comp_Name'].unique().tolist())

    if args.list:
        print("\nAvailable measurement columns:")
        for i, col in enumerate(measurement_cols, 1):
            print(f"  {i:2d}. {col}")
        print("\nAvailable components (Comp_Name):")
        for i, comp in enumerate(components, 1):
            print(f"  {i:2d}. {comp}")
        return

    # Validate selection
    measurement = args.algo
    if measurement not in measurement_cols:
        print(f"\n❌ Error: '{measurement}' not found in data!")
        print("\nAvailable columns:")
        for col in measurement_cols:
            print(f"  - {col}")
        return

    component = args.component or (components[0] if components else None)
    if component is None:
        print("\n❌ Error: No components found in data.")
        return

    print(f"\nSelection:")
    print(f"  - Measurement: {measurement}")
    print(f"  - Component: {component}")
    if args.operator and 'Operator' in df.columns:
        print(f"  - Operator filter: {args.operator}")

    # Subset values
    subset = df[df['Comp_Name'] == component].copy()
    if args.operator and 'Operator' in subset.columns:
        subset = subset[subset['Operator'] == args.operator]

    values = subset[measurement].dropna()
    if len(values) < 2:
        print("\n❌ Error: Need at least 2 readings for the selected component/operator.")
        return

    # Compute metrics
    metrics = compute_type1_metrics(values, sv=args.sv, tol=args.tol, target=args.target, alpha=args.av)

    # Output prefix
    prefix = args.output_prefix

    # Summary CSV
    summary_df = create_type1_summary_df(metrics)
    summary_path = f"{prefix}type1_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")

    # JSON
    json_path = f"{prefix}type1_results.json"
    with open(json_path, 'w') as f:
        json.dump({'measurement': measurement, 'component': component, 'metrics': metrics}, f, indent=2)
    print(f"Results saved: {json_path}")

    # Plots
    if args.merge:
        plot_merged(values, metrics, measurement, f"{prefix}type1_merged.png")
    else:
        plot_distribution_vs_tolerance(values, metrics, f"{prefix}distribution_vs_tolerance.png")
        plot_individuals_chart(values, metrics, f"{prefix}individuals_chart.png")
        plot_moving_range_chart(values, f"{prefix}moving_range_chart.png")
        plot_bias_chart(metrics, f"{prefix}bias_plot.png")

    print(f"\n{'='*70}")
    print("TYPE 1 STUDY COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


