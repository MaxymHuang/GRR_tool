"""
Gage R&R Analysis Application (Type 2 — crossed ANOVA)
Performs ANOVA-based Gage R&R with automated or column-mapped study design
and generates visualization charts plus summary tables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import json
import warnings
import argparse
import sys
from data_parser import (
    load_and_clean_data,
    apply_component_filters,
    remove_outliers_iqr,
    get_measurement_columns,
    apply_study_design,
)
from grr_tool.msa import (
    perform_anova_grr,
    create_anova_table,
    create_variance_summary_df,
    DesignMode,
)
from grr_tool.msa.gage_rr import ReproMode
from grr_tool.msa.xbar_r import perform_xbar_r
from grr_tool.msa.nested import perform_nested_grr
from grr_tool.msa.reml_check import reml_variance_components

warnings.filterwarnings('ignore')


def plot_components_of_variation(results: Dict, output_path: str):
    """
    Generate Components of Variation bar chart.
    
    Chart 1: Bar chart showing %SV and %Contribution side-by-side for each component
    """
    if results is None:
        print(f"Skipping plot for {output_path} - no valid results")
        return
    
    pct_sv = results['pct_study_var']
    pct_contrib = results['pct_contribution']
    
    components = ['Gage R&R', 'Repeat', 'Reproduce', 'Part-to-Part']
    contribution_values = [
        pct_contrib['grr'],
        pct_contrib['repeatability'],
        pct_contrib['reproducibility'],
        pct_contrib['part']
    ]
    study_var_values = [
        pct_sv['grr'],
        pct_sv['repeatability'],
        pct_sv['reproducibility'],
        pct_sv['part']
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(components))
    width = 0.35
    
    # Create side-by-side bars
    bars1 = ax.bar(x - width/2, contribution_values, width, 
                   label='% Contribution', color='orange', alpha=0.8)
    bars2 = ax.bar(x + width/2, study_var_values, width, 
                   label='% Study Var', color='green', alpha=0.8)
    
    ax.set_ylabel('Percent', fontsize=12)
    ax.set_title(f"Components of Variation", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:  # Only show labels for visible bars
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_algorithm_by_component(df: pd.DataFrame, measurement_col: str, output_path: str):
    """
    Generate Algorithm values by Component chart.
    
    Chart 2: Line chart showing measurement values across distinct component names
    """
    # Filter valid data
    data = df[['Component', measurement_col]].copy()
    data = data.dropna()
    
    if len(data) == 0:
        print(f"Skipping plot for {output_path} - no valid data")
        return
    
    # Get unique components and their mean values
    component_means = data.groupby('Component')[measurement_col].mean().sort_index()
    
    # Limit to reasonable number of components for readability
    if len(component_means) > 50:
        print(f"Note: Limiting display to first 50 components (out of {len(component_means)})")
        component_means = component_means.head(50)
    
    components = component_means.index.tolist()
    values = component_means.values
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create line chart
    ax.plot(components, values, marker='o', linewidth=2, markersize=6, 
            color='#1f77b4', markerfacecolor='white', markeredgewidth=2, 
            markeredgecolor='#1f77b4')
    
    ax.set_ylabel(measurement_col, fontsize=12)
    ax.set_xlabel('Comp_Name', fontsize=12)
    ax.set_title(f"{measurement_col} by Comp_Name", fontsize=14, fontweight='bold')
    ax.grid(axis='both', alpha=0.3)
    
    # Add value labels on points (every nth point to avoid crowding)
    label_frequency = max(1, len(components) // 20)  # Label ~20 points max
    for i in range(0, len(components), label_frequency):
        ax.text(i, values[i], f'{values[i]:.1f}',
                ha='center', va='bottom', fontsize=8, rotation=45)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_s_chart_by_operator(df: pd.DataFrame, measurement_col: str, output_path: str):
    """
    Generate S Chart by Operator.
    
    Chart 3: Control chart showing standard deviation by operator with UCL
    """
    # Filter valid data
    data = df[['Part', 'Operator', measurement_col]].copy()
    data = data.dropna()
    
    if len(data) == 0:
        print(f"Skipping S chart for {measurement_col} - no valid data")
        return
    
    # Check if we have replicates within part-operator groups
    group_sizes = data.groupby(['Operator', 'Part']).size()
    has_replicates = (group_sizes > 1).any()
    
    if not has_replicates:
        # Alternative: compute moving range between consecutive measurements per operator
        print(f"Note: Using moving range method for S chart (no replicates within part-operator groups)")
        
        # Calculate moving ranges for each operator
        mr_data = []
        for operator in data['Operator'].unique():
            op_data = data[data['Operator'] == operator].sort_index()
            measurements = op_data[measurement_col].values
            if len(measurements) > 1:
                moving_ranges = np.abs(np.diff(measurements))
                for i, mr in enumerate(moving_ranges):
                    mr_data.append({'Operator': operator, 'Index': i, 'MR': mr})
        
        mr_df = pd.DataFrame(mr_data)
        
        # Calculate mean moving range, UCL, and LCL
        mr_bar = mr_df['MR'].mean()
        # For moving range with n=2, d2=1.128
        s_bar = mr_bar / 1.128
        # UCL and LCL for moving range chart
        # D4 = 3.267, D3 = 0 for n=2 (LCL is typically 0 for small sample sizes)
        ucl = 3.267 * mr_bar / 1.128  # D4 constant for n=2
        lcl = 0  # D3 = 0 for n=2, so LCL is 0
        
        s_values = mr_df.rename(columns={'MR': 'S'})
    else:
        # Calculate standard deviation for each part-operator combination
        s_values = data.groupby(['Operator', 'Part'])[measurement_col].std().reset_index()
        s_values.columns = ['Operator', 'Part', 'S']
        s_values = s_values.dropna()
        
        # Calculate overall mean S, UCL, and LCL
        s_bar = s_values['S'].mean()
        n = data.groupby(['Operator', 'Part']).size().mean()  # Average sample size per group
        
        # Control chart constants for S chart
        # UCL = S̄ × B4, LCL = S̄ × B3
        # For approximation: UCL = S̄ × (1 + 3 / √(2(n − 1)))
        # LCL = S̄ × (1 - 3 / √(2(n − 1))), but cannot be negative
        if n > 1:
            factor = 3 / np.sqrt(2 * (n - 1))
            ucl = s_bar * (1 + factor)
            lcl = max(0, s_bar * (1 - factor))  # LCL cannot be negative
        else:
            ucl = s_bar * 2
            lcl = 0
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot points by operator
    operators = sorted(s_values['Operator'].unique())
    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
    
    x_pos = 0
    x_ticks = []
    x_labels = []
    operator_boundaries = []
    
    for i, operator in enumerate(operators):
        op_data = s_values[s_values['Operator'] == operator].copy()
        n_points = len(op_data)
        
        x_positions = np.arange(x_pos, x_pos + n_points)
        ax.scatter(x_positions, op_data['S'].values, 
                  color=colors.get(operator, 'gray'), 
                  label=f'Operator {operator}', s=50, alpha=0.7)
        
        # Connect points with lines
        ax.plot(x_positions, op_data['S'].values, 
               color=colors.get(operator, 'gray'), alpha=0.3, linewidth=1)
        
        x_ticks.append(x_pos + n_points / 2)
        x_labels.append(f'Operator {operator}')
        
        if i < len(operators) - 1:
            operator_boundaries.append(x_pos + n_points - 0.5)
        
        x_pos += n_points
    
    # Draw horizontal lines for S̄, UCL, and LCL
    ax.axhline(y=s_bar, color='green', linestyle='-', linewidth=2, label=f'S̄ = {s_bar:.4f}')
    ax.axhline(y=ucl, color='red', linestyle='--', linewidth=2, label=f'UCL = {ucl:.4f}')
    ax.axhline(y=lcl, color='red', linestyle='--', linewidth=2, label=f'LCL = {lcl:.4f}')
    
    # Draw vertical lines to separate operators
    for boundary in operator_boundaries:
        ax.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_ylabel('Sample Standard Deviation (S)', fontsize=12)
    ax.set_xlabel('Operator', fontsize=12)
    ax.set_title(f"S Chart by Operator\n{measurement_col}", fontsize=14, fontweight='bold')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_algo_by_operator(df: pd.DataFrame, measurement_col: str, output_path: str):
    """
    Generate Algorithm Type by Operator chart.
    
    Chart 4: Box-and-whisker plot showing measurement distribution by operator
    """
    # Filter valid data
    data = df[['Operator', measurement_col]].copy()
    data = data.dropna()
    
    if len(data) == 0:
        print(f"Skipping operator plot for {measurement_col} - no valid data")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for box plot
    operators = sorted(data['Operator'].unique())
    plot_data = [data[data['Operator'] == op][measurement_col].values for op in operators]
    
    # Create box plot
    bp = ax.boxplot(plot_data, labels=operators, patch_artist=True,
                    showmeans=True, meanline=True,
                    widths=0.6)
    
    # Color the boxes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Customize the plot
    for median in bp['medians']:
        median.set(color='red', linewidth=2)
    
    for mean in bp['means']:
        mean.set(color='black', linewidth=2)
    
    ax.set_xlabel('OP', fontsize=12)
    ax.set_ylabel(measurement_col, fontsize=12)
    ax.set_title(f"{measurement_col} by OP", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    for i, op in enumerate(operators):
        op_data = data[data['Operator'] == op][measurement_col]
        mean_val = op_data.mean()
        median_val = op_data.median()
        ax.text(i+1, ax.get_ylim()[1]*0.98, f'μ={mean_val:.2f}\nM={median_val:.2f}',
                ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_merged_charts(df: pd.DataFrame, results: Dict, measurement_col: str, output_path: str):
    """
    Generate merged chart combining all 4 visualizations into a single image.
    
    Creates a 2x2 grid with:
    - Top-left: Components of Variation
    - Top-right: Algorithm by Component
    - Bottom-left: S Chart by Operator
    - Bottom-right: Algorithm Type by Operator
    """
    fig = plt.figure(figsize=(20, 16))
    
    # Create 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Components of Variation (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    pct_sv = results['pct_study_var']
    pct_contrib = results['pct_contribution']
    
    components = ['Gage R&R', 'Repeat', 'Reproduce', 'Part-to-Part']
    contribution_values = [
        pct_contrib['grr'], pct_contrib['repeatability'],
        pct_contrib['reproducibility'], pct_contrib['part']
    ]
    study_var_values = [
        pct_sv['grr'], pct_sv['repeatability'],
        pct_sv['reproducibility'], pct_sv['part']
    ]
    
    x = np.arange(len(components))
    width = 0.35
    bars1 = ax1.bar(x - width/2, contribution_values, width, 
                    label='% Contribution', color='orange', alpha=0.8)
    bars2 = ax1.bar(x + width/2, study_var_values, width, 
                    label='% Study Var', color='green', alpha=0.8)
    
    ax1.set_ylabel('Percent', fontsize=11)
    ax1.set_title("Components of Variation", fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(components)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 110)
    ax1.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Algorithm by Component (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    data = df[['Component', measurement_col]].copy().dropna()
    component_means = data.groupby('Component')[measurement_col].mean().sort_index()
    
    if len(component_means) > 50:
        component_means = component_means.head(50)
    
    components_list = component_means.index.tolist()
    values = component_means.values
    
    ax2.plot(components_list, values, marker='o', linewidth=2, markersize=5, 
            color='#1f77b4', markerfacecolor='white', markeredgewidth=2, 
            markeredgecolor='#1f77b4')
    
    ax2.set_ylabel(measurement_col, fontsize=11)
    ax2.set_xlabel('Comp_Name', fontsize=11)
    ax2.set_title(f"{measurement_col} by Comp_Name", fontsize=12, fontweight='bold')
    ax2.grid(axis='both', alpha=0.3)
    
    label_frequency = max(1, len(components_list) // 15)
    for i in range(0, len(components_list), label_frequency):
        ax2.text(i, values[i], f'{values[i]:.1f}',
                ha='center', va='bottom', fontsize=7, rotation=45)
    
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90, ha='right', fontsize=7)
    
    # 3. S Chart by Operator (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    data_s = df[['Part', 'Operator', measurement_col]].copy().dropna()
    
    if len(data_s) > 0:
        group_sizes = data_s.groupby(['Operator', 'Part']).size()
        has_replicates = (group_sizes > 1).any()
        
        if not has_replicates:
            mr_data = []
            for operator in data_s['Operator'].unique():
                op_data = data_s[data_s['Operator'] == operator].sort_index()
                measurements = op_data[measurement_col].values
                if len(measurements) > 1:
                    moving_ranges = np.abs(np.diff(measurements))
                    for i, mr in enumerate(moving_ranges):
                        mr_data.append({'Operator': operator, 'Index': i, 'MR': mr})
            
            mr_df = pd.DataFrame(mr_data)
            mr_bar = mr_df['MR'].mean()
            s_bar = mr_bar / 1.128
            ucl = 3.267 * mr_bar / 1.128
            lcl = 0
            s_values = mr_df.rename(columns={'MR': 'S'})
        else:
            s_values = data_s.groupby(['Operator', 'Part'])[measurement_col].std().reset_index()
            s_values.columns = ['Operator', 'Part', 'S']
            s_values = s_values.dropna()
            s_bar = s_values['S'].mean()
            n = data_s.groupby(['Operator', 'Part']).size().mean()
            if n > 1:
                factor = 3 / np.sqrt(2 * (n - 1))
                ucl = s_bar * (1 + factor)
                lcl = max(0, s_bar * (1 - factor))
            else:
                ucl = s_bar * 2
                lcl = 0
        
        operators = sorted(s_values['Operator'].unique())
        colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
        
        x_pos = 0
        x_ticks = []
        x_labels = []
        operator_boundaries = []
        
        for i, operator in enumerate(operators):
            op_data = s_values[s_values['Operator'] == operator].copy()
            n_points = len(op_data)
            x_positions = np.arange(x_pos, x_pos + n_points)
            ax3.scatter(x_positions, op_data['S'].values, 
                       color=colors.get(operator, 'gray'), 
                       label=f'Operator {operator}', s=30, alpha=0.7)
            ax3.plot(x_positions, op_data['S'].values, 
                    color=colors.get(operator, 'gray'), alpha=0.3, linewidth=1)
            
            x_ticks.append(x_pos + n_points / 2)
            x_labels.append(f'OP {operator}')
            if i < len(operators) - 1:
                operator_boundaries.append(x_pos + n_points - 0.5)
            x_pos += n_points
        
        ax3.axhline(y=s_bar, color='green', linestyle='-', linewidth=1.5, label=f'S̄={s_bar:.3f}')
        ax3.axhline(y=ucl, color='red', linestyle='--', linewidth=1.5, label=f'UCL={ucl:.3f}')
        ax3.axhline(y=lcl, color='red', linestyle='--', linewidth=1.5, label=f'LCL={lcl:.3f}')
        
        for boundary in operator_boundaries:
            ax3.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5)
        
        ax3.set_ylabel('Sample Std Dev', fontsize=11)
        ax3.set_xlabel('Operator', fontsize=11)
        ax3.set_title("S Chart by Operator", fontsize=12, fontweight='bold')
        ax3.set_xticks(x_ticks)
        ax3.set_xticklabels(x_labels)
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. Algorithm Type by Operator (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    data_box = df[['Operator', measurement_col]].copy().dropna()
    
    if len(data_box) > 0:
        operators = sorted(data_box['Operator'].unique())
        plot_data = [data_box[data_box['Operator'] == op][measurement_col].values for op in operators]
        
        bp = ax4.boxplot(plot_data, labels=operators, patch_artist=True,
                        showmeans=True, meanline=True, widths=0.6)
        
        colors_box = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        for median in bp['medians']:
            median.set(color='red', linewidth=2)
        for mean in bp['means']:
            mean.set(color='black', linewidth=2)
        
        ax4.set_xlabel('OP', fontsize=11)
        ax4.set_ylabel(measurement_col, fontsize=11)
        ax4.set_title(f"{measurement_col} by OP", fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        for i, op in enumerate(operators):
            op_data = data_box[data_box['Operator'] == op][measurement_col]
            mean_val = op_data.mean()
            median_val = op_data.median()
            ax4.text(i+1, ax4.get_ylim()[1]*0.97, f'μ={mean_val:.2f}\nM={median_val:.2f}',
                    ha='center', va='top', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Add overall title
    fig.suptitle(f'Gage R&R Analysis - {measurement_col}', 
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved merged chart: {output_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Gage R&R Analysis - ANOVA-based measurement system evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python gage_rr_analysis.py
  
  # Specify input file and algorithm
  python gage_rr_analysis.py -f data.txt --algo Solder_Area_Layer1
  
  # Custom study variation and alpha
  python gage_rr_analysis.py --sv 5.15 --av 0.05 --algo Solder_ThicknessN1_Layer2
  
  # List available algorithms
  python gage_rr_analysis.py --list
  
  # Parse data and save to CSV without analysis
  python gage_rr_analysis.py -p
  python gage_rr_analysis.py -p -o MyData_
        """
    )
    
    parser.add_argument('-f', '--file', 
                       default='/Users/maxymhuang/GRR_plots/EMI_20um_SV.txt',
                       help='Input data file path (default: EMI_20um_SV.txt)')
    
    parser.add_argument('--algo', '--algorithm',
                       default='Solder_ThicknessN1_Layer3',
                       help='Measurement column to analyze (default: Solder_ThicknessN1_Layer3)')
    
    parser.add_argument('--sv', '--study-var',
                       type=float,
                       default=6.0,
                       help='Study variation multiplier (default: 6.0)')
    
    parser.add_argument('--av', '--alpha',
                       type=float,
                       default=0.025,
                       help='Alpha value for confidence intervals (default: 0.025)')
    
    parser.add_argument('--list',
                       action='store_true',
                       help='List available measurement columns and exit')
    
    parser.add_argument('-o', '--output-prefix',
                       default='',
                       help='Prefix for output files (default: none)')
    
    parser.add_argument('-m', '--merge',
                       action='store_true',
                       help='Merge all 4 charts into a single image (only outputs merged chart)')
    
    parser.add_argument('-p', '--parse',
                       action='store_true',
                       help='Parse and save cleaned data to CSV file, then exit (no analysis)')
    
    parser.add_argument('--exclude',
                       nargs='*',
                       default=[],
                       help='Components (Comp_Name) to exclude; accept space or comma separated values')

    parser.add_argument('--display-comp',
                       action='store_true',
                       help='Preview components (Comp_Name) in a 1x60 row and exit')

    parser.add_argument('--include',
                       nargs='*',
                       default=[],
                       help='Components (Comp_Name) to include; accept space or comma separated values')

    parser.add_argument('--rm', '--remove-outliers',
                       action='store_true',
                       help='Remove outliers using IQR (1.5*IQR) for the selected measurement before analysis')
    parser.add_argument('-n', '--num-measurements',
                       type=int,
                       default=0,
                       help='Number of measurements to include per component (applies before operator assignment). Use 0 for all.')
    parser.add_argument('--operator',
                       type=int,
                       default=3,
                       help='Number of operators to simulate (1–10). Default: 3')

    parser.add_argument('--tol', type=float, default=None,
                       help='Total tolerance width for %%GRR(Tol) metrics')

    parser.add_argument('--design',
                       choices=['sequential', 'columns', 'comp_name'],
                       default='sequential',
                       help='Study design: sequential row assignment, CSV columns, or Comp_Name as parts')

    parser.add_argument('--part-col', default=None,
                       help='Column name for Part (with --design columns)')

    parser.add_argument('--operator-col', default=None,
                       help='Column name for Operator (with --design columns)')

    parser.add_argument('--replicate-col', default=None,
                       help='Optional replicate/trial column (with --design columns)')

    parser.add_argument('--repro-mode',
                       choices=['operator_only', 'operator_plus_interaction'],
                       default='operator_only',
                       help='Reproducibility: operator only (default) or operator+interaction')

    parser.add_argument('--method',
                       choices=['anova', 'xbar_r', 'nested'],
                       default='anova',
                       help='Analysis method (default: anova)')

    parser.add_argument('--reml-check', action='store_true',
                       help='Run optional statsmodels REML cross-check (requires statsmodels)')

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("="*70)
    print("GAGE R&R (TYPE 2) ANALYSIS")
    print("="*70)
    
    # Load and clean data
    print(f"\nInput file: {args.file}")
    df = load_and_clean_data(args.file)
    
    # Apply component filters
    df = apply_component_filters(df, include=args.include, exclude=args.exclude)
    
    # Optionally limit number of measurements per component prior to operator assignment
    if args.num_measurements and args.num_measurements > 0 and 'Comp_Name' in df.columns:
        df = df.groupby('Comp_Name', group_keys=False).apply(lambda g: g.head(args.num_measurements)).reset_index(drop=True)
    design_mode = DesignMode(args.design)
    df = apply_study_design(
        df,
        mode=design_mode,
        n_operators=args.operator,
        part_col=args.part_col,
        operator_col=args.operator_col,
        replicate_col=args.replicate_col,
    )

    # Preview components and exit if requested
    if args.display_comp:
        comps = sorted(df['Comp_Name'].dropna().astype(str).unique().tolist())
        preview = comps[:60]
        print("\nComponents (up to 60):")
        if preview:
            print("  " + " | ".join(preview))
        else:
            print("  (none)")
        return
    
    # If -p flag is set, save parsed data and exit
    if args.parse:
        prefix = args.output_prefix
        output_file = f'{prefix}parsed_data.csv'
        df.to_csv(output_file, index=False)
        print(f"\n{'='*70}")
        print("PARSED DATA SAVED")
        print(f"{'='*70}")
        print(f"\nCleaned data saved to: {output_file}")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nColumns: {list(df.columns)}")
        return
    
    # Identify measurement columns (all numeric columns except Comp_Name)
    measurement_cols = get_measurement_columns(df)
    
    print(f"\nFound {len(measurement_cols)} measurement columns")
    
    # If --list flag, show available columns and exit
    if args.list:
        print("\nAvailable measurement columns:")
        for i, col in enumerate(measurement_cols, 1):
            print(f"  {i:2d}. {col}")
        return
    
    print(f"Measurement columns: {measurement_cols[:5]}... (showing first 5)")
    
    # Use target measurement from CLI
    target_measurement = args.algo
    
    if target_measurement not in measurement_cols:
        print(f"\n❌ Error: '{target_measurement}' not found in data!")
        print("\nAvailable columns:")
        for col in measurement_cols:
            print(f"  - {col}")
        print("\nUse --list to see all available columns")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {target_measurement}")
    print(f"{'='*70}")
    print(f"Parameters:")
    print(f"  - Study Variation: {args.sv}")
    print(f"  - Alpha Value: {args.av}")
    print(f"  - Tolerance: {args.tol if args.tol else '(not set)'}")
    print(f"  - Design: {args.design}")
    print(f"  - Method: {args.method}")
    print(f"  - Repro Mode: {args.repro_mode}")
    print(f"  - Output Prefix: '{args.output_prefix}'")
    print(f"  - Operators: {max(1, min(int(getattr(args, 'operator', 3) or 1), 10))}")
    
    # Optional outlier removal on the selected measurement
    if args.rm:
        before = len(df)
        df, removed = remove_outliers_iqr(df, target_measurement)
        after = len(df)
        print(f"\nOutlier removal (IQR) for {target_measurement}: removed {removed} rows (from {before} to {after})")

    repro_mode: ReproMode = args.repro_mode  # type: ignore[assignment]
    tol = args.tol

    if args.method == 'xbar_r':
        alt = perform_xbar_r(df, target_measurement, study_var=args.sv, tolerance=tol)
        if alt is None:
            print("Xbar-R analysis failed.")
            return
        print("\nXbar-R method results (%Study Var):")
        for k, v in alt['pct_study_var'].items():
            print(f"  {k}: {v:.2f}%")
        if tol and 'pct_tolerance' in alt:
            print(f"  GRR %Tol: {alt['pct_tolerance']['grr']:.2f}%")
        return

    if args.method == 'nested':
        alt = perform_nested_grr(df, target_measurement, study_var=args.sv, tolerance=tol)
        if alt is None:
            print("Nested GRR analysis failed.")
            return
        print("\nNested GRR results (%Study Var):")
        for k, v in alt['pct_study_var'].items():
            print(f"  {k}: {v:.2f}%")
        return

    results = perform_anova_grr(
        df,
        target_measurement,
        study_var=args.sv,
        tolerance=tol,
        alpha=args.av,
        repro_mode=repro_mode,
    )

    if args.reml_check:
        reml = reml_variance_components(df, target_measurement)
        if reml:
            print("\nREML cross-check (statsmodels):")
            for k, v in reml.items():
                print(f"  {k}: {v:.6g}")
        else:
            print("\nREML cross-check skipped (statsmodels not installed or fit failed).")

    if results:
        anova_table = create_anova_table(results)
        print(f"\n{target_measurement} - Gage R&R (Type 2) Results:")
        print(anova_table.to_string(index=False))
        print(f"\nNumber of Distinct Categories = {results['ndc']}")
        if results.get('acceptance'):
            acc = results['acceptance']
            print(f"  %SV verdict: {acc.get('pct_sv_verdict')}  |  NDC verdict: {acc.get('ndc_verdict')}")
            if 'pct_tol_verdict' in acc:
                print(f"  %Tol verdict: {acc.get('pct_tol_verdict')} ({acc.get('pct_tol_grr', 0):.2f}%)")

        full_anova = results.get('full_anova_table')
        if full_anova is not None:
            print("\nFull ANOVA Table:")
            print(full_anova.to_string(index=False))

        prefix = args.output_prefix
        anova_table.to_csv(f'{prefix}anova_table.csv', index=False)
        if full_anova is not None:
            full_anova.to_csv(f'{prefix}full_anova_table.csv', index=False)

        variance_summary = create_variance_summary_df(results)
        print("\nVariance Components Summary:")
        print(variance_summary.to_string(index=False))
        
        # Save variance summary to CSV
        variance_summary.to_csv(f'{prefix}variance_components.csv', index=False)
        
        # Verification against HTML report
        print(f"\n{'='*70}")
        print("VERIFICATION AGAINST HTML REPORT")
        print(f"{'='*70}")
        print(f"Expected Repeatability StdDev: 0.5208685")
        print(f"Calculated Repeatability StdDev: {results['std_dev']['repeatability']:.7f}")
        print(f"\nExpected %SV: 4.96%")
        print(f"Calculated %SV: {results['pct_study_var']['grr']:.2f}%")
        print(f"\nExpected NDC: 28")
        print(f"Calculated NDC: {results['ndc']}")
        
        # Generate visualizations
        print(f"\n{'='*70}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*70}")
        
        if args.merge:
            # Generate merged chart only
            plot_merged_charts(df, results, target_measurement, f'{prefix}merged_analysis.png')
        else:
            # Generate individual charts
            plot_components_of_variation(results, f'{prefix}components_of_variation.png')
            plot_algorithm_by_component(df, target_measurement, f'{prefix}algorithm_by_component.png')
            plot_s_chart_by_operator(df, target_measurement, f'{prefix}s_chart_by_operator.png')
            plot_algo_by_operator(df, target_measurement, f'{prefix}algo_by_operator.png')
        
        # Save results to JSON
        results_dict = {
            'study_type': results.get('study_type', 'gage_rr_type2'),
            'measurement': results['measurement'],
            'statistics': {
                'n_parts': results['n_parts'],
                'n_operators': results['n_operators'],
                'n_measurements': results['n_measurements'],
                'ndc': results['ndc'],
            },
            'variance_components': results['variance_components'],
            'std_dev': results['std_dev'],
            'study_var': results['study_var'],
            'pct_contribution': results['pct_contribution'],
            'pct_study_var': results['pct_study_var'],
            'acceptance': results.get('acceptance'),
        }
        if results.get('pct_tolerance'):
            results_dict['pct_tolerance'] = results['pct_tolerance']
        if results.get('tolerance'):
            results_dict['tolerance'] = results['tolerance']
        
        with open(f'{prefix}grr_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to {prefix}grr_results.json")
        
        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print("\nOutput files generated:")
        
        if args.merge:
            print(f"  - {prefix}merged_analysis.png")
        else:
            print(f"  - {prefix}components_of_variation.png")
            print(f"  - {prefix}algorithm_by_component.png")
            print(f"  - {prefix}s_chart_by_operator.png")
            print(f"  - {prefix}algo_by_operator.png")
        
        print(f"  - {prefix}anova_table.csv")
        if full_anova is not None:
            print(f"  - {prefix}full_anova_table.csv")
        print(f"  - {prefix}variance_components.csv")
        print(f"  - {prefix}grr_results.json")
    else:
        print(f"Could not perform analysis for {target_measurement}")


if __name__ == "__main__":
    main()

