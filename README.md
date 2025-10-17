# Gage R&R Analysis Application

A Python application for performing ANOVA-based Gage R&R (Repeatability and Reproducibility) analysis on measurement data with automated operator assignment and comprehensive visualization.

## Features

- **Automated Data Cleaning**: Removes redundant headers, empty columns/rows, and irrelevant fields
- **Sequential Operator Assignment**: Automatically assigns operators A, B, C sequentially to measurement rows
- **ANOVA-Based Variance Decomposition**: Calculates variance components for Repeatability, Reproducibility, Part-to-Part variation
- **Four Comprehensive Charts**:
  1. Components of Variation - Bar chart showing percent contribution
  2. Algorithm by Component - Standard deviation comparison across components
  3. S Chart by Operator - Control chart with UCL and centerline
  4. Algorithm Type by Operator - Distribution plots showing measurement patterns

## Installation

### Using uv (recommended)

```bash
uv pip install -r requirements.txt
```

### Using pip

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
# Basic usage with default settings
python gage_rr_analysis.py

# Show help and available options
python gage_rr_analysis.py --help

# List all available measurement columns
python gage_rr_analysis.py --list

# Specify measurement algorithm
python gage_rr_analysis.py --algo Solder_Area_Layer1

# Custom study variation multiplier (default: 6.0)
python gage_rr_analysis.py --sv 5.15

# Custom alpha value for confidence intervals (default: 0.25)
python gage_rr_analysis.py --av 0.05

# Specify input file
python gage_rr_analysis.py -f /path/to/data.txt

# Add output file prefix
python gage_rr_analysis.py --algo Solder_ThicknessN1_Layer1 -o Layer1_

# Merge all charts into a single image
python gage_rr_analysis.py --algo Solder_Area_Layer2 -m

# Parse and save cleaned data to CSV (no analysis)
python gage_rr_analysis.py -p
python gage_rr_analysis.py -p -o MyData_

# Combine multiple options
python gage_rr_analysis.py -f data.txt --algo Solder_Area_Layer2 --sv 5.15 --av 0.05 -o output_
```

### Command Line Arguments

| Argument | Alias | Default | Description |
|----------|-------|---------|-------------|
| `-f FILE` | `--file` | `EMI_20um_SV.txt` | Input data file path |
| `--algo ALGO` | `--algorithm` | `Solder_ThicknessN1_Layer3` | Measurement column to analyze |
| `--sv SV` | `--study-var` | `6.0` | Study variation multiplier (typically 5.15 or 6.0) |
| `--av AV` | `--alpha` | `0.25` | Alpha value for confidence intervals |
| `-o PREFIX` | `--output-prefix` | `` | Prefix for output files |
| `-m` | `--merge` | | Merge all 4 charts into a single image |
| `-p` | `--parse` | | Parse and save cleaned data to CSV file, then exit (no analysis) |
| `--list` | | | List available measurement columns and exit |

### Basic Execution

The script will:
1. Load and clean data from the input file
2. Assign operators (A, B, C) sequentially to measurements
3. Perform ANOVA-based Gage R&R analysis
4. Generate four PNG visualization files
5. Export results to CSV and JSON formats

### Output Files

The application generates the following files:

#### Visualizations (PNG)
- `components_of_variation.png` - Variance component breakdown
- `algorithm_by_component.png` - Standard deviation by component
- `s_chart_by_operator.png` - Statistical process control chart
- `algo_by_operator.png` - Operator measurement distributions
- `merged_analysis.png` - All 4 charts combined (when using `-m` flag)

#### Data Files
- `anova_table.csv` - ANOVA summary table (similar to HTML report format)
- `variance_components.csv` - Detailed variance component breakdown
- `grr_results.json` - Complete analysis results in JSON format
- `parsed_data.csv` - Cleaned and formatted raw data (when using `-p` flag)

## Methodology

### ANOVA-Based Gage R&R

The application uses Type I ANOVA to decompose measurement variance into:

- **Repeatability (Equipment Variation)**: Within-operator variation
- **Reproducibility (Operator Variation)**: Between-operator variation including interaction effects
- **Part-to-Part Variation**: Variation due to actual part differences
- **Total Variation**: Overall measurement system variation

### Variance Components

```
Var(Total) = Var(Repeatability) + Var(Reproducibility) + Var(Part)
Var(GR&R) = Var(Repeatability) + Var(Reproducibility)
```

### Number of Distinct Categories (NDC)

```
NDC = floor(1.41 × σ_part / σ_GR&R)
```

**Interpretation**:
- NDC ≥ 5: Acceptable measurement system
- NDC < 5: Measurement system needs improvement

### %Study Variation (%SV)

```
%SV = (6σ_component / 6σ_total) × 100%
```

**Acceptance Criteria**:
- %SV < 10%: Excellent
- 10% ≤ %SV < 30%: Acceptable
- %SV ≥ 30%: Unacceptable (measurement system needs improvement)

## Data Format

### Input File Requirements

- Tab-separated text file
- First column: `Comp_Name` (part identifier)
- Subsequent columns: Numeric measurement values
- The script automatically handles:
  - Redundant header rows
  - Empty columns and rows
  - Non-numeric data conversion

### Columns Removed During Cleaning

- Box_Name
- Subtype_NO
- BoardIn_NO
- Scan_NO
- CAD_X, CAD_Y
- CAD_Width, CAD_Height
- Board_Side
- Any empty columns

## Verification

The script includes verification against reference HTML report values:

For `Solder_ThicknessN1_Layer3`:
- Expected Repeatability StdDev: 0.5208685
- Expected %SV: 4.96%
- Expected NDC: 28

Results are displayed in the console output for comparison.

## Customization

### Analyzing Different Measurements

Use the `--algo` flag to specify which measurement to analyze:

```bash
python gage_rr_analysis.py --algo Solder_Area_Layer1
```

To see all available measurements:

```bash
python gage_rr_analysis.py --list
```

### Study Variation Multiplier

The study variation (SV) multiplier determines the range of measurement spread:
- **6.0σ** (default): Captures 99.73% of variation (6 sigma approach)
- **5.15σ**: Common in automotive industry (AIAG MSA standard)

```bash
python gage_rr_analysis.py --sv 5.15
```

## Requirements

- Python 3.8+
- pandas ≥ 2.0.0
- numpy ≥ 1.24.0
- matplotlib ≥ 3.7.0
- scipy ≥ 1.10.0

## License

This application is provided as-is for quality control and measurement system analysis purposes.

## References

- AIAG Measurement Systems Analysis (MSA) Manual
- ISO 5725 - Accuracy (trueness and precision) of measurement methods and results
- ASTM E2782 - Standard Guide for Measurement Systems Analysis

