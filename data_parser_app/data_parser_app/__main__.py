"""CLI entry for ``python -m data_parser_app``."""

import argparse

from data_parser_app.parser import (
    apply_component_filters,
    assign_operators_sequential,
    load_and_clean_data,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Parse and prepare measurement data with optional component filters and operator assignment',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-f', '--file',
                        required=True,
                        help='Input data file path (.txt multi-section or .csv)')
    parser.add_argument('--include',
                        nargs='*',
                        default=[],
                        help='Components (Comp_Name) to include; space or comma separated')
    parser.add_argument('--exclude',
                        nargs='*',
                        default=[],
                        help='Components (Comp_Name) to exclude; space or comma separated')
    parser.add_argument('--operator',
                        type=int,
                        default=3,
                        help='Number of operators to simulate (1–10). Default: 3')
    parser.add_argument('-o', '--output',
                        default='',
                        help='Output CSV file path. If omitted, no file is written')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print("=" * 70)
    print("DATA PARSER")
    print("=" * 70)
    print(f"\nInput file: {args.file}")
    df = load_and_clean_data(args.file)
    df = apply_component_filters(df, include=args.include, exclude=args.exclude)
    df = assign_operators_sequential(df, n_operators=args.operator)
    print(f"\nFinal shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved CSV to: {args.output}")


if __name__ == '__main__':
    main()
