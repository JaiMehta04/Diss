"""Randomly sample rows from a CSV and save to a new file.

Usage examples:
    python sample_csv.py --input combined_station_data_with_coords.csv --n 500 --output outputs/sample_combined.csv
    python sample_csv.py --input band_stats.csv --frac 0.1 --output outputs/sample_band_stats.csv

Notes:
- By default, sampling is without replacement.
- Use --frac to sample a fraction of rows (0<frac<=1) instead of a fixed count.
- Creates parent directories of output if needed.

Dependencies: pandas (pip install pandas)
"""
import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly sample rows from a CSV")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--n", type=int, default=None, help="Number of rows to sample")
    parser.add_argument("--frac", type=float, default=None, help="Fraction of rows to sample (0<frac<=1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.output)

    if not inp.exists():
        raise SystemExit(f"Input CSV not found: {inp}")

    df = pd.read_csv(inp)

    if args.frac is not None:
        if not (0 < args.frac <= 1):
            raise SystemExit("--frac must be in (0, 1]")
        sampled = df.sample(frac=args.frac, random_state=args.seed)
    elif args.n is not None:
        if args.n <= 0:
            raise SystemExit("--n must be positive")
        sampled = df.sample(n=min(args.n, len(df)), random_state=args.seed)
    else:
        raise SystemExit("Specify either --n or --frac")

    out.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out, index=False)
    print(f"Wrote {len(sampled)} rows to {out}")


if __name__ == "__main__":
    main()
