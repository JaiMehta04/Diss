"""Analyze band_stats.csv from exported GeoTIFFs.

What it does:
- Reads band_stats.csv (from analyze_exports.py).
- Flags low-coverage scenes (few valid pixels).
- Summarizes per-file band medians for quick comparison across dates/stations.
- Writes two outputs to outputs/:
    * band_summary.csv  (per file+band stats with flags)
    * file_quality.csv  (per file rollup of coverage and key index medians)

Usage:
    python analyze_band_stats.py

Dependencies: pandas (install if needed: pip install pandas)
"""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BAND_STATS_CSV = PROJECT_ROOT / "band_stats.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOW_COUNT_THRESHOLD = 10_000  # pixels; adjust based on your patch size


def load_band_stats(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    df = pd.read_csv(path)
    required = {"file", "band", "count", "mean", "std", "p5", "p50", "p95"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in {path}: {sorted(missing)}")
    return df


def summarize_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["low_count"] = df["count"] < LOW_COUNT_THRESHOLD
    # Select columns to keep
    cols = [
        "file",
        "band",
        "count",
        "low_count",
        "mean",
        "std",
        "min",
        "p5",
        "p50",
        "p95",
        "max",
    ]
    return df[cols].sort_values(["file", "band"])


def summarize_files(df: pd.DataFrame) -> pd.DataFrame:
    # Pivot to bring key medians per band onto one row per file
    pivot = df.pivot_table(index="file", columns="band", values="p50", aggfunc="first")
    counts = df.pivot_table(index="file", values="count", aggfunc="sum")
    low_counts = df.pivot_table(index="file", values="low_count", aggfunc="any")
    out = pivot.merge(counts, left_index=True, right_index=True, how="left")
    out = out.merge(low_counts, left_index=True, right_index=True, how="left", suffixes=("", "_any_low"))
    out = out.rename(columns={"count": "total_valid_pixels", "low_count": "any_low_count"})
    out = out.reset_index()
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_band_stats(BAND_STATS_CSV)
    band_summary = summarize_bands(df)
    file_quality = summarize_files(band_summary)

    band_summary_path = OUTPUT_DIR / "band_summary.csv"
    file_quality_path = OUTPUT_DIR / "file_quality.csv"

    band_summary.to_csv(band_summary_path, index=False)
    file_quality.to_csv(file_quality_path, index=False)

    print(f"Wrote per-band summary to {band_summary_path}")
    print(f"Wrote per-file quality rollup to {file_quality_path}")


if __name__ == "__main__":
    main()
