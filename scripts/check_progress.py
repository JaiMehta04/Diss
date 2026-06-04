"""
check_progress.py — Show training dataset build progress.

Usage:
    python check_progress.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

# Paths
WQ_CSV = PROJECT_ROOT / "data" / "rtwqms_historical" / "rtwqms_daily_wide.csv"
CHECKPOINT = PROJECT_ROOT / "training_dataset" / "checkpoint.json"
SAT_CSV = PROJECT_ROOT / "training_dataset" / "satellite_indices.csv"
MERGED_CSV = PROJECT_ROOT / "training_dataset" / "merged_training_dataset.csv"
PATCH_DIR = PROJECT_ROOT / "training_dataset" / "patches"


def main():
    print("=" * 60)
    print("  TRAINING DATASET BUILD — PROGRESS REPORT")
    print("=" * 60)

    # Total jobs (from WQ CSV)
    if not WQ_CSV.exists():
        print(f"\n[!] WQ source not found: {WQ_CSV}")
        sys.exit(1)

    wq_df = pd.read_csv(WQ_CSV)
    total_jobs = len(wq_df)
    n_stations = wq_df["station_no"].nunique()
    date_min = wq_df["date"].min()
    date_max = wq_df["date"].max()

    print(f"\n  Source: {total_jobs:,} station-day pairs")
    print(f"  Stations: {n_stations}")
    print(f"  Date range: {date_min} → {date_max}")

    # Checkpoint (completed)
    if CHECKPOINT.exists():
        data = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        completed = data.get("completed", [])
        n_done = len(completed)
    else:
        n_done = 0

    remaining = total_jobs - n_done
    pct = (n_done / total_jobs * 100) if total_jobs > 0 else 0

    print(f"\n  {'─' * 40}")
    print(f"  Completed:  {n_done:,} / {total_jobs:,}  ({pct:.1f}%)")
    print(f"  Remaining:  {remaining:,}")
    print(f"  {'─' * 40}")

    # Progress bar
    bar_width = 40
    filled = int(bar_width * pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\n  [{bar}] {pct:.1f}%")

    # Output files
    print(f"\n  Output files:")
    if SAT_CSV.exists():
        sat_df = pd.read_csv(SAT_CSV)
        print(f"    satellite_indices.csv : {len(sat_df):,} rows")
    else:
        print(f"    satellite_indices.csv : (not yet created)")

    if MERGED_CSV.exists():
        merged_df = pd.read_csv(MERGED_CSV)
        print(f"    merged_training.csv   : {len(merged_df):,} rows")
    else:
        print(f"    merged_training.csv   : (not yet created)")

    # Patches
    if PATCH_DIR.exists():
        n_patches = len(list(PATCH_DIR.glob("*.npy")))
        print(f"    patches/*.npy         : {n_patches:,} files")
    else:
        print(f"    patches/              : (not yet created)")

    # Per-station breakdown
    if n_done > 0 and CHECKPOINT.exists():
        from collections import Counter
        station_counts = Counter(k.split("_")[0] for k in completed)
        station_total = wq_df.groupby("station_id").size().to_dict()

        print(f"\n  Per-station progress ({len(station_counts)} active):")
        print(f"  {'Station':<12} {'Done':>7} {'Total':>7} {'%':>7}")
        print(f"  {'─'*12} {'─'*7} {'─'*7} {'─'*7}")

        for sid, done_ct in sorted(station_counts.items(), key=lambda x: -x[1]):
            tot = station_total.get(int(sid) if sid.isdigit() else sid, "?")
            if isinstance(tot, int):
                spct = f"{done_ct/tot*100:.0f}%"
            else:
                spct = "?"
            # Get station_no for readability
            match = wq_df.loc[wq_df["station_id"].astype(str) == sid, "station_no"]
            sno = match.iloc[0] if len(match) > 0 else sid
            print(f"  {sno:<12} {done_ct:>7,} {tot:>7,} {spct:>7}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
