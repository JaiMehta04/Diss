"""
Merge & transpose hourly water-quality CSV snapshots into a single wide CSV.

Input format (one row per measurement)
--------------------------------------
    stationId, timestamp, timestampDate, value, unit, parameterNo, parameterName

Each `water_data_YYYY-MM-DD_HH-MM.csv` in the input folder is a snapshot taken
at that wall-clock time, but inside it the rows are observations whose own
`timestamp` (= measurement time) is what actually identifies the reading.
The same (stationId, timestamp, parameterNo) reading therefore repeats across
many snapshot files - that's expected and we de-duplicate.

Output (wide format)
--------------------
    stationId, timestamp, timestampDate, <param1>, <param2>, ...
e.g. columns: pH, WT, BOD, DO, NO3, EC, COD, CL, TOC, WTb, S, Depth, ...

Side-car files written next to the wide CSV:
    parameter_metadata.csv   parameterNo -> parameterName, unit (canonical)
    _manifest.json           list of source CSVs already merged + counts

Re-running is idempotent: only files not yet in the manifest are read; their
new (station, timestamp, param) tuples are appended to the existing wide CSV.

Usage
-----
    python -m src.preprocessing.merge_hourly_data \
        --input-dir  data/Hourly_data \
        --output-dir data/Hourly_data_processed \
        [--rebuild]        # ignore manifest and rebuild from scratch

Adding new CSVs later: just drop them in `--input-dir` and rerun the script -
only the new files will be read and merged.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

LOG = logging.getLogger("merge_hourly")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
EXPECTED_COLS = [
    "stationId", "timestamp", "timestampDate",
    "value", "unit", "parameterNo", "parameterName",
]
KEY_COLS = ["stationId", "timestamp", "parameterNo"]   # uniqueness key
INDEX_COLS = ["stationId", "timestamp", "timestampDate"]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
@dataclass
class Paths:
    input_dir: Path
    output_dir: Path

    @property
    def wide_csv(self) -> Path:
        return self.output_dir / "hourly_wide.csv"

    @property
    def long_csv(self) -> Path:
        return self.output_dir / "hourly_long.csv"

    @property
    def param_meta(self) -> Path:
        return self.output_dir / "parameter_metadata.csv"

    @property
    def manifest(self) -> Path:
        return self.output_dir / "_manifest.json"


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"files": {}}


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------
def discover_csvs(input_dir: Path, manifest: dict, rebuild: bool) -> list[Path]:
    """Return CSVs in input_dir that still need to be processed."""
    all_csvs = sorted(input_dir.glob("water_data_*.csv"))
    if rebuild:
        return all_csvs
    seen = set(manifest.get("files", {}).keys())
    return [p for p in all_csvs if p.name not in seen]


def read_long(files: Iterable[Path]) -> pd.DataFrame:
    """Read & concat snapshot CSVs into one long DataFrame."""
    frames: list[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as exc:                         # noqa: BLE001
            LOG.warning("skipping unreadable file %s: %s", fp.name, exc)
            continue
        missing = set(EXPECTED_COLS) - set(df.columns)
        if missing:
            LOG.warning("skipping %s - missing cols: %s", fp.name, missing)
            continue
        df = df[EXPECTED_COLS]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=EXPECTED_COLS)
    out = pd.concat(frames, ignore_index=True)
    LOG.info("read %d rows from %d files", len(out), len(frames))
    return out


def clean_long(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and drop dupes / unusable rows."""
    if df.empty:
        return df
    df["stationId"] = pd.to_numeric(df["stationId"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestampDate"] = df["timestampDate"].astype(str)
    df = df.dropna(subset=["stationId", "timestamp", "parameterNo", "value"])
    # Keep last value if the same (station, ts, param) appears multiple times
    df = df.sort_values(["stationId", "timestamp", "parameterNo"]) \
           .drop_duplicates(subset=KEY_COLS, keep="last")
    return df


def build_param_metadata(df_long: pd.DataFrame) -> pd.DataFrame:
    """parameterNo -> (parameterName, unit). Most common value wins on conflict."""
    if df_long.empty:
        return pd.DataFrame(columns=["parameterNo", "parameterName", "unit"])
    meta = (
        df_long.groupby("parameterNo")
        .agg(parameterName=("parameterName", lambda s: s.mode().iat[0]),
             unit=("unit", lambda s: s.mode().iat[0]))
        .reset_index()
    )
    return meta


def pivot_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Long -> wide: one row per (stationId, timestamp), columns = parameterNo."""
    if df_long.empty:
        return pd.DataFrame(columns=INDEX_COLS)
    wide = (
        df_long.pivot_table(
            index=INDEX_COLS,
            columns="parameterNo",
            values="value",
            aggfunc="last",        # already de-duped, but be safe
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    return wide


def merge_with_existing(new_wide: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    """Union new_wide with existing wide CSV (if any), preferring new values."""
    if not existing_path.exists():
        return new_wide
    old = pd.read_csv(existing_path)
    # Normalize timestamp dtype on both sides for stable de-dup
    for d in (old, new_wide):
        if "timestamp" in d.columns:
            d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce", utc=True)
    combined = pd.concat([old, new_wide], ignore_index=True, sort=False)
    combined = (
        combined.sort_values(["stationId", "timestamp"])
                .drop_duplicates(subset=["stationId", "timestamp"], keep="last")
    )
    return combined


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run(input_dir: Path, output_dir: Path, rebuild: bool = False) -> None:
    paths = Paths(input_dir, output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"files": {}} if rebuild else load_manifest(paths.manifest)
    todo = discover_csvs(paths.input_dir, manifest, rebuild)
    LOG.info("%d new CSV(s) to process (rebuild=%s)", len(todo), rebuild)
    if not todo:
        LOG.info("nothing to do - exiting")
        return

    long_df = clean_long(read_long(todo))
    LOG.info("after cleaning: %d unique long rows", len(long_df))

    # Update parameter metadata (merge with existing if not rebuilding)
    new_meta = build_param_metadata(long_df)
    if paths.param_meta.exists() and not rebuild:
        old_meta = pd.read_csv(paths.param_meta)
        new_meta = (
            pd.concat([old_meta, new_meta], ignore_index=True)
              .drop_duplicates(subset=["parameterNo"], keep="last")
        )
    new_meta.sort_values("parameterNo").to_csv(paths.param_meta, index=False)

    wide_new = pivot_wide(long_df)
    wide_final = wide_new if rebuild else merge_with_existing(wide_new, paths.wide_csv)

    # Stable column order: index cols first, parameters alphabetised
    param_cols = [c for c in wide_final.columns if c not in INDEX_COLS]
    wide_final = wide_final[INDEX_COLS + sorted(param_cols)]
    wide_final.to_csv(paths.wide_csv, index=False)
    LOG.info("wrote wide CSV: %s  (%d rows x %d params)",
             paths.wide_csv, len(wide_final), len(param_cols))

    # Update manifest
    for fp in todo:
        manifest["files"][fp.name] = {"size": fp.stat().st_size}
    manifest["last_run_added"] = [p.name for p in todo]
    manifest["wide_rows"] = int(len(wide_final))
    manifest["param_count"] = int(len(param_cols))
    save_manifest(paths.manifest, manifest)
    LOG.info("manifest updated: %s", paths.manifest)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("data/Hourly_data"))
    p.add_argument("--output-dir", type=Path, default=Path("data/Hourly_data_processed"))
    p.add_argument("--rebuild", action="store_true", help="ignore manifest, reprocess everything")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(args.input_dir.resolve(), args.output_dir.resolve(), rebuild=args.rebuild)


if __name__ == "__main__":
    _cli()
