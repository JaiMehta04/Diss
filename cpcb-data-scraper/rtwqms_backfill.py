"""RTWQMS historical backfill — download full Period-of-Record daily data.

Fetches Day.Mean, Day.Min, Day.Max series for all 12 water-quality parameters
across all 40 CPCB RTWQMS stations on the Ganga basin and writes them to a
single consolidated CSV plus per-station CSVs.

API pattern
-----------
  Stations list : GET /data/internet/stations/stations.json
  Timeseries    : GET /data/internet/stations/DSP_SWAN/{station_no}/{param}/year.json

Each timeseries endpoint returns a list of 3 dicts (Day.Min, Day.Mean, Day.Max),
each containing a `data` key with rows of [timestamp_iso, value, aggregation_%].

Usage
-----
    python rtwqms_backfill.py [--output-dir DIR] [--station STATION_NO]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

warnings.simplefilter("ignore", category=InsecureRequestWarning)

BASE_URL = "https://rtwqmsdb1.cpcb.gov.in/data/"

PARAMETERS = ["pH", "DO", "BOD", "WT", "EC", "NO3", "COD", "CL", "TOC", "WTb", "S", "Depth"]

PARAM_LONG_NAMES = {
    "pH": "pH",
    "DO": "Dissolved Oxygen",
    "BOD": "Biochemical Oxygen Demand",
    "WT": "Water Temperature",
    "EC": "Electrical Conductivity",
    "NO3": "Nitrate",
    "COD": "Chemical Oxygen Demand",
    "CL": "Chloride",
    "TOC": "Total Organic Carbon",
    "WTb": "Turbidity",
    "S": "Specific Conductance",
    "Depth": "Water Level",
}

SESSION = requests.Session()
SESSION.verify = False


def _get_json(url: str, retries: int = 3) -> list | dict | None:
    """GET with retries and back-off. Returns parsed JSON or None."""
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=120)
            if r.status_code == 200:
                return r.json()
            print(f"  HTTP {r.status_code} for {url}")
        except requests.exceptions.RequestException as exc:
            print(f"  Request error (attempt {attempt+1}): {exc}")
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return None


def fetch_stations() -> list[dict]:
    """Return list of station metadata dicts from the RTWQMS API."""
    url = BASE_URL + "internet/stations/stations.json"
    data = _get_json(url)
    if not isinstance(data, list) or not data:
        print("ERROR: Could not fetch station list.")
        sys.exit(1)
    return data


def fetch_timeseries(station_no: str, param: str) -> list[dict]:
    """Fetch daily timeseries for one station-parameter combo.

    Returns a list of row dicts ready for DataFrame construction.
    """
    url = f"{BASE_URL}internet/stations/DSP_SWAN/{station_no}/{param}/year.json"
    data = _get_json(url)
    if not isinstance(data, list):
        return []

    rows: list[dict] = []
    for series in data:
        ts_name = series.get("ts_name", "")       # Day.Mean / Day.Min / Day.Max
        unit = series.get("ts_unitsymbol", "")
        precision = series.get("ts_precision", "")
        series_data = series.get("data", [])

        for point in series_data:
            if len(point) < 3:
                continue
            rows.append({
                "station_no": station_no,
                "parameter": param,
                "parameter_long": PARAM_LONG_NAMES.get(param, param),
                "aggregation": ts_name,
                "timestamp": point[0],
                "value": point[1],
                "quality_pct": point[2],
                "unit": unit,
            })
    return rows


def backfill(stations: list[dict], output_dir: Path,
             only_station: str | None = None) -> pd.DataFrame:
    """Download all historical data and return consolidated DataFrame."""
    output_dir.mkdir(parents=True, exist_ok=True)
    per_station_dir = output_dir / "per_station"
    per_station_dir.mkdir(exist_ok=True)

    all_rows: list[dict] = []
    station_meta: list[dict] = []

    targets = stations
    if only_station:
        targets = [s for s in stations if s["station_no"] == only_station]
        if not targets:
            print(f"Station {only_station} not found in API.")
            sys.exit(1)

    total = len(targets)
    for idx, st in enumerate(sorted(targets, key=lambda x: x["station_no"]), 1):
        sno = st["station_no"]
        sid = st["station_id"]
        sname = st["station_name"]
        state = st.get("territory_name", "")
        lat = st.get("station_latitude", "")
        lon = st.get("station_longitude", "")

        print(f"[{idx}/{total}] {sno} — {sname}")

        station_rows: list[dict] = []
        for param in PARAMETERS:
            rows = fetch_timeseries(sno, param)
            # Enrich with station metadata
            for r in rows:
                r["station_id"] = sid
                r["station_name"] = sname
                r["state"] = state
                r["latitude"] = lat
                r["longitude"] = lon
            station_rows.extend(rows)
            print(f"    {param}: {len(rows)} rows")

        if station_rows:
            df_st = pd.DataFrame(station_rows)
            df_st.to_csv(per_station_dir / f"{sno}.csv", index=False)
            all_rows.extend(station_rows)

        station_meta.append({
            "station_id": sid,
            "station_no": sno,
            "station_name": sname,
            "state": state,
            "latitude": lat,
            "longitude": lon,
            "rows_downloaded": len(station_rows),
        })

    # Consolidated output
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No data downloaded.")
        return df

    df["date"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.date
    df.sort_values(["station_no", "parameter", "aggregation", "date"], inplace=True)

    consolidated_path = output_dir / "rtwqms_historical_daily.csv"
    df.to_csv(consolidated_path, index=False)
    print(f"\nConsolidated CSV: {consolidated_path} ({len(df):,} rows)")

    # Station metadata
    meta_df = pd.DataFrame(station_meta)
    meta_df.to_csv(output_dir / "station_metadata.csv", index=False)

    # Summary
    print("\n=== Backfill Summary ===")
    print(f"  Stations : {df['station_no'].nunique()}")
    print(f"  Parameters: {df['parameter'].nunique()}")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"  Total rows: {len(df):,}")

    # Pivot to wide format (Day.Mean only) for analysis convenience
    mean_df = df[df["aggregation"] == "Day.Mean"].copy()
    if not mean_df.empty:
        wide = mean_df.pivot_table(
            index=["station_no", "station_id", "station_name", "state",
                   "latitude", "longitude", "date"],
            columns="parameter",
            values="value",
            aggfunc="first",
        ).reset_index()
        wide.columns.name = None
        wide_path = output_dir / "rtwqms_daily_wide.csv"
        wide.to_csv(wide_path, index=False)
        print(f"  Wide CSV (Day.Mean): {wide_path} ({len(wide):,} rows)")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="RTWQMS historical backfill")
    parser.add_argument("--output-dir", default="data/rtwqms_historical",
                        help="Output directory (default: data/rtwqms_historical)")
    parser.add_argument("--station", default=None,
                        help="Download only this station_no (e.g. WB89)")
    args = parser.parse_args()

    print("Fetching station list …")
    stations = fetch_stations()
    print(f"Found {len(stations)} stations.\n")

    t0 = time.time()
    backfill(stations, Path(args.output_dir), only_station=args.station)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
