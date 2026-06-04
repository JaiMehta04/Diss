"""
extract_tapi_data.py
====================
Download and filter CPCB surface water quality data for the **Tapi (Tapti) River**
from the NWDP CKAN API (nwdp.nwic.gov.in).

Tapi flows through: Maharashtra and Gujarat.

Downloads chemical + physical parameter CSVs for those states, filters to
stations on the Tapi river, and produces a consolidated output CSV.

Usage:
    python extract_tapi_data.py
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "tapi"
FINAL_CSV = OUTPUT_DIR / "tapi_water_quality.csv"

NWDP_API = "https://nwdp.nwic.gov.in/api/3/action/datastore_search"

# NWDP CKAN resource IDs for Maharashtra and Gujarat (CPCB surface water)
RESOURCES = {
    # Chemical parameters
    "chem_MH_1961_2020": "e0aea216-fcf6-479e-910c-b99bc4d2f9fd",
    "chem_MH_2021_2025": "4593ba3e-3097-4f26-800c-b25c692adaef",
    "chem_MH_2026_2030": "a3b61440-bc51-401c-872c-f7414d62afa3",
    "chem_GJ_1961_2020": "90de9aa8-b079-4ad3-81d2-b3b1c92d0739",
    "chem_GJ_2021_2025": "35b88783-4d47-44db-aef9-526bbe63fc96",
    "chem_GJ_2026_2030": "41939d3e-925e-4993-aca7-6a813d48eb9b",
    # Physical parameters
    "phys_MH_1961_2020": "380cc112-a27a-475c-a097-e1f4b7120b31",
    "phys_MH_2021_2025": "e333c91d-5e2f-4e0e-992d-5256ad49f10c",
    "phys_MH_2026_2030": "0a65b1c9-a18b-414e-97be-86145457ebcd",
    "phys_GJ_1961_2020": "5f28f607-583d-4b88-8caa-facd8fb0db4a",
    "phys_GJ_2021_2025": "b7c5018d-842d-4b1c-96cf-4c1d3a55807d",
    "phys_GJ_2026_2030": "e408ffcd-99d5-473e-8f5a-08416eed0168",
}

# Keywords to identify Tapi/Tapti river stations
RIVER_KEYWORDS = ["tapi", "tapti"]

BATCH_SIZE = 32000  # CKAN datastore max per request


def download_resource(resource_id: str, label: str) -> pd.DataFrame:
    """Download all rows from a CKAN datastore resource."""
    all_rows = []
    offset = 0
    while True:
        params = {"resource_id": resource_id, "limit": BATCH_SIZE, "offset": offset}
        try:
            r = requests.get(NWDP_API, params=params, timeout=120)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    ERROR downloading {label}: {e}")
            break

        records = data.get("result", {}).get("records", [])
        if not records:
            break
        all_rows.extend(records)
        total = data.get("result", {}).get("total", 0)
        offset += len(records)
        print(f"    {label}: {offset}/{total} rows", end="\r")
        if offset >= total:
            break
        time.sleep(0.3)

    print(f"    {label}: {len(all_rows)} rows downloaded         ")
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def filter_tapi_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows mentioning Tapi/Tapti river in station/location columns."""
    if df.empty:
        return df

    # Look for river name in likely columns
    text_cols = [c for c in df.columns if any(kw in c.lower() for kw in
                 ["station", "location", "site", "river", "water_body", "waterbody", "name"])]
    if not text_cols:
        # Fallback: search all string columns
        text_cols = [c for c in df.columns if df[c].dtype == object]

    mask = pd.Series(False, index=df.index)
    for col in text_cols:
        col_lower = df[col].astype(str).str.lower()
        for kw in RIVER_KEYWORDS:
            mask |= col_lower.str.contains(kw, na=False)

    filtered = df[mask].copy()
    return filtered


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  TAPI RIVER — WATER QUALITY DATA EXTRACTION")
    print("=" * 60)

    all_dfs = []
    for label, rid in RESOURCES.items():
        print(f"\n  [{label}]")
        df = download_resource(rid, label)
        if df.empty:
            continue

        # Save raw
        raw_path = OUTPUT_DIR / f"raw_{label}.csv"
        df.to_csv(raw_path, index=False)

        # Filter to Tapi
        tapi_df = filter_tapi_stations(df)
        print(f"    Tapi stations: {len(tapi_df)} / {len(df)} rows")

        if not tapi_df.empty:
            tapi_df["_source"] = label
            all_dfs.append(tapi_df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)

        # Remove CKAN internal columns
        drop_cols = [c for c in combined.columns if c.startswith("_")]
        combined.drop(columns=drop_cols, inplace=True, errors="ignore")

        combined.to_csv(FINAL_CSV, index=False)
        print(f"\n{'=' * 60}")
        print(f"  DONE — Tapi river water quality data")
        print(f"  Rows: {len(combined):,}")
        print(f"  Columns: {combined.shape[1]}")
        print(f"  Output: {FINAL_CSV}")
        print(f"{'=' * 60}")
    else:
        print("\n  WARNING: No Tapi river stations found in any dataset.")


if __name__ == "__main__":
    main()
