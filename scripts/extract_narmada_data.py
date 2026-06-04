"""
extract_narmada_data.py
=======================
Download and filter CPCB surface water quality data for the **Narmada River**
from the NWDP CKAN API (nwdp.nwic.gov.in).

Narmada flows through: Madhya Pradesh, Maharashtra, and Gujarat.

Downloads chemical + physical parameter CSVs for those states, filters to
stations on the Narmada river, and produces a consolidated output CSV.

Usage:
    python extract_narmada_data.py
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
OUTPUT_DIR = PROJECT_ROOT / "data" / "narmada"
FINAL_CSV = OUTPUT_DIR / "narmada_water_quality.csv"

NWDP_API = "https://nwdp.nwic.gov.in/api/3/action/datastore_search"

# NWDP CKAN resource IDs for Madhya Pradesh, Maharashtra, and Gujarat
RESOURCES = {
    # Chemical parameters
    "chem_MP_1961_2020": "ca8196be-05b9-4597-af2e-5617d61fdbfb",
    "chem_MP_2021_2025": "d092a7d1-7824-49d4-a021-5fb5c86bec25",
    "chem_MP_2026_2030": "89e55506-8819-4f35-84bd-210d66cc224c",
    "chem_MH_1961_2020": "e0aea216-fcf6-479e-910c-b99bc4d2f9fd",
    "chem_MH_2021_2025": "4593ba3e-3097-4f26-800c-b25c692adaef",
    "chem_MH_2026_2030": "a3b61440-bc51-401c-872c-f7414d62afa3",
    "chem_GJ_1961_2020": "90de9aa8-b079-4ad3-81d2-b3b1c92d0739",
    "chem_GJ_2021_2025": "35b88783-4d47-44db-aef9-526bbe63fc96",
    "chem_GJ_2026_2030": "41939d3e-925e-4993-aca7-6a813d48eb9b",
    # Physical parameters
    "phys_MP_1961_2020": "845edfa8-113b-4d53-8b7d-37de5d596e2d",
    "phys_MP_2021_2025": "a37764dc-a5b3-4bba-bbfc-231783eca2d1",
    "phys_MP_2026_2030": "0ba55947-f0a0-4dcf-a8f8-4c5aed7e5584",
    "phys_MH_1961_2020": "380cc112-a27a-475c-a097-e1f4b7120b31",
    "phys_MH_2021_2025": "e333c91d-5e2f-4e0e-992d-5256ad49f10c",
    "phys_MH_2026_2030": "0a65b1c9-a18b-414e-97be-86145457ebcd",
    "phys_GJ_1961_2020": "5f28f607-583d-4b88-8caa-facd8fb0db4a",
    "phys_GJ_2021_2025": "b7c5018d-842d-4b1c-96cf-4c1d3a55807d",
    "phys_GJ_2026_2030": "e408ffcd-99d5-473e-8f5a-08416eed0168",
}

# Keywords to identify Narmada river stations
RIVER_KEYWORDS = ["narmada"]

BATCH_SIZE = 32000


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


def filter_narmada_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows mentioning Narmada river in station/location columns."""
    if df.empty:
        return df

    text_cols = [c for c in df.columns if any(kw in c.lower() for kw in
                 ["station", "location", "site", "river", "water_body", "waterbody", "name"])]
    if not text_cols:
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
    print("  NARMADA RIVER — WATER QUALITY DATA EXTRACTION")
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

        # Filter to Narmada
        narmada_df = filter_narmada_stations(df)
        print(f"    Narmada stations: {len(narmada_df)} / {len(df)} rows")

        if not narmada_df.empty:
            narmada_df["_source"] = label
            all_dfs.append(narmada_df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)

        # Remove CKAN internal columns
        drop_cols = [c for c in combined.columns if c.startswith("_")]
        combined.drop(columns=drop_cols, inplace=True, errors="ignore")

        combined.to_csv(FINAL_CSV, index=False)
        print(f"\n{'=' * 60}")
        print(f"  DONE — Narmada river water quality data")
        print(f"  Rows: {len(combined):,}")
        print(f"  Columns: {combined.shape[1]}")
        print(f"  Output: {FINAL_CSV}")
        print(f"{'=' * 60}")
    else:
        print("\n  WARNING: No Narmada river stations found in any dataset.")


if __name__ == "__main__":
    main()
