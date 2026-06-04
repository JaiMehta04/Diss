"""
Download ALL CPCB surface water quality data from the National Water Data Portal (NWDP)
for Ganga-basin states (Uttarakhand, UP, Bihar, Jharkhand, West Bengal).

Source: https://nwdp.nwic.gov.in (CKAN-based)
Datasets: Biological, Chemical, Physical parameters from CPCB (1961-2025)
"""
import csv
import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "data" / "web_extracted" / "nwdp_cpcb"
OUT_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.verify = False
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0",
    "Accept": "*/*",
})

NWDP_BASE = "https://nwdp.nwic.gov.in"
NWDP_API = f"{NWDP_BASE}/api/3/action"

# Ganga-basin states
GANGA_STATES = ["uttarakhand", "uttar pradesh", "bihar", "jharkhand", "west bengal"]

# The three CPCB surface water quality packages
CPCB_PACKAGES = [
    "surface-water-quality-manual-biological-parameters-cpcb",
    "surface-water-quality-manual-chemical-parameters-cpcb",
    "surface-water-quality-manual-physical-parameters-cpcb",
]


def get_ganga_resources():
    """Get all CPCB resources for Ganga-basin states."""
    resources = []
    for pkg_name in CPCB_PACKAGES:
        logger.info(f"Fetching package: {pkg_name}")
        r = session.get(f"{NWDP_API}/package_show?id={pkg_name}", timeout=30)
        r.raise_for_status()
        pkg = r.json().get("result", {})
        param_type = "biological" if "biological" in pkg_name else (
            "chemical" if "chemical" in pkg_name else "physical"
        )

        for res in pkg.get("resources", []):
            name = res.get("name", "").lower()
            # Only Ganga-basin states, skip 2026-2030 (empty placeholders)
            if any(st in name for st in GANGA_STATES) and "2026" not in name:
                resources.append({
                    "id": res["id"],
                    "name": res["name"],
                    "url": res["url"],
                    "size": res.get("size", 0),
                    "param_type": param_type,
                    "period": "2021-2025" if "2021" in name else "1961-2020",
                    "state": next(
                        (st for st in GANGA_STATES if st in name), "unknown"
                    ),
                })
    return resources


def download_resource(res, out_dir):
    """Download a single CSV resource."""
    state_abbr = {
        "uttarakhand": "UK", "uttar pradesh": "UP",
        "bihar": "BR", "jharkhand": "JH", "west bengal": "WB",
    }
    abbr = state_abbr.get(res["state"], "XX")
    fname = f"cpcb_{res['param_type']}_{abbr}_{res['period'].replace('-', '_')}.csv"
    outpath = out_dir / fname

    if outpath.exists() and outpath.stat().st_size > 500:
        logger.info(f"  Already exists: {fname} ({outpath.stat().st_size:,} bytes)")
        return outpath

    logger.info(f"  Downloading: {res['name'][:70]}...")
    try:
        r = session.get(res["url"], timeout=120)
        r.raise_for_status()
        with open(outpath, "wb") as f:
            f.write(r.content)
        logger.info(f"  Saved: {fname} ({len(r.content):,} bytes)")
        return outpath
    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return None


def main():
    print("=" * 70)
    print("NWDP CPCB Water Quality Data Downloader")
    print("States: Uttarakhand, UP, Bihar, Jharkhand, West Bengal")
    print("Parameters: Biological, Chemical, Physical")
    print("=" * 70)

    # Step 1: Get all resources
    resources = get_ganga_resources()
    print(f"\nFound {len(resources)} resources for Ganga-basin states")
    for res in resources:
        print(f"  [{res['param_type'][:4]:4}] {res['state']:15} {res['period']:10} ({res['size']:>10,} bytes)")

    # Step 2: Download all
    print(f"\n--- Downloading to {OUT_DIR} ---")
    downloaded = []
    for res in resources:
        time.sleep(1)  # polite delay
        path = download_resource(res, OUT_DIR)
        if path:
            downloaded.append((res, path))

    # Step 3: Preview each downloaded file
    print(f"\n--- Downloaded {len(downloaded)}/{len(resources)} files ---")
    all_dfs = []
    for res, path in downloaded:
        try:
            df = pd.read_csv(path)
            print(f"\n  {path.name}: {len(df)} rows × {len(df.columns)} cols")
            print(f"    Columns: {list(df.columns)[:12]}")
            if len(df) > 0:
                # Show date range if date column exists
                for dc in ["Observation Date", "observation_date", "Date", "date"]:
                    if dc in df.columns:
                        print(f"    Date range: {df[dc].min()} → {df[dc].max()}")
                        break
                all_dfs.append((res, df))
        except Exception as e:
            print(f"  {path.name}: ERROR {e}")

    # Step 4: Merge all into combined dataset
    if all_dfs:
        print(f"\n--- Merging {len(all_dfs)} datasets ---")
        combined_parts = []
        for res, df in all_dfs:
            df = df.copy()
            df["_param_type"] = res["param_type"]
            df["_state"] = res["state"]
            df["_period"] = res["period"]
            combined_parts.append(df)

        # Group by param_type and merge
        for ptype in ["biological", "chemical", "physical"]:
            parts = [df for res, df in all_dfs if res["param_type"] == ptype]
            if parts:
                merged = pd.concat(parts, ignore_index=True)
                outfile = OUT_DIR / f"cpcb_{ptype}_ganga_all.csv"
                merged.to_csv(outfile, index=False)
                print(f"  {ptype}: {len(merged)} rows → {outfile.name}")

        # Also save a manifest
        manifest = []
        for res, path in downloaded:
            manifest.append({
                "file": path.name,
                "param_type": res["param_type"],
                "state": res["state"],
                "period": res["period"],
                "size_bytes": path.stat().st_size,
                "rows": len(pd.read_csv(path)),
            })
        manifest_df = pd.DataFrame(manifest)
        manifest_df.to_csv(OUT_DIR / "download_manifest.csv", index=False)
        print(f"\n  Manifest: {OUT_DIR / 'download_manifest.csv'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
