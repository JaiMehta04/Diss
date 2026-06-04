"""
build_training_dataset.py
=========================
Build the full historical training dataset by aligning:
  • RTWQMS daily water quality  (Nov 2021 – present, from backfill)
  • Sentinel-2 imagery           (Jun 2015 – present, ~5-day revisit)

The overlap period is **Dec 2021 – present** (~4.4 years).

Strategy
--------
1. Load the RTWQMS daily-wide CSV (Day.Mean values for 40 stations).
2. For each (station, date) pair, query Google Earth Engine for the
   best available Sentinel-2 composite within ±3 / ±10 / ±30 days.
3. Extract 10 reflectance bands + 8 spectral indices (tabular)
   and optionally download 64×64 image patches (.npy).
4. Merge WQ + satellite features → final training CSV.

The script is **checkpoint-resumable**: completed (station, date) pairs
are tracked so reruns skip finished work.

Sentinel-2 background
---------------------
• Two satellites: S2A (Jun 2015) + S2B (Mar 2017)
• Combined revisit: ~5 days at equator, ~3–5 days at India latitudes
• Overpass time: ~10:30 AM local solar time (IST ≈ 05:00 UTC)
• After cloud filtering, expect usable imagery every 10–15 days
• Collection: COPERNICUS/S2_SR_HARMONIZED (L2A Surface Reflectance)

RTWQMS alignment
----------------
Since Sentinel-2 overpasses at ~10:30 AM IST, the Day.Mean from RTWQMS
(aggregated over 24h) is a suitable match.  The `year.json` endpoint
already provides pre-aggregated daily statistics, so there is **no need
to scrape hourly data** — the daily means we already have are optimal.

Usage
-----
  python build_training_dataset.py                          # tabular only
  python build_training_dataset.py --patches                # + 64×64 .npy
  python build_training_dataset.py --workers 8              # 8 threads
  python build_training_dataset.py --start-date 2022-01-01  # custom range
  python build_training_dataset.py --station WB89           # one station
  python build_training_dataset.py --fresh                  # ignore checkpoint
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "compat"))

import ee

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ID = "dissertation-ganga"

# Input: RTWQMS historical daily-wide CSV from the backfill script
WQ_CSV = PROJECT_ROOT / "data" / "rtwqms_historical" / "rtwqms_daily_wide.csv"
STATION_META_CSV = PROJECT_ROOT / "data" / "rtwqms_historical" / "station_metadata.csv"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "training_dataset"
CHECKPOINT = OUTPUT_DIR / "checkpoint.json"
SAT_CSV = OUTPUT_DIR / "satellite_indices.csv"
MERGED_CSV = OUTPUT_DIR / "merged_training_dataset.csv"
PATCH_DIR = OUTPUT_DIR / "patches"
META_CSV = OUTPUT_DIR / "patches_meta.csv"

BUFFER_METRES = 500
SCALE = 10
PATCH_SIZE = 64

SEARCH_WINDOWS = [
    (3, 20),     # ±3 days, <20% cloud
    (10, 60),    # ±10 days, <60% cloud
    (30, 100),   # ±30 days, any cloud level
]

BASE_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
INDEX_NAMES = ["NDWI", "MNDWI", "NDTI", "NDVI", "NDCI", "MCI", "FAI", "SABI"]
ALL_BANDS = BASE_BANDS + INDEX_NAMES

# WQ columns in the daily-wide CSV
WQ_PARAMS = ["BOD", "CL", "COD", "DO", "Depth", "EC", "NO3", "S", "TOC", "WT", "WTb", "pH"]

MAX_WORKERS = 4
RETRY_LIMIT = 3
RETRY_DELAY = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────────────────────────────────────
def load_checkpoint(path: Path) -> set:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("completed", []))
    return set()


def save_checkpoint(path: Path, completed: set) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"completed": sorted(completed)}, indent=0),
        encoding="utf-8",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Earth Engine helpers (same logic as existing download_all_indices.py)
# ──────────────────────────────────────────────────────────────────────────────
def mask_clouds_and_scale(image: ee.Image) -> ee.Image:
    qa = image.select("QA60")
    clear = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(clear).divide(10000).select(BASE_BANDS)


def compute_indices(image: ee.Image) -> ee.Image:
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = image.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndti = image.normalizedDifference(["B4", "B3"]).rename("NDTI")
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndci = image.normalizedDifference(["B5", "B4"]).rename("NDCI")

    r665 = image.select("B4")
    r705 = image.select("B5")
    r740 = image.select("B6")
    lam = ee.Number(705 - 665).divide(740 - 665)
    mci = r705.subtract(r665).subtract(r740.subtract(r665).multiply(lam)).rename("MCI")

    r842 = image.select("B8")
    r1610 = image.select("B11")
    fai_lam = ee.Number(842 - 665).divide(1610 - 665)
    fai = r842.subtract(r665).subtract(r1610.subtract(r665).multiply(fai_lam)).rename("FAI")

    sabi = (
        image.select("B8").subtract(image.select("B4"))
        .divide(image.select("B3").add(image.select("B2")))
        .rename("SABI")
    )
    return image.addBands([ndwi, mndwi, ndti, ndvi, ndci, mci, fai, sabi])


def build_composite(
    point: ee.Geometry, date_str: str
) -> Optional[Tuple[ee.Image, int]]:
    for days, cloud_pct in SEARCH_WINDOWS:
        start = ee.Date(date_str).advance(-days, "day")
        end = ee.Date(date_str).advance(days, "day")
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point.buffer(BUFFER_METRES))
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
            .map(mask_clouds_and_scale)
        )
        count = col.size().getInfo()
        if count and count > 0:
            composite = compute_indices(col.median().toFloat())
            return composite, count
    return None


def extract_tabular(composite: ee.Image, region: ee.Geometry) -> Optional[Dict[str, float]]:
    result = (
        composite.select(ALL_BANDS)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=SCALE,
            maxPixels=1e8,
        )
        .getInfo()
    )
    if not result or all(v is None for v in result.values()):
        return None
    return {k: round(v, 6) if v is not None else None for k, v in result.items()}


def extract_patch(
    composite: ee.Image, point: ee.Geometry, patch_size: int
) -> Optional[np.ndarray]:
    half = (patch_size * SCALE) / 2.0
    region = point.buffer(half).bounds()
    try:
        url = composite.select(BASE_BANDS).getDownloadURL({
            "region": region,
            "dimensions": f"{patch_size}x{patch_size}",
            "format": "NPY",
        })
        import requests
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        arr = np.load(resp.content if hasattr(resp.content, "read") else
                       __import__("io").BytesIO(resp.content))
        # Structured array → (H, W, C) → (C, H, W)
        if arr.dtype.names:
            arr = np.stack([arr[b].astype(np.float32) for b in BASE_BANDS], axis=0)
        elif arr.ndim == 3 and arr.shape[-1] == len(BASE_BANDS):
            arr = arr.transpose(2, 0, 1).astype(np.float32)
        elif arr.ndim == 3 and arr.shape[0] == len(BASE_BANDS):
            arr = arr.astype(np.float32)
        else:
            log.warning(f"Unexpected patch shape: {arr.shape}")
            return None
        return arr
    except Exception as exc:
        log.warning(f"Patch download failed: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Worker function
# ──────────────────────────────────────────────────────────────────────────────
def process_station_day(
    station_id: str, station_no: str, lat: float, lon: float,
    date_str: str, wq_values: Dict[str, Any],
    download_patches: bool,
) -> Optional[Dict[str, Any]]:
    """Process one (station, date) pair: fetch satellite + merge with WQ."""
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(BUFFER_METRES)

    for attempt in range(RETRY_LIMIT):
        try:
            result = build_composite(point, date_str)
            if result is None:
                return None

            composite, scene_count = result

            # Tabular extraction
            vals = extract_tabular(composite, region)
            if vals is None:
                return None

            row = {
                "stationId": station_id,
                "station_no": station_no,
                "date": date_str,
                "lat": lat,
                "lon": lon,
                "scene_count": scene_count,
            }
            row.update(vals)
            row.update({f"wq_{k}": v for k, v in wq_values.items()})

            # Patch download (optional)
            if download_patches:
                patch = extract_patch(composite, point, PATCH_SIZE)
                if patch is not None:
                    patch_file = PATCH_DIR / f"{station_id}_{date_str}.npy"
                    np.save(patch_file, patch)
                    row["patch_file"] = str(patch_file.name)

            return row

        except Exception as exc:
            if attempt < RETRY_LIMIT - 1:
                log.warning(f"  Retry {attempt+1} for {station_no}/{date_str}: {exc}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                log.error(f"  Failed {station_no}/{date_str}: {exc}")
                return None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────
def build_job_list(
    wq_df: pd.DataFrame,
    only_station: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    completed: set,
) -> List[Dict[str, Any]]:
    """Build list of (station, date) jobs to process."""
    jobs = []
    for _, row in wq_df.iterrows():
        sid = str(row["station_id"])
        sno = row["station_no"]
        d = str(row["date"])
        key = f"{sid}_{d}"

        if key in completed:
            continue
        if only_station and sno != only_station:
            continue
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue

        wq_vals = {}
        for p in WQ_PARAMS:
            val = row.get(p)
            wq_vals[p] = float(val) if pd.notna(val) else None

        jobs.append({
            "station_id": sid,
            "station_no": sno,
            "lat": float(row["latitude"]),
            "lon": float(row["longitude"]),
            "date": d,
            "wq_values": wq_vals,
        })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RTWQMS + Sentinel-2 training dataset")
    parser.add_argument("--patches", action="store_true", help="Also download 64×64 .npy image patches")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel GEE workers")
    parser.add_argument("--station", default=None, help="Only process this station_no (e.g. WB89)")
    parser.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD (default: earliest)")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD (default: latest)")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoint, start fresh")
    args = parser.parse_args()

    # ── Initialise Earth Engine ───────────────────────────────────────────────
    log.info("Initialising Earth Engine …")
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info(f"Loading WQ data from {WQ_CSV}")
    wq_df = pd.read_csv(WQ_CSV)
    log.info(f"  {len(wq_df):,} station-days, {wq_df['station_no'].nunique()} stations")
    log.info(f"  Date range: {wq_df['date'].min()} → {wq_df['date'].max()}")

    # ── Setup output dirs ─────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.patches:
        PATCH_DIR.mkdir(parents=True, exist_ok=True)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    completed = set() if args.fresh else load_checkpoint(CHECKPOINT)
    log.info(f"  Checkpoint: {len(completed)} previously completed")

    # ── Build job list ────────────────────────────────────────────────────────
    jobs = build_job_list(wq_df, args.station, args.start_date, args.end_date, completed)
    log.info(f"  Jobs to process: {len(jobs)}")
    if not jobs:
        log.info("Nothing to do.")
        return

    # ── Estimate ──────────────────────────────────────────────────────────────
    # Sentinel-2 revisits India every ~5 days.  With ±3 day window and cloud
    # filtering, ~60-70% of days should find usable imagery.
    # GEE queries take ~1-3s each, so with 4 workers → ~0.5s/job average.
    log.info(f"  Estimated: {len(jobs)} GEE queries with {args.workers} workers")

    # ── Process ───────────────────────────────────────────────────────────────
    results = []
    sat_rows = []
    failed = 0
    no_imagery = 0
    t0 = time.time()

    # Append header to SAT_CSV if new
    sat_header_written = SAT_CSV.exists() and not args.fresh

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {}
        for job in jobs:
            fut = pool.submit(
                process_station_day,
                station_id=job["station_id"],
                station_no=job["station_no"],
                lat=job["lat"],
                lon=job["lon"],
                date_str=job["date"],
                wq_values=job["wq_values"],
                download_patches=args.patches,
            )
            future_map[fut] = job

        for i, fut in enumerate(as_completed(future_map), 1):
            job = future_map[fut]
            key = f"{job['station_id']}_{job['date']}"

            try:
                row = fut.result()
                if row is None:
                    no_imagery += 1
                    log.info(f"  [{i}/{len(jobs)}] {job['station_no']}/{job['date']}: no imagery")
                else:
                    results.append(row)
                    sat_rows.append(row)
                    log.info(
                        f"  [{i}/{len(jobs)}] {job['station_no']}/{job['date']}: "
                        f"{row.get('scene_count', 0)} scenes"
                    )
            except Exception as exc:
                failed += 1
                log.error(f"  [{i}/{len(jobs)}] {job['station_no']}/{job['date']}: {exc}")

            completed.add(key)

            # Periodic checkpoint save (every 50 jobs)
            if i % 50 == 0:
                save_checkpoint(CHECKPOINT, completed)
                log.info(f"  Checkpoint saved ({len(completed)} total)")

    save_checkpoint(CHECKPOINT, completed)

    # ── Write outputs ─────────────────────────────────────────────────────────
    if results:
        new_df = pd.DataFrame(results)

        # Append to or create satellite CSV
        if SAT_CSV.exists() and not args.fresh:
            existing = pd.read_csv(SAT_CSV)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset=["stationId", "date"], keep="last", inplace=True)
        else:
            combined = new_df

        combined.sort_values(["station_no", "date"], inplace=True)
        combined.to_csv(SAT_CSV, index=False)

        # Build merged training CSV (WQ + satellite)
        # Only keep rows that have both WQ and satellite data
        merged = combined.copy()
        merged.to_csv(MERGED_CSV, index=False)

        elapsed = time.time() - t0
        log.info(f"\n{'='*60}")
        log.info(f"DONE in {elapsed:.0f}s")
        log.info(f"  Successful:  {len(results)}")
        log.info(f"  No imagery:  {no_imagery}")
        log.info(f"  Failed:      {failed}")
        log.info(f"  Total in dataset: {len(combined)}")
        log.info(f"  Output: {MERGED_CSV}")
        log.info(f"{'='*60}")
    else:
        log.warning("No new results collected.")


if __name__ == "__main__":
    main()
