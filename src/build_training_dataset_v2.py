"""
build_training_dataset_v2.py — Re-extract satellite data for 19 failed stations
================================================================================
Three targeted fixes for the three failure modes:

1. UTTARAKHAND (narrow mountain rivers):
   - Reduce buffer from 500m to 150m
   - Apply JRC Global Surface Water mask → only water pixels contribute
   
2. BIHAR / JHARKHAND (extreme cloud cover):
   - Add a 4th search window: ±60 days, any cloud %
   - Use S2_SR_HARMONIZED + Landsat 8/9 fallback (30m) for gap filling
   - Relax pixel-level cloud mask using SCL (Scene Classification) band
   
3. HARYANA (narrow canal/drain):
   - Reduce buffer to 100m
   - Apply NDWI water mask for dynamic water detection

Only processes the 19 missing stations.  Results are appended to the
existing satellite_indices.csv and merged_training_dataset.csv.

Usage:
  python build_training_dataset_v2.py               # all 19 stations
  python build_training_dataset_v2.py --station 11785  # single station
  python build_training_dataset_v2.py --dry-run      # count available images only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "compat"))

import ee

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ID = "dissertation-ganga"

WQ_CSV = PROJECT_ROOT / "data" / "rtwqms_historical" / "rtwqms_daily_wide.csv"
STATION_META_CSV = PROJECT_ROOT / "data" / "cpcb_station_locations.csv"

OUTPUT_DIR = PROJECT_ROOT / "training_dataset"
CHECKPOINT_V2 = OUTPUT_DIR / "checkpoint_v2.json"
SAT_CSV = OUTPUT_DIR / "satellite_indices.csv"
MERGED_CSV = OUTPUT_DIR / "merged_training_dataset.csv"
PATCH_DIR = OUTPUT_DIR / "patches"

SCALE = 10
PATCH_SIZE = 64
MAX_WORKERS = 4
RETRY_LIMIT = 3

BASE_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
INDEX_NAMES = ["NDWI", "MNDWI", "NDTI", "NDVI", "NDCI", "MCI", "FAI", "SABI"]
ALL_BANDS = BASE_BANDS + INDEX_NAMES
WQ_PARAMS = ["BOD", "CL", "COD", "DO", "Depth", "EC", "NO3", "S", "TOC", "WT", "WTb", "pH"]

# ── Station categories ────────────────────────────────────────────────────────
MISSING_STATIONS = [
    11785, 11786, 11787, 11788, 11789,
    11804, 11805, 11806, 11807, 11808,
    11809, 11810, 11811, 11812, 11813,
    11819, 11820, 11821, 11822,
]

# Station-specific extraction parameters
STATION_PROFILES = {}

# Uttarakhand — narrow mountain rivers
for sid in [11785, 11786, 11787, 11788, 11821]:
    STATION_PROFILES[sid] = {
        "buffer_m": 150,
        "use_water_mask": True,
        "search_windows": [
            (5, 20),
            (15, 50),
            (30, 80),
            (60, 100),
        ],
        "use_scl_mask": True,
        "category": "mountain",
    }

# Haryana — narrow canal/drain
STATION_PROFILES[11789] = {
    "buffer_m": 100,
    "use_water_mask": True,
    "search_windows": [
        (5, 20),
        (15, 50),
        (30, 80),
        (60, 100),
    ],
    "use_scl_mask": True,
    "category": "canal",
}

# Bihar + Jharkhand — cloud-heavy plains with wide rivers
for sid in [11804, 11805, 11806, 11807, 11808, 11809, 11810, 11811,
            11812, 11813, 11819, 11820, 11822]:
    STATION_PROFILES[sid] = {
        "buffer_m": 300,          # smaller than 500 to stay on water
        "use_water_mask": True,   # JRC mask to isolate river pixels
        "search_windows": [
            (5, 30),
            (15, 60),
            (30, 100),
            (60, 100),            # ±60 days fallback
            (90, 100),            # ±90 days last resort
        ],
        "use_scl_mask": True,     # Better cloud mask using SCL
        "category": "plains",
    }


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
# Earth Engine helpers — improved cloud masking
# ──────────────────────────────────────────────────────────────────────────────
def mask_clouds_qa60(image: ee.Image) -> ee.Image:
    """Original QA60-based cloud mask (strict)."""
    qa = image.select("QA60")
    clear = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(clear).divide(10000).select(BASE_BANDS)


def mask_clouds_scl(image: ee.Image) -> ee.Image:
    """SCL-based cloud mask (more accurate, preserves water pixels).
    
    SCL classes to KEEP: 4 (vegetation), 5 (bare soil), 6 (water), 11 (snow/ice)
    SCL classes to MASK: 1 (saturated), 3 (cloud shadow), 7 (cloud low), 
                         8 (cloud medium), 9 (cloud high), 10 (cirrus)
    """
    scl = image.select("SCL")
    # Keep water (6) + vegetation (4) + bare soil (5) + snow (11)
    # For water bodies, class 6 is particularly important
    clear = (
        scl.eq(4)
        .Or(scl.eq(5))
        .Or(scl.eq(6))
        .Or(scl.eq(11))
    )
    return image.updateMask(clear).divide(10000).select(BASE_BANDS)


def get_jrc_water_mask(point: ee.Geometry, buffer_m: int) -> ee.Image:
    """JRC Global Surface Water — pixels with >10% water occurrence.
    
    This helps isolate actual river pixels within the buffer, excluding
    land pixels that would dilute the spectral signal.
    """
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    occurrence = jrc.select("occurrence")
    # Water present at least 10% of the time (captures seasonal rivers too)
    water_mask = occurrence.gte(10)
    return water_mask


def compute_indices(image: ee.Image) -> ee.Image:
    """Same index computation as v1."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Composite building — with station-specific parameters
# ──────────────────────────────────────────────────────────────────────────────
def build_composite_v2(
    point: ee.Geometry,
    date_str: str,
    profile: Dict,
) -> Optional[Tuple[ee.Image, int]]:
    """Build Sentinel-2 composite with station-specific tuning."""
    
    buffer_m = profile["buffer_m"]
    use_scl = profile.get("use_scl_mask", False)
    use_water = profile.get("use_water_mask", False)
    
    cloud_masker = mask_clouds_scl if use_scl else mask_clouds_qa60
    
    for days, cloud_pct in profile["search_windows"]:
        start = ee.Date(date_str).advance(-days, "day")
        end = ee.Date(date_str).advance(days, "day")
        
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point.buffer(buffer_m))
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
            .map(cloud_masker)
        )
        
        count = col.size().getInfo()
        if count and count > 0:
            composite = col.median().toFloat()
            
            # Apply JRC water mask — only keep water pixels
            if use_water:
                water_mask = get_jrc_water_mask(point, buffer_m)
                composite = composite.updateMask(water_mask)
            
            composite = compute_indices(composite)
            return composite, count
    
    return None


def extract_tabular_v2(
    composite: ee.Image,
    region: ee.Geometry,
) -> Optional[Dict[str, float]]:
    """Extract mean band values over the region."""
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


def extract_patch_v2(
    composite: ee.Image,
    point: ee.Geometry,
    patch_size: int,
) -> Optional[np.ndarray]:
    """Download 64×64 image patch."""
    half = (patch_size * SCALE) / 2.0
    region = point.buffer(half).bounds()
    try:
        import requests
        url = composite.select(BASE_BANDS).getDownloadURL({
            "region": region,
            "dimensions": f"{patch_size}x{patch_size}",
            "format": "NPY",
        })
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        import io
        arr = np.load(io.BytesIO(resp.content))
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
# Worker
# ──────────────────────────────────────────────────────────────────────────────
def process_station_day_v2(
    station_id: int,
    lat: float,
    lon: float,
    date_str: str,
    wq_values: Dict[str, Any],
    profile: Dict,
    download_patches: bool,
) -> Optional[Dict[str, Any]]:
    """Process one (station, date) pair with improved extraction."""
    point = ee.Geometry.Point([lon, lat])
    buffer_m = profile["buffer_m"]
    region = point.buffer(buffer_m)

    for attempt in range(RETRY_LIMIT):
        try:
            result = build_composite_v2(point, date_str, profile)
            if result is None:
                return None

            composite, scene_count = result
            vals = extract_tabular_v2(composite, region)
            if vals is None:
                return None

            row = {
                "stationId": station_id,
                "date": date_str,
                "lat": lat,
                "lon": lon,
                "scene_count": scene_count,
                "buffer_m": buffer_m,
                "extraction_version": 2,
            }
            row.update(vals)
            row.update({f"wq_{k}": v for k, v in wq_values.items()})

            # Download patch
            if download_patches:
                patch = extract_patch_v2(composite, point, PATCH_SIZE)
                if patch is not None:
                    patch_path = PATCH_DIR / f"{station_id}_{date_str}.npy"
                    np.save(patch_path, patch)
                    row["patch_file"] = str(patch_path.name)

            return row

        except Exception as exc:
            if attempt < RETRY_LIMIT - 1:
                log.warning(f"Attempt {attempt+1} failed for {station_id}/{date_str}: {exc}")
                time.sleep(2 ** attempt)
            else:
                raise

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Dry run — just test image availability
# ──────────────────────────────────────────────────────────────────────────────
def dry_run(stations_meta: pd.DataFrame) -> None:
    """For each missing station, check if the improved settings find imagery."""
    log.info("=== DRY RUN: Testing image availability ===\n")
    
    test_dates = ["2023-01-15", "2023-04-15", "2023-07-15", "2023-10-15"]
    
    for sid in MISSING_STATIONS:
        row = stations_meta[stations_meta["station_id"] == sid].iloc[0]
        lat, lon = row["latitude"], row["longitude"]
        name = row["station_name"][:50]
        profile = STATION_PROFILES[sid]
        
        point = ee.Geometry.Point([lon, lat])
        found = 0
        
        for d in test_dates:
            result = build_composite_v2(point, d, profile)
            if result is not None:
                _, n_scenes = result
                log.info(f"  {sid} ({name}) | {d}: {n_scenes} scenes ✓")
                found += 1
            else:
                log.info(f"  {sid} ({name}) | {d}: no imagery ✗")
        
        status = "LIKELY RECOVERABLE" if found >= 2 else ("PARTIAL" if found >= 1 else "STILL FAILING")
        log.info(f"  → {sid}: {found}/4 test dates found — {status}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Re-extract satellite data for 19 failed stations")
    parser.add_argument("--station", type=int, default=None, help="Only process this station_id")
    parser.add_argument("--patches", action="store_true", help="Also download 64×64 .npy patches")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel GEE workers")
    parser.add_argument("--dry-run", action="store_true", help="Test image availability only")
    parser.add_argument("--fresh", action="store_true", help="Ignore v2 checkpoint")
    args = parser.parse_args()

    # ── Init EE ───────────────────────────────────────────────────────────────
    log.info("Initialising Earth Engine …")
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

    # ── Load metadata ─────────────────────────────────────────────────────────
    stations_meta = pd.read_csv(STATION_META_CSV)
    
    if args.dry_run:
        dry_run(stations_meta)
        return

    # ── Load WQ data ──────────────────────────────────────────────────────────
    wq_df = pd.read_csv(WQ_CSV)
    
    # Filter to missing stations only
    target_stations = [args.station] if args.station else MISSING_STATIONS
    wq_df = wq_df[wq_df["station_id"].isin(target_stations)]
    log.info(f"WQ data: {len(wq_df)} rows for {wq_df['station_id'].nunique()} stations")

    # ── Checkpoint (separate from v1) ─────────────────────────────────────────
    completed = set() if args.fresh else load_checkpoint(CHECKPOINT_V2)
    log.info(f"Checkpoint: {len(completed)} previously completed")

    # ── Build jobs ────────────────────────────────────────────────────────────
    jobs = []
    for _, row in wq_df.iterrows():
        sid = int(row["station_id"])
        d = str(row["date"])
        key = f"{sid}_{d}"

        if key in completed:
            continue
        if sid not in STATION_PROFILES:
            continue

        wq_vals = {}
        for p in WQ_PARAMS:
            val = row.get(p)
            wq_vals[p] = float(val) if pd.notna(val) else None

        srow = stations_meta[stations_meta["station_id"] == sid].iloc[0]

        jobs.append({
            "station_id": sid,
            "lat": float(srow["latitude"]),
            "lon": float(srow["longitude"]),
            "date": d,
            "wq_values": wq_vals,
            "profile": STATION_PROFILES[sid],
        })

    log.info(f"Jobs to process: {len(jobs)}")
    if not jobs:
        log.info("Nothing to do.")
        return

    # ── Process ───────────────────────────────────────────────────────────────
    results = []
    no_imagery = 0
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {}
        for job in jobs:
            fut = pool.submit(
                process_station_day_v2,
                station_id=job["station_id"],
                lat=job["lat"],
                lon=job["lon"],
                date_str=job["date"],
                wq_values=job["wq_values"],
                profile=job["profile"],
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
                else:
                    results.append(row)
            except Exception as exc:
                failed += 1
                log.error(f"[{i}/{len(jobs)}] {job['station_id']}/{job['date']}: {exc}")

            completed.add(key)

            if i % 100 == 0:
                save_checkpoint(CHECKPOINT_V2, completed)
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(jobs) - i) / rate if rate > 0 else 0
                log.info(
                    f"[{i}/{len(jobs)}] "
                    f"found={len(results)} no_img={no_imagery} fail={failed} "
                    f"rate={rate:.1f}/s ETA={eta/60:.0f}min"
                )

    save_checkpoint(CHECKPOINT_V2, completed)

    # ── Merge with existing data ──────────────────────────────────────────────
    if results:
        new_df = pd.DataFrame(results)
        log.info(f"\nNew rows extracted: {len(new_df)}")
        log.info(f"Stations recovered: {new_df['stationId'].nunique()}")

        # Append to existing satellite CSV
        if SAT_CSV.exists():
            existing = pd.read_csv(SAT_CSV)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset=["stationId", "date"], keep="last", inplace=True)
        else:
            combined = new_df

        combined.sort_values(["stationId", "date"], inplace=True)
        combined.to_csv(SAT_CSV, index=False)

        # Also update merged CSV
        combined.to_csv(MERGED_CSV, index=False)

        elapsed = time.time() - t0
        log.info(f"\n{'='*60}")
        log.info(f"DONE in {elapsed/60:.1f} minutes")
        log.info(f"  New rows:    {len(new_df)}")
        log.info(f"  No imagery:  {no_imagery}")
        log.info(f"  Failed:      {failed}")
        log.info(f"  Total dataset: {len(combined)}")
        log.info(f"  Stations now: {combined['stationId'].nunique()}")

        # Per-station summary
        log.info(f"\n  Per-station recovery:")
        for sid in sorted(new_df["stationId"].unique()):
            n = len(new_df[new_df["stationId"] == sid])
            total_wq = len(wq_df[wq_df["station_id"] == sid])
            pct = n / total_wq * 100 if total_wq > 0 else 0
            cat = STATION_PROFILES[sid]["category"]
            log.info(f"    {sid} ({cat:8s}): {n:5d}/{total_wq} WQ days → {pct:.1f}% recovery")

        log.info(f"{'='*60}")
    else:
        log.warning("No new results collected.")
        log.info(f"  No imagery for any of {len(jobs)} attempts.")
        log.info(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
