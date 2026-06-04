"""
download_patches.py
===================
Download Sentinel-2 image patches (10-band, 64×64 @ 10 m) for every
station-day in the CSV dataset.  Patches are saved as .npy files and
are **skipped** if they already exist on disk, so the script is fully
resumable.

Each patch  → (10, 64, 64) float32  (channels-first: B2–B12)
Metadata    → patches_meta.csv  (stationId, date, lat, lon, wq_*, file)

Bands used (all resampled to 10 m):
  B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12

Usage
-----
  python download_patches.py                    # download all
  python download_patches.py --workers 6        # 6 parallel threads
  python download_patches.py --patch_size 32    # 32×32 patches
  python download_patches.py --fresh            # re-download everything
"""

import argparse
import csv
import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Project root (two levels up from src/data_collection/) ────────────────────
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "compat"))

import ee

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ID = "dissertation-ganga"
CSV_PATH = PROJECT_ROOT / "data" / "combined_station_data_with_coords.csv"
PATCH_DIR = PROJECT_ROOT / "SentinelPatches"
META_CSV = PATCH_DIR / "patches_meta.csv"

SCALE = 10                  # metres per pixel
PATCH_SIZE_DEFAULT = 64     # 64 px → 640 m × 640 m at 10 m
MAX_WORKERS = 4
RETRY_LIMIT = 3
RETRY_DELAY = 5

# Sentinel-2 bands to download (all resampled to 10 m inside GEE)
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
N_BANDS = len(BANDS)

# Cascading search windows: (± days, max cloud %)
SEARCH_WINDOWS = [
    (3, 20),
    (10, 60),
    (30, 100),
]

# Water quality columns (same as download_all_indices.py)
WQ_COLS = [
    "Biochemical Oxygen Demand", "Chemical Oxygen Demand", "Chloride",
    "Conductivity", "Depth", "Dissolved Oxygen", "Nitrate",
    "Total Organic Carbon", "Water Level", "Water Temperature",
    "Water Turbidity", "pH",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CSV loading (reused logic from download_all_indices.py)
# ──────────────────────────────────────────────────────────────────────────────
def load_station_days(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Group CSV rows by (stationId, date) → dict with lat, lon, WQ means."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    wq_accum: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in rows:
        sid = row.get("stationId", "").strip()
        ts_raw = row.get("timestamp", "").strip()
        if not sid or not ts_raw:
            continue
        try:
            lat, lon = float(row["latitude"]), float(row["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except ValueError:
            continue

        date_str = dt.date().isoformat()
        key = (sid, date_str)

        if key not in groups:
            groups[key] = {
                "stationId": sid,
                "date": date_str,
                "lat": lat,
                "lon": lon,
            }

        for col in WQ_COLS:
            val = row.get(col, "").strip()
            if val:
                try:
                    wq_accum[key][col].append(float(val))
                except ValueError:
                    pass

    for key, info in groups.items():
        for col in WQ_COLS:
            vals = wq_accum[key].get(col, [])
            info[f"wq_{col}"] = round(sum(vals) / len(vals), 4) if vals else None

    return groups


# ──────────────────────────────────────────────────────────────────────────────
# Earth Engine helpers
# ──────────────────────────────────────────────────────────────────────────────
def _init_ee():
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
    log.info("Earth Engine initialized (project=%s)", PROJECT_ID)


def _mask_clouds_scale(image: ee.Image) -> ee.Image:
    qa = image.select("QA60")
    clear = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(clear).divide(10000).select(BANDS)


def _find_composite(
    lat: float, lon: float, date_str: str, patch_half_m: float,
) -> Optional[Tuple[ee.Image, int]]:
    """Find best Sentinel-2 composite using cascading search windows."""
    point = ee.Geometry.Point([lon, lat])
    aoi = point.buffer(patch_half_m).bounds()

    for days, cloud_pct in SEARCH_WINDOWS:
        start = ee.Date(date_str).advance(-days, "day")
        end = ee.Date(date_str).advance(days, "day")
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
            .map(_mask_clouds_scale)
        )
        count = col.size().getInfo()
        if count and count > 0:
            composite = col.median().toFloat().clip(aoi)
            return composite, count
    return None


def _download_patch(
    composite: ee.Image, lat: float, lon: float,
    patch_size: int, scale: int,
) -> Optional[np.ndarray]:
    """Download a (N_BANDS, patch_size, patch_size) numpy array via
    sampleRectangle().

    sampleRectangle() returns a Feature whose properties are 2-D arrays
    (one per band).  We stack them into a single numpy array.
    """
    half_m = (patch_size * scale) / 2.0
    point = ee.Geometry.Point([lon, lat])
    rect = point.buffer(half_m).bounds()

    # Reproject every band to a common 10 m grid before sampling
    composite_reproj = composite.reproject(crs="EPSG:4326", scale=scale)

    try:
        feat = composite_reproj.sampleRectangle(
            region=rect, defaultValue=0
        ).getInfo()
    except Exception as exc:
        log.warning("sampleRectangle failed: %s", exc)
        return None

    if feat is None or "properties" not in feat:
        return None

    props = feat["properties"]
    arrays = []
    for band in BANDS:
        arr = props.get(band)
        if arr is None:
            return None
        arrays.append(np.array(arr, dtype=np.float32))

    patch = np.stack(arrays, axis=0)  # (N_BANDS, H, W)

    # Pad or crop to exact patch_size × patch_size
    _, h, w = patch.shape
    if h < patch_size or w < patch_size:
        padded = np.zeros((N_BANDS, patch_size, patch_size), dtype=np.float32)
        padded[:, :min(h, patch_size), :min(w, patch_size)] = patch[
            :, :min(h, patch_size), :min(w, patch_size)
        ]
        patch = padded
    elif h > patch_size or w > patch_size:
        sh = (h - patch_size) // 2
        sw = (w - patch_size) // 2
        patch = patch[:, sh:sh + patch_size, sw:sw + patch_size]

    return patch


# ──────────────────────────────────────────────────────────────────────────────
# Per station-day worker
# ──────────────────────────────────────────────────────────────────────────────
def _patch_filename(sid: str, date_str: str) -> str:
    safe_sid = sid.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return f"{safe_sid}_{date_str}.npy"


def process_one(
    info: Dict[str, Any], patch_size: int, scale: int,
) -> Optional[Dict[str, Any]]:
    """Download one patch. Returns metadata dict or None on failure."""
    sid = info["stationId"]
    date_str = info["date"]
    lat, lon = info["lat"], info["lon"]
    fname = _patch_filename(sid, date_str)
    fpath = PATCH_DIR / fname

    # Skip if already exists
    if fpath.exists():
        return {**info, "file": fname, "status": "cached"}

    half_m = (patch_size * scale) / 2.0

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            result = _find_composite(lat, lon, date_str, half_m)
            if result is None:
                return {**info, "file": None, "status": "no_imagery"}

            composite, scene_count = result
            patch = _download_patch(composite, lat, lon, patch_size, scale)
            if patch is None:
                return {**info, "file": None, "status": "download_fail"}

            np.save(fpath, patch)
            return {**info, "file": fname, "status": "ok", "scenes": scene_count}

        except Exception as exc:
            log.warning(
                "  [%s %s] attempt %d/%d failed: %s",
                sid, date_str, attempt, RETRY_LIMIT, exc,
            )
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)

    return {**info, "file": None, "status": "error"}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download Sentinel-2 patches")
    parser.add_argument("--patch_size", type=int, default=PATCH_SIZE_DEFAULT,
                        help="Patch size in pixels (default 64 → 640 m)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing patches and re-download")
    args = parser.parse_args()

    PATCH_DIR.mkdir(parents=True, exist_ok=True)

    _init_ee()

    station_days = load_station_days(CSV_PATH)
    total = len(station_days)
    log.info("Loaded %d station-days from CSV", total)

    # Prepare task list
    tasks = list(station_days.values())

    # If --fresh, remove existing patches
    if args.fresh:
        for f in PATCH_DIR.glob("*.npy"):
            f.unlink()
        log.info("Cleared existing patches (--fresh)")

    # Count already cached
    cached = sum(1 for t in tasks if (PATCH_DIR / _patch_filename(t["stationId"], t["date"])).exists())
    log.info("Already cached: %d / %d  (will skip)", cached, total)

    # Process
    results = []
    ok = 0
    skipped = 0
    cached_count = 0
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, info, args.patch_size, SCALE): info
            for info in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res is None:
                failed += 1
                continue
            results.append(res)

            status = res.get("status", "?")
            if status == "ok":
                ok += 1
            elif status == "cached":
                cached_count += 1
            elif status == "no_imagery":
                skipped += 1
            else:
                failed += 1

            if i % 50 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                log.info(
                    "  [%d/%d]  ok=%d  cached=%d  skip=%d  fail=%d  "
                    "(%.1f/min, %.1f min elapsed)",
                    i, total, ok, cached_count, skipped, failed,
                    rate * 60, elapsed / 60,
                )

    # Write metadata CSV
    if results:
        meta_fields = (
            ["stationId", "date", "lat", "lon", "file", "status"]
            + [f"wq_{c}" for c in WQ_COLS]
        )
        with META_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=meta_fields, extrasaction="ignore"
            )
            writer.writeheader()
            for r in sorted(results, key=lambda x: (x["stationId"], x["date"])):
                writer.writerow(r)
        log.info("Saved metadata → %s", META_CSV)

    elapsed = time.time() - t0
    log.info(
        "DONE  ok=%d  cached=%d  skipped=%d  failed=%d  total_time=%.1f min",
        ok, cached_count, skipped, failed, elapsed / 60,
    )
    log.info(
        "Patches saved in: %s   shape: (%d, %d, %d)",
        PATCH_DIR, N_BANDS, args.patch_size, args.patch_size,
    )


if __name__ == "__main__":
    main()
