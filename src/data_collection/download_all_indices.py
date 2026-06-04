"""
download_all_indices.py
=======================
Extract ALL water-quality-relevant Sentinel-2 indices for every station-day
in the CSV dataset.  No artificial limits.

Two modes
---------
  tabular  (default) — Extract numeric index values via reduceRegion → local CSV.
                        Best for Bayesian Network training.
  image              — Export GeoTIFF composites to Google Drive folder.
                        Best for CNN / spatial analysis.

Outputs (in GangaIndices_All/)
-------------------------------
  satellite_indices.csv   — Sentinel-2 index values per station-day
  water_quality_daily.csv — Daily-averaged water quality per station-day
  merged_bn_dataset.csv   — Joined dataset ready for Bayesian Network training

Usage
-----
  python download_all_indices.py                     # tabular, all station-days
  python download_all_indices.py --mode image        # image export to Drive
  python download_all_indices.py --workers 8         # 8 parallel workers
  python download_all_indices.py --fresh             # ignore checkpoint, start fresh
"""

import argparse
import csv
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Project root (two levels up from src/data_collection/) ────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "compat"))

import ee

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ID = "dissertation-ganga"
CSV_PATH = PROJECT_ROOT / "data" / "combined_station_data_with_coords.csv"
OUTPUT_DIR = PROJECT_ROOT / "GangaIndices_All"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.json"
SAT_CSV_PATH = OUTPUT_DIR / "satellite_indices.csv"
WQ_CSV_PATH = OUTPUT_DIR / "water_quality_daily.csv"
MERGED_CSV_PATH = OUTPUT_DIR / "merged_bn_dataset.csv"

BUFFER_METERS = 500          # buffer radius around each station (metres)
SCALE = 10                   # Sentinel-2 native resolution

# Cascading search windows: (± days around timestamp, max cloud %)
SEARCH_WINDOWS = [
    (3, 20),
    (10, 60),
    (30, 100),
]

DRIVE_FOLDER = "GangaIndices_All"      # Google Drive folder for image mode
MAX_WORKERS_DEFAULT = 4
RETRY_LIMIT = 3
RETRY_DELAY_SEC = 5

# Water quality columns to aggregate to daily means
WQ_NUMERIC_COLS = [
    "Biochemical Oxygen Demand",
    "Chemical Oxygen Demand",
    "Chloride",
    "Conductivity",
    "Depth",
    "Dissolved Oxygen",
    "Nitrate",
    "Total Organic Carbon",
    "Water Level",
    "Water Temperature",
    "Water Turbidity",
    "pH",
]

# Sentinel-2 bands to retain after cloud masking
BASE_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Names of all bands/indices we extract
INDEX_NAMES = ["NDWI", "MNDWI", "NDTI", "NDVI", "NDCI", "MCI", "FAI", "SABI"]
EXTRACT_BANDS = BASE_BANDS + INDEX_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CSV loading & grouping
# ──────────────────────────────────────────────────────────────────────────────
def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_station_days(
    rows: List[Dict[str, str]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Group rows by (stationId, date).

    Returns a dict keyed by (stationId, date_iso) with representative metadata
    and daily-averaged water quality values.
    """
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
            lat = float(row["latitude"])
            lon = float(row["longitude"])
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
                "timestamp": ts_raw,
            }

        for col in WQ_NUMERIC_COLS:
            val = row.get(col, "").strip()
            if val:
                try:
                    wq_accum[key][col].append(float(val))
                except ValueError:
                    pass

    # Attach daily-mean WQ values
    for key, info in groups.items():
        for col in WQ_NUMERIC_COLS:
            vals = wq_accum[key].get(col, [])
            info[f"wq_{col}"] = round(sum(vals) / len(vals), 4) if vals else None

    return groups


# ──────────────────────────────────────────────────────────────────────────────
# Earth Engine helpers
# ──────────────────────────────────────────────────────────────────────────────
def mask_clouds_and_scale(image: ee.Image) -> ee.Image:
    """Mask opaque & cirrus clouds via QA60 and scale reflectance to [0, 1]."""
    qa = image.select("QA60")
    clear = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(clear).divide(10000).select(BASE_BANDS)


def compute_indices(image: ee.Image) -> ee.Image:
    """Compute comprehensive water-quality-relevant spectral indices."""
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = image.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndti = image.normalizedDifference(["B4", "B3"]).rename("NDTI")
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndci = image.normalizedDifference(["B5", "B4"]).rename("NDCI")

    # MCI — Maximum Chlorophyll Index
    r665 = image.select("B4")
    r705 = image.select("B5")
    r740 = image.select("B6")
    lam = ee.Number(705 - 665).divide(740 - 665)
    mci = r705.subtract(r665).subtract(
        r740.subtract(r665).multiply(lam)
    ).rename("MCI")

    # FAI — Floating Algae Index
    r842 = image.select("B8")
    r1610 = image.select("B11")
    fai_lam = ee.Number(842 - 665).divide(1610 - 665)
    fai = r842.subtract(r665).subtract(
        r1610.subtract(r665).multiply(fai_lam)
    ).rename("FAI")

    # SABI — Surface Algal Bloom Index
    sabi = (
        image.select("B8").subtract(image.select("B4"))
        .divide(image.select("B3").add(image.select("B2")))
        .rename("SABI")
    )

    return image.addBands([ndwi, mndwi, ndti, ndvi, ndci, mci, fai, sabi])


def build_composite(
    point: ee.Geometry, timestamp_str: str
) -> Optional[Tuple[ee.Image, int]]:
    """Build a cloud-free median composite with cascading window fallbacks.

    Returns (composite_image, scene_count) or None when no imagery exists
    in any search window.
    """
    for days, cloud_pct in SEARCH_WINDOWS:
        start = ee.Date(timestamp_str).advance(-days, "day")
        end = ee.Date(timestamp_str).advance(days, "day")
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point.buffer(BUFFER_METERS))
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
            .map(mask_clouds_and_scale)
        )
        count = col.size().getInfo()
        if count and count > 0:
            composite = compute_indices(col.median().toFloat())
            return composite, count
    return None


def extract_values(
    composite: ee.Image, region: ee.Geometry
) -> Optional[Dict[str, float]]:
    """Extract spatial-mean values of all bands/indices over the region."""
    result = (
        composite.select(EXTRACT_BANDS)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=SCALE,
            maxPixels=1e8,
        )
        .getInfo()
    )
    if result is None or all(v is None for v in result.values()):
        return None
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Per-station-day processing
# ──────────────────────────────────────────────────────────────────────────────
def process_station_day(info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build composite → extract index values for one station-day."""
    point = ee.Geometry.Point([info["lon"], info["lat"]])
    result_data = build_composite(point, info["timestamp"])
    if result_data is None:
        return None
    composite, scene_count = result_data
    region = point.buffer(BUFFER_METERS)
    values = extract_values(composite, region)
    if values is None:
        return None
    values["scene_count"] = scene_count
    values["stationId"] = info["stationId"]
    values["date"] = info["date"]
    values["lat"] = info["lat"]
    values["lon"] = info["lon"]
    return values


def process_with_retry(info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Wrap process_station_day with automatic retries."""
    key = f"{info['stationId']}_{info['date']}"
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            return process_station_day(info)
        except Exception as exc:
            if attempt < RETRY_LIMIT:
                log.warning(
                    "%s: attempt %d failed (%s), retrying in %ds …",
                    key, attempt, exc, RETRY_DELAY_SEC,
                )
                time.sleep(RETRY_DELAY_SEC)
            else:
                log.error("%s: all %d attempts failed: %s", key, RETRY_LIMIT, exc)
                return None


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint management  —  allows resuming after interruptions
# ──────────────────────────────────────────────────────────────────────────────
def load_checkpoint() -> set:
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        return set(data.get("completed", []))
    return set()


def save_checkpoint(completed: set) -> None:
    CHECKPOINT_PATH.write_text(
        json.dumps({"completed": sorted(completed)}, indent=2),
        encoding="utf-8",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tabular extraction mode
# ──────────────────────────────────────────────────────────────────────────────
SAT_HEADER = ["stationId", "date", "lat", "lon", "scene_count"] + EXTRACT_BANDS


def run_tabular(
    pending: List[Dict],
    completed: set,
    max_workers: int,
    station_days: Dict,
) -> None:
    """Extract index values for every pending station-day → CSV."""
    file_exists = SAT_CSV_PATH.exists() and len(completed) > 0
    mode = "a" if file_exists else "w"
    csvfile = SAT_CSV_PATH.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(csvfile, fieldnames=SAT_HEADER, extrasaction="ignore")
    if not file_exists:
        writer.writeheader()

    total = len(pending) + len(completed)
    done_count = len(completed)
    extracted = 0
    skipped = 0
    start_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(process_with_retry, info): info for info in pending
            }
            for future in as_completed(futures):
                info = futures[future]
                key = f"{info['stationId']}_{info['date']}"
                result = future.result()
                done_count += 1

                if result:
                    writer.writerow(result)
                    csvfile.flush()
                    extracted += 1
                    elapsed = time.time() - start_time
                    rate = (done_count - len(load_checkpoint()) + extracted) / max(elapsed, 1)
                    eta_sec = (total - done_count) / max(rate, 0.01)
                    log.info(
                        "[%d/%d] OK  %s  (scenes=%s, elapsed=%dm, ~%dm left)",
                        done_count, total, key,
                        result.get("scene_count", "?"),
                        int(elapsed // 60),
                        int(eta_sec // 60),
                    )
                else:
                    skipped += 1
                    log.warning("[%d/%d] SKIP %s — no Sentinel-2 data", done_count, total, key)

                completed.add(key)

                # Save checkpoint periodically
                if done_count % 10 == 0:
                    save_checkpoint(completed)

    except KeyboardInterrupt:
        log.warning("Interrupted!  Saving checkpoint …")
    finally:
        csvfile.close()
        save_checkpoint(completed)

    elapsed_total = time.time() - start_time
    log.info(
        "Tabular extraction complete: %d extracted, %d skipped, %.1f min elapsed.",
        extracted, skipped, elapsed_total / 60,
    )
    merge_outputs(station_days)


# ──────────────────────────────────────────────────────────────────────────────
# Image export mode
# ──────────────────────────────────────────────────────────────────────────────
def export_image_to_drive(info: Dict[str, Any]) -> Optional[str]:
    """Export a composite GeoTIFF to Google Drive and return the task ID."""
    point = ee.Geometry.Point([info["lon"], info["lat"]])
    result_data = build_composite(point, info["timestamp"])
    if result_data is None:
        return None
    composite, _ = result_data
    region = point.buffer(BUFFER_METERS).bounds()
    name = f"station{info['stationId']}_{info['date']}"
    task = ee.batch.Export.image.toDrive(
        image=composite.select(EXTRACT_BANDS),
        description=name,
        folder=DRIVE_FOLDER,
        fileNamePrefix=name,
        scale=SCALE,
        region=region,
        maxPixels=1e13,
    )
    task.start()
    return task.id


def run_image_export(pending: List[Dict], completed: set) -> None:
    """Queue image-export tasks for every pending station-day → Google Drive."""
    total = len(pending) + len(completed)
    started = len(completed)
    skipped = 0

    for info in pending:
        key = f"{info['stationId']}_{info['date']}"
        try:
            task_id = export_image_to_drive(info)
            if task_id:
                started += 1
                completed.add(key)
                log.info("[%d/%d] Export started: %s (task %s)", started, total, key, task_id)
            else:
                skipped += 1
                completed.add(key)
                log.warning("[%d/%d] SKIP %s — no data", started + skipped, total, key)
        except Exception as exc:
            log.error("Failed: %s — %s", key, exc)

        # Save checkpoint periodically
        if (started + skipped) % 50 == 0:
            save_checkpoint(completed)

    save_checkpoint(completed)
    log.info(
        "Queued %d export tasks (%d skipped). "
        "Monitor at https://code.earthengine.google.com/tasks",
        started - len(load_checkpoint()), skipped,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Merge satellite + water quality into BN-ready dataset
# ──────────────────────────────────────────────────────────────────────────────
def merge_outputs(station_days: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
    """Join satellite index CSV with daily WQ averages → merged_bn_dataset.csv."""
    if not SAT_CSV_PATH.exists():
        log.warning("No satellite CSV found — skipping merge.")
        return

    sat_data: Dict[Tuple[str, str], Dict[str, str]] = {}
    with SAT_CSV_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sat_data[(row["stationId"], row["date"])] = row

    merged_rows: List[Dict[str, Any]] = []
    for (sid, date), info in station_days.items():
        sat = sat_data.get((sid, date))
        if not sat:
            continue
        merged: Dict[str, Any] = {
            "stationId": sid,
            "date": date,
            "lat": info["lat"],
            "lon": info["lon"],
        }
        for col in WQ_NUMERIC_COLS:
            merged[f"wq_{col}"] = info.get(f"wq_{col}")
        for band in EXTRACT_BANDS:
            merged[band] = sat.get(band)
        merged["scene_count"] = sat.get("scene_count")
        merged_rows.append(merged)

    if not merged_rows:
        log.warning("No rows to merge.")
        return

    header = (
        ["stationId", "date", "lat", "lon"]
        + [f"wq_{c}" for c in WQ_NUMERIC_COLS]
        + EXTRACT_BANDS
        + ["scene_count"]
    )
    with MERGED_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)

    log.info(
        "Merged BN dataset written: %s (%d rows, %d columns)",
        MERGED_CSV_PATH, len(merged_rows), len(header),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Write daily water quality CSV
# ──────────────────────────────────────────────────────────────────────────────
def write_wq_daily(station_days: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
    """Write daily-averaged water quality to a standalone CSV."""
    header = ["stationId", "date", "lat", "lon"] + [f"wq_{c}" for c in WQ_NUMERIC_COLS]
    with WQ_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for info in station_days.values():
            writer.writerow({k: info.get(k) for k in header})
    log.info("Daily water quality CSV: %s (%d rows)", WQ_CSV_PATH, len(station_days))


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 indices for ALL station-days (no limit)."
    )
    parser.add_argument(
        "--mode", choices=["tabular", "image"], default="tabular",
        help="tabular = extract values to CSV (default); image = GeoTIFFs to Drive",
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS_DEFAULT,
        help="parallel GEE workers (tabular mode only, default: 4)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="ignore existing checkpoint and start from scratch",
    )
    args = parser.parse_args()

    # ── Initialise Earth Engine ────────────────────────────────────────────
    ee.Initialize(project=PROJECT_ID)
    log.info("Earth Engine initialised (project=%s).", PROJECT_ID)

    # ── Load & group data ──────────────────────────────────────────────────
    rows = load_csv(CSV_PATH)
    station_days = group_station_days(rows)
    log.info(
        "CSV loaded: %d measurement rows → %d unique station-days across %d stations.",
        len(rows),
        len(station_days),
        len({k[0] for k in station_days}),
    )

    # ── Prepare output directory ───────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Write daily WQ CSV ─────────────────────────────────────────────────
    write_wq_daily(station_days)

    # ── Load checkpoint ────────────────────────────────────────────────────
    completed = set() if args.fresh else load_checkpoint()
    if completed:
        log.info("Resuming from checkpoint: %d station-days already processed.", len(completed))

    pending = [
        info
        for (sid, date), info in station_days.items()
        if f"{info['stationId']}_{info['date']}" not in completed
    ]
    log.info("Pending: %d station-days to process.", len(pending))

    if not pending:
        log.info("All station-days already processed. Merging outputs …")
        merge_outputs(station_days)
        return

    # ── Dispatch ───────────────────────────────────────────────────────────
    if args.mode == "tabular":
        run_tabular(pending, completed, args.workers, station_days)
    else:
        run_image_export(pending, completed)


if __name__ == "__main__":
    main()
