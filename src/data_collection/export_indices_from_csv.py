"""Export NDWI/NDTI/MCI composites for every measurement row in a CSV.

Steps
- Ensure `combined_station_data_with_coords.csv` has `stationId`, `timestamp`, `latitude`, `longitude` columns.
- Authenticate once with `earthengine authenticate` or uncomment `ee.Authenticate()`.
- Adjust DATE_WINDOW_DAYS and BUFFER_METERS if you want a different window or footprint.
"""
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "compat"))

import ee

# Uncomment if you need interactive auth
# ee.Authenticate()
ee.Initialize(project="dissertation-ganga")

CSV_PATH = PROJECT_ROOT / "data" / "combined_station_data_with_coords.csv"
DATE_WINDOW_DAYS = 3  # ± days around each timestamp row
FALLBACK_WINDOW_DAYS = 10  # used if the narrow window returns no scenes
WIDE_WINDOW_DAYS = 30  # last-resort window
BUFFER_METERS = 3000  # 3 km footprint similar to the JS example
CLOUDY_PIXEL_PERCENTAGE = 20
FALLBACK_CLOUDY_PIXEL_PERCENTAGE = 60  # used if strict cloud filter yields none
WIDE_CLOUDY_PIXEL_PERCENTAGE = 100  # last-resort cloud filter
EXPORT_FOLDER = "GangaIndices"  # Google Drive folder; create it manually if absent
EXPORT_SCALE = 10
MAX_EXPORTS = 0  # 0 or None to export all station-day groups (beware task quotas)

BASE_BANDS = ["B2", "B3", "B4", "B5", "B6", "B8"]
REQUIRED_COLUMNS = {"stationId", "timestamp", "latitude", "longitude"}


def normalize_station_id(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def load_measurements(path: Path) -> Iterable[Dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV missing required columns: {sorted(missing)}. Add timestamp/latitude/longitude before running."
            )

        for row in reader:
            key = normalize_station_id(row.get("stationId"))
            if not key:
                continue
            ts_raw = str(row.get("timestamp", "")).strip()
            if not ts_raw:
                continue
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (TypeError, ValueError):
                continue
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            yield {
                "stationId": key,
                "lat": lat,
                "lon": lon,
                "timestamp": ts_raw,
                "date": ts.date().isoformat(),
            }


def group_by_station_day(rows: Iterable[Dict]) -> Dict[Tuple[str, str], Dict]:
    grouped: Dict[Tuple[str, str], Dict] = {}
    for row in rows:
        key = (row["stationId"], row["date"])
        if key not in grouped:
            grouped[key] = {
                "stationId": row["stationId"],
                "date": row["date"],
                "lat": row["lat"],
                "lon": row["lon"],
                "representative_ts": row["timestamp"],
            }
    return grouped


def mask_and_scale(image: ee.Image) -> ee.Image:
    qa = image.select("QA60")
    cloud_free = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_free).divide(10000)


def add_indices(image: ee.Image) -> ee.Image:
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndti = image.normalizedDifference(["B4", "B3"]).rename("NDTI")
    r705 = image.select("B5")
    r665 = image.select("B4")
    r740 = image.select("B6")
    coeff = ee.Number(705 - 665).divide(740 - 665)
    mci = r705.subtract(r665).subtract(r740.subtract(r665).multiply(coeff)).rename("MCI")
    return image.addBands([ndwi, ndti, mci])


def build_collection(point: ee.Geometry, timestamp_str: str, days: int, cloud_pct: int) -> ee.ImageCollection:
    start = ee.Date(timestamp_str).advance(-days, "day")
    end = ee.Date(timestamp_str).advance(days, "day")
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point.buffer(BUFFER_METERS))
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(mask_and_scale)
    )


def build_composite(point: ee.Geometry, timestamp_str: str) -> ee.Image:
    primary = build_collection(point, timestamp_str, DATE_WINDOW_DAYS, CLOUDY_PIXEL_PERCENTAGE)
    size_primary = primary.size().getInfo()

    collection = primary
    if not size_primary or size_primary == 0:
        # Relax filters if nothing found
        fallback = build_collection(
            point, timestamp_str, FALLBACK_WINDOW_DAYS, FALLBACK_CLOUDY_PIXEL_PERCENTAGE
        )
        size_fallback = fallback.size().getInfo()
        if size_fallback and size_fallback > 0:
            collection = fallback
            print(
                f"Using fallback window/clouds for timestamp {timestamp_str}: "
                f"{size_fallback} scenes (±{FALLBACK_WINDOW_DAYS}d, cloud<{FALLBACK_CLOUDY_PIXEL_PERCENTAGE}%)"
            )
        else:
            # Last-resort wide search
            wide = build_collection(
                point, timestamp_str, WIDE_WINDOW_DAYS, WIDE_CLOUDY_PIXEL_PERCENTAGE
            )
            size_wide = wide.size().getInfo()
            if size_wide and size_wide > 0:
                collection = wide
                print(
                    f"Using wide window/clouds for timestamp {timestamp_str}: "
                    f"{size_wide} scenes (±{WIDE_WINDOW_DAYS}d, cloud<{WIDE_CLOUDY_PIXEL_PERCENTAGE}%)"
                )
            else:
                return None

    composite = collection.median()
    composite = add_indices(composite).toFloat()
    return composite.select(["NDWI", "NDTI", "MCI", "B4", "B3", "B2"])


def export_indices(measurement: Dict) -> bool:
    point = ee.Geometry.Point([measurement["lon"], measurement["lat"]])
    composite = build_composite(point, measurement["representative_ts"])
    if composite is None:
        print(
            f"Skip station {measurement['stationId']} @ {measurement['date']}: "
            f"no scenes even after wide window (±{WIDE_WINDOW_DAYS}d, cloud<{WIDE_CLOUDY_PIXEL_PERCENTAGE}%)."
        )
        return False
    region = point.buffer(BUFFER_METERS).bounds()
    name = (
        f"station{measurement['stationId']}_"
        f"{measurement['date']}_"
        f"pm{DATE_WINDOW_DAYS}d"
    )

    size = ee.Number(composite.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=region,
        scale=EXPORT_SCALE,
        maxPixels=1e8,
    ).get("NDWI"))
    try:
        if size.getInfo() is None:
            print(f"Skip station {measurement['stationId']} @ {measurement['timestamp']}: no pixels in window.")
            return False
    except Exception:
        # If reduceRegion fails (rare for very small geometries), proceed with export anyway
        pass

    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=name,
        folder=EXPORT_FOLDER,
        fileNamePrefix=name,
        scale=EXPORT_SCALE,
        region=region,
        maxPixels=1e13,
    )
    task.start()
    print(f"Started export: {name} (task {task.id})")
    return True


def main() -> None:
    rows = list(load_measurements(CSV_PATH))
    grouped = group_by_station_day(rows)
    started = 0
    for measurement in grouped.values():
        if MAX_EXPORTS and started >= MAX_EXPORTS:
            break
        if export_indices(measurement):
            started += 1
    print(
        f"Queued {started} export task(s). Monitor https://code.earthengine.google.com/tasks for progress."
    )


if __name__ == "__main__":
    main()
