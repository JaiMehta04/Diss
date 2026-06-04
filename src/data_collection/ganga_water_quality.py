"""Export Sentinel-2 patches (bands + indices) aligned to station measurements.

Assumptions
- The CSV now contains latitude/longitude columns named `latitude` and `longitude`.
- `timestamp` stays in ISO-8601 (e.g., 2025-07-29T09:30:00.000Z).
- Authenticate once with `earthengine authenticate` or by uncommenting ee.Authenticate().
"""
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "compat"))

import ee

# Uncomment if you need interactive auth
# ee.Authenticate()
ee.Initialize(project="dissertation-ganga")

CSV_PATH = str(PROJECT_ROOT / "data" / "combined_station_data.csv")
EXPORT_FOLDER = "GangaExports"  # Google Drive folder; create it manually if it does not exist
DATE_WINDOW_DAYS = 3  # how wide a window (before/after) to search for a scene around each timestamp
PATCH_RADIUS_M = 500  # buffer radius around each station point
CLOUDY_PIXEL_PERCENTAGE = 60
MAX_EXPORTS = 20  # safety stop so you do not enqueue thousands of tasks accidentally
BASE_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
REQUIRED_COLUMNS = {"stationId", "timestamp", "latitude", "longitude"}


def parse_measurements(path: str) -> Iterable[Dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV missing required columns: {sorted(missing)}. "
                "Add latitude/longitude columns before running."
            )

        for row in reader:
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (TypeError, ValueError):
                continue

            ts_raw = row["timestamp"]
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                continue

            yield {
                "stationId": row["stationId"],
                "lat": lat,
                "lon": lon,
                "timestamp_str": ts_raw,
                "timestamp": ts,
            }


def mask_and_scale(image: ee.Image) -> ee.Image:
    qa = image.select("QA60")
    cloud_free = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    scaled = image.updateMask(cloud_free).divide(10000).select(BASE_BANDS)
    return scaled.copyProperties(image, image.propertyNames())


def add_indices(image: ee.Image) -> ee.Image:
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndti = image.normalizedDifference(["B4", "B3"]).rename("NDTI")
    r705 = image.select("B5")
    r665 = image.select("B4")
    r740 = image.select("B6")
    mci = r705.subtract(r665).subtract(
        ee.Number(705 - 665).divide(740 - 665).multiply(r740.subtract(r665))
    ).rename("MCI")
    return image.addBands([ndwi, ndti, mci])


def prepare_collection(point: ee.Geometry, timestamp_str: str) -> ee.ImageCollection:
    start = ee.Date(timestamp_str).advance(-DATE_WINDOW_DAYS, "day")
    end = ee.Date(timestamp_str).advance(DATE_WINDOW_DAYS, "day")
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point.buffer(PATCH_RADIUS_M))
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUDY_PIXEL_PERCENTAGE))
        .map(mask_and_scale)
        .map(add_indices)
    )


def export_patch(measurement: Dict) -> None:
    point = ee.Geometry.Point([measurement["lon"], measurement["lat"]])
    collection = prepare_collection(point, measurement["timestamp_str"])

    if collection.size().getInfo() == 0:
        print(
            f"Skip station {measurement['stationId']} @ {measurement['timestamp_str']}: no scenes in window."
        )
        return

    image = collection.sort("CLOUDY_PIXEL_PERCENTAGE").first()
    region = point.buffer(PATCH_RADIUS_M).bounds()
    name = f"station{measurement['stationId']}_{measurement['timestamp'].strftime('%Y%m%dT%H%M')}"

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=name,
        folder=EXPORT_FOLDER,
        fileNamePrefix=name,
        scale=10,
        region=region,
        maxPixels=1e13,
    )
    task.start()
    print(f"Started export: {name} (task {task.id})")


def main() -> None:
    measurements = parse_measurements(CSV_PATH)
    started = 0

    for measurement in measurements:
        if MAX_EXPORTS and started >= MAX_EXPORTS:
            break
        export_patch(measurement)
        started += 1

    print(f"Queued {started} export task(s). Check https://code.earthengine.google.com/tasks to monitor.")


if __name__ == "__main__":
    main()
