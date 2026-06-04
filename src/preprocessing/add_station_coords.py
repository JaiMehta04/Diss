"""Add latitude/longitude to the measurements CSV using a station lookup.

Steps:
1) Create station_coords.csv with columns: stationId,latitude,longitude
2) Run this script (after activating your venv):
   python add_station_coords.py
3) It produces combined_station_data_with_coords.csv, which you can feed to ganga_water_quality.py
"""
import csv
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = str(PROJECT_ROOT / "data" / "combined_station_data.csv")
COORDS_CSV = str(PROJECT_ROOT / "data" / "station_coords.csv")
OUTPUT_CSV = str(PROJECT_ROOT / "data" / "combined_station_data_with_coords.csv")


def load_coords(path: Path) -> Dict[str, Tuple[float, float]]:
    coords: Dict[str, Tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"stationId", "latitude", "longitude"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"station coords file missing columns: {sorted(missing)}; "
                "expected stationId, latitude, longitude"
            )
        for row in reader:
            sid = (row.get("stationId") or "").strip()
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (TypeError, ValueError):
                continue
            if sid:
                coords[sid] = (lat, lon)
    if not coords:
        raise ValueError("No valid station coordinates found in station_coords.csv")
    return coords


def enrich_measurements(input_path: Path, coords_path: Path, output_path: Path) -> None:
    coords = load_coords(coords_path)

    with input_path.open(newline="", encoding="utf-8") as input_handle, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header")

        fieldnames = list(reader.fieldnames)
        for col in ("latitude", "longitude"):
            if col not in fieldnames:
                fieldnames.append(col)

        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()

        filled = 0
        missing_ids = set()
        total = 0

        for row in reader:
            total += 1
            sid = (row.get("stationId") or "").strip()
            lat_lon = coords.get(sid)
            if lat_lon:
                lat, lon = lat_lon
                if not row.get("latitude"):
                    row["latitude"] = f"{lat:.6f}"
                if not row.get("longitude"):
                    row["longitude"] = f"{lon:.6f}"
                filled += 1
            else:
                missing_ids.add(sid)
            writer.writerow(row)

    print(f"Wrote {output_path} with {filled} row(s) updated out of {total}.")
    if missing_ids:
        print(
            f"Warning: {len(missing_ids)} stationId(s) had no coordinates. "
            f"Example missing IDs: {sorted(list(missing_ids))[:5]}"
        )


def main() -> None:
    input_path = Path(INPUT_CSV)
    coords_path = Path(COORDS_CSV)
    output_path = Path(OUTPUT_CSV)

    if not input_path.exists():
        raise FileNotFoundError(f"Input measurements CSV not found: {input_path}")
    if not coords_path.exists():
        raise FileNotFoundError(f"Station coords CSV not found: {coords_path}")

    enrich_measurements(input_path, coords_path, output_path)


if __name__ == "__main__":
    main()
