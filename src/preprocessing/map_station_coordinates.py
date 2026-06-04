import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMBINED_CSV = PROJECT_ROOT / "data" / "combined_station_data.csv"
LOCATIONS_CSV = PROJECT_ROOT / "data" / "cpcb_station_locations.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "combined_station_data_with_coords.csv"

EXTRA_FIELDS = [
    "station_no",
    "station_name",
    "latitude",
    "longitude",
    "state",
    "site_name",
    "catchment",
    "status_remark",
]

def normalize_station_id(value: str) -> str:
    """Normalize station id values to string integers for consistent matching."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text

def load_locations(path: Path) -> dict:
    mapping: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = normalize_station_id(row.get("station_id"))
            if key:
                mapping[key] = row
    return mapping

def merge_coordinates(combined_path: Path, locations: dict, output_path: Path) -> None:
    with combined_path.open(newline="", encoding="utf-8") as combined_file, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        combined_reader = csv.DictReader(combined_file)
        fieldnames = list(combined_reader.fieldnames or []) + [f for f in EXTRA_FIELDS if f not in combined_reader.fieldnames]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in combined_reader:
            key = normalize_station_id(row.get("stationId"))
            location = locations.get(key, {})
            for field in EXTRA_FIELDS:
                row[field] = location.get(field, "")
            writer.writerow(row)

def main() -> None:
    locations = load_locations(LOCATIONS_CSV)
    merge_coordinates(COMBINED_CSV, locations, OUTPUT_CSV)

if __name__ == "__main__":
    main()
