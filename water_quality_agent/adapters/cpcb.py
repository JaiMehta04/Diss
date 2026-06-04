"""
CPCB (Central Pollution Control Board) data adapter.

Handles:
  1. Local CSV files already downloaded from CPCB ENVIS portal
  2. Live scraping of CPCB water quality data (with rate limiting)

CPCB CSV format typically has columns like:
  stationId, timestamp, Biochemical Oxygen Demand, Chemical Oxygen Demand,
  Chloride, Conductivity, Dissolved Oxygen, Fecal Coliform, Nitrate,
  pH, Temperature, Total Organic Carbon, Turbidity, Water Level, ...
"""

import csv
import json
import logging
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from .base import BaseAdapter
from ..config.schema import CanonicalRecord

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Known CPCB station metadata
# We maintain a lookup of station IDs to names/rivers/coordinates.
# This is populated from the station locations CSV if available.
# ──────────────────────────────────────────────────────────────────────────────
_KNOWN_RIVERS = {
    "ganga", "yamuna", "brahmaputra", "godavari", "narmada",
    "krishna", "kaveri", "mahanadi", "sutlej", "beas", "chenab",
    "ravi", "jhelum", "chambal", "betwa", "gomti", "ghaghra",
    "kosi", "gandak", "son", "damodar", "hooghly", "sabarmati",
    "mahi", "tapi", "pennar", "tungabhadra", "bhima",
}

# Parameters as they appear in CPCB CSVs (column names)
_CPCB_PARAM_UNIT_MAP = {
    "Biochemical Oxygen Demand":    ("bod",         "mg/L"),
    "Chemical Oxygen Demand":       ("cod",         "mg/L"),
    "Chloride":                     ("chloride",    "mg/L"),
    "Conductivity":                 ("conductivity","µS/cm"),
    "Dissolved Oxygen":             ("do",          "mg/L"),
    "Fecal Coliform":               ("fecal_coliform", "MPN/100mL"),
    "Nitrate":                      ("nitrate",     "mg/L"),
    "pH":                           ("ph",          ""),
    "Temperature":                  ("temperature", "°C"),
    "Water Temperature":            ("temperature", "°C"),
    "Total Organic Carbon":         ("toc",         "mg/L"),
    "Turbidity":                    ("turbidity",   "NTU"),
    "Water Turbidity":              ("turbidity",   "NTU"),
    "Water Level":                  ("water_level", "m"),
    "Depth":                        ("depth",       "m"),
    # Additional parameters sometimes present
    "Total Dissolved Solids":       ("tds",         "mg/L"),
    "Total Suspended Solids":       ("tss",         "mg/L"),
    "Alkalinity":                   ("alkalinity",  "mg/L"),
    "Hardness":                     ("hardness",    "mg/L"),
    "Fluoride":                     ("fluoride",    "mg/L"),
    "Sulphate":                     ("sulphate",    "mg/L"),
    "Phosphate":                    ("phosphate",   "mg/L"),
    "Iron":                         ("iron",        "mg/L"),
    "Ammonia":                      ("ammonia",     "mg/L"),
    "Total Coliform":               ("total_coliform", "MPN/100mL"),
    "Calcium":                      ("calcium",     "mg/L"),
    "Magnesium":                    ("magnesium",   "mg/L"),
    "Sodium":                       ("sodium",      "mg/L"),
    "Potassium":                    ("potassium",   "mg/L"),
}


class CPCBAdapter(BaseAdapter):
    """Adapter for CPCB ENVIS water quality data."""

    source_id = "cpcb"
    source_name = "Central Pollution Control Board (CPCB)"

    def __init__(
        self,
        csv_path: Optional[Path] = None,
        station_locations_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        super().__init__(output_dir)
        self.csv_path = csv_path
        self.station_locations_path = station_locations_path
        self._station_meta: Dict[str, Dict] = {}

    def _load_station_metadata(self):
        """Load station locations from a separate CSV if available."""
        if self.station_locations_path and self.station_locations_path.exists():
            with open(self.station_locations_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = self._normalize_station_id(
                        row.get("station_id") or row.get("stationId") or ""
                    )
                    if sid:
                        self._station_meta[sid] = {
                            "station_name": row.get("station_name", ""),
                            "latitude":     self.safe_float(row.get("latitude")),
                            "longitude":    self.safe_float(row.get("longitude")),
                            "state":        row.get("state", ""),
                            "river":        self._extract_river(row),
                        }
            logger.info(
                f"Loaded metadata for {len(self._station_meta)} CPCB stations"
            )

    @staticmethod
    def _normalize_station_id(value: str) -> str:
        text = str(value).strip()
        if not text:
            return ""
        try:
            return str(int(float(text)))
        except ValueError:
            return text

    @staticmethod
    def _extract_river(row: dict) -> str:
        """Try to extract river name from station metadata."""
        for field in ("river", "river_name", "site_name", "catchment"):
            val = row.get(field, "").strip().lower()
            if val:
                for river in _KNOWN_RIVERS:
                    if river in val:
                        return river.title()
        return ""

    def fetch(self, **kwargs) -> List[Dict]:
        """Read raw data from a local CPCB CSV file.

        The CSV should have columns: stationId, timestamp, and various
        water quality parameter columns.
        """
        self._load_station_metadata()

        if not self.csv_path or not self.csv_path.exists():
            raise FileNotFoundError(
                f"CPCB CSV not found: {self.csv_path}. "
                "Provide a valid path to the CPCB data file."
            )

        rows = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._raw_columns = list(reader.fieldnames or [])
            for row in reader:
                rows.append(dict(row))

        logger.info(f"Read {len(rows)} rows from {self.csv_path}")
        logger.info(f"Columns: {self._raw_columns}")
        return rows

    def parse(self, raw_data: List[Dict]) -> List[CanonicalRecord]:
        """Parse CPCB CSV rows into canonical records.

        Each row can contain multiple parameters (one per column),
        so we produce multiple CanonicalRecords per input row.
        """
        records = []

        # Detect which columns are WQ parameters
        param_columns = {}
        for col in self._raw_columns:
            col_clean = col.strip()
            if col_clean in _CPCB_PARAM_UNIT_MAP:
                param_columns[col] = _CPCB_PARAM_UNIT_MAP[col_clean]
            elif col_clean.startswith("wq_"):
                # Handle prefixed columns like wq_Biochemical Oxygen Demand
                unprefixed = col_clean[3:]
                if unprefixed in _CPCB_PARAM_UNIT_MAP:
                    param_columns[col] = _CPCB_PARAM_UNIT_MAP[unprefixed]

        if not param_columns:
            logger.warning("No recognized WQ parameter columns found in CSV!")
            return records

        logger.info(
            f"Detected {len(param_columns)} parameter columns: "
            f"{list(param_columns.keys())}"
        )

        for row in raw_data:
            # Extract station info
            station_id = self._normalize_station_id(
                row.get("stationId") or row.get("station_id") or ""
            )
            if not station_id:
                continue

            # Get coordinates and metadata from the row or station lookup
            meta = self._station_meta.get(station_id, {})
            lat = self.safe_float(row.get("latitude")) or meta.get("latitude")
            lon = self.safe_float(row.get("longitude")) or meta.get("longitude")
            station_name = (
                row.get("station_name", "").strip()
                or meta.get("station_name", "")
            )
            state = row.get("state", "").strip() or meta.get("state", "")
            river = meta.get("river", "")

            if lat is None or lon is None:
                continue

            # Parse date
            sample_date = self.safe_date(row.get("timestamp"))
            if sample_date is None:
                continue

            # Extract each parameter value
            for col, (canonical_param, canonical_unit) in param_columns.items():
                raw_val = row.get(col, "").strip()
                value = self.safe_float(raw_val)
                if value is None:
                    continue

                records.append(CanonicalRecord(
                    station_id=f"cpcb_{station_id}",
                    station_name=station_name,
                    river=river,
                    state=state,
                    latitude=lat,
                    longitude=lon,
                    sample_date=sample_date,
                    parameter=canonical_param,
                    value=value,
                    unit=canonical_unit,
                    source=self.source_id,
                    source_station_id=station_id,
                    source_parameter=col.strip(),
                    source_unit=canonical_unit,
                    source_value=value,
                ))

        return records
