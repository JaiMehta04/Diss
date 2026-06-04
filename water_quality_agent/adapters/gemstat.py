"""
GEMStat (Global Environment Monitoring System for Water) adapter.

GEMStat is a UN database with water quality data from monitoring stations
worldwide, including many Indian stations.

Data download: https://gemstat.org/data/data-portal/
Format: CSV with standardized columns.
"""

import csv
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAdapter
from ..config.schema import CanonicalRecord
from ..config.parameters import normalize_parameter_name

logger = logging.getLogger(__name__)


class GEMStatAdapter(BaseAdapter):
    """Adapter for GEMStat CSV data."""

    source_id = "gemstat"
    source_name = "GEMStat (UNEP)"

    def __init__(
        self,
        csv_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        super().__init__(output_dir)
        self.csv_path = csv_path

    def fetch(self, **kwargs) -> Any:
        """Read GEMStat CSV file."""
        path = kwargs.get("path") or self.csv_path
        if not path:
            logger.warning("No GEMStat CSV path provided, skipping.")
            return []
        path = Path(path)
        if not path.exists():
            logger.warning(f"GEMStat file not found: {path}")
            return []

        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._raw_columns = list(reader.fieldnames or [])
            for row in reader:
                rows.append(dict(row))

        logger.info(f"Read {len(rows)} GEMStat rows from {path}")
        return rows

    def parse(self, raw_data: List[Dict]) -> List[CanonicalRecord]:
        """Parse GEMStat CSV rows.

        GEMStat typically uses a long format:
          GEMS Station Number, Station Name, Country, Water Type,
          Sample Date, Parameter, Analysis Method, Value, Unit,
          Latitude, Longitude, ...
        """
        records = []
        if not raw_data:
            return records

        for row in raw_data:
            # GEMStat column names (may vary slightly)
            station_id_raw = self.clean_string(
                row.get("GEMS Station Number", "")
                or row.get("Station_ID", "")
            )
            station_name = self.clean_string(
                row.get("Station Name", "")
                or row.get("Station_Name", "")
            )
            country = self.clean_string(
                row.get("Country", "") or row.get("Country_Name", "")
            )

            # Filter to India only
            if country and "india" not in country.lower():
                continue

            sample_date = self.safe_date(
                row.get("Sample Date", "")
                or row.get("Sampling_Date", "")
                or row.get("Date", "")
            )
            if sample_date is None:
                continue

            lat = self.safe_float(
                row.get("Latitude", "") or row.get("Lat", "")
            )
            lon = self.safe_float(
                row.get("Longitude", "") or row.get("Lon", "")
            )

            # Parameter (long format: one param per row)
            raw_param = self.clean_string(
                row.get("Parameter", "")
                or row.get("Parameter_Name", "")
            )
            raw_value = self.safe_float(
                row.get("Value", "")
                or row.get("Result_Value", "")
            )
            raw_unit = self.clean_string(
                row.get("Unit", "")
                or row.get("Unit_Name", "")
            )

            if not raw_param or raw_value is None:
                continue

            canon_param = normalize_parameter_name(raw_param)
            if canon_param is None:
                continue

            # Try to extract river and state from metadata
            water_body = self.clean_string(
                row.get("Water Body", "")
                or row.get("Water_Body", "")
            )
            state = self.clean_string(row.get("State", ""))

            records.append(CanonicalRecord(
                station_id=f"gems_{station_id_raw}" if station_id_raw else f"gems_{station_name[:20]}",
                station_name=station_name,
                river=water_body,
                state=state,
                latitude=lat or 0.0,
                longitude=lon or 0.0,
                sample_date=sample_date,
                parameter=canon_param,
                value=raw_value,
                unit=raw_unit,
                source=self.source_id,
                source_station_id=station_id_raw,
                source_parameter=raw_param,
                source_unit=raw_unit,
                source_value=raw_value,
            ))

        return records
