"""
data.gov.in Open Data adapter.

India's Open Government Data platform provides water quality datasets
via a REST API and downloadable CSV/JSON files.

API documentation: https://data.gov.in/ogpl_apis
Rate limit: Typically requires an API key with limited requests per day.

Common datasets:
  - Water Quality of Rivers (CPCB)
  - Ground Water Quality
  - State-wise River Water Quality
"""

import csv
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import urlopen, Request

from .base import BaseAdapter
from ..config.schema import CanonicalRecord
from ..config.parameters import normalize_parameter_name

logger = logging.getLogger(__name__)


# API settings
_API_BASE = "https://api.data.gov.in/resource"
_RATE_LIMIT_SECONDS = 1.5   # minimum gap between API calls

# Known dataset resource IDs on data.gov.in
KNOWN_DATASETS = {
    "river_wq_realtime": {
        "resource_id": "b7a1cda7-3c04-49e4-9a7a-ee2b24c4a534",
        "description": "Real-time River Water Quality",
    },
    "nwmp_monitoring": {
        "resource_id": "5a2100a0-a4e9-4757-ab8d-e90e0b9cae02",
        "description": "National Water Monitoring Programme data",
    },
}


class DataGovAdapter(BaseAdapter):
    """Adapter for data.gov.in water quality datasets."""

    source_id = "data_gov_in"
    source_name = "data.gov.in"

    def __init__(
        self,
        api_key: Optional[str] = None,
        csv_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        super().__init__(output_dir)
        self.api_key = api_key
        self.csv_path = csv_path

    def fetch(self, **kwargs) -> Any:
        """Fetch data from local CSV or data.gov.in API.

        Priority:
          1. Local CSV file (if csv_path provided)
          2. API fetch (if api_key provided)
        """
        # Local file first
        if self.csv_path and Path(self.csv_path).exists():
            return self._fetch_from_csv(Path(self.csv_path))

        # API fetch
        if self.api_key:
            resource_id = kwargs.get("resource_id", "")
            if not resource_id and kwargs.get("dataset") in KNOWN_DATASETS:
                resource_id = KNOWN_DATASETS[kwargs["dataset"]]["resource_id"]
            if resource_id:
                return self._fetch_from_api(resource_id, kwargs.get("limit", 1000))

        logger.warning("No CSV path or API key for data.gov.in, skipping.")
        return []

    def _fetch_from_csv(self, path: Path) -> List[Dict]:
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._raw_columns = list(reader.fieldnames or [])
            for row in reader:
                rows.append(dict(row))
        logger.info(f"Read {len(rows)} rows from {path}")
        return rows

    def _fetch_from_api(
        self, resource_id: str, limit: int = 1000
    ) -> List[Dict]:
        """Fetch records from data.gov.in REST API with pagination."""
        all_records = []
        offset = 0

        while True:
            params = {
                "api-key": self.api_key,
                "format": "json",
                "offset": offset,
                "limit": min(limit - len(all_records), 500),
            }
            url = f"{_API_BASE}/{resource_id}?{urlencode(params)}"

            try:
                req = Request(url, headers={"User-Agent": "WQAgent/1.0"})
                with urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                records = data.get("records", [])
                if not records:
                    break

                all_records.extend(records)
                logger.info(
                    f"data.gov.in: fetched {len(all_records)}/{data.get('total', '?')} records"
                )

                if len(all_records) >= limit:
                    break

                offset += len(records)
                time.sleep(_RATE_LIMIT_SECONDS)

            except Exception as e:
                logger.error(f"data.gov.in API error: {e}")
                break

        self._raw_columns = list(all_records[0].keys()) if all_records else []
        return all_records

    def parse(self, raw_data: List[Dict]) -> List[CanonicalRecord]:
        """Parse data.gov.in records.

        data.gov.in datasets vary widely in schema, so we use
        flexible column detection.
        """
        records = []
        if not raw_data:
            return records

        columns = list(raw_data[0].keys())

        # Auto-detect metadata columns
        col_map = self._auto_detect_columns(columns)
        logger.info(f"data.gov.in column mapping: {col_map}")

        # Detect parameter columns (anything not metadata)
        meta_cols = set(col_map.values())
        param_cols = [c for c in columns if c not in meta_cols]

        for row in raw_data:
            station_name = self.clean_string(
                row.get(col_map.get("station", ""), "")
            )
            sample_date = self.safe_date(
                row.get(col_map.get("date", ""), "")
            )
            if not station_name or sample_date is None:
                continue

            lat = self.safe_float(row.get(col_map.get("latitude", ""), ""))
            lon = self.safe_float(row.get(col_map.get("longitude", ""), ""))
            state = self.clean_string(row.get(col_map.get("state", ""), ""))
            river = self.clean_string(row.get(col_map.get("river", ""), ""))

            station_id = f"dgov_{station_name.lower().replace(' ', '_')[:30]}"

            for col in param_cols:
                value = self.safe_float(row.get(col, ""))
                if value is None:
                    continue

                canon_param = normalize_parameter_name(col)
                if canon_param is None:
                    continue  # Unknown parameter — LLM agent handles later

                records.append(CanonicalRecord(
                    station_id=station_id,
                    station_name=station_name,
                    river=river,
                    state=state,
                    latitude=lat or 0.0,
                    longitude=lon or 0.0,
                    sample_date=sample_date,
                    parameter=canon_param,
                    value=value,
                    unit="",  # unit determined by normalization pipeline
                    source=self.source_id,
                    source_station_id=station_name,
                    source_parameter=col,
                    source_unit="",
                    source_value=value,
                ))

        return records

    @staticmethod
    def _auto_detect_columns(columns: List[str]) -> Dict[str, str]:
        """Automatically detect metadata columns from header names."""
        mapping = {}
        patterns = {
            "station":   ["station", "site", "location_name", "monitoring_station"],
            "date":      ["date", "sampling_date", "sample_date", "observation_date"],
            "latitude":  ["lat", "latitude"],
            "longitude": ["lon", "long", "longitude"],
            "state":     ["state"],
            "river":     ["river", "water_body", "waterbody"],
        }
        for key, pats in patterns.items():
            for col in columns:
                col_lower = col.lower().strip()
                if any(p in col_lower for p in pats):
                    mapping[key] = col
                    break
        return mapping
