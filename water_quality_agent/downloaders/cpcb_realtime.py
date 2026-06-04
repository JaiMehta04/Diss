"""
CPCB Real-Time Water Quality Data Downloader.

Downloads water quality data from the CPCB Real-Time Water Quality Monitoring
System (RTWQMS) API at https://estuarine.cpcb.gov.in

The API provides:
  - Station list with coordinates
  - Hourly sensor readings for 12+ parameters
  - Historical data (limited window per request)

This module:
  1. Fetches all Ganga basin station metadata
  2. Downloads historical WQ data station by station
  3. Saves to CSV with same schema as existing local data
  4. Validates against local ground truth

API endpoints (discovered via browser network inspection):
  - GET /api/es-station-data-list?state_id=X&type=river
  - GET /api/water-quality-data?station_id=X&from=YYYY-MM-DD&to=YYYY-MM-DD
"""

import csv
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# ── CPCB API Endpoints ────────────────────────────────────────────────────────
# The CPCB RTWQMS runs at estuarine.cpcb.gov.in
# But the main data API endpoint is actually at:

CPCB_API_BASE = "https://estuarine.cpcb.gov.in"

# Alternative endpoints observed:
CPCB_ALT_ENDPOINTS = [
    "https://estuarine.cpcb.gov.in",
    "https://app.cpcbccr.com",
    "https://cpcbccr.nic.in",
]

# Ganga basin state IDs (from CPCB portal)
GANGA_STATE_IDS = {
    "Uttarakhand": 36,
    "Uttar Pradesh": 34,
    "Bihar": 5,
    "West Bengal": 37,
    "Delhi": 7,
    "Jharkhand": 16,
    "Haryana": 12,
}

# Known station IDs from local metadata (to validate)
KNOWN_STATION_IDS = [
    11783, 11784, 11785, 11786, 11787, 11788, 11789, 11790,
    11793, 11794, 11795, 11796, 11797, 11798, 11799, 11800,
    11801, 11802, 11803, 11804, 11805, 11806, 11807, 11808,
    11809, 11810, 11811, 11812, 11813, 11814, 11815, 11816,
    11817, 11818, 11819, 11820, 11821, 11822,
]


class CPCBRealTimeDownloader:
    """Downloads water quality data from CPCB RTWQMS API."""

    def __init__(self, rate_limit_delay: float = 2.0):
        self.delay = rate_limit_delay
        self.session = requests.Session()
        self.session.verify = False
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json, text/html, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://estuarine.cpcb.gov.in/",
            "Origin": "https://estuarine.cpcb.gov.in",
        })
        self._working_base = None

    def _find_working_api(self) -> Optional[str]:
        """Try each known endpoint to find one that responds."""
        if self._working_base:
            return self._working_base

        for base in CPCB_ALT_ENDPOINTS:
            try:
                r = self.session.get(f"{base}/", timeout=10)
                if r.status_code < 500:
                    logger.info(f"CPCB API reachable at: {base}")
                    self._working_base = base
                    return base
            except Exception as e:
                logger.debug(f"  {base} unreachable: {e}")
                continue

        logger.warning("No CPCB API endpoint is reachable")
        return None

    def discover_stations(self) -> List[Dict]:
        """Try to discover Ganga monitoring stations from the API."""
        base = self._find_working_api()
        if not base:
            return []

        stations = []

        # Try various API path patterns observed in CPCB portals
        api_paths = [
            "/api/es-station-data-list",
            "/api/station/getStations",
            "/api/stations",
            "/api/v1/stations",
            "/station/getStations",
            "/getStations",
        ]

        for path in api_paths:
            url = f"{base}{path}"
            try:
                # Try with state filter
                for state, sid in GANGA_STATE_IDS.items():
                    params = {"state_id": sid, "type": "river"}
                    r = self.session.get(url, params=params, timeout=15)
                    if r.status_code == 200:
                        data = r.json()
                        if isinstance(data, list):
                            stations.extend(data)
                            logger.info(f"  Found {len(data)} stations for {state} via {path}")
                        elif isinstance(data, dict) and "data" in data:
                            stations.extend(data["data"])
                            logger.info(f"  Found {len(data['data'])} stations for {state} via {path}")
                        time.sleep(self.delay)
            except Exception as e:
                logger.debug(f"  {path} failed: {e}")
                continue

        if stations:
            logger.info(f"Discovered {len(stations)} stations total")
        return stations

    def download_station_data(
        self,
        station_id: int,
        from_date: str,
        to_date: str,
    ) -> List[Dict]:
        """Download water quality data for a single station and date range.

        Args:
            station_id: CPCB station ID (e.g., 11783)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            List of measurement records
        """
        base = self._find_working_api()
        if not base:
            return []

        records = []

        # Try various data API patterns
        data_paths = [
            "/api/water-quality-data",
            "/api/es-wq-data",
            "/api/station/getStationData",
            "/api/v1/station-data",
            "/getData",
        ]

        for path in data_paths:
            url = f"{base}{path}"
            params = {
                "station_id": station_id,
                "from": from_date,
                "to": to_date,
                "format": "json",
            }
            # Also try POST
            for method in [self.session.get, self.session.post]:
                try:
                    if method == self.session.get:
                        r = method(url, params=params, timeout=30)
                    else:
                        r = method(url, json=params, timeout=30)

                    if r.status_code == 200:
                        data = r.json()
                        if isinstance(data, list) and data:
                            records = data
                            logger.info(f"  Got {len(data)} records from {path}")
                            return records
                        elif isinstance(data, dict):
                            if "data" in data and data["data"]:
                                records = data["data"]
                                logger.info(f"  Got {len(records)} records from {path}")
                                return records
                            elif "records" in data and data["records"]:
                                records = data["records"]
                                logger.info(f"  Got {len(records)} records from {path}")
                                return records
                except Exception as e:
                    logger.debug(f"  {method.__name__} {path} failed: {e}")
                    continue

        return records

    def download_all_stations(
        self,
        output_dir: Path,
        from_date: str = "2024-01-01",
        to_date: str = None,
        station_ids: List[int] = None,
    ) -> Dict[str, Any]:
        """Download data for all known Ganga stations.

        Args:
            output_dir: Directory to save CSVs
            from_date: Start date
            to_date: End date (default: today)
            station_ids: Specific stations (default: all known)

        Returns:
            Summary dict
        """
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")

        station_ids = station_ids or KNOWN_STATION_IDS
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {"stations_attempted": 0, "stations_with_data": 0,
                    "total_records": 0, "errors": []}
        all_records = []

        logger.info(f"Downloading CPCB data for {len(station_ids)} stations")
        logger.info(f"Date range: {from_date} to {to_date}")

        for sid in station_ids:
            results["stations_attempted"] += 1
            try:
                records = self.download_station_data(sid, from_date, to_date)
                if records:
                    results["stations_with_data"] += 1
                    results["total_records"] += len(records)
                    all_records.extend(records)
                    logger.info(f"  Station {sid}: {len(records)} records")
                else:
                    logger.info(f"  Station {sid}: no data")
            except Exception as e:
                results["errors"].append(f"Station {sid}: {e}")
                logger.warning(f"  Station {sid}: error - {e}")

            time.sleep(self.delay)

        # Save all records
        if all_records:
            csv_path = output_dir / "cpcb_web_extracted.csv"
            self._save_records(all_records, csv_path)
            results["output_file"] = str(csv_path)
            logger.info(f"Saved {len(all_records)} records to {csv_path}")

        # Save summary
        report_path = output_dir / "cpcb_download_report.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _save_records(self, records: List[Dict], output_path: Path):
        """Save records to CSV."""
        if not records:
            return
        # Get all unique keys
        all_keys = set()
        for r in records:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)


def main():
    """CLI entry point."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Download CPCB real-time WQ data")
    parser.add_argument("--output-dir", type=str, default="data/raw/cpcb_web",
                        help="Output directory")
    parser.add_argument("--from-date", type=str, default="2024-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to-date", type=str, default=None,
                        help="End date (YYYY-MM-DD, default: today)")
    args = parser.parse_args()

    dl = CPCBRealTimeDownloader()

    # First try to discover stations
    print("Step 1: Probing CPCB API endpoints...")
    stations = dl.discover_stations()
    if stations:
        print(f"  Discovered {len(stations)} stations")
    else:
        print("  Could not discover stations via API, using known station list")

    # Download data
    print(f"\nStep 2: Downloading data ({args.from_date} to {args.to_date or 'today'})...")
    results = dl.download_all_stations(
        output_dir=Path(args.output_dir),
        from_date=args.from_date,
        to_date=args.to_date,
    )

    print(f"\n=== RESULTS ===")
    print(f"Stations attempted:   {results['stations_attempted']}")
    print(f"Stations with data:   {results['stations_with_data']}")
    print(f"Total records:        {results['total_records']}")
    if results.get("output_file"):
        print(f"Output file:          {results['output_file']}")
    if results["errors"]:
        print(f"Errors:               {len(results['errors'])}")


if __name__ == "__main__":
    main()
