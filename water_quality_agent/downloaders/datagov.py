"""
data.gov.in API downloader.

Uses the Open Government Data (OGD) Platform REST API to search and
download water quality datasets published by CPCB, MoEFCC, and state
pollution control boards.

Uses the OGD platform's default public API key — NO per-dataset key needed.
The catalog search at /lists also needs no key at all.

Usage:
    downloader = DataGovDownloader()  # No API key needed!
    downloader.download_all(output_dir="data/raw/datagov")
"""

import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ── OGD Platform default API key (public, works for all datasets) ─────────
# This is the default key published in data.gov.in API documentation/examples.
OGD_DEFAULT_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"

# ── Verified Ganga water quality dataset resource IDs ─────────────────────
# These are verified working as of April 2026.

GANGA_RESOURCES = [
    {
        "id": "c321b776-06c6-464a-a0a5-7fb7525431e8",
        "name": "Water quality of River Ganga - 2012",
        "total": 66,
    },
    {
        "id": "3aa4a3c3-6e22-4ced-b76b-033e9c9e542d",
        "name": "River Ganga Water Quality 2011-2015",
        "total": None,
    },
    {
        "id": "95273406-7394-489b-ba92-475595008d10",
        "name": "Station-wise River Ganga WQ 2018-2020",
        "total": 93,
    },
    {
        "id": "c1c01473-f81d-4919-85d0-6d9f9a3d804d",
        "name": "State-wise River Ganga WQ Median 2017-2022",
        "total": 554,
    },
    {
        "id": "ed5d2b18-acda-4d21-99e9-552792f32e67",
        "name": "Station-wise River Ganga WQ 2021",
        "total": None,
    },
]

# Surface water quality (covers Ganga basin + all India)
SURFACE_WQ_RESOURCES = [
    {
        "id": "19697d76-442e-4d76-aeae-13f8a17c91e1",
        "name": "Surface Water Quality March 2018 CPCB",
        "total": 378926,
    },
    {
        "id": "81cbae68-7675-4b64-8c40-5980515b709a",
        "name": "Surface Water Quality July 2020 CPCB",
        "total": None,
    },
    {
        "id": "9314665e-c201-4839-85d7-e108e647ae35",
        "name": "Surface Water Quality UP 1996-2006",
        "total": None,
    },
    {
        "id": "3c2187bf-3f96-4f0c-9639-0aae9f5ac238",
        "name": "Surface Water Quality UP 1985-1995",
        "total": None,
    },
]

# Other river datasets (for multi-river support)
OTHER_RIVER_RESOURCES = [
    {
        "id": "64c533a8-3f95-49e0-835b-b81f6d1f94b7",
        "name": "Water quality of River Yamuna 2014",
        "total": None,
    },
    {
        "id": "3ae8274c-f1a1-45b6-9798-d06ec4341a1f",
        "name": "Water quality of River Cauvery 2012",
        "total": None,
    },
    {
        "id": "f9ebfddf-936d-4d04-9ff6-8a41b2128ff2",
        "name": "Water quality of River Krishna 2012",
        "total": None,
    },
]

# data.gov.in API base URL
API_BASE = "https://api.data.gov.in/resource"
CATALOG_API = "https://api.data.gov.in/catalog"


class DataGovDownloader:
    """Downloads water quality data from data.gov.in REST API.
    
    No API key needed — uses the OGD platform default key automatically.
    """

    def __init__(self, api_key: str = None, rate_limit_delay: float = 2.5):
        """
        Args:
            api_key: Optional API key (uses OGD default if not provided)
            rate_limit_delay: Seconds between API calls to avoid throttling
        """
        self.api_key = api_key or OGD_DEFAULT_KEY
        self.delay = rate_limit_delay
        self.session = requests.Session()
        self.session.verify = False  # Handle corporate proxy / SSL issues
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.session.headers.update({
            "User-Agent": "WaterQualityAgent/1.0 (Research)",
            "Accept": "application/json",
        })

    def search_datasets(self, query: str = "water quality", limit: int = 20) -> List[Dict]:
        """Search data.gov.in catalog for water quality datasets.

        Uses the /lists endpoint which works without API key for discovery,
        then uses resource API with key for actual data download.

        Returns list of {title, index_name, ...}
        """
        # The catalog search uses /lists endpoint
        url = "https://api.data.gov.in/lists"
        params = {
            "format": "json",
            "filters[title]": query,
            "limit": limit,
            "offset": 0,
        }

        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("records", []):
                results.append({
                    "title": item.get("title", ""),
                    "resource_id": item.get("resource_id", ""),
                    "description": item.get("description", ""),
                    "org": item.get("org", ""),
                    "sector": item.get("sector", ""),
                })
            logger.info(f"Found {len(results)} datasets matching '{query}'")
            return results

        except Exception as e:
            logger.error(f"Catalog search failed: {e}")
            return []

    def download_resource(
        self,
        resource_id: str,
        output_path: Path,
        limit_per_page: int = 10,
        max_records: int = 50000,
        filters: Dict[str, str] = None,
    ) -> int:
        """Download a single resource (paginated) and save as CSV.

        Note: The OGD default API key caps at 10 records/request.
        Pagination handles this automatically.

        Args:
            resource_id: data.gov.in resource UUID
            output_path: Path to save CSV
            limit_per_page: Records per API page (default: 10, OGD limit)
            max_records: Max total records to download
            filters: Optional dict of {field_name: value} for server-side filtering

        Returns:
            Number of records downloaded
        """
        all_records = []
        offset = 0
        fields = None
        rate_limit_retries = 0
        max_rate_retries = 5

        while offset < max_records:
            url = f"{API_BASE}/{resource_id}"
            params = {
                "api-key": self.api_key,
                "format": "json",
                "offset": offset,
                "limit": limit_per_page,
            }
            if filters:
                for k, v in filters.items():
                    params[f"filters[{k}]"] = v

            try:
                resp = self.session.get(url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                rate_limit_retries = 0  # Reset on success
            except requests.exceptions.ReadTimeout:
                rate_limit_retries += 1
                if rate_limit_retries > max_rate_retries:
                    logger.warning(
                        f"Timed out {max_rate_retries} times. "
                        f"Saving {len(all_records)} records so far."
                    )
                    break
                wait = 15 * rate_limit_retries
                logger.warning(f"Timeout ({rate_limit_retries}/{max_rate_retries}). Retrying in {wait}s...")
                time.sleep(wait)
                continue
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 404:
                    logger.warning(f"Resource {resource_id} not found (404). Skipping.")
                    return 0
                elif resp.status_code == 429:
                    rate_limit_retries += 1
                    if rate_limit_retries > max_rate_retries:
                        logger.warning(f"Rate limited {max_rate_retries} times. Moving on with {len(all_records)} records.")
                        break
                    wait = 10 * rate_limit_retries
                    logger.warning(f"Rate limited ({rate_limit_retries}/{max_rate_retries}). Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                elif resp.status_code == 400:
                    logger.debug(f"HTTP 400 for {resource_id}: {resp.text[:100]}")
                    break
                else:
                    logger.error(f"HTTP error for {resource_id}: {e}")
                    break
            except requests.exceptions.ConnectionError:
                rate_limit_retries += 1
                if rate_limit_retries > max_rate_retries:
                    logger.warning(f"Connection failed {max_rate_retries} times. Saving {len(all_records)} records.")
                    break
                wait = 20 * rate_limit_retries
                logger.warning(f"Connection error ({rate_limit_retries}/{max_rate_retries}). Retrying in {wait}s...")
                time.sleep(wait)
                continue
            except Exception as e:
                logger.error(f"Request failed for {resource_id}: {e}")
                break

            # Extract field names from first response
            if fields is None:
                fields = [f.get("id", f.get("name", ""))
                          for f in data.get("fields", data.get("field", []))]
                if not fields and data.get("records"):
                    fields = list(data["records"][0].keys())

            records = data.get("records", [])
            if not records:
                break

            all_records.extend(records)
            offset += limit_per_page

            total = data.get("total", data.get("count", 0))
            if len(all_records) % 100 < limit_per_page:  # Log every ~100 records
                logger.info(
                    f"  Downloaded {len(all_records)}/{total} records "
                    f"from resource {resource_id[:8]}..."
                )

            # Incremental save every 500 records
            if len(all_records) % 500 < limit_per_page and fields:
                self._save_csv(all_records, fields, output_path)

            # Stop if we got everything
            if len(records) < limit_per_page:
                break
            if total and len(all_records) >= total:
                break

            time.sleep(self.delay)

        if not all_records:
            logger.warning(f"No records from resource {resource_id}")
            return 0

        # Final save
        if not fields:
            fields = list(all_records[0].keys())
        self._save_csv(all_records, fields, output_path)
        logger.info(f"Saved {len(all_records)} records to {output_path}")
        return len(all_records)

    @staticmethod
    def _save_csv(records: list, fields: list, output_path: Path):
        """Save records to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

    def download_all(self, output_dir: str = "data/raw/datagov", river: str = "ganga") -> Dict[str, int]:
        """Download all water quality resources — no API key needed.

        Downloads:
          1. Verified Ganga-specific datasets (most relevant)
          2. Surface water quality datasets (covers Ganga basin)
          3. Catalog search for additional datasets

        Args:
            output_dir: Directory to save CSV files
            river: River name to focus on (default: ganga)

        Returns:
            dict of {dataset_name: record_count}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        seen_ids = set()

        # 1. Download verified Ganga-specific datasets
        logger.info("Step 1: Downloading verified Ganga water quality datasets...")
        for resource in GANGA_RESOURCES:
            name_slug = re.sub(r"[^a-z0-9_]", "", resource["name"].lower().replace(" ", "_"))[:60]
            output_path = output_dir / f"ganga_{name_slug}.csv"

            if output_path.exists() and output_path.stat().st_size > 100:
                logger.info(f"  Already downloaded: {output_path.name}")
                results[resource["name"]] = -1  # Already exists
                seen_ids.add(resource["id"])
                continue

            logger.info(f"  Downloading: {resource['name']}")
            count = self.download_resource(resource["id"], output_path)
            results[resource["name"]] = count
            seen_ids.add(resource["id"])
            time.sleep(self.delay)

        # 2. Download surface water quality datasets
        # Use Basin filter for Ganga to avoid downloading 378K all-India records
        logger.info("\nStep 2: Downloading surface water quality datasets (filtered for Ganga basin)...")
        basin_filter = {"Basin": river.capitalize()} if river else None
        for resource in SURFACE_WQ_RESOURCES:
            if resource["id"] in seen_ids:
                continue

            name_slug = re.sub(r"[^a-z0-9_]", "", resource["name"].lower().replace(" ", "_"))[:60]
            output_path = output_dir / f"surface_{name_slug}.csv"

            if output_path.exists() and output_path.stat().st_size > 100:
                logger.info(f"  Already downloaded: {output_path.name}")
                results[resource["name"]] = -1
                seen_ids.add(resource["id"])
                continue

            logger.info(f"  Downloading: {resource['name']} (Basin={river.capitalize()})")
            count = self.download_resource(
                resource["id"], output_path,
                max_records=10000, filters=basin_filter,
            )
            results[resource["name"]] = count
            seen_ids.add(resource["id"])
            time.sleep(self.delay)

        # 3. Search catalog for additional datasets
        logger.info("\nStep 3: Searching catalog for more datasets...")
        search_queries = [
            f"water quality river {river}",
            "water quality CPCB river",
            "surface water quality monitoring",
        ]

        for query in search_queries:
            logger.info(f"  Searching: '{query}'...")
            extra_datasets = self.search_datasets(query, limit=5)

            for ds in extra_datasets:
                rid = ds.get("index_name", ds.get("resource_id", ""))
                if not rid or rid in seen_ids:
                    continue
                seen_ids.add(rid)

                title = ds.get("title", "unknown")[:50]
                name_slug = re.sub(r"[^a-z0-9_]", "", title.lower().replace(" ", "_"))[:50]
                output_path = output_dir / f"search_{name_slug}.csv"

                if output_path.exists():
                    continue

                logger.info(f"    Trying: {title}")
                count = self.download_resource(rid, output_path)
                if count > 0:
                    results[title] = count
                time.sleep(self.delay)

        return results


def download_datagov(output_dir: str = "data/raw/datagov", api_key: str = None) -> Dict[str, int]:
    """Convenience function — no API key needed."""
    dl = DataGovDownloader(api_key=api_key)
    return dl.download_all(output_dir=output_dir)
