"""
Kaggle dataset downloader for Indian water quality data.

Downloads curated water quality datasets from Kaggle. These are often
cleaned versions of CPCB/CWC data with good coverage.

Requirements:
    pip install kaggle
    Set KAGGLE_USERNAME and KAGGLE_KEY env vars
    (or place kaggle.json in ~/.kaggle/)
"""

import csv
import logging
import os
import zipfile
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# Known Kaggle datasets with Indian water quality data
KAGGLE_DATASETS = [
    {
        "slug": "anbarivan/indian-water-quality-data",
        "name": "Indian Water Quality Data (CPCB)",
        "description": "Historical CPCB water quality data for Indian rivers/stations",
    },
    {
        "slug": "venkatramakrishnan/india-water-quality-data",
        "name": "India Water Quality Data",
        "description": "Water quality data with pH, DO, BOD, COD, etc.",
    },
    {
        "slug": "adityakadiwal/water-potability",
        "name": "Water Potability",
        "description": "Water quality metrics for potability classification",
    },
]


class KaggleDownloader:
    """Downloads water quality datasets from Kaggle."""

    def __init__(self, username: str = None, key: str = None):
        """
        Args:
            username: Kaggle username (or set KAGGLE_USERNAME env var)
            key: Kaggle API key (or set KAGGLE_KEY env var)
        """
        if username:
            os.environ["KAGGLE_USERNAME"] = username
        if key:
            os.environ["KAGGLE_KEY"] = key

        self._api = None

    def _get_api(self):
        """Lazy-load Kaggle API client."""
        if self._api is None:
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                self._api = KaggleApi()
                self._api.authenticate()
                logger.info("Kaggle API authenticated")
            except ImportError:
                raise ImportError(
                    "kaggle package not installed. Install with: pip install kaggle"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Kaggle authentication failed: {e}. "
                    "Set KAGGLE_USERNAME and KAGGLE_KEY env vars, or "
                    "place kaggle.json in ~/.kaggle/"
                )
        return self._api

    def download_dataset(self, slug: str, output_dir: Path) -> List[Path]:
        """Download and extract a Kaggle dataset.

        Args:
            slug: Kaggle dataset slug (e.g., "user/dataset-name")
            output_dir: Directory to extract files to

        Returns:
            List of extracted file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        api = self._get_api()
        logger.info(f"Downloading Kaggle dataset: {slug}")

        try:
            api.dataset_download_files(slug, path=str(output_dir), unzip=True)
        except Exception as e:
            logger.error(f"Failed to download {slug}: {e}")
            return []

        # Find all CSV/Excel files extracted
        files = (
            list(output_dir.rglob("*.csv")) +
            list(output_dir.rglob("*.xlsx")) +
            list(output_dir.rglob("*.xls"))
        )
        logger.info(f"Extracted {len(files)} data files from {slug}")
        return files

    def search_datasets(self, query: str = "india water quality") -> List[Dict]:
        """Search Kaggle for water quality datasets."""
        api = self._get_api()
        try:
            results = api.dataset_list(search=query, sort_by="relevance")
            datasets = []
            for ds in results[:20]:
                datasets.append({
                    "slug": str(ds),
                    "title": ds.title if hasattr(ds, "title") else str(ds),
                    "size": ds.size if hasattr(ds, "size") else "unknown",
                })
            return datasets
        except Exception as e:
            logger.error(f"Kaggle search failed: {e}")
            return []

    def download_all(self, output_dir: str = "data/raw/kaggle") -> Dict[str, int]:
        """Download all known water quality datasets from Kaggle.

        Returns dict of {dataset_name: file_count}
        """
        output_dir = Path(output_dir)
        results = {}

        for ds in KAGGLE_DATASETS:
            slug = ds["slug"]
            ds_dir = output_dir / slug.replace("/", "_")

            try:
                files = self.download_dataset(slug, ds_dir)
                results[ds["name"]] = len(files)
            except Exception as e:
                logger.error(f"Failed: {ds['name']}: {e}")
                results[ds["name"]] = 0

        return results


def download_kaggle(output_dir: str = "data/raw/kaggle") -> Dict[str, int]:
    """Convenience function to download all Kaggle water quality datasets."""
    dl = KaggleDownloader()
    return dl.download_all(output_dir=output_dir)
