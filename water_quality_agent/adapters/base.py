"""
Base adapter class for all water quality data sources.

Each source adapter inherits from BaseAdapter and implements:
  - fetch()    → download/read raw data from the source
  - parse()    → convert raw data into CanonicalRecord list
  - source_id  → unique identifier for this source
"""

import csv
import logging
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.schema import CanonicalRecord


logger = logging.getLogger(__name__)


class BaseAdapter(ABC):
    """Abstract base class for data source adapters."""

    source_id: str = "unknown"
    source_name: str = "Unknown Source"

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._raw_data: Any = None
        self._records: List[CanonicalRecord] = []

    @abstractmethod
    def fetch(self, **kwargs) -> Any:
        """Fetch raw data from the source.

        This can mean reading a local file, calling an API, or scraping
        a website. Returns the raw data in whatever form the source
        provides (DataFrame, list of dicts, etc.).
        """
        ...

    @abstractmethod
    def parse(self, raw_data: Any) -> List[CanonicalRecord]:
        """Parse raw data into canonical records.

        The records at this stage have source_parameter and source_unit
        filled in. Normalization to canonical names/units happens later
        in the pipeline.
        """
        ...

    def run(self, **kwargs) -> List[CanonicalRecord]:
        """Fetch + parse in one call."""
        logger.info(f"[{self.source_id}] Fetching data...")
        raw = self.fetch(**kwargs)
        logger.info(f"[{self.source_id}] Parsing {type(raw).__name__}...")
        records = self.parse(raw)
        logger.info(f"[{self.source_id}] Produced {len(records)} records")
        self._records = records
        return records

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def safe_float(value: Any) -> Optional[float]:
        """Convert a value to float, returning None on failure."""
        if value is None:
            return None
        s = str(value).strip()
        if not s or s.lower() in ("", "-", "na", "n/a", "nan", "null", "nil",
                                    "bdl", "below detection", "nd", "not detected"):
            return None
        # Handle ranges like "2.5-3.0" → take midpoint
        if "-" in s and not s.startswith("-"):
            parts = s.split("-")
            if len(parts) == 2:
                try:
                    a, b = float(parts[0].strip()), float(parts[1].strip())
                    return (a + b) / 2
                except ValueError:
                    pass
        # Handle "<0.5" or ">100" → use the number with flag
        s_clean = s.lstrip("<>≤≥")
        try:
            return float(s_clean.replace(",", ""))
        except ValueError:
            return None

    @staticmethod
    def safe_date(value: Any) -> Optional[date]:
        """Parse a date from various formats."""
        if value is None:
            return None
        from ..utils.date_parser import parse_date
        return parse_date(str(value))

    @staticmethod
    def clean_string(value: Any) -> str:
        """Clean and normalize a string value."""
        if value is None:
            return ""
        return " ".join(str(value).strip().split())

    def save_raw(self, filename: str, records: List[CanonicalRecord]):
        """Save parsed records to a CSV for inspection."""
        path = self.output_dir / filename
        if not records:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].to_dict().keys()))
            writer.writeheader()
            for r in records:
                writer.writerow(r.to_dict())
        logger.info(f"[{self.source_id}] Saved {len(records)} records → {path}")
