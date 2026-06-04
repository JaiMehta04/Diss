"""
Canonical schema for the unified water quality data lake.

Every record from every source is transformed into this schema before
being written to the consolidated output.
"""

from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from typing import Optional


class QualityFlag(str, Enum):
    """Data quality flags assigned during validation."""
    VALID = "valid"
    SUSPECT_RANGE = "suspect_range"           # outside expected physical range
    SUSPECT_OUTLIER = "suspect_outlier"        # statistical outlier
    SUSPECT_CONSISTENCY = "suspect_consistency"  # cross-parameter inconsistency
    SUSPECT_DUPLICATE = "suspect_duplicate"    # possible duplicate record
    MISSING = "missing"                        # imputed or placeholder
    REJECTED = "rejected"                      # physically impossible


@dataclass
class CanonicalRecord:
    """A single water quality observation in canonical form."""

    # ── Identity ──────────────────────────────────────────────────────────────
    station_id: str               # Unique canonical station ID
    station_name: str             # Human-readable station name
    river: str                    # River name (canonical, e.g. "Ganga")
    state: str                    # Indian state
    latitude: float               # WGS84 decimal degrees
    longitude: float              # WGS84 decimal degrees

    # ── Temporal ──────────────────────────────────────────────────────────────
    sample_date: date             # Date of sample collection

    # ── Measurement ───────────────────────────────────────────────────────────
    parameter: str                # Canonical parameter name (e.g. "bod")
    value: float                  # Numeric value in canonical unit
    unit: str                     # Canonical unit (e.g. "mg/L")

    # ── Optional temporal ─────────────────────────────────────────────────────
    sample_time: Optional[str] = None  # HH:MM if available

    # ── Provenance ────────────────────────────────────────────────────────────
    source: str = ""              # Data source (e.g. "cpcb", "cwc", "gemstat")
    source_station_id: str = ""   # Original station ID from source
    source_parameter: str = ""    # Original parameter name from source
    source_unit: str = ""         # Original unit from source
    source_value: float = 0.0     # Original value before conversion

    # ── Quality ───────────────────────────────────────────────────────────────
    quality_flag: str = QualityFlag.VALID
    confidence: float = 1.0       # 0.0–1.0 confidence score
    notes: str = ""               # Validation/agent reasoning

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sample_date"] = self.sample_date.isoformat()
        return d


# Column order for the output CSV
CANONICAL_SCHEMA = [
    "station_id",
    "station_name",
    "river",
    "state",
    "latitude",
    "longitude",
    "sample_date",
    "sample_time",
    "parameter",
    "value",
    "unit",
    "source",
    "source_station_id",
    "source_parameter",
    "source_unit",
    "source_value",
    "quality_flag",
    "confidence",
    "notes",
]
