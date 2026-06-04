"""
Geospatial utilities for station matching and deduplication.
"""

import math
from typing import Tuple


def haversine_km(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """Haversine distance between two WGS84 points in kilometres."""
    R = 6371.0  # Earth radius in km
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def dms_to_decimal(degrees: float, minutes: float, seconds: float, direction: str) -> float:
    """Convert DMS (degrees, minutes, seconds) to decimal degrees."""
    dd = abs(degrees) + minutes / 60 + seconds / 3600
    if direction.upper() in ("S", "W"):
        dd = -dd
    return dd


def parse_coordinate(value: str) -> float:
    """Parse a coordinate string in various formats.

    Handles:
      - "26.85" (decimal)
      - "26°51'N" (DMS)
      - "26°51'00\"N"
      - "26 51 00 N"
    """
    import re

    value = value.strip()

    # Already decimal
    try:
        return float(value)
    except ValueError:
        pass

    # DMS pattern: 26°51'00"N or 26 51 00 N
    m = re.match(
        r"(\d+)[°\s]+(\d+)['\s]+(\d+(?:\.\d+)?)[\"″\s]*([NSEW])?",
        value, re.IGNORECASE,
    )
    if m:
        d, mi, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
        direction = (m.group(4) or "N").upper()
        return dms_to_decimal(d, mi, s, direction)

    # DM pattern: 26°51.5'N
    m = re.match(
        r"(\d+)[°\s]+(\d+(?:\.\d+)?)['\s]*([NSEW])?",
        value, re.IGNORECASE,
    )
    if m:
        d, mi = float(m.group(1)), float(m.group(2))
        direction = (m.group(3) or "N").upper()
        return dms_to_decimal(d, mi, 0, direction)

    raise ValueError(f"Cannot parse coordinate: {value!r}")


def is_in_india(lat: float, lon: float) -> bool:
    """Quick bounding-box check for Indian coordinates."""
    return 6.0 <= lat <= 37.0 and 68.0 <= lon <= 98.0
