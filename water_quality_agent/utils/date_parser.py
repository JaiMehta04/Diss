"""
Robust date parser for heterogeneous Indian water quality data.

Handles: ISO-8601, DD/MM/YYYY, MM/DD/YYYY, DD-Mon-YYYY, "Jan 2024",
"2024-Q1", Excel serial dates, and many more.
"""

import re
from datetime import date, datetime
from typing import Optional


# Pre-compiled patterns ordered by specificity
_DATE_PATTERNS = [
    # ISO 8601 with time
    (re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})[T ]"), "ymd"),
    # ISO 8601 date only
    (re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$"), "ymd"),
    # DD/MM/YYYY or DD-MM-YYYY
    (re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})$"), "dmy"),
    # YYYY/MM/DD
    (re.compile(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$"), "ymd"),
    # DD/MM/YY
    (re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})$"), "dmy_short"),
]

_MONTH_NAMES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def parse_date(value: str) -> Optional[date]:
    """Parse a date string into a Python date object.

    Tries multiple formats commonly found in Indian data sources.
    Returns None if parsing fails.
    """
    if not value:
        return None

    text = value.strip()

    # Remove trailing timezone markers
    text = re.sub(r"\s*[+-]\d{2}:\d{2}$", "", text)
    text = text.replace("Z", "").strip()

    # Try compiled patterns
    for pattern, fmt in _DATE_PATTERNS:
        m = pattern.match(text)
        if m:
            groups = m.groups()
            try:
                if fmt == "ymd":
                    return date(int(groups[0]), int(groups[1]), int(groups[2]))
                elif fmt == "dmy":
                    d, mo, y = int(groups[0]), int(groups[1]), int(groups[2])
                    # Heuristic: if d > 12, it's definitely DD/MM/YYYY
                    # If both <=12, assume DD/MM/YYYY (Indian convention)
                    return date(y, mo, d)
                elif fmt == "dmy_short":
                    d, mo, y = int(groups[0]), int(groups[1]), int(groups[2])
                    y = y + 2000 if y < 50 else y + 1900
                    return date(y, mo, d)
            except ValueError:
                continue

    # Try "DD Mon YYYY" or "Mon DD, YYYY"
    m = re.match(
        r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", text
    )
    if m:
        d, mon_str, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        mo = _MONTH_NAMES.get(mon_str[:3])
        if mo:
            try:
                return date(y, mo, d)
            except ValueError:
                pass

    m = re.match(
        r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", text
    )
    if m:
        mon_str, d, y = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        mo = _MONTH_NAMES.get(mon_str[:3])
        if mo:
            try:
                return date(y, mo, d)
            except ValueError:
                pass

    # Try "Mon YYYY" or "YYYY Mon" → use 1st of month
    m = re.match(r"([A-Za-z]+)\s+(\d{4})", text)
    if m:
        mon_str, y = m.group(1).lower(), int(m.group(2))
        mo = _MONTH_NAMES.get(mon_str[:3])
        if mo:
            return date(y, mo, 1)

    m = re.match(r"(\d{4})\s+([A-Za-z]+)", text)
    if m:
        y, mon_str = int(m.group(1)), m.group(2).lower()
        mo = _MONTH_NAMES.get(mon_str[:3])
        if mo:
            return date(y, mo, 1)

    # Try "YYYY-Q1" format → map to quarter start
    m = re.match(r"(\d{4})-?[Qq](\d)", text)
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        quarter_starts = {1: 1, 2: 4, 3: 7, 4: 10}
        if q in quarter_starts:
            return date(y, quarter_starts[q], 1)

    # Try Excel serial date (numeric)
    try:
        serial = float(text)
        if 30000 < serial < 60000:  # Plausible range: ~1982–2063
            # Excel epoch: 1899-12-30
            from datetime import timedelta
            return (datetime(1899, 12, 30) + timedelta(days=int(serial))).date()
    except (ValueError, OverflowError):
        pass

    # Last resort: try Python dateutil
    try:
        from dateutil import parser as dateutil_parser
        dt = dateutil_parser.parse(text, dayfirst=True)
        return dt.date()
    except Exception:
        pass

    return None
