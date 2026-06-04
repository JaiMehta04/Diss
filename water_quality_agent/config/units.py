"""
Unit conversion engine.

Converts any recognised source unit to the canonical unit defined in
config.parameters.CANONICAL_PARAMETERS.
"""

import re
from typing import Optional, Tuple

from .parameters import CANONICAL_PARAMETERS


# ──────────────────────────────────────────────────────────────────────────────
# Unit aliases → canonical unit string
# ──────────────────────────────────────────────────────────────────────────────
_UNIT_ALIASES: dict[str, str] = {
    # Concentration
    "mg/l":     "mg/L",
    "mgl":      "mg/L",
    "mg/lit":   "mg/L",
    "mg/litre": "mg/L",
    "mg/liter": "mg/L",
    "ppm":      "mg/L",   # 1 ppm ≈ 1 mg/L for dilute aqueous
    "µg/l":     "µg/L",
    "ug/l":     "µg/L",
    "ug/L":     "µg/L",
    "ppb":      "µg/L",
    "g/l":      "g/L",

    # Conductivity
    "µs/cm":    "µS/cm",
    "us/cm":    "µS/cm",
    "umho/cm":  "µS/cm",
    "µmho/cm":  "µS/cm",
    "ms/cm":    "mS/cm",
    "mmho/cm":  "mS/cm",

    # Turbidity
    "ntu":      "NTU",
    "jtu":      "JTU",
    "ftu":      "FTU",
    "fnu":      "FNU",

    # Temperature
    "°c":       "°C",
    "deg c":    "°C",
    "degc":     "°C",
    "celsius":  "°C",
    "c":        "°C",
    "°f":       "°F",
    "fahrenheit": "°F",

    # Colour
    "hazen":    "Hazen",
    "pt-co":    "Hazen",
    "pcu":      "Hazen",
    "tcu":      "Hazen",

    # Biological
    "mpn/100ml":  "MPN/100mL",
    "mpn/100 ml": "MPN/100mL",
    "cfu/100ml":  "CFU/100mL",
    "cfu/100 ml": "CFU/100mL",
    "cfu/ml":     "CFU/mL",

    # Flow
    "m³/s":     "m³/s",
    "m3/s":     "m³/s",
    "cumec":    "m³/s",
    "cusec":    "ft³/s",
    "ft³/s":    "ft³/s",
    "ft3/s":    "ft³/s",

    # Length
    "m":        "m",
    "metre":    "m",
    "meter":    "m",
    "cm":       "cm",
    "mm":       "mm",
    "ft":       "ft",

    # Dimensionless
    "":         "",
    "-":        "",
    "none":     "",
}


# ──────────────────────────────────────────────────────────────────────────────
# Conversion factors: (from_unit, to_unit) → multiplier
# ──────────────────────────────────────────────────────────────────────────────
UNIT_CONVERSIONS: dict[Tuple[str, str], float] = {
    # Concentration
    ("µg/L",    "mg/L"):     0.001,
    ("mg/L",    "µg/L"):     1000.0,
    ("g/L",     "mg/L"):     1000.0,
    ("mg/L",    "g/L"):      0.001,

    # Conductivity
    ("mS/cm",   "µS/cm"):    1000.0,
    ("µS/cm",   "mS/cm"):    0.001,

    # Turbidity (approximate, flagged with lower confidence)
    ("JTU",     "NTU"):      1.0,    # roughly equivalent
    ("FTU",     "NTU"):      1.0,
    ("FNU",     "NTU"):      1.0,

    # Temperature
    ("°F",      "°C"):       None,   # special: (F-32)*5/9

    # Biological
    ("CFU/100mL", "MPN/100mL"): 1.0,   # approximate equivalence
    ("CFU/mL",    "MPN/100mL"): 100.0,

    # Flow
    ("ft³/s",   "m³/s"):     0.0283168,

    # Length
    ("cm",      "m"):        0.01,
    ("mm",      "m"):        0.001,
    ("ft",      "m"):        0.3048,
}


def normalize_unit(raw_unit: str) -> str:
    """Normalize a raw unit string to canonical form."""
    cleaned = raw_unit.strip().lower()
    cleaned = re.sub(r"\s+", "", cleaned)
    return _UNIT_ALIASES.get(cleaned, raw_unit.strip())


def convert_value(
    value: float,
    from_unit: str,
    to_unit: str,
) -> Tuple[float, float]:
    """Convert a value between units.

    Returns:
        (converted_value, confidence)
        confidence is 1.0 for exact conversions, 0.8 for approximate ones.
    """
    if from_unit == to_unit:
        return value, 1.0

    # Special case: Fahrenheit → Celsius
    if from_unit == "°F" and to_unit == "°C":
        return (value - 32) * 5 / 9, 1.0

    key = (from_unit, to_unit)
    if key in UNIT_CONVERSIONS:
        factor = UNIT_CONVERSIONS[key]
        # Approximate conversions get lower confidence
        approximate = {
            ("JTU", "NTU"), ("FTU", "NTU"), ("FNU", "NTU"),
            ("CFU/100mL", "MPN/100mL"), ("CFU/mL", "MPN/100mL"),
        }
        conf = 0.85 if key in approximate else 1.0
        return value * factor, conf

    raise ValueError(f"No conversion rule from '{from_unit}' to '{to_unit}'")
