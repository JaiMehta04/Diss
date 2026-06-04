"""
Physical bounds for water quality parameters.

Used by the validation engine to flag or reject records that fall outside
physically possible or statistically expected ranges.
"""

from typing import NamedTuple, Optional


class Bounds(NamedTuple):
    """Physical bounds for a water quality parameter."""
    hard_min: Optional[float]   # Physically impossible below this → REJECTED
    soft_min: Optional[float]   # Unusual below this → SUSPECT_RANGE
    soft_max: Optional[float]   # Unusual above this → SUSPECT_RANGE
    hard_max: Optional[float]   # Physically impossible above this → REJECTED


# ──────────────────────────────────────────────────────────────────────────────
# Bounds dictionary: canonical_parameter → Bounds
# Values are based on literature ranges for Indian rivers.
# ──────────────────────────────────────────────────────────────────────────────
PHYSICAL_BOUNDS: dict[str, Bounds] = {
    # Physical
    "temperature":      Bounds(0,     5,     40,    50),
    "turbidity":        Bounds(0,     0,     2000,  10000),
    "conductivity":     Bounds(0,     10,    5000,  50000),
    "tds":              Bounds(0,     10,    3000,  50000),
    "tss":              Bounds(0,     0,     2000,  20000),
    "water_level":      Bounds(-10,   0,     30,    100),

    # Chemical
    "ph":               Bounds(0,     4,     10,    14),
    "do":               Bounds(0,     0,     15,    25),
    "bod":              Bounds(0,     0,     100,   1000),
    "cod":              Bounds(0,     0,     500,   5000),
    "toc":              Bounds(0,     0,     50,    500),
    "alkalinity":       Bounds(0,     0,     1000,  5000),
    "hardness":         Bounds(0,     0,     2000,  10000),

    # Nutrients
    "nitrate":          Bounds(0,     0,     100,   500),
    "nitrite":          Bounds(0,     0,     10,    50),
    "ammonia":          Bounds(0,     0,     50,    200),
    "ammoniacal_nitrogen": Bounds(0,  0,     50,    200),
    "total_nitrogen":   Bounds(0,     0,     100,   500),
    "phosphate":        Bounds(0,     0,     30,    100),
    "total_phosphorus": Bounds(0,     0,     30,    100),

    # Ions
    "chloride":         Bounds(0,     0,     2000,  10000),
    "fluoride":         Bounds(0,     0,     10,    50),
    "sulphate":         Bounds(0,     0,     1000,  5000),
    "sodium":           Bounds(0,     0,     1000,  10000),
    "potassium":        Bounds(0,     0,     200,   1000),
    "calcium":          Bounds(0,     0,     1000,  5000),
    "magnesium":        Bounds(0,     0,     500,   2000),
    "iron":             Bounds(0,     0,     20,    100),

    # Heavy metals (typically µg/L converted to mg/L)
    "arsenic":          Bounds(0,     0,     0.5,   5),
    "cadmium":          Bounds(0,     0,     0.1,   1),
    "chromium":         Bounds(0,     0,     1,     10),
    "copper":           Bounds(0,     0,     5,     50),
    "lead":             Bounds(0,     0,     0.5,   5),
    "mercury":          Bounds(0,     0,     0.01,  0.1),
    "nickel":           Bounds(0,     0,     1,     10),
    "zinc":             Bounds(0,     0,     10,    50),
    "manganese":        Bounds(0,     0,     5,     20),

    # Biological (MPN/100mL — log-scale)
    "fecal_coliform":   Bounds(0,     0,     1e7,   1e9),
    "total_coliform":   Bounds(0,     0,     1e8,   1e10),
    "e_coli":           Bounds(0,     0,     1e7,   1e9),

    # Hydrological
    "discharge":        Bounds(0,     0,     50000, 200000),
    "velocity":         Bounds(0,     0,     10,    30),
    "depth":            Bounds(0,     0,     50,    200),
    "sediment_load":    Bounds(0,     0,     10000, 100000),
}


def check_bounds(parameter: str, value: float) -> tuple[str, str]:
    """Check if a value is within physical bounds.

    Returns:
        (quality_flag, reason)
    """
    if parameter not in PHYSICAL_BOUNDS:
        return "valid", ""

    b = PHYSICAL_BOUNDS[parameter]

    if b.hard_min is not None and value < b.hard_min:
        return "rejected", f"Below hard minimum ({value} < {b.hard_min})"
    if b.hard_max is not None and value > b.hard_max:
        return "rejected", f"Above hard maximum ({value} > {b.hard_max})"
    if b.soft_min is not None and value < b.soft_min:
        return "suspect_range", f"Below typical range ({value} < {b.soft_min})"
    if b.soft_max is not None and value > b.soft_max:
        return "suspect_range", f"Above typical range ({value} > {b.soft_max})"

    return "valid", ""
