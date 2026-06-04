"""
Canonical parameter names and mapping from source variations.

Every parameter in the data lake uses lowercase snake_case canonical names.
This module maps the hundreds of variations found across Indian WQ data
sources to those canonical names.
"""

import re
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Canonical parameter definitions
# ──────────────────────────────────────────────────────────────────────────────
# Mapping: canonical_name → (display_name, canonical_unit)
CANONICAL_PARAMETERS = {
    # Physical
    "temperature":      ("Temperature",           "°C"),
    "turbidity":        ("Turbidity",             "NTU"),
    "conductivity":     ("Conductivity",          "µS/cm"),
    "tds":              ("Total Dissolved Solids", "mg/L"),
    "tss":              ("Total Suspended Solids", "mg/L"),
    "colour":           ("Colour",                "Hazen"),
    "odour":            ("Odour",                 ""),
    "water_level":      ("Water Level",           "m"),

    # Chemical — general
    "ph":               ("pH",                    ""),
    "do":               ("Dissolved Oxygen",      "mg/L"),
    "bod":              ("Biochemical Oxygen Demand", "mg/L"),
    "cod":              ("Chemical Oxygen Demand", "mg/L"),
    "toc":              ("Total Organic Carbon",  "mg/L"),
    "alkalinity":       ("Total Alkalinity",      "mg/L"),
    "hardness":         ("Total Hardness",        "mg/L"),

    # Nutrients
    "nitrate":          ("Nitrate",               "mg/L"),
    "nitrite":          ("Nitrite",               "mg/L"),
    "ammonia":          ("Ammonia",               "mg/L"),
    "ammoniacal_nitrogen": ("Ammoniacal Nitrogen","mg/L"),
    "total_nitrogen":   ("Total Nitrogen",        "mg/L"),
    "phosphate":        ("Phosphate",             "mg/L"),
    "total_phosphorus": ("Total Phosphorus",      "mg/L"),

    # Ions
    "chloride":         ("Chloride",              "mg/L"),
    "fluoride":         ("Fluoride",              "mg/L"),
    "sulphate":         ("Sulphate",              "mg/L"),
    "sodium":           ("Sodium",                "mg/L"),
    "potassium":        ("Potassium",             "mg/L"),
    "calcium":          ("Calcium",               "mg/L"),
    "magnesium":        ("Magnesium",             "mg/L"),
    "iron":             ("Iron",                  "mg/L"),

    # Metals
    "arsenic":          ("Arsenic",               "mg/L"),
    "cadmium":          ("Cadmium",               "mg/L"),
    "chromium":         ("Chromium",              "mg/L"),
    "copper":           ("Copper",                "mg/L"),
    "lead":             ("Lead",                  "mg/L"),
    "mercury":          ("Mercury",               "mg/L"),
    "nickel":           ("Nickel",                "mg/L"),
    "zinc":             ("Zinc",                  "mg/L"),
    "manganese":        ("Manganese",             "mg/L"),
    "boron":            ("Boron",                 "mg/L"),
    "selenium":         ("Selenium",              "mg/L"),

    # Biological
    "fecal_coliform":   ("Fecal Coliform",        "MPN/100mL"),
    "total_coliform":   ("Total Coliform",        "MPN/100mL"),
    "e_coli":           ("E. Coli",               "MPN/100mL"),

    # Organic
    "oil_grease":       ("Oil & Grease",          "mg/L"),
    "phenol":           ("Phenol",                "mg/L"),
    "detergent":        ("Detergent (MBAS)",      "mg/L"),
    "pesticides":       ("Pesticides",            "µg/L"),

    # Hydrological
    "discharge":        ("Discharge",             "m³/s"),
    "velocity":         ("Flow Velocity",         "m/s"),
    "depth":            ("River Depth",           "m"),
    "sediment_load":    ("Sediment Load",         "mg/L"),
}


# ──────────────────────────────────────────────────────────────────────────────
# Fuzzy mapping: source parameter name → canonical name
# ──────────────────────────────────────────────────────────────────────────────
# Each entry is (regex_pattern, canonical_name).
# Patterns are matched case-insensitively against the source parameter name.
_PARAMETER_MAP: list[tuple[str, str]] = [
    # Temperature
    (r"^temp(?:erature)?(?:\s*(?:of\s+)?water)?$", "temperature"),
    (r"^water\s*temp", "temperature"),

    # pH
    (r"^ph$", "ph"),
    (r"^p\.?h\.?$", "ph"),
    (r"^hydrogen\s*ion", "ph"),

    # DO
    (r"^d\.?o\.?$", "do"),
    (r"^dissolved\s*oxygen", "do"),

    # BOD
    (r"^b\.?o\.?d\.?(?:\s*[35])?$", "bod"),
    (r"^biochem(?:ical)?\s*oxygen\s*demand", "bod"),

    # COD
    (r"^c\.?o\.?d\.?$", "cod"),
    (r"^chem(?:ical)?\s*oxygen\s*demand", "cod"),

    # Conductivity / EC
    (r"^(?:e\.?c\.?|conductivity|elec(?:trical)?\s*cond)", "conductivity"),
    (r"^specific\s*cond", "conductivity"),

    # Turbidity
    (r"^turbidity", "turbidity"),
    (r"^turb$", "turbidity"),

    # TDS
    (r"^t\.?d\.?s\.?$", "tds"),
    (r"^total\s*dissolved\s*solid", "tds"),

    # TSS
    (r"^t\.?s\.?s\.?$", "tss"),
    (r"^total\s*suspended\s*solid", "tss"),

    # Nitrate
    (r"^nitrate", "nitrate"),
    (r"^no3", "nitrate"),

    # Nitrite
    (r"^nitrite", "nitrite"),
    (r"^no2", "nitrite"),

    # Ammonia / Ammoniacal Nitrogen
    (r"^ammonia(?:cal)?\s*(?:nitrogen|n)?$", "ammoniacal_nitrogen"),
    (r"^nh[34]", "ammoniacal_nitrogen"),
    (r"^ammoniacal", "ammoniacal_nitrogen"),

    # Total Nitrogen
    (r"^total\s*nitrogen", "total_nitrogen"),
    (r"^t\.?n\.?$", "total_nitrogen"),

    # Phosphate
    (r"^phosphate", "phosphate"),
    (r"^po4", "phosphate"),

    # Total Phosphorus
    (r"^total\s*phospho", "total_phosphorus"),
    (r"^t\.?p\.?$", "total_phosphorus"),

    # Chloride
    (r"^chloride", "chloride"),
    (r"^cl\s*[-–]?$", "chloride"),

    # Fluoride
    (r"^fluoride", "fluoride"),

    # Sulphate / Sulfate
    (r"^sul[fp]h?ate", "sulphate"),
    (r"^so4", "sulphate"),

    # Alkalinity
    (r"^(?:total\s*)?alkalinity", "alkalinity"),

    # Hardness
    (r"^(?:total\s*)?hardness", "hardness"),

    # Calcium
    (r"^calcium", "calcium"),
    (r"^ca$", "calcium"),

    # Magnesium
    (r"^magnesium", "magnesium"),
    (r"^mg$", "magnesium"),

    # Sodium
    (r"^sodium", "sodium"),
    (r"^na$", "sodium"),

    # Potassium
    (r"^potassium", "potassium"),
    (r"^k$", "potassium"),

    # Iron
    (r"^iron", "iron"),
    (r"^fe$", "iron"),

    # Heavy metals
    (r"^arsenic", "arsenic"),
    (r"^cadmium", "cadmium"),
    (r"^chromium", "chromium"),
    (r"^copper", "copper"),
    (r"^cu$", "copper"),
    (r"^lead", "lead"),
    (r"^pb$", "lead"),
    (r"^mercury", "mercury"),
    (r"^hg$", "mercury"),
    (r"^nickel", "nickel"),
    (r"^ni$", "nickel"),
    (r"^zinc", "zinc"),
    (r"^zn$", "zinc"),
    (r"^manganese", "manganese"),
    (r"^mn$", "manganese"),
    (r"^boron", "boron"),
    (r"^selenium", "selenium"),
    (r"^se$", "selenium"),

    # Biological
    (r"^f(?:a?ec|aec)al\s*coli", "fecal_coliform"),
    (r"^fc$", "fecal_coliform"),
    (r"^total\s*coli", "total_coliform"),
    (r"^tc$", "total_coliform"),
    (r"^e\.?\s*coli", "e_coli"),

    # Colour / Color
    (r"^colou?r", "colour"),

    # TOC
    (r"^t\.?o\.?c\.?$", "toc"),
    (r"^total\s*organic\s*carbon", "toc"),

    # Oil & Grease
    (r"^oil", "oil_grease"),

    # Phenol
    (r"^phenol", "phenol"),

    # Detergent
    (r"^detergent", "detergent"),
    (r"^mbas", "detergent"),

    # Water Level
    (r"^water\s*level", "water_level"),
    (r"^gauge\s*level", "water_level"),
    (r"^stage", "water_level"),

    # Discharge
    (r"^discharge", "discharge"),
    (r"^flow$", "discharge"),
    (r"^stream\s*flow", "discharge"),

    # Sediment
    (r"^sediment", "sediment_load"),
    (r"^suspended\s*sediment", "sediment_load"),
]

# Pre-compile regexes
PARAMETER_MAP = [(re.compile(pat, re.IGNORECASE), canon) for pat, canon in _PARAMETER_MAP]


def normalize_parameter_name(raw_name: str) -> Optional[str]:
    """Map a raw parameter name to its canonical form.

    Returns None if no match is found (the LLM agent should handle these).
    """
    cleaned = raw_name.strip()
    # Remove common suffixes like "(mg/L)", "(NTU)" etc.
    cleaned = re.sub(r"\s*\(.*?\)\s*$", "", cleaned)
    cleaned = cleaned.strip()

    for pattern, canonical in PARAMETER_MAP:
        if pattern.search(cleaned):
            return canonical

    return None
