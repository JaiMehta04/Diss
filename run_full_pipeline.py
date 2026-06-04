"""
Full pipeline: Data Sources → Ingestion & Standardization → QA/QC & Trust Scoring
→ Ontology Mapping → Fusion → Final consolidated CSV.

Follows the architecture:
  1. Data Sources: CPCB, data.gov.in (API), NWMP PDFs, existing CSVs
  2. Ingestion & Standardization: Schema alignment, unit conversion
  3. QA/QC & Trust Scoring: Range checks, outlier detection, trust weights
  4. Ontology & Semantic Mapping: Parameter name normalization
  5. Fusion & Analytics: Merge multi-source, resolve conflicts
  6. Governance: Provenance tracking, FAIR compliance

Usage:
    python run_full_pipeline.py
"""

import csv
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Setup ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("full_pipeline")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
NWMP_PDF_DIR = RAW_DIR / "cpcb" / "nwmp_pdfs"
DATAGOV_DIR = RAW_DIR / "datagov"
CPCB_CSV = DATA_DIR / "combined_station_data_with_coords.csv"
STATION_META = DATA_DIR / "cpcb_station_locations.csv"
OUTPUT_DIR = BASE_DIR / "data" / "final"
FINAL_CSV = OUTPUT_DIR / "ganga_water_quality_final.csv"

# ── Trust scores per source ───────────────────────────────────────────────────
SOURCE_TRUST = {
    "cpcb": 1.0,           # Lab-verified, highest trust
    "data_gov_in": 0.90,   # Government API, structured
    "nwmp_pdf": 0.85,      # PDF extraction, may have artifacts
}

# ── Physical bounds for QA/QC ─────────────────────────────────────────────────
BOUNDS = {
    "temperature":    (0, 50),
    "ph":             (0, 14),
    "conductivity":   (0, 50000),
    "do":             (0, 25),
    "bod":            (0, 1000),
    "cod":            (0, 5000),
    "nitrate":        (0, 500),
    "fecal_coliform": (0, 1e9),
    "total_coliform": (0, 1e10),
    "tds":            (0, 50000),
    "fluoride":       (0, 50),
    "arsenic":        (0, 5),
    "turbidity":      (0, 10000),
    "chloride":       (0, 10000),
    "hardness":       (0, 10000),
    "alkalinity":     (0, 5000),
    "sulphate":       (0, 5000),
    "iron":           (0, 100),
    "calcium":        (0, 5000),
    "magnesium":      (0, 2000),
    "sodium":         (0, 10000),
    "potassium":      (0, 1000),
    "ammonia":        (0, 200),
    "tss":            (0, 20000),
    "toc":            (0, 500),
}

# ── Canonical parameters we want in the final output ──────────────────────────
CANONICAL_PARAMS = [
    "temperature", "ph", "conductivity", "do", "bod", "cod", "nitrate",
    "fecal_coliform", "total_coliform", "tds", "fluoride", "arsenic",
    "turbidity", "chloride", "hardness", "alkalinity", "sulphate",
    "iron", "calcium", "magnesium", "sodium", "potassium", "ammonia",
    "tss", "toc",
]

# ── Column name normalization map ─────────────────────────────────────────────
# Maps various column names from different sources to canonical parameter names
_COLUMN_NORM: Dict[str, str] = {}


def _build_column_norm():
    """Build a comprehensive column-name → canonical-parameter mapping."""
    mappings = {
        "temperature": [
            "temperature", "temp", "water_temperature", "temperature_c",
            "temperature_in_degree_centigrade", "water_temp",
        ],
        "ph": ["ph", "ph_value", "ph_gen", "ph_general"],
        "conductivity": [
            "conductivity", "cond", "ec", "electrical_conductivity",
            "conductivity_mhos_cm", "specific_conductance",
        ],
        "do": [
            "do", "_do", "dissolved_oxygen", "d_o", "diss_oxygen",
            "dissolved_oxygen_mg_l",
        ],
        "bod": ["bod", "b_o_d", "biochemical_oxygen_demand", "bod_mg_l"],
        "cod": ["cod", "c_o_d", "chemical_oxygen_demand", "cod_mg_l"],
        "nitrate": [
            "nitrate", "no3", "nitrate_n", "nitratenitrogen",
            "nitrate_nitrogen",
        ],
        "fecal_coliform": [
            "fecal_coliform", "faecal_coliform", "fc", "f_coliform",
            "fc_mpn_100ml", "fecal_coli",
        ],
        "total_coliform": [
            "total_coliform", "tc", "t_coliform", "tcol",
            "tcol_mpn", "total_coli",
        ],
        "tds": ["tds", "total_dissolved_solids"],
        "fluoride": ["fluoride", "f", "fl"],
        "arsenic": ["arsenic", "as"],
        "turbidity": ["turbidity", "turb", "ntu"],
        "chloride": ["chloride", "cl"],
        "hardness": ["hardness", "total_hardness", "th"],
        "alkalinity": ["alkalinity", "total_alkalinity"],
        "sulphate": ["sulphate", "sulfate", "so4"],
        "iron": ["iron", "fe"],
        "calcium": ["calcium", "ca"],
        "magnesium": ["magnesium", "mg"],
        "sodium": ["sodium", "na"],
        "potassium": ["potassium", "k"],
        "ammonia": ["ammonia", "nh3", "ammoniacal_nitrogen"],
        "tss": ["tss", "total_suspended_solids"],
        "toc": ["toc", "total_organic_carbon"],
    }
    for canonical, aliases in mappings.items():
        for alias in aliases:
            _COLUMN_NORM[alias.lower()] = canonical


_build_column_norm()


def normalize_column_name(col: str) -> Optional[str]:
    """Normalize a column name to a canonical parameter name."""
    # Clean the column name
    col_clean = col.lower().strip()
    col_clean = re.sub(r"[^a-z0-9_]", "_", col_clean)
    col_clean = re.sub(r"_+", "_", col_clean).strip("_")

    # Direct match
    if col_clean in _COLUMN_NORM:
        return _COLUMN_NORM[col_clean]

    # Partial match
    for alias, canonical in _COLUMN_NORM.items():
        if alias in col_clean or col_clean in alias:
            return canonical

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: INGESTION — Load from all sources
# ═══════════════════════════════════════════════════════════════════════════════


def ingest_cpcb_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Ingest CPCB monitoring data from combined CSV."""
    if not csv_path.exists():
        logger.warning(f"CPCB CSV not found: {csv_path}")
        return []

    import pandas as pd

    # Load station metadata for ID → name mapping
    station_meta = {}
    if STATION_META.exists():
        meta_df = pd.read_csv(STATION_META)
        for _, row in meta_df.iterrows():
            sid = str(int(row["station_id"])) if pd.notna(row.get("station_id")) else ""
            if sid:
                station_meta[sid] = {
                    "station_name": str(row.get("station_name", "")).strip(),
                    "state": str(row.get("state", "")).strip(),
                }
        logger.info(f"  Loaded metadata for {len(station_meta)} CPCB stations")

    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"CPCB CSV: {len(df)} rows, {len(df.columns)} columns")

    records = []
    # Identify station and parameter columns
    station_col = None
    for c in ["station_name", "stationName", "station", "stationId"]:
        if c in df.columns:
            station_col = c
            break

    lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
    state_col = next((c for c in df.columns if c.lower() in ("state", "state_name")), None)
    date_col = next(
        (c for c in df.columns if c.lower() in ("timestamp", "date", "sample_date", "timestampdate")),
        None,
    )

    # Map parameter columns — also try full CPCB column names
    _CPCB_DIRECT_MAP = {
        "Biochemical Oxygen Demand": "bod",
        "Chemical Oxygen Demand": "cod",
        "Chloride": "chloride",
        "Conductivity": "conductivity",
        "Dissolved Oxygen": "do",
        "Nitrate": "nitrate",
        "Total Organic Carbon": "toc",
        "Water Temperature": "temperature",
        "Water Turbidity": "turbidity",
        "pH": "ph",
        "Depth": "depth",
        "Water Level": "water_level",
        "Fecal Coliform": "fecal_coliform",
        "Total Coliform": "total_coliform",
        "Total Dissolved Solids": "tds",
        "Hardness": "hardness",
        "Alkalinity": "alkalinity",
        "Fluoride": "fluoride",
        "Sulphate": "sulphate",
        "Iron": "iron",
        "Ammonia": "ammonia",
        "Calcium": "calcium",
        "Magnesium": "magnesium",
        "Sodium": "sodium",
        "Potassium": "potassium",
    }
    param_cols = {}
    for col in df.columns:
        if col in _CPCB_DIRECT_MAP:
            param_cols[col] = _CPCB_DIRECT_MAP[col]
        else:
            canonical = normalize_column_name(col)
            if canonical:
                param_cols[col] = canonical

    logger.info(f"  Mapped {len(param_cols)} parameter columns: {list(param_cols.values())[:10]}")

    for _, row in df.iterrows():
        station_id = str(row.get(station_col, "")).strip() if station_col else ""
        # Convert float station IDs like 11783.0 → "11783"
        try:
            station_id = str(int(float(station_id)))
        except (ValueError, TypeError):
            pass

        # Look up station name from metadata
        meta = station_meta.get(station_id, {})
        station_name = meta.get("station_name", station_id)

        base = {
            "station_name": station_name,
            "station_code": station_id,
            "latitude": _safe_float(row.get(lat_col)) if lat_col else None,
            "longitude": _safe_float(row.get(lon_col)) if lon_col else None,
            "state": meta.get("state") or (str(row.get(state_col, "")).strip() if state_col else ""),
            "date": _safe_date(row.get(date_col)) if date_col else None,
            "source": "cpcb",
        }

        # Skip if no station info
        if not base["station_name"]:
            continue

        for col, param in param_cols.items():
            val = _safe_float(row.get(col))
            if val is not None:
                rec = dict(base)
                rec["parameter"] = param
                rec["value"] = val
                rec["unit"] = _default_unit(param)
                records.append(rec)

    logger.info(f"  Ingested {len(records)} CPCB records")
    return records


def ingest_datagov_csvs(datagov_dir: Path) -> List[Dict[str, Any]]:
    """Ingest all CSVs from data.gov.in downloads."""
    if not datagov_dir.exists():
        logger.warning(f"DataGov directory not found: {datagov_dir}")
        return []

    import pandas as pd
    records = []

    for csv_path in sorted(datagov_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            logger.debug(f"  Skip {csv_path.name}: {e}")
            continue

        # Normalize column names
        col_map = {}
        for col in df.columns:
            canonical = normalize_column_name(col)
            if canonical:
                col_map[col] = canonical

        # Find station/location columns — prefer name/location over code
        station_col = None
        _station_priority = [
            "station_name", "location", "locations", "site_name",
            "station", "site", "station_code",
        ]
        for preferred in _station_priority:
            for c in df.columns:
                if c.lower().strip() == preferred:
                    station_col = c
                    break
            if station_col:
                break
        # Fallback: any column with station/location/site keyword
        if not station_col:
            for c in df.columns:
                if any(kw in c.lower() for kw in ["station", "location", "site"]):
                    station_col = c
                    break

        lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
        lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
        state_col = next(
            (c for c in df.columns if c.lower().strip() in ("state", "state_name")),
            None,
        )
        date_col = next(
            (c for c in df.columns if any(kw in c.lower() for kw in ["date", "year", "period"])),
            None,
        )

        for _, row in df.iterrows():
            base = {
                "station_name": str(row.get(station_col, "")).strip() if station_col else "",
                "latitude": _safe_float(row.get(lat_col)) if lat_col else None,
                "longitude": _safe_float(row.get(lon_col)) if lon_col else None,
                "state": str(row.get(state_col, "")).strip() if state_col else "",
                "date": _safe_date(row.get(date_col)) if date_col else None,
                "source": "data_gov_in",
                "source_file": csv_path.name,
            }

            if not base["station_name"]:
                continue

            for col, param in col_map.items():
                val = _safe_float(row.get(col))
                if val is not None:
                    rec = dict(base)
                    rec["parameter"] = param
                    rec["value"] = val
                    rec["unit"] = _default_unit(param)
                    records.append(rec)

        logger.info(f"  {csv_path.name}: {len(df)} rows")

    logger.info(f"  Ingested {len(records)} data.gov.in records")
    return records


def ingest_nwmp_pdfs(pdf_dir: Path) -> List[Dict[str, Any]]:
    """Ingest water quality data from NWMP PDF reports using robust extractor."""
    if not pdf_dir.exists():
        logger.warning(f"NWMP PDF directory not found: {pdf_dir}")
        return []

    from water_quality_agent.downloaders.nwmp_extractor import NWMPExtractor

    extractor = NWMPExtractor()
    raw_records = extractor.extract_all(str(pdf_dir), river="ganga")

    # Convert NWMP flat records to long-form (one row per parameter)
    records = []
    for raw in raw_records:
        base = {
            "station_name": raw.get("station_name", ""),
            "station_code": raw.get("station_code", ""),
            "state": raw.get("state", ""),
            "year": raw.get("year"),
            "date": f"{raw.get('year', '')}-01-01" if raw.get("year") else None,
            "source": "nwmp_pdf",
            "source_page": raw.get("source_page"),
        }

        if not base["station_name"] and not base["station_code"]:
            continue

        for param in CANONICAL_PARAMS:
            # Check for direct value, or min/max
            val = raw.get(param)
            val_min = raw.get(f"{param}_min")
            val_max = raw.get(f"{param}_max")

            if val is not None:
                rec = dict(base)
                rec["parameter"] = param
                rec["value"] = val
                rec["unit"] = _default_unit(param)
                records.append(rec)
            elif val_min is not None and val_max is not None:
                # Use mean of min/max
                rec = dict(base)
                rec["parameter"] = param
                rec["value"] = (val_min + val_max) / 2.0
                rec["value_min"] = val_min
                rec["value_max"] = val_max
                rec["unit"] = _default_unit(param)
                records.append(rec)
            elif val_min is not None:
                rec = dict(base)
                rec["parameter"] = param
                rec["value"] = val_min
                rec["unit"] = _default_unit(param)
                records.append(rec)
            elif val_max is not None:
                rec = dict(base)
                rec["parameter"] = param
                rec["value"] = val_max
                rec["unit"] = _default_unit(param)
                records.append(rec)

    logger.info(f"  Ingested {len(records)} NWMP PDF records")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: STANDARDIZATION — Schema alignment, unit conversion
# ═══════════════════════════════════════════════════════════════════════════════


def standardize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Standardize all records to canonical schema."""
    standardized = []
    dropped = 0

    for rec in records:
        param = rec.get("parameter", "")
        if param not in CANONICAL_PARAMS:
            dropped += 1
            continue

        # Ensure numeric value
        val = rec.get("value")
        if val is None:
            dropped += 1
            continue

        try:
            val = float(val)
        except (ValueError, TypeError):
            dropped += 1
            continue

        rec["value"] = val
        rec["unit"] = _default_unit(param)

        # Normalize station name
        rec["station_name"] = _clean_station_name(rec.get("station_name", ""))

        # Normalize state
        rec["state"] = _normalize_state(rec.get("state", ""))

        # Filter NWMP PDF records: only keep Ganga basin states
        # (NWMP PDFs cover all of India; keyword matching picks up pages
        #  that mention "Ganga" but contain non-Ganga station data too)
        if rec.get("source") == "nwmp_pdf":
            state = rec.get("state", "")
            station = rec.get("station_name", "").lower()
            # Keep if in Ganga basin state OR station name mentions Ganga
            ganga_keywords = ["ganga", "ganges", "yamuna", "gomti", "ghaghra",
                              "gandak", "kosi", "son ", "damodar", "hooghly",
                              "ramganga", "hindon", "varuna"]
            in_basin = state in _GANGA_BASIN_STATES
            name_match = any(kw in station for kw in ganga_keywords)
            if not in_basin and not name_match:
                dropped += 1
                continue

        # Normalize date
        if rec.get("date"):
            rec["date"] = _safe_date(rec["date"])
        elif rec.get("year"):
            rec["date"] = f"{rec['year']}-01-01"

        # Extract year if not present
        if not rec.get("year") and rec.get("date"):
            try:
                rec["year"] = int(str(rec["date"])[:4])
            except (ValueError, TypeError):
                pass

        standardized.append(rec)

    logger.info(
        f"Standardization: {len(standardized)} valid, {dropped} dropped"
    )
    return standardized


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2b: SPATIAL VALIDATION — Ganga basin bounding box filter
# ═══════════════════════════════════════════════════════════════════════════════


def spatial_filter(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter records with coordinates outside the Ganga basin bounding box.

    Records WITHOUT coordinates are kept (no grounds to reject them).
    Records WITH coordinates clearly outside the basin are removed.
    """
    lat_min = _GANGA_BBOX["lat_min"]
    lat_max = _GANGA_BBOX["lat_max"]
    lon_min = _GANGA_BBOX["lon_min"]
    lon_max = _GANGA_BBOX["lon_max"]

    kept = []
    rejected = 0

    for rec in records:
        lat = rec.get("latitude")
        lon = rec.get("longitude")

        # No coords — keep (benefit of the doubt)
        if lat is None or lon is None:
            kept.append(rec)
            continue

        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except (ValueError, TypeError):
            kept.append(rec)
            continue

        # Check bounding box
        if lat_min <= lat_f <= lat_max and lon_min <= lon_f <= lon_max:
            kept.append(rec)
        else:
            rejected += 1

    logger.info(
        f"Spatial filter: {len(kept)} kept, {rejected} rejected "
        f"(outside bbox {lat_min}–{lat_max}°N, {lon_min}–{lon_max}°E)"
    )
    return kept


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: QA/QC & TRUST SCORING
# ═══════════════════════════════════════════════════════════════════════════════


def qaqc_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply QA/QC checks and assign quality flags + trust scores."""
    valid = []
    stats = defaultdict(int)

    for rec in records:
        param = rec.get("parameter", "")
        val = rec.get("value")
        source = rec.get("source", "")

        # Range check
        if param in BOUNDS:
            lo, hi = BOUNDS[param]
            if val < lo or val > hi:
                stats["rejected_range"] += 1
                continue

        # Assign trust score based on source
        base_trust = SOURCE_TRUST.get(source, 0.5)

        # Reduce trust for extreme values (soft bounds)
        if param in BOUNDS:
            lo, hi = BOUNDS[param]
            mid = (lo + hi) / 2
            spread = (hi - lo) / 2
            if spread > 0:
                distance = abs(val - mid) / spread
                if distance > 0.9:
                    base_trust *= 0.7

        # Negative values (except temperature) are suspect
        if val < 0 and param != "temperature":
            stats["rejected_negative"] += 1
            continue

        rec["quality_flag"] = "valid"
        rec["trust_score"] = round(base_trust, 3)
        valid.append(rec)
        stats["valid"] += 1

    logger.info(
        f"QA/QC: {stats['valid']} valid, "
        f"{stats.get('rejected_range', 0)} rejected (range), "
        f"{stats.get('rejected_negative', 0)} rejected (negative)"
    )
    return valid


def detect_outliers(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flag statistical outliers using IQR method."""
    # Group by parameter
    by_param: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records:
        by_param[rec["parameter"]].append(rec)

    flagged = 0
    for param, recs in by_param.items():
        if len(recs) < 20:
            continue

        values = sorted(r["value"] for r in recs)
        n = len(values)
        q1 = values[n // 4]
        q3 = values[3 * n // 4]
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr

        for rec in recs:
            if rec["value"] < lower or rec["value"] > upper:
                rec["quality_flag"] = "suspect_outlier"
                rec["trust_score"] *= 0.5
                flagged += 1

    logger.info(f"Outlier detection: {flagged} records flagged")
    return records


def cross_param_consistency(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Check cross-parameter consistency (BOD ≤ COD, etc.)."""
    # Group by (station, date)
    groups: Dict[Tuple, Dict[str, float]] = defaultdict(dict)
    group_records: Dict[Tuple, List[Dict]] = defaultdict(list)

    for rec in records:
        key = (rec.get("station_name", ""), rec.get("date", ""))
        groups[key][rec["parameter"]] = rec["value"]
        group_records[key].append(rec)

    flagged = 0
    for key, params in groups.items():
        issues = []

        # BOD ≤ COD (relaxed: BOD can legitimately approach COD in highly
        # biodegradable wastewater; flag only extreme violations > 2×)
        if "bod" in params and "cod" in params:
            if params["bod"] > params["cod"] * 2.0:
                issues.append("BOD > 2*COD")

        # TDS vs conductivity
        if "tds" in params and "conductivity" in params and params["conductivity"] > 0:
            ratio = params["tds"] / params["conductivity"]
            if ratio < 0.2 or ratio > 1.5:
                issues.append(f"TDS/EC ratio={ratio:.2f}")

        if issues:
            for rec in group_records[key]:
                if rec["quality_flag"] == "valid":
                    rec["quality_flag"] = "suspect_consistency"
                    rec["trust_score"] *= 0.7
                    rec["qc_notes"] = "; ".join(issues)
                    flagged += 1

    logger.info(f"Consistency checks: {flagged} records flagged")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: FUSION — Merge, deduplicate, resolve conflicts
# ═══════════════════════════════════════════════════════════════════════════════


def fuse_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fuse multi-source records: deduplicate and resolve conflicts.

    When multiple sources report the same parameter for the same
    station + date, we keep the record with the highest trust score.
    """
    # Group by (station_name, date, parameter)
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for rec in records:
        station = _normalize_for_dedup(rec.get("station_name", ""))
        dt = rec.get("date", "")
        param = rec.get("parameter", "")
        key = (station, dt, param)
        groups[key].append(rec)

    fused = []
    duplicates_resolved = 0

    for key, group in groups.items():
        if len(group) == 1:
            fused.append(group[0])
        else:
            # Multiple sources — pick highest trust, note provenance
            group.sort(key=lambda r: r.get("trust_score", 0), reverse=True)
            best = group[0]
            sources = list(set(r.get("source", "") for r in group))
            best["all_sources"] = "; ".join(sources)
            best["n_sources"] = len(sources)

            # If values agree (within 10%), boost trust
            values = [r["value"] for r in group]
            if len(values) > 1 and max(values) > 0:
                cv = (max(values) - min(values)) / max(max(values), 0.001)
                if cv < 0.1:
                    best["trust_score"] = min(1.0, best["trust_score"] * 1.2)
                    best["quality_flag"] = "valid"

            fused.append(best)
            duplicates_resolved += 1

    logger.info(
        f"Fusion: {len(fused)} records after dedup "
        f"({duplicates_resolved} conflicts resolved)"
    )
    return fused


def _normalize_for_dedup(name: str) -> str:
    """Normalize station name for deduplication."""
    s = name.lower().strip()
    s = re.sub(r"\b(station|monitoring|wq|water quality|cpcb|spcb)\b", "", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5: PIVOT TO WIDE FORMAT — One row per station-date
# ═══════════════════════════════════════════════════════════════════════════════


def pivot_to_wide(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pivot long-form records to wide format (one row per station-date).

    This is the final format: each row has station info + all parameter
    values as columns.
    """
    # Group by (station_name, date)
    groups: Dict[Tuple, Dict[str, Any]] = {}

    for rec in records:
        station = rec.get("station_name", "")
        dt = rec.get("date", "")
        key = (station, dt)

        if key not in groups:
            groups[key] = {
                "station_name": station,
                "station_code": rec.get("station_code", ""),
                "state": rec.get("state", ""),
                "latitude": rec.get("latitude"),
                "longitude": rec.get("longitude"),
                "date": dt,
                "year": rec.get("year"),
                "source": rec.get("source", ""),
                "all_sources": rec.get("all_sources", rec.get("source", "")),
                "n_params": 0,
                "min_trust": 1.0,
                "mean_trust": 0.0,
            }

        row = groups[key]
        param = rec["parameter"]
        row[param] = rec["value"]
        if rec.get("value_min") is not None:
            row[f"{param}_min"] = rec["value_min"]
        if rec.get("value_max") is not None:
            row[f"{param}_max"] = rec["value_max"]
        row["n_params"] += 1
        row["min_trust"] = min(row["min_trust"], rec.get("trust_score", 1.0))
        row["mean_trust"] += rec.get("trust_score", 1.0)

        # Update metadata if better
        if not row.get("latitude") and rec.get("latitude"):
            row["latitude"] = rec["latitude"]
        if not row.get("longitude") and rec.get("longitude"):
            row["longitude"] = rec["longitude"]
        if not row.get("state") and rec.get("state"):
            row["state"] = rec["state"]
        # Track all sources
        sources = set((row.get("all_sources") or "").split("; "))
        sources.add(rec.get("source", ""))
        sources.discard("")
        row["all_sources"] = "; ".join(sorted(sources))

    # Finalize
    rows = []
    for row in groups.values():
        if row["n_params"] > 0:
            row["mean_trust"] = round(row["mean_trust"] / row["n_params"], 3)
        rows.append(row)

    rows.sort(key=lambda r: (r.get("station_name", ""), str(r.get("date", ""))))
    logger.info(f"Pivoted to {len(rows)} station-date rows")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════


def save_final_csv(rows: List[Dict[str, Any]], output_path: Path):
    """Save the final fused dataset as CSV."""
    if not rows:
        logger.error("No records to save!")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine columns
    meta_cols = [
        "station_name", "station_code", "state", "latitude", "longitude",
        "date", "year", "all_sources", "n_params", "min_trust", "mean_trust",
    ]
    param_cols = []
    for p in CANONICAL_PARAMS:
        if any(p in r for r in rows):
            param_cols.append(p)

    all_cols = meta_cols + param_cols

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logger.info(f"Saved {len(rows)} rows → {output_path}")
    logger.info(f"  Columns: {len(all_cols)} ({len(meta_cols)} meta + {len(param_cols)} params)")


def save_long_csv(records: List[Dict[str, Any]], output_path: Path):
    """Save long-form records (one per observation) for the pipeline adapter."""
    if not records:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "station_name", "station_code", "state", "latitude", "longitude",
        "date", "year", "parameter", "value", "unit",
        "source", "quality_flag", "trust_score",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    logger.info(f"Saved {len(records)} long-form records → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        import math
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (ValueError, TypeError):
        return None


def _safe_date(val) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nat", "nan", "none", ""):
        return None
    # Try ISO format
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            dt = datetime.strptime(s[:10], fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Try just year
    m = re.match(r"^(\d{4})", s)
    if m:
        return f"{m.group(1)}-01-01"
    return None


def _default_unit(param: str) -> str:
    units = {
        "temperature": "°C",
        "ph": "",
        "conductivity": "µS/cm",
        "do": "mg/L",
        "bod": "mg/L",
        "cod": "mg/L",
        "nitrate": "mg/L",
        "fecal_coliform": "MPN/100mL",
        "total_coliform": "MPN/100mL",
        "tds": "mg/L",
        "fluoride": "mg/L",
        "arsenic": "mg/L",
        "turbidity": "NTU",
        "chloride": "mg/L",
        "hardness": "mg/L",
        "alkalinity": "mg/L",
        "sulphate": "mg/L",
        "iron": "mg/L",
        "calcium": "mg/L",
        "magnesium": "mg/L",
        "sodium": "mg/L",
        "potassium": "mg/L",
        "ammonia": "mg/L",
        "tss": "mg/L",
        "toc": "mg/L",
    }
    return units.get(param, "")


def _clean_station_name(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r"\s+", " ", name).strip()
    # Title case but preserve abbreviations
    words = []
    for w in name.split():
        if w.isupper() and len(w) <= 4:
            words.append(w)
        else:
            words.append(w.title())
    return " ".join(words)


# State name normalization map
_STATE_NORM = {
    "uttar pradesh": "Uttar Pradesh",
    "uttar": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand",
    "uttaranchal": "Uttarakhand",
    "uttrakhand": "Uttarakhand",
    "bihar": "Bihar",
    "west bengal": "West Bengal",
    "delhi": "Delhi",
    "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh",
    "jharkhand": "Jharkhand",
    "rajasthan": "Rajasthan",
    "madhya pradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "pradesh": "Uttar Pradesh",
    "andhra pradesh": "Andhra Pradesh",
    "assam": "Assam",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "jammu & kashmir": "Jammu & Kashmir",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "meghalaya": "Meghalaya",
    "odisha": "Odisha",
    "tamil nadu": "Tamil Nadu",
    "telangana": "Telangana",
    "tripura": "Tripura",
}

# States in the Ganga basin (for filtering)
_GANGA_BASIN_STATES = {
    "Uttar Pradesh", "Uttarakhand", "Bihar", "West Bengal",
    "Delhi", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Madhya Pradesh", "Rajasthan",
}

# Ganga basin approximate bounding box (lat/lon)
# Covers from Gangotri (31°N, 78.9°E) to Bay of Bengal (21.5°N, 89°E)
_GANGA_BBOX = {
    "lat_min": 21.0,
    "lat_max": 31.5,
    "lon_min": 73.0,
    "lon_max": 90.0,
}


def _normalize_state(state: str) -> str:
    if not state:
        return ""
    return _STATE_NORM.get(state.lower().strip(), state.strip().title())


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def run_pipeline():
    """Execute the full data integration pipeline."""
    logger.info("=" * 70)
    logger.info("GANGA WATER QUALITY — FULL DATA INTEGRATION PIPELINE")
    logger.info("=" * 70)

    # ── Stage 1: Ingestion ────────────────────────────────────────────────
    logger.info("\n[1/6] INGESTION — Loading from all data sources")
    all_records = []

    # Source 1: CPCB CSV
    logger.info("  Loading CPCB CSV...")
    cpcb_records = ingest_cpcb_csv(CPCB_CSV)
    all_records.extend(cpcb_records)

    # Source 2: data.gov.in
    logger.info("  Loading data.gov.in CSVs...")
    datagov_records = ingest_datagov_csvs(DATAGOV_DIR)
    all_records.extend(datagov_records)

    # Source 3: NWMP PDFs
    logger.info("  Loading NWMP PDFs (robust extraction)...")
    nwmp_records = ingest_nwmp_pdfs(NWMP_PDF_DIR)
    all_records.extend(nwmp_records)

    logger.info(f"  Total raw records: {len(all_records)}")

    if not all_records:
        logger.error("No records ingested from any source!")
        return

    # ── Stage 2: Standardization ──────────────────────────────────────────
    logger.info("\n[2/7] STANDARDIZATION — Schema alignment & unit conversion")
    records = standardize_records(all_records)

    # ── Stage 2b: Spatial validation ──────────────────────────────────────
    logger.info("\n[2b/7] SPATIAL VALIDATION — Ganga basin bounding box filter")
    records = spatial_filter(records)

    # ── Stage 3: QA/QC ───────────────────────────────────────────────────
    logger.info("\n[3/7] QA/QC — Range checks & trust scoring")
    records = qaqc_records(records)

    logger.info("\n[3b/7] QA/QC — Statistical outlier detection")
    records = detect_outliers(records)

    logger.info("\n[3c/7] QA/QC — Cross-parameter consistency")
    records = cross_param_consistency(records)

    # ── Stage 4: Fusion ──────────────────────────────────────────────────
    logger.info("\n[4/7] FUSION — Deduplication & conflict resolution")
    records = fuse_records(records)

    # ── Stage 5: Save long-form ──────────────────────────────────────────
    logger.info("\n[5/7] SAVING — Long-form records (pipeline-compatible)")
    long_csv = OUTPUT_DIR / "ganga_water_quality_long.csv"
    save_long_csv(records, long_csv)

    # ── Stage 6: Pivot & save wide ────────────────────────────────────────
    logger.info("\n[6/7] PIVOT & SAVE — Wide-format final CSV")
    wide_rows = pivot_to_wide(records)
    save_final_csv(wide_rows, FINAL_CSV)

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    # Print summary stats
    sources = defaultdict(int)
    params = defaultdict(int)
    for rec in records:
        sources[rec.get("source", "?")] += 1
        params[rec.get("parameter", "?")] += 1

    logger.info(f"Records by source:")
    for s, n in sorted(sources.items(), key=lambda x: -x[1]):
        logger.info(f"  {s}: {n}")

    logger.info(f"\nRecords by parameter (top 15):")
    for p, n in sorted(params.items(), key=lambda x: -x[1])[:15]:
        logger.info(f"  {p}: {n}")

    logger.info(f"\nFinal outputs:")
    logger.info(f"  Wide format: {FINAL_CSV}")
    logger.info(f"  Long format: {long_csv}")
    logger.info(f"  Wide rows: {len(wide_rows)}")
    logger.info(f"  Long records: {len(records)}")

    # Save pipeline report
    report = {
        "timestamp": datetime.now().isoformat(),
        "sources": dict(sources),
        "parameters": dict(params),
        "total_records_long": len(records),
        "total_rows_wide": len(wide_rows),
        "output_files": {
            "wide": str(FINAL_CSV),
            "long": str(long_csv),
        },
    }
    report_path = OUTPUT_DIR / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Report: {report_path}")


if __name__ == "__main__":
    run_pipeline()
