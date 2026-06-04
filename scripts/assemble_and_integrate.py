"""
Assemble all water quality data sources into water_quality_agent/multi_source_data/
and produce a clean integrated dataset with cross-source validation.

Sources:
  1. CPCB Real-Time 2025 (local CSV)
  2. NWDP-CPCB Bio/Chem/Phys (downloaded from nwdp.nwic.gov.in)
  3. NWMP HTML annual reports (parsed from CPCB ENVIS HTM files)
  4. data.gov.in API downloads

Run:
  python assemble_and_integrate.py
"""
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
NWDP = DATA / "web_extracted" / "nwdp_cpcb"
NWMP = DATA / "web_extracted" / "nwmp"
DATAGOV = DATA / "raw" / "datagov"

MS_DIR = BASE / "water_quality_agent" / "multi_source_data"
OUT = MS_DIR / "integrated"

# Canonical parameter names used across all sources
CANONICAL = {
    "DO": "Dissolved_Oxygen_mg_l",
    "BOD": "BOD_mg_l",
    "COD": "COD_mg_l",
    "pH": "pH",
    "Conductivity": "Conductivity_uS_cm",
    "Temperature": "Temperature_C",
    "Turbidity": "Turbidity_NTU",
    "Nitrate": "Nitrate_mg_l",
    "Chloride": "Chloride_mg_l",
    "Fecal_Coliform": "Fecal_Coliform_MPN_100ml",
    "Total_Coliform": "Total_Coliform_MPN_100ml",
    "TDS": "TDS_mg_l",
    "TOC": "TOC_mg_l",
}

# Physical bounds for QA
BOUNDS = {
    "Dissolved_Oxygen_mg_l": (0, 20),
    "BOD_mg_l": (0, 100),
    "COD_mg_l": (0, 500),
    "pH": (4, 11),
    "Conductivity_uS_cm": (0, 5000),
    "Temperature_C": (0, 50),
    "Turbidity_NTU": (0, 5000),
    "Nitrate_mg_l": (0, 100),
    "Chloride_mg_l": (0, 1000),
    "Fecal_Coliform_MPN_100ml": (0, 1e8),
    "Total_Coliform_MPN_100ml": (0, 1e8),
    "TDS_mg_l": (0, 5000),
    "TOC_mg_l": (0, 100),
}


# ══════════════════════════════════════════════════════════════════════════
# STAGE 1 — Assemble multi-source data
# ══════════════════════════════════════════════════════════════════════════
def assemble_sources():
    """Copy/link all source data into the multi_source_data folder."""
    log.info("STAGE 1: Assembling multi-source data")

    # ── 1a. CPCB Real-Time ────────────────────────────────────────────────
    d1 = MS_DIR / "01_cpcb_realtime"
    d1.mkdir(parents=True, exist_ok=True)
    for src, dst in [
        (DATA / "combined_station_data_with_coords.csv", d1 / "station_data.csv"),
        (DATA / "cpcb_station_locations.csv", d1 / "station_locations.csv"),
    ]:
        if src.exists():
            shutil.copy2(src, dst)
            log.info(f"  Copied {src.name} → 01_cpcb_realtime/")

    # ── 1b. NWDP-CPCB ────────────────────────────────────────────────────
    d2 = MS_DIR / "02_nwdp_cpcb"
    d2.mkdir(parents=True, exist_ok=True)
    for ptype in ["biological", "chemical", "physical"]:
        src = NWDP / f"cpcb_{ptype}_ganga_all.csv"
        if src.exists():
            shutil.copy2(src, d2 / f"{ptype}_ganga.csv")
            log.info(f"  Copied {src.name} → 02_nwdp_cpcb/")

    # ── 1c. NWMP HTML parsed ─────────────────────────────────────────────
    d3 = MS_DIR / "03_nwmp_html"
    d3.mkdir(parents=True, exist_ok=True)
    src = NWMP / "nwmp_ganga_all_years.csv"
    if src.exists():
        shutil.copy2(src, d3 / "nwmp_ganga_all_years.csv")
        log.info(f"  Copied {src.name} → 03_nwmp_html/")

    # ── 1c-bis. NWMP PDF extracted ───────────────────────────────────────
    src_pdf = NWMP / "nwmp_pdf_ganga_all_years.csv"
    if src_pdf.exists():
        shutil.copy2(src_pdf, d3 / "nwmp_pdf_ganga_all_years.csv")
        log.info(f"  Copied {src_pdf.name} → 03_nwmp_html/")

    # ── 1d. data.gov.in ──────────────────────────────────────────────────
    d4 = MS_DIR / "04_datagov"
    d4.mkdir(parents=True, exist_ok=True)
    renames = {
        "ganga_river_ganga_water_quality_20112015.csv": "ganga_wq_2011_2015.csv",
        "ganga_statewise_river_ganga_wq_median_20172022.csv": "ganga_wq_median_2017_2022.csv",
        "ganga_stationwise_river_ganga_wq_20182020.csv": "ganga_wq_stationwise_2018_2020.csv",
        "ganga_stationwise_river_ganga_wq_2021.csv": "ganga_wq_stationwise_2021.csv",
        "ganga_water_quality_of_river_ganga__2012.csv": "ganga_wq_2012.csv",
        "surface_surface_water_quality_march_2018_cpcb.csv": "surface_wq_march_2018_cpcb.csv",
    }
    for orig, new in renames.items():
        src = DATAGOV / orig
        if src.exists():
            shutil.copy2(src, d4 / new)
            log.info(f"  Copied {orig} → 04_datagov/{new}")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 2 — Standardize each source to canonical schema
# ══════════════════════════════════════════════════════════════════════════
def _to_numeric(series):
    """Coerce to numeric, stripping common non-numeric markers."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "").str.strip()
        .replace({"": np.nan, "-": np.nan, "NR": np.nan, "NA": np.nan,
                  "NM": np.nan, "N/A": np.nan, "*": np.nan, "BDL": np.nan}),
        errors="coerce",
    )


def standardize_cpcb_realtime() -> pd.DataFrame:
    """Source 1: CPCB real-time sensor data (2025)."""
    f = MS_DIR / "01_cpcb_realtime" / "station_data.csv"
    if not f.exists():
        return pd.DataFrame()

    df = pd.read_csv(f, low_memory=False)
    df["date"] = pd.to_datetime(df["timestampDate"], format="mixed", errors="coerce")

    col_map = {
        "Dissolved Oxygen": "Dissolved_Oxygen_mg_l",
        "Biochemical Oxygen Demand": "BOD_mg_l",
        "Chemical Oxygen Demand": "COD_mg_l",
        "pH": "pH",
        "Conductivity": "Conductivity_uS_cm",
        "Water Temperature": "Temperature_C",
        "Water Turbidity": "Turbidity_NTU",
        "Nitrate": "Nitrate_mg_l",
        "Chloride": "Chloride_mg_l",
        "Total Organic Carbon": "TOC_mg_l",
    }

    out = pd.DataFrame()
    out["station_id"] = df["stationId"].astype(str)
    out["station_name"] = df["stationId"].astype(str)  # will enrich later
    out["latitude"] = df["latitude"]
    out["longitude"] = df["longitude"]
    out["date"] = df["date"]
    out["source"] = "cpcb_realtime"
    out["year"] = df["date"].dt.year

    for orig, canon in col_map.items():
        if orig in df.columns:
            out[canon] = _to_numeric(df[orig])

    log.info(f"  CPCB Real-Time: {len(out)} rows")
    return out


def standardize_nwdp() -> pd.DataFrame:
    """Source 2: NWDP-CPCB (Biological + Chemical + Physical)."""
    d = MS_DIR / "02_nwdp_cpcb"
    frames = []

    # Biological
    f = d / "biological_ganga.csv"
    if f.exists():
        df = pd.read_csv(f, low_memory=False)
        # Filter for Ganga stations
        mask = df["Station"].str.contains("GANGA", case=False, na=False)
        df = df[mask].copy()
        out = pd.DataFrame()
        out["station_id"] = df["Station"].str.strip()
        out["station_name"] = df["Station"].str.strip()
        out["latitude"] = _to_numeric(df["Latitude"])
        out["longitude"] = _to_numeric(df["Longitude"])
        out["date"] = pd.to_datetime(df["Data Acquisition Time"], format="mixed",
                                      errors="coerce", dayfirst=True)
        out["source"] = "nwdp_cpcb"
        out["year"] = out["date"].dt.year
        out["BOD_mg_l"] = _to_numeric(df.get("Biochemical Oxygen Demand (mg/L)", pd.Series()))
        out["COD_mg_l"] = _to_numeric(df.get("Chemical Oxygen Demand (mg/L)", pd.Series()))
        out["Fecal_Coliform_MPN_100ml"] = _to_numeric(
            df.get("Fecal Coliform (MPN/100mL)", pd.Series()))
        out["Total_Coliform_MPN_100ml"] = _to_numeric(
            df.get("Total Coliform (MPN/100mL)", pd.Series()))
        frames.append(out)

    # Chemical
    f = d / "chemical_ganga.csv"
    if f.exists():
        df = pd.read_csv(f, low_memory=False)
        mask = df["Station"].str.contains("GANGA", case=False, na=False)
        df = df[mask].copy()
        out = pd.DataFrame()
        out["station_id"] = df["Station"].str.strip()
        out["station_name"] = df["Station"].str.strip()
        out["latitude"] = _to_numeric(df["Latitude"])
        out["longitude"] = _to_numeric(df["Longitude"])
        out["date"] = pd.to_datetime(df["Data Acquisition Time"], format="mixed",
                                      errors="coerce", dayfirst=True)
        out["source"] = "nwdp_cpcb"
        out["year"] = out["date"].dt.year
        chem_map = {
            "Dissolved oxygen (mg/L)": "Dissolved_Oxygen_mg_l",
            "Potential of Hydrogen (pH)": "pH",
            "Chloride (mg/L)": "Chloride_mg_l",
            "Nitrate (mg/L)": "Nitrate_mg_l",
            "Nitrate N (mgN/L)": "Nitrate_mg_l",
            "Total Dissolved Solids (mg/L)": "TDS_mg_l",
            "Amonia N (mgN/L)": "Ammonia_mg_l",
        }
        for orig, canon in chem_map.items():
            if orig in df.columns:
                out[canon] = _to_numeric(df[orig])
        frames.append(out)
        log.info(f"    Chemical: {len(out)} Ganga rows")

    # Physical
    f = d / "physical_ganga.csv"
    if f.exists():
        df = pd.read_csv(f, low_memory=False)
        mask = df["Station"].str.contains("GANGA", case=False, na=False)
        df = df[mask].copy()
        out = pd.DataFrame()
        out["station_id"] = df["Station"].str.strip()
        out["station_name"] = df["Station"].str.strip()
        out["latitude"] = _to_numeric(df["Latitude"])
        out["longitude"] = _to_numeric(df["Longitude"])
        out["date"] = pd.to_datetime(df["Data Acquisition Time"], format="mixed",
                                      errors="coerce", dayfirst=True)
        out["source"] = "nwdp_cpcb"
        out["year"] = out["date"].dt.year
        phys_map = {
            "Electric Conductivity (μS/cm)": "Conductivity_uS_cm",
            "Turbidity (NTU)": "Turbidity_NTU",
            "Temperature (ºC)": "Temperature_C",
            "Total Solids (mg/L)": "TDS_mg_l",
        }
        for orig, canon in phys_map.items():
            if orig in df.columns:
                out[canon] = _to_numeric(df[orig])
        frames.append(out)

    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    log.info(f"  NWDP-CPCB: {len(merged)} rows")
    return merged


def standardize_nwmp() -> pd.DataFrame:
    """Source 3: NWMP HTML annual reports (2012–2014)."""
    f = MS_DIR / "03_nwmp_html" / "nwmp_ganga_all_years.csv"
    if not f.exists():
        return pd.DataFrame()

    df = pd.read_csv(f)
    out = pd.DataFrame()
    out["station_id"] = df["Station_Code"].astype(str)
    out["station_name"] = df["Location"]
    out["latitude"] = np.nan  # NWMP data doesn't have coords
    out["longitude"] = np.nan
    # Use July 1 of each year as a representative date
    out["date"] = pd.to_datetime(df["Year"].astype(str) + "-07-01")
    out["source"] = "nwmp_html"
    out["year"] = df["Year"]

    # Map Mean values to canonical columns
    nwmp_map = {
        "Temperature_C_Mean": "Temperature_C",
        "DO_mg_l_Mean": "Dissolved_Oxygen_mg_l",
        "pH_Mean": "pH",
        "Conductivity_umhos_cm_Mean": "Conductivity_uS_cm",
        "BOD_mg_l_Mean": "BOD_mg_l",
        "Nitrate_N_mg_l_Mean": "Nitrate_mg_l",
        "Fecal_Coliform_MPN_100ml_Mean": "Fecal_Coliform_MPN_100ml",
        "Total_Coliform_MPN_100ml_Mean": "Total_Coliform_MPN_100ml",
    }
    for orig, canon in nwmp_map.items():
        if orig in df.columns:
            out[canon] = _to_numeric(df[orig])

    log.info(f"  NWMP HTML: {len(out)} rows")
    return out


def standardize_nwmp_pdf() -> pd.DataFrame:
    """Source 5: NWMP PDF annual reports (2015–2024)."""
    f = MS_DIR / "03_nwmp_html" / "nwmp_pdf_ganga_all_years.csv"
    if not f.exists():
        return pd.DataFrame()

    df = pd.read_csv(f, low_memory=False)
    out = pd.DataFrame()
    out["station_id"] = df["station_code"].astype(str) if "station_code" in df.columns else "unknown"
    out["station_name"] = df.get("station_name", pd.Series("unknown")).astype(str)
    out["latitude"] = np.nan
    out["longitude"] = np.nan
    out["date"] = pd.to_datetime(df["year"].astype(str) + "-07-01", errors="coerce")
    out["source"] = "nwmp_pdf"
    out["year"] = df["year"]

    pdf_map = {
        "Temperature_C": "Temperature_C",
        "Dissolved_Oxygen_mg_l": "Dissolved_Oxygen_mg_l",
        "pH": "pH",
        "BOD_mg_l": "BOD_mg_l",
        "Nitrate_mg_l": "Nitrate_mg_l",
        "TDS_mg_l": "TDS_mg_l",
        "Conductivity_uS_cm": "Conductivity_uS_cm",
        "Fecal_Coliform_MPN_100ml": "Fecal_Coliform_MPN_100ml",
        "Total_Coliform_MPN_100ml": "Total_Coliform_MPN_100ml",
        "Fluoride_mg_l": "Fluoride_mg_l",
    }
    for orig, canon in pdf_map.items():
        if orig in df.columns:
            out[canon] = _to_numeric(df[orig])

    log.info(f"  NWMP PDF: {len(out)} rows")
    return out


def standardize_datagov() -> pd.DataFrame:
    """Source 4: data.gov.in API downloads."""
    d = MS_DIR / "04_datagov"
    frames = []

    # ── ganga_wq_2012.csv (annual Min/Max/Mean per station) ───────────────
    f = d / "ganga_wq_2012.csv"
    if f.exists():
        df = pd.read_csv(f, low_memory=False)
        if len(df) > 0:
            out = pd.DataFrame()
            out["station_id"] = df.get("station_code", pd.Series("unknown")).astype(str)
            out["station_name"] = df.get("locations", pd.Series("unknown")).astype(str)
            out["latitude"] = np.nan
            out["longitude"] = np.nan
            out["date"] = pd.Timestamp("2012-07-01")
            out["source"] = "datagov"
            out["year"] = 2012
            # Mean columns
            for col in df.columns:
                cl = col.lower()
                if not cl.endswith("_mean") and "mean" not in cl:
                    continue
                if "temperature" in cl:
                    out["Temperature_C"] = _to_numeric(df[col])
                elif "d_o" in cl or "do" in cl:
                    out["Dissolved_Oxygen_mg_l"] = _to_numeric(df[col])
                elif "ph" in cl:
                    out["pH"] = _to_numeric(df[col])
                elif "conductiv" in cl:
                    out["Conductivity_uS_cm"] = _to_numeric(df[col])
                elif "b_o_d" in cl or "bod" in cl:
                    out["BOD_mg_l"] = _to_numeric(df[col])
                elif "nitrate" in cl:
                    out["Nitrate_mg_l"] = _to_numeric(df[col])
                elif "fecal" in cl or "faecal" in cl:
                    out["Fecal_Coliform_MPN_100ml"] = _to_numeric(df[col])
                elif "total_coliform" in cl:
                    out["Total_Coliform_MPN_100ml"] = _to_numeric(df[col])
            frames.append(out)
            log.info(f"    {f.name}: {len(out)} rows")

    # ── ganga_wq_median_2017_2022.csv ─────────────────────────────────────
    f = d / "ganga_wq_median_2017_2022.csv"
    if f.exists():
        df = pd.read_csv(f, low_memory=False)
        if len(df) > 0:
            out = pd.DataFrame()
            out["station_id"] = df.get("station_code", pd.Series("unknown")).astype(str)
            out["station_name"] = df.get("station_name", pd.Series("unknown")).astype(str)
            out["latitude"] = np.nan
            out["longitude"] = np.nan
            yr = pd.to_numeric(df.get("_year", pd.Series()), errors="coerce")
            out["date"] = pd.to_datetime(yr.astype("Int64").astype(str) + "-07-01",
                                          errors="coerce")
            out["source"] = "datagov"
            out["year"] = yr
            out["Dissolved_Oxygen_mg_l"] = _to_numeric(df.get("do__mg_l_", pd.Series()))
            out["BOD_mg_l"] = _to_numeric(df.get("bod__mg_l_", pd.Series()))
            out["pH"] = _to_numeric(df.get("ph", pd.Series()))
            out["Fecal_Coliform_MPN_100ml"] = _to_numeric(df.get("fc__mpn_100ml_", pd.Series()))
            frames.append(out)
            log.info(f"    {f.name}: {len(out)} rows")

    # ── ganga_wq_stationwise_2018_2020.csv ────────────────────────────────
    f = d / "ganga_wq_stationwise_2018_2020.csv"
    if f.exists():
        df = pd.read_csv(f, low_memory=False)
        if len(df) > 0:
            out = pd.DataFrame()
            out["station_id"] = df.get("station_name", pd.Series("unknown")).astype(str)
            out["station_name"] = df.get("station_name", pd.Series("unknown")).astype(str)
            out["latitude"] = np.nan
            out["longitude"] = np.nan
            out["date"] = pd.Timestamp("2019-07-01")  # midpoint
            out["source"] = "datagov"
            out["year"] = 2019
            # Take 2020 DO column if exists
            do_cols = [c for c in df.columns if "dissolved_oxygen" in c.lower() and "2020" in c]
            if do_cols:
                out["Dissolved_Oxygen_mg_l"] = _to_numeric(df[do_cols[0]])
            fc_cols = [c for c in df.columns if "faecal" in c.lower() and "2020" in c]
            if fc_cols:
                out["Fecal_Coliform_MPN_100ml"] = _to_numeric(df[fc_cols[0]])
            frames.append(out)
            log.info(f"    {f.name}: {len(out)} rows")

    # ── ganga_wq_stationwise_2021.csv ─────────────────────────────────────
    f = d / "ganga_wq_stationwise_2021.csv"
    if f.exists():
        df = pd.read_csv(f, low_memory=False)
        if len(df) > 0:
            out = pd.DataFrame()
            out["station_id"] = df.get("station_code", df.get("station_name", pd.Series("unknown"))).astype(str)
            out["station_name"] = df.get("station_name", pd.Series("unknown")).astype(str)
            out["latitude"] = np.nan
            out["longitude"] = np.nan
            out["date"] = pd.Timestamp("2021-07-01")
            out["source"] = "datagov"
            out["year"] = 2021
            do_cols = [c for c in df.columns if "dissolved_oxygen" in c.lower()]
            if do_cols:
                out["Dissolved_Oxygen_mg_l"] = _to_numeric(df[do_cols[0]])
            fc_cols = [c for c in df.columns if "faecal" in c.lower()]
            if fc_cols:
                out["Fecal_Coliform_MPN_100ml"] = _to_numeric(df[fc_cols[0]])
            frames.append(out)
            log.info(f"    {f.name}: {len(out)} rows")

    # ── surface_wq_march_2018_cpcb.csv (the big one) ─────────────────────
    f = d / "surface_wq_march_2018_cpcb.csv"
    if f.exists():
        df = pd.read_csv(f, low_memory=False)
        # Filter for Ganga basin
        if "Basin" in df.columns:
            mask = df["Basin"].str.contains("Ganga", case=False, na=False)
            df = df[mask].copy()
        if len(df) > 0:
            out = pd.DataFrame()
            out["station_id"] = df.get("Station_Name", pd.Series("unknown")).astype(str)
            out["station_name"] = df.get("Station_Name", pd.Series("unknown")).astype(str)
            out["latitude"] = _to_numeric(df.get("Latitude", pd.Series()))
            out["longitude"] = _to_numeric(df.get("Longitude", pd.Series()))
            out["date"] = pd.to_datetime(df.get("Date", pd.Series()), format="mixed",
                                          errors="coerce")
            out["source"] = "datagov"
            out["year"] = _to_numeric(df.get("Year", pd.Series()))

            col_map = {
                "_do": "Dissolved_Oxygen_mg_l",
                "bod": "BOD_mg_l",
                "cod": "COD_mg_l",
                "ph_gen": "pH",
                "ec_gen": "Conductivity_uS_cm",
                "temp": "Temperature_C",
                "turb": "Turbidity_NTU",
                "no3_n": "Nitrate_mg_l",
                "cl": "Chloride_mg_l",
                "fcol_mpn": "Fecal_Coliform_MPN_100ml",
                "tcol_mpn": "Total_Coliform_MPN_100ml",
                "tds": "TDS_mg_l",
                "toc": "TOC_mg_l",
            }
            for orig, canon in col_map.items():
                if orig in df.columns:
                    out[canon] = _to_numeric(df[orig])
            frames.append(out)
            log.info(f"    {f.name}: {len(out)} Ganga rows")

    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    log.info(f"  data.gov.in: {len(merged)} total rows")
    return merged


# ══════════════════════════════════════════════════════════════════════════
# STAGE 3 — QA/QC: bounds check, deduplication
# ══════════════════════════════════════════════════════════════════════════
def qa_qc(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply QA/QC: bounds, BOD≤COD, dedup."""
    log.info("STAGE 3: QA/QC")
    stats = {"rows_in": len(df), "bounds_clipped": 0, "bod_cod_flagged": 0}

    # Bounds check
    for param, (lo, hi) in BOUNDS.items():
        if param in df.columns:
            mask = df[param].notna() & ((df[param] < lo) | (df[param] > hi))
            n_bad = mask.sum()
            if n_bad > 0:
                df.loc[mask, param] = np.nan
                stats["bounds_clipped"] += n_bad
                log.info(f"  Bounds: {param} clipped {n_bad} values")

    # BOD ≤ 2×COD rule (relaxed)
    if "BOD_mg_l" in df.columns and "COD_mg_l" in df.columns:
        mask = (df["BOD_mg_l"].notna() & df["COD_mg_l"].notna() &
                (df["BOD_mg_l"] > 2 * df["COD_mg_l"]))
        stats["bod_cod_flagged"] = int(mask.sum())
        if mask.sum() > 0:
            df.loc[mask, "BOD_mg_l"] = np.nan
            log.info(f"  BOD>2×COD: flagged {mask.sum()} rows")

    # Dedup: same station+date+source
    before = len(df)
    df = df.drop_duplicates(subset=["station_id", "date", "source"], keep="first")
    stats["deduped"] = before - len(df)
    log.info(f"  Dedup: removed {stats['deduped']} duplicates")

    stats["rows_out"] = len(df)
    return df, stats


# ══════════════════════════════════════════════════════════════════════════
# STAGE 4 — Cross-source validation
# ══════════════════════════════════════════════════════════════════════════
def validate(df: pd.DataFrame) -> dict:
    """Cross-source validation statistics."""
    log.info("STAGE 4: Cross-source validation")

    param_cols = [c for c in CANONICAL.values() if c in df.columns]
    df = df[df["source"].notna()].copy()
    sources = sorted(df["source"].dropna().unique())

    report = {
        "total_rows": len(df),
        "total_stations": int(df["station_id"].nunique()),
        "sources": {},
        "cross_source_comparison": {},
        "parameter_coverage": {},
    }

    # Per-source stats
    for src in sources:
        sdf = df[df["source"] == src]
        report["sources"][src] = {
            "rows": len(sdf),
            "stations": int(sdf["station_id"].nunique()),
            "date_range": [str(sdf["date"].min()), str(sdf["date"].max())],
            "year_range": [int(sdf["year"].min()) if sdf["year"].notna().any() else None,
                           int(sdf["year"].max()) if sdf["year"].notna().any() else None],
        }

    # Parameter coverage per source
    for param in param_cols:
        coverage = {}
        for src in sources:
            sdf = df[df["source"] == src]
            if param in sdf.columns:
                n = sdf[param].notna().sum()
                mean_val = sdf[param].mean() if n > 0 else None
                coverage[src] = {
                    "count": int(n),
                    "mean": round(float(mean_val), 3) if mean_val is not None else None,
                }
        report["parameter_coverage"][param] = coverage

    # Cross-source comparison: compare parameter means
    log.info("\n  Cross-source parameter means:")
    header = f"  {'Parameter':<30}"
    for src in sources:
        header += f" {src[:15]:>15}"
    log.info(header)
    log.info("  " + "-" * (30 + 16 * len(sources)))

    for param in param_cols:
        line = f"  {param:<30}"
        means = {}
        for src in sources:
            sdf = df[df["source"] == src]
            if param in sdf.columns:
                vals = sdf[param].dropna()
                if len(vals) > 0:
                    m = vals.mean()
                    means[src] = m
                    line += f" {m:>15.2f}"
                else:
                    line += f" {'—':>15}"
            else:
                line += f" {'—':>15}"
        log.info(line)

        # Compute pairwise differences
        if len(means) >= 2:
            src_list = list(means.keys())
            for i in range(len(src_list)):
                for j in range(i + 1, len(src_list)):
                    s1, s2 = src_list[i], src_list[j]
                    diff_pct = abs(means[s1] - means[s2]) / max(abs(means[s1]), 1e-6) * 100
                    key = f"{param}|{s1}_vs_{s2}"
                    report["cross_source_comparison"][key] = {
                        "param": param,
                        "source_1": s1,
                        "source_2": s2,
                        "mean_1": round(means[s1], 3),
                        "mean_2": round(means[s2], 3),
                        "diff_pct": round(diff_pct, 1),
                    }

    return report


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  Multi-Source Ganga Water Quality Integration Pipeline")
    print("=" * 70)

    # Stage 1: Assemble
    assemble_sources()

    # Stage 2: Standardize
    log.info("\nSTAGE 2: Standardizing all sources to canonical schema")
    dfs = []

    df1 = standardize_cpcb_realtime()
    if len(df1) > 0:
        dfs.append(df1)

    df2 = standardize_nwdp()
    if len(df2) > 0:
        dfs.append(df2)

    df3 = standardize_nwmp()
    if len(df3) > 0:
        dfs.append(df3)

    df4 = standardize_datagov()
    if len(df4) > 0:
        dfs.append(df4)

    df5 = standardize_nwmp_pdf()
    if len(df5) > 0:
        dfs.append(df5)

    if not dfs:
        log.error("No data sources loaded!")
        return

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"\n  Combined: {len(combined)} rows, {combined['source'].nunique()} sources")
    log.info(f"  Sources: {dict(combined['source'].value_counts())}")

    # Stage 3: QA/QC
    combined, qa_stats = qa_qc(combined)

    # Stage 4: Validate
    val_report = validate(combined)
    val_report["qa_qc"] = qa_stats

    # Save outputs
    OUT.mkdir(parents=True, exist_ok=True)

    # Integrated CSV — write to temp first, then replace (avoids lock issues)
    outfile = OUT / "ganga_integrated.csv"
    tmpfile = OUT / "ganga_integrated_tmp.csv"
    combined.to_csv(tmpfile, index=False)
    try:
        if outfile.exists():
            outfile.unlink()
        tmpfile.rename(outfile)
    except PermissionError:
        log.warning(f"  Could not replace {outfile} (file locked). Saved as {tmpfile}")
        outfile = tmpfile
    log.info(f"\n  Saved integrated dataset: {outfile}")
    log.info(f"  Shape: {combined.shape}")

    # Validation report JSON
    report_file = OUT / "validation_report.json"
    with open(report_file, "w") as f:
        json.dump(val_report, f, indent=2, default=str)
    log.info(f"  Saved validation report: {report_file}")

    # Summary table
    summary = []
    for src in sorted(combined["source"].dropna().unique()):
        sdf = combined[combined["source"] == src]
        param_cols = [c for c in CANONICAL.values() if c in sdf.columns]
        n_params = sum(1 for c in param_cols if sdf[c].notna().any())
        summary.append({
            "Source": src,
            "Rows": len(sdf),
            "Stations": sdf["station_id"].nunique(),
            "Year_Min": int(sdf["year"].min()) if sdf["year"].notna().any() else None,
            "Year_Max": int(sdf["year"].max()) if sdf["year"].notna().any() else None,
            "Parameters_With_Data": n_params,
        })
    summary_df = pd.DataFrame(summary)
    summary_file = OUT / "source_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    log.info(f"  Saved source summary: {summary_file}")

    # Print final summary
    print("\n" + "=" * 70)
    print("  INTEGRATION SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    print(f"\n  Total rows: {len(combined):,}")
    print(f"  Total stations: {combined['station_id'].nunique()}")
    print(f"  Year range: {int(combined['year'].min())} – {int(combined['year'].max())}")
    print(f"  QA/QC: {qa_stats['bounds_clipped']} bounds violations, "
          f"{qa_stats['bod_cod_flagged']} BOD>2×COD, "
          f"{qa_stats['deduped']} duplicates removed")

    # Print key validation comparison
    print("\n  Key Cross-Source Parameter Means:")
    for param in ["Dissolved_Oxygen_mg_l", "BOD_mg_l", "pH",
                   "Conductivity_uS_cm", "Temperature_C"]:
        if param in val_report["parameter_coverage"]:
            line = f"    {param:<30}"
            for src, stats in val_report["parameter_coverage"][param].items():
                if stats["mean"] is not None:
                    line += f"  {src}={stats['mean']:.2f}"
            print(line)

    print("\n  Output files:")
    print(f"    {outfile}")
    print(f"    {report_file}")
    print(f"    {summary_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
