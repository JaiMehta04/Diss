"""
Extract Ganga water quality data from ALL NWMP PDF reports (2015–2024).

Scans two directories:
  - NWMP data/               (2016-2023 PDFs with varied naming)
  - data/raw/cpcb/nwmp_pdfs/ (2015, 2018-2022, 2024)

Uses the NWMPExtractor from water_quality_agent for robust table parsing,
plus a direct pdfplumber fallback for PDFs with different naming conventions.

Output:  data/web_extracted/nwmp/nwmp_pdf_ganga_all_years.csv
"""

import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from water_quality_agent.downloaders.nwmp_extractor import NWMPExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Collect all NWMP PDF paths ────────────────────────────────────────────────
PDF_DIRS = [
    ROOT / "NWMP data",
    ROOT / "data" / "raw" / "cpcb" / "nwmp_pdfs",
]

GANGA_KEYWORDS = [
    "ganga", "ganges", "yamuna", "gomti", "ghaghra", "gandak", "kosi",
    "son", "damodar", "hooghly", "ramganga", "betwa", "chambal",
    "tons", "hindon", "kali", "pandu", "varuna",
]


def find_all_pdfs() -> dict:
    """Return {year: pdf_path} for all NWMP WQ PDFs."""
    pdfs = {}
    for d in PDF_DIRS:
        if not d.exists():
            continue
        for f in d.glob("*.pdf"):
            m = re.search(r"(\d{4})", f.stem)
            if m:
                year = int(m.group(1))
                if 2015 <= year <= 2024:
                    if year not in pdfs:
                        pdfs[year] = f
    return dict(sorted(pdfs.items()))


def extract_with_extractor(pdf_path: Path, year: int) -> list:
    """Use the NWMPExtractor on a single PDF."""
    import pdfplumber

    extractor = NWMPExtractor()

    # The extractor's _extract_single_pdf does the real work
    records = extractor._extract_single_pdf(pdf_path, year, "ganga", False)
    return records


def extract_ganga_pages_direct(pdf_path: Path, year: int) -> list:
    """
    Direct pdfplumber fallback: extract tables from Ganga-relevant pages
    and attempt to parse station/parameter data.
    """
    import pdfplumber

    records = []
    with pdfplumber.open(pdf_path) as pdf:
        log.info(f"  Direct extraction: {pdf_path.name}, {len(pdf.pages)} pages")

        for pi, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").lower()
            # Only process pages mentioning Ganga system rivers
            if not any(kw in text for kw in GANGA_KEYWORDS):
                continue

            tables = page.extract_tables({
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 4,
            })
            if not tables:
                tables = page.extract_tables({
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 6,
                })
            if not tables:
                continue

            for table in tables:
                if not table or len(table) < 3:
                    continue

                # Find the header row by looking for WQ parameter keywords
                header_idx = None
                for ri, row in enumerate(table[:5]):
                    row_text = " ".join(str(c or "").lower() for c in row)
                    wq_hits = sum(1 for kw in ["temp", "ph", "do", "bod", "conduct",
                                                "nitrat", "coliform", "oxygen"]
                                  if kw in row_text)
                    if wq_hits >= 2:
                        header_idx = ri
                        break

                if header_idx is None:
                    continue

                header = [str(c or "").strip() for c in table[header_idx]]

                # Build column mapping
                col_map = {}
                for ci, h in enumerate(header):
                    hl = h.lower()
                    if "code" in hl:
                        col_map[ci] = "station_code"
                    elif "name" in hl or "location" in hl:
                        col_map[ci] = "station_name"
                    elif "state" in hl:
                        col_map[ci] = "state"
                    elif "temp" in hl or "ºc" in hl or "°c" in hl:
                        col_map[ci] = "temperature"
                    elif "ph" in hl and len(hl) < 8:
                        col_map[ci] = "ph"
                    elif "conduct" in hl or "µmhos" in hl:
                        col_map[ci] = "conductivity"
                    elif "d.o" in hl or "dissolv" in hl or "oxygen" in hl:
                        col_map[ci] = "do"
                    elif "b.o.d" in hl or "bod" in hl:
                        col_map[ci] = "bod"
                    elif "nitrat" in hl:
                        col_map[ci] = "nitrate"
                    elif "fecal" in hl or "faecal" in hl:
                        col_map[ci] = "fecal_coliform"
                    elif "total" in hl and "coli" in hl:
                        col_map[ci] = "total_coliform"

                # Parse data rows
                for row in table[header_idx + 1:]:
                    if not row or len(row) < 3:
                        continue
                    rec = {"year": year, "source": "nwmp_pdf", "source_page": pi}
                    has_data = False

                    for ci, val in enumerate(row):
                        if ci not in col_map or not val:
                            continue
                        val = str(val).strip()
                        field = col_map[ci]
                        if field in ("station_code", "station_name", "state"):
                            rec[field] = val
                        else:
                            try:
                                val_clean = re.sub(r"[<>≤≥~,]", "", val).strip()
                                if val_clean and val_clean not in ("", "-", "BDL", "NR"):
                                    rec[field] = float(val_clean)
                                    has_data = True
                            except ValueError:
                                pass

                    if has_data and (rec.get("station_name") or rec.get("station_code")):
                        # Check if it's Ganga-related
                        search_text = " ".join(str(rec.get(k, "")).lower()
                                               for k in ("station_name", "state"))
                        if any(kw in search_text for kw in GANGA_KEYWORDS):
                            records.append(rec)

    return records


def records_to_dataframe(records: list) -> pd.DataFrame:
    """Convert list of record dicts to standardized DataFrame."""
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Canonical parameter mapping
    rename = {
        "temperature": "Temperature_C",
        "temperature_min": "Temperature_C_min",
        "temperature_max": "Temperature_C_max",
        "temperature_mean": "Temperature_C",
        "do": "Dissolved_Oxygen_mg_l",
        "do_min": "Dissolved_Oxygen_mg_l_min",
        "do_max": "Dissolved_Oxygen_mg_l_max",
        "do_mean": "Dissolved_Oxygen_mg_l",
        "ph": "pH",
        "ph_min": "pH_min",
        "ph_max": "pH_max",
        "ph_mean": "pH",
        "conductivity": "Conductivity_uS_cm",
        "conductivity_min": "Conductivity_uS_cm_min",
        "conductivity_max": "Conductivity_uS_cm_max",
        "conductivity_mean": "Conductivity_uS_cm",
        "bod": "BOD_mg_l",
        "bod_min": "BOD_mg_l_min",
        "bod_max": "BOD_mg_l_max",
        "bod_mean": "BOD_mg_l",
        "cod": "COD_mg_l",
        "cod_mean": "COD_mg_l",
        "nitrate": "Nitrate_mg_l",
        "nitrate_min": "Nitrate_mg_l_min",
        "nitrate_max": "Nitrate_mg_l_max",
        "nitrate_mean": "Nitrate_mg_l",
        "fecal_coliform": "Fecal_Coliform_MPN_100ml",
        "fecal_coliform_min": "Fecal_Coliform_MPN_100ml_min",
        "fecal_coliform_max": "Fecal_Coliform_MPN_100ml_max",
        "fecal_coliform_mean": "Fecal_Coliform_MPN_100ml",
        "total_coliform": "Total_Coliform_MPN_100ml",
        "total_coliform_min": "Total_Coliform_MPN_100ml_min",
        "total_coliform_max": "Total_Coliform_MPN_100ml_max",
        "total_coliform_mean": "Total_Coliform_MPN_100ml",
        "tds": "TDS_mg_l",
        "tds_mean": "TDS_mg_l",
        "fluoride": "Fluoride_mg_l",
        "fluoride_mean": "Fluoride_mg_l",
        "chloride": "Chloride_mg_l",
        "chloride_mean": "Chloride_mg_l",
        "hardness": "Hardness_mg_l",
        "hardness_mean": "Hardness_mg_l",
        "turbidity": "Turbidity_NTU",
        "turbidity_mean": "Turbidity_NTU",
    }

    existing_renames = {k: v for k, v in rename.items() if k in df.columns}
    df = df.rename(columns=existing_renames)

    # For parameters with min/max but no mean, compute mean
    param_bases = set()
    for col in df.columns:
        if col.endswith("_min"):
            param_bases.add(col[:-4])

    for base in param_bases:
        min_col = f"{base}_min"
        max_col = f"{base}_max"
        if min_col in df.columns and max_col in df.columns and base not in df.columns:
            df[base] = df[[min_col, max_col]].mean(axis=1)

    return df


def main():
    print("=" * 70)
    print("  NWMP PDF Extraction — Ganga Water Quality (2015–2024)")
    print("=" * 70)

    pdfs = find_all_pdfs()
    if not pdfs:
        log.error("No NWMP PDFs found!")
        return

    log.info(f"Found {len(pdfs)} NWMP PDFs: {list(pdfs.keys())}")

    all_records = []
    summary = []

    for year, pdf_path in pdfs.items():
        log.info(f"\n{'─'*50}")
        log.info(f"Processing {year}: {pdf_path.name}")

        # Try the robust NWMPExtractor first
        records = extract_with_extractor(pdf_path, year)

        # If extractor got very few, try direct fallback
        if len(records) < 5:
            log.info(f"  Extractor found only {len(records)}, trying direct method...")
            direct_records = extract_ganga_pages_direct(pdf_path, year)
            if len(direct_records) > len(records):
                log.info(f"  Direct method found {len(direct_records)} records (better)")
                records = direct_records
            else:
                log.info(f"  Direct method found {len(direct_records)} (keeping extractor result)")

        log.info(f"  → {year}: {len(records)} Ganga records extracted")
        all_records.extend(records)
        summary.append({"year": year, "pdf": pdf_path.name, "records": len(records)})

    if not all_records:
        log.warning("No Ganga records extracted from any PDF!")
        return

    # Convert to DataFrame
    df = records_to_dataframe(all_records)

    # Clean up: drop internal columns
    drop_cols = [c for c in df.columns if c.startswith("source_page") or c == "_has_data"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Save
    out_dir = ROOT / "data" / "web_extracted" / "nwmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nwmp_pdf_ganga_all_years.csv"
    df.to_csv(out_path, index=False)

    # Also save to multi_source_data
    ms_dir = ROOT / "water_quality_agent" / "multi_source_data" / "03_nwmp_html"
    ms_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(ms_dir / "nwmp_pdf_ganga_all_years.csv", index=False)

    print("\n" + "=" * 70)
    print("  EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"\n  Total records: {len(df)}")
    print(f"  Years covered: {sorted(df['year'].unique())}")
    print(f"  Unique stations: {df['station_name'].nunique() if 'station_name' in df.columns else '?'}")

    # Parameter coverage
    param_cols = [c for c in df.columns if c not in
                  ("station_code", "station_name", "state", "year", "source",
                   "source_page", "river") and not c.endswith("_min") and not c.endswith("_max")]
    print(f"\n  Parameters extracted:")
    for p in sorted(param_cols):
        n = df[p].notna().sum()
        if n > 0:
            print(f"    {p}: {n} values ({n/len(df)*100:.1f}%)")

    print(f"\n  Per-year breakdown:")
    for row in summary:
        print(f"    {row['year']}: {row['records']:>4} records  ({row['pdf']})")

    print(f"\n  Output: {out_path}")
    print("  Done!\n")


if __name__ == "__main__":
    main()
