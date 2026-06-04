"""
Robust NWMP PDF extractor for water quality data.

The NWMP (National Water Monitoring Programme) annual reports published by
CPCB contain water quality tables in landscape-oriented PDF pages. This
module:
  1. Identifies Ganga-relevant pages via text search
  2. Extracts tables using pdfplumber with layout-aware settings
  3. Detects the header schema per-PDF (varies by year)
  4. Reassembles fragmented headers and merged cells
  5. Outputs clean, row-per-observation records

Handles known layout variations:
  - 2015: Multi-line headers with CODE/LOCATIONS/STATE prefix columns
  - 2018-2019: "Station Code | Station Name | State" + split sub-headers (Min/Max)
  - 2020-2022: Well-structured tables, sometimes split into many sub-tables per page
  - 2024: Cleaner layout with inline units

Usage:
    from water_quality_agent.downloaders.nwmp_extractor import NWMPExtractor
    extractor = NWMPExtractor()
    records = extractor.extract_all("data/raw/cpcb/nwmp_pdfs", river="ganga")
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Header patterns per NWMP year format ──────────────────────────────────────
# These are the canonical column names we look for in table headers.
# The order varies by year, so we detect them dynamically.

_PARAM_PATTERNS: Dict[str, List[str]] = {
    "station_code": [r"stat?i(?:on)?\s*code", r"^s\s*\.?\s*no", r"code"],
    "station_name": [r"(?:station|monitoring)\s*(?:name|location)", r"location",
                     r"name\s*of\s*monitoring"],
    "state": [r"state(?:\s*name)?"],
    "temperature": [r"temp(?:erature)?", r"⁰?\s*c"],
    "ph": [r"^ph$"],
    "conductivity": [r"conduct(?:ivity)?", r"µ?m?hos"],
    "do": [r"dissolv(?:ed)?\s*ox(?:ygen)?", r"^d\.?o\.?$", r"^ox$"],
    "bod": [r"b\.?o\.?d", r"biochem"],
    "cod": [r"c\.?o\.?d", r"chem.*oxygen.*demand"],
    "nitrate": [r"nitrat", r"no3", r"nitrate[\-\s]*n"],
    "fecal_coliform": [r"f(?:a?e)cal\s*coli", r"fc\b"],
    "total_coliform": [r"total\s*coli", r"tc\b"],
    "tds": [r"t\.?d\.?s", r"total\s*dissolv"],
    "fluoride": [r"fluorid"],
    "arsenic": [r"arsenic", r"^as\b"],
    "turbidity": [r"turbid"],
    "chloride": [r"chlorid"],
    "hardness": [r"hardness"],
    "alkalinity": [r"alkalin"],
    "sulphate": [r"sulph"],
    "iron": [r"^iron$", r"^fe\b"],
}

# Rivers/tributaries that belong to the Ganga system
_GANGA_KEYWORDS = [
    "ganga", "ganges", "yamuna", "gomti", "ghaghra", "gandak", "kosi",
    "son", "damodar", "hooghly", "ramganga", "betwa", "chambal",
    "tons", "hindon", "kali", "pandu", "varuna",
]


class NWMPExtractor:
    """Robust extractor for NWMP annual water quality PDF reports."""

    def __init__(self):
        try:
            import pdfplumber
            self._pdfplumber = pdfplumber
        except ImportError:
            raise ImportError("pdfplumber required: pip install pdfplumber")

    def extract_all(
        self,
        pdf_dir: str,
        river: str = "ganga",
        include_all_rivers: bool = False,
    ) -> List[Dict[str, Any]]:
        """Extract water quality records from all NWMP PDFs in a directory.

        Args:
            pdf_dir: Directory containing NWMP_DATA_YYYY.pdf files
            river: Filter to this river system (default: ganga)
            include_all_rivers: If True, extract all rivers, not just Ganga

        Returns:
            List of dicts, each representing one station-observation.
        """
        pdf_dir = Path(pdf_dir)
        all_records = []

        pdf_files = sorted(pdf_dir.glob("NWMP_DATA_*.pdf"))
        if not pdf_files:
            logger.warning(f"No NWMP PDFs found in {pdf_dir}")
            return []

        logger.info(f"Found {len(pdf_files)} NWMP PDFs to process")

        for pdf_path in pdf_files:
            year = self._extract_year(pdf_path.stem)
            if not year:
                continue

            records = self._extract_single_pdf(
                pdf_path, year, river, include_all_rivers
            )
            all_records.extend(records)
            logger.info(f"  {pdf_path.name}: {len(records)} Ganga records")

        logger.info(f"Total NWMP records extracted: {len(all_records)}")
        return all_records

    def _extract_year(self, stem: str) -> Optional[int]:
        """Extract year from filename like 'NWMP_DATA_2015'."""
        m = re.search(r"(\d{4})", stem)
        return int(m.group(1)) if m else None

    def _extract_single_pdf(
        self,
        pdf_path: Path,
        year: int,
        river: str,
        include_all: bool,
    ) -> List[Dict[str, Any]]:
        """Extract records from a single NWMP PDF."""
        records = []

        with self._pdfplumber.open(pdf_path) as pdf:
            n_pages = len(pdf.pages)
            logger.info(f"Processing {pdf_path.name}: {n_pages} pages, year={year}")

            # Find relevant pages (containing river keyword)
            if include_all:
                target_pages = list(range(n_pages))
            else:
                target_pages = self._find_river_pages(pdf, river)
                if not target_pages:
                    logger.warning(f"  No '{river}' pages found in {pdf_path.name}")
                    return []
                logger.info(f"  Found {len(target_pages)} relevant pages: {target_pages[:20]}")

            for pi in target_pages:
                page = pdf.pages[pi]
                page_records = self._extract_page(page, year, pi, river, include_all)
                records.extend(page_records)

        return records

    def _find_river_pages(self, pdf, river: str) -> List[int]:
        """Find all pages that mention the target river system."""
        keywords = _GANGA_KEYWORDS if river.lower() == "ganga" else [river.lower()]
        pages = []
        for pi, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").lower()
            if any(kw in text for kw in keywords):
                pages.append(pi)
        return pages

    def _extract_page(
        self,
        page,
        year: int,
        page_idx: int,
        river: str,
        include_all: bool,
    ) -> List[Dict[str, Any]]:
        """Extract records from a single page."""
        records = []

        # Try different extraction strategies
        tables = self._extract_tables_robust(page)

        if not tables:
            return []

        for table in tables:
            if len(table) < 2:
                continue

            # Detect and normalize headers
            header_idx, headers = self._detect_headers(table)
            if headers is None:
                continue

            # Extract data rows
            for row_idx in range(header_idx + 1, len(table)):
                row = table[row_idx]
                record = self._parse_data_row(row, headers, year, page_idx)
                if record and record.get("_has_data"):
                    del record["_has_data"]
                    # Filter by river if needed
                    if include_all or self._is_ganga_record(record, river):
                        records.append(record)

        return records

    def _extract_tables_robust(self, page) -> List[List[List[str]]]:
        """Try multiple extraction strategies and pick the best result."""
        results = []

        # Strategy 1: Strict line detection (best for well-formed tables)
        try:
            tables = page.extract_tables({
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 4,
                "join_tolerance": 4,
                "edge_min_length": 10,
                "min_words_vertical": 1,
                "min_words_horizontal": 1,
            })
            if tables:
                cleaned = [self._clean_table(t) for t in tables]
                cleaned = [t for t in cleaned if t and len(t) >= 2]
                if cleaned:
                    results.extend(cleaned)
        except Exception:
            pass

        # Strategy 2: Text-based detection (for tables without clear lines)
        if not results:
            try:
                tables = page.extract_tables({
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 6,
                    "join_tolerance": 6,
                })
                if tables:
                    cleaned = [self._clean_table(t) for t in tables]
                    cleaned = [t for t in cleaned if t and len(t) >= 2]
                    if cleaned:
                        results.extend(cleaned)
            except Exception:
                pass

        # Strategy 3: Explicit settings (fallback)
        if not results:
            try:
                tables = page.extract_tables({
                    "vertical_strategy": "lines_strict",
                    "horizontal_strategy": "lines_strict",
                })
                if tables:
                    cleaned = [self._clean_table(t) for t in tables]
                    cleaned = [t for t in cleaned if t and len(t) >= 2]
                    if cleaned:
                        results.extend(cleaned)
            except Exception:
                pass

        # If we got multiple small tables from the same page, try to merge them
        if len(results) > 1:
            merged = self._try_merge_tables(results)
            if merged:
                return merged

        return results

    def _try_merge_tables(
        self, tables: List[List[List[str]]]
    ) -> Optional[List[List[List[str]]]]:
        """Try to merge fragmented tables that were split by pdfplumber.

        Some PDFs (2020, 2022) split a single logical table into many
        sub-tables. We detect this by checking if they have the same
        column count and compatible headers.
        """
        if not tables:
            return None

        # Group by column count
        by_ncols: Dict[int, List[List[List[str]]]] = {}
        for t in tables:
            if t:
                nc = len(t[0])
                by_ncols.setdefault(nc, []).append(t)

        merged_tables = []
        for nc, group in by_ncols.items():
            if len(group) <= 1:
                merged_tables.extend(group)
                continue

            # Check if all share similar headers
            base_header = group[0][0] if group[0] else []
            compatible = True
            for t in group[1:]:
                if not t:
                    compatible = False
                    break
                # Check if first row looks like a header repeat
                if not self._rows_similar(base_header, t[0]):
                    compatible = False
                    break

            if compatible and len(group) > 1:
                # Merge: keep header from first, skip headers in rest
                merged = list(group[0])
                for t in group[1:]:
                    # Skip the header row if it looks like a repeat
                    start = 1 if self._rows_similar(base_header, t[0]) else 0
                    merged.extend(t[start:])
                merged_tables.append(merged)
            else:
                merged_tables.extend(group)

        return merged_tables if merged_tables else None

    def _rows_similar(self, row1: List[str], row2: List[str]) -> bool:
        """Check if two rows are similar (header repeat detection)."""
        if len(row1) != len(row2):
            return False
        matches = 0
        for a, b in zip(row1, row2):
            a_clean = self._normalize_cell(a)
            b_clean = self._normalize_cell(b)
            if a_clean and b_clean and (a_clean in b_clean or b_clean in a_clean):
                matches += 1
        return matches >= len(row1) * 0.5

    def _clean_table(self, table: List[List[Optional[str]]]) -> List[List[str]]:
        """Clean extracted table."""
        cleaned = []
        for row in table:
            if row is None:
                continue
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    text = re.sub(r"\s+", " ", str(cell).strip())
                    cleaned_row.append(text)
            if any(c.strip() for c in cleaned_row):
                cleaned.append(cleaned_row)
        return cleaned

    def _detect_headers(
        self, table: List[List[str]]
    ) -> Tuple[int, Optional[List[Dict[str, str]]]]:
        """Detect the header row and map columns to canonical parameters.

        NWMP tables have complex multi-line headers. We search the first
        few rows for parameter keywords and build a column mapping.

        Returns:
            (header_row_index, list_of_column_mappings)
            Each column mapping is {"param": canonical_name, "sub": "min"|"max"|"mean"|""}
        """
        # Gather text from first few rows to build a composite header
        header_candidates = min(5, len(table))
        best_mapping = None
        best_score = 0
        best_idx = 0

        # Try single-row headers first
        for i in range(header_candidates):
            row = table[i]
            mapping = self._map_columns(row)
            score = sum(1 for m in mapping if m["param"] != "unknown")
            if score > best_score:
                best_score = score
                best_mapping = mapping
                best_idx = i

        # Try combining consecutive rows (multi-line headers)
        for i in range(min(3, len(table) - 1)):
            for j in range(i + 1, min(i + 3, len(table))):
                combined = self._combine_header_rows(table[i], table[j])
                mapping = self._map_columns(combined)
                score = sum(1 for m in mapping if m["param"] != "unknown")
                if score > best_score:
                    best_score = score
                    best_mapping = mapping
                    best_idx = j

        if best_score < 3:
            return 0, None

        return best_idx, best_mapping

    def _combine_header_rows(
        self, row1: List[str], row2: List[str]
    ) -> List[str]:
        """Combine two header rows (e.g., parameter name + unit/min/max)."""
        result = []
        max_len = max(len(row1), len(row2))
        for i in range(max_len):
            a = row1[i] if i < len(row1) else ""
            b = row2[i] if i < len(row2) else ""
            combined = f"{a} {b}".strip()
            result.append(combined)
        return result

    def _map_columns(self, header_row: List[str]) -> List[Dict[str, str]]:
        """Map a header row to canonical parameter names."""
        mappings = []

        for cell in header_row:
            cell_lower = cell.lower().strip()
            mapped = {"param": "unknown", "sub": "", "raw": cell}

            # Check for Min/Max sub-headers
            sub = ""
            if re.search(r"\bmin\b", cell_lower):
                sub = "min"
                cell_lower = re.sub(r"\bmin\b", "", cell_lower).strip()
            elif re.search(r"\bmax\b", cell_lower):
                sub = "max"
                cell_lower = re.sub(r"\bmax\b", "", cell_lower).strip()
            elif re.search(r"\bmean\b|\bavg\b|\bmedian\b", cell_lower):
                sub = "mean"
                cell_lower = re.sub(r"\bmean\b|\bavg\b|\bmedian\b", "", cell_lower).strip()

            # Match against known parameter patterns
            for param_name, patterns in _PARAM_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, cell_lower):
                        mapped = {"param": param_name, "sub": sub, "raw": cell}
                        break
                if mapped["param"] != "unknown":
                    break

            mappings.append(mapped)

        return mappings

    def _parse_data_row(
        self,
        row: List[str],
        headers: List[Dict[str, str]],
        year: int,
        page_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Parse a data row using the detected header mapping."""
        if not row or len(row) == 0:
            return None

        record: Dict[str, Any] = {
            "year": year,
            "source": "nwmp_pdf",
            "source_page": page_idx,
        }

        has_data = False
        n_cols = min(len(row), len(headers))

        for i in range(n_cols):
            cell = row[i].strip() if i < len(row) and row[i] else ""
            header = headers[i]
            param = header["param"]
            sub = header["sub"]

            if param == "unknown":
                continue

            if param in ("station_code", "station_name", "state"):
                if cell and not self._is_header_text(cell):
                    record[param] = cell
            else:
                # Numeric parameter
                value = self._parse_numeric(cell)
                if value is not None:
                    key = f"{param}_{sub}" if sub else param
                    record[key] = value
                    has_data = True

        record["_has_data"] = has_data
        return record if (record.get("station_name") or record.get("station_code")) else None

    def _is_header_text(self, text: str) -> bool:
        """Check if a cell value is actually a repeated header."""
        lower = text.lower()
        header_words = [
            "station", "code", "location", "state", "temperature",
            "conductivity", "dissolved", "oxygen", "coliform",
            "nitrate", "monitoring", "parameter", "min", "max",
        ]
        matches = sum(1 for w in header_words if w in lower)
        return matches >= 2

    def _parse_numeric(self, text: str) -> Optional[float]:
        """Parse a numeric value from a cell, handling common artifacts."""
        if not text:
            return None

        # Remove common non-numeric markers
        text = text.strip()
        text = re.sub(r"[<>≤≥~]", "", text)
        text = re.sub(r"\s+", "", text)

        # Handle "BDL" (Below Detection Limit), "NR", "NA", "-", etc.
        if re.match(r"^(bdl|nd|nr|na|nil|n/?a|\-+|\*+)$", text, re.IGNORECASE):
            return None

        # Handle range like "12.3-15.6" → take mean
        range_match = re.match(r"^(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)$", text)
        if range_match:
            a, b = float(range_match.group(1)), float(range_match.group(2))
            return (a + b) / 2.0

        # Handle comma-separated thousands
        text = text.replace(",", "")

        try:
            return float(text)
        except ValueError:
            # Try extracting first number
            m = re.search(r"(\d+\.?\d*)", text)
            if m:
                return float(m.group(1))
            return None

    def _is_ganga_record(self, record: Dict[str, Any], river: str) -> bool:
        """Check if a record belongs to the Ganga river system."""
        keywords = _GANGA_KEYWORDS if river.lower() == "ganga" else [river.lower()]
        searchable = " ".join(
            str(v).lower()
            for k, v in record.items()
            if k in ("station_name", "state", "river") and v
        )
        return any(kw in searchable for kw in keywords)

    @staticmethod
    def _normalize_cell(text: str) -> str:
        """Normalize cell text for comparison."""
        if not text:
            return ""
        return re.sub(r"[^a-z0-9]", "", text.lower())


def extract_nwmp_to_csv(
    pdf_dir: str = "data/raw/cpcb/nwmp_pdfs",
    output_path: str = "data/raw/pdf_extracted/nwmp_ganga_extracted.csv",
    river: str = "ganga",
) -> str:
    """Convenience function: extract all NWMP data and save as CSV.

    Returns path to saved CSV.
    """
    import csv as csv_mod

    extractor = NWMPExtractor()
    records = extractor.extract_all(pdf_dir, river=river)

    if not records:
        logger.warning("No records extracted!")
        return ""

    # Collect all unique columns
    all_cols = set()
    for r in records:
        all_cols.update(r.keys())

    # Priority ordering
    priority = [
        "year", "station_code", "station_name", "state",
        "temperature", "temperature_min", "temperature_max",
        "ph", "ph_min", "ph_max",
        "conductivity", "conductivity_min", "conductivity_max",
        "do", "do_min", "do_max",
        "bod", "bod_min", "bod_max",
        "cod", "cod_min", "cod_max",
        "nitrate", "nitrate_min", "nitrate_max",
        "fecal_coliform", "fecal_coliform_min", "fecal_coliform_max",
        "total_coliform", "total_coliform_min", "total_coliform_max",
        "tds", "fluoride", "arsenic", "turbidity",
        "chloride", "hardness", "alkalinity", "sulphate", "iron",
        "source", "source_page",
    ]
    ordered = [c for c in priority if c in all_cols]
    ordered += sorted(all_cols - set(ordered))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    logger.info(f"Saved {len(records)} NWMP records → {output}")
    return str(output)
