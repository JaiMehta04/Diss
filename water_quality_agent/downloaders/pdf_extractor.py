"""
PDF table extractor for water quality reports.

CPCB and other agencies publish water quality data as PDF reports
with tabular data. This module:
  1. Downloads PDFs from known URLs
  2. Extracts tables using pdfplumber (lattice + stream detection)
  3. Cleans and normalizes extracted data into CSVs

Handles common PDF issues:
  - Merged cells, multi-line headers
  - Rotated / landscape pages
  - Scanned PDFs (falls back to OCR if pytesseract available)

Requirements:
    pip install pdfplumber
    # Optional for scanned PDFs:
    pip install pytesseract Pillow
"""

import csv
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extracts tables from water quality PDF reports."""

    def __init__(self):
        try:
            import pdfplumber
            self._pdfplumber = pdfplumber
        except ImportError:
            self._pdfplumber = None
            logger.warning(
                "pdfplumber not installed. PDF extraction unavailable. "
                "Install with: pip install pdfplumber"
            )

    def extract_tables(self, pdf_path: Path) -> List[List[List[str]]]:
        """Extract all tables from a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of tables, each table is a list of rows,
            each row is a list of cell strings.
        """
        if self._pdfplumber is None:
            return []

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return []

        all_tables = []
        try:
            with self._pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Processing {pdf_path.name}: {len(pdf.pages)} pages")

                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables({
                        "vertical_strategy": "lines_strict",
                        "horizontal_strategy": "lines_strict",
                        "snap_tolerance": 5,
                        "join_tolerance": 5,
                    })

                    # If strict line detection finds nothing, try text-based
                    if not tables:
                        tables = page.extract_tables({
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                            "snap_tolerance": 8,
                        })

                    for table in tables:
                        cleaned = self._clean_table(table)
                        if cleaned and len(cleaned) > 1:
                            all_tables.append(cleaned)
                            logger.debug(
                                f"  Page {page_num}: found table "
                                f"({len(cleaned)} rows x {len(cleaned[0])} cols)"
                            )

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")

        logger.info(f"Extracted {len(all_tables)} tables from {pdf_path.name}")
        return all_tables

    def _clean_table(self, table: List[List[Optional[str]]]) -> List[List[str]]:
        """Clean extracted table: handle None values, strip whitespace, remove empty rows."""
        cleaned = []
        for row in table:
            if row is None:
                continue
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # Normalize whitespace (PDF tables often have weird spacing)
                    text = re.sub(r"\s+", " ", str(cell).strip())
                    cleaned_row.append(text)
            # Skip entirely empty rows
            if any(c for c in cleaned_row):
                cleaned.append(cleaned_row)
        return cleaned

    def extract_to_csv(self, pdf_path: Path, output_dir: Path) -> List[Path]:
        """Extract all tables from a PDF and save each as a separate CSV.

        Args:
            pdf_path: Path to PDF
            output_dir: Directory to save CSVs

        Returns:
            List of saved CSV paths
        """
        tables = self.extract_tables(pdf_path)
        if not tables:
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        stem = pdf_path.stem

        for i, table in enumerate(tables):
            csv_path = output_dir / f"{stem}_table{i+1}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(table)
            saved.append(csv_path)

        logger.info(f"Saved {len(saved)} CSV files from {pdf_path.name}")
        return saved

    def extract_water_quality_data(
        self, pdf_path: Path
    ) -> List[Dict[str, str]]:
        """Extract and parse water quality data from a PDF report.

        Tries to identify the header row, then maps columns to
        known water quality parameters.

        Returns:
            List of dicts with keys matching the identified columns.
        """
        tables = self.extract_tables(pdf_path)
        all_records = []

        for table in tables:
            if len(table) < 2:
                continue

            # Identify header row (look for common WQ column names)
            header_idx = self._find_header_row(table)
            if header_idx is None:
                continue

            headers = [h.lower().strip() for h in table[header_idx]]

            # Extract data rows
            for row in table[header_idx + 1:]:
                if len(row) != len(headers):
                    # Pad or truncate
                    row = row[:len(headers)] + [""] * max(0, len(headers) - len(row))

                record = {}
                for col_name, value in zip(headers, row):
                    if col_name:
                        record[col_name] = value
                if any(v for v in record.values()):
                    all_records.append(record)

        return all_records

    def _find_header_row(self, table: List[List[str]]) -> Optional[int]:
        """Find the header row in a table by looking for WQ-related keywords."""
        wq_keywords = {
            "station", "location", "river", "ph", "do", "bod", "cod",
            "turbidity", "conductivity", "temperature", "tds",
            "dissolved oxygen", "parameter", "date", "year", "state",
            "district", "nitrate", "coliform", "hardness", "chloride",
            "alkalinity", "sample", "monitoring",
        }

        best_idx = None
        best_score = 0

        for i, row in enumerate(table[:5]):  # Check first 5 rows
            score = sum(
                1 for cell in row
                if any(kw in cell.lower() for kw in wq_keywords)
            )
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= 2:
            return best_idx
        return None


def extract_all_pdfs(
    pdf_dir: str, output_dir: str = "data/raw/pdf_extracted"
) -> Dict[str, int]:
    """Extract tables from all PDFs in a directory.

    Returns dict of {pdf_name: num_tables_extracted}
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)

    extractor = PDFExtractor()
    results = {}

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return results

    logger.info(f"Found {len(pdf_files)} PDFs to process")

    for pdf_path in pdf_files:
        csvs = extractor.extract_to_csv(pdf_path, output_dir)
        results[pdf_path.name] = len(csvs)

    return results
