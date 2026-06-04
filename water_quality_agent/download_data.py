"""
Master data download script.

Automatically downloads water quality data from all available sources:
  1. data.gov.in — REST API (needs free API key)
  2. CPCB — Web scraping + ENVIS PDF reports
  3. Kaggle — Curated datasets (needs free Kaggle account)
  4. PDF extraction — Parses downloaded PDFs into CSVs

Then consolidates everything into adapter-compatible CSVs that the
main pipeline (water_quality_agent.main) can process.

Usage:
    # Full auto-download (needs API keys)
    python -m water_quality_agent.download_data --datagov-key YOUR_KEY

    # Just CPCB scraping (no API key needed)
    python -m water_quality_agent.download_data --sources cpcb

    # Everything including Kaggle
    python -m water_quality_agent.download_data --datagov-key KEY --kaggle --all
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from water_quality_agent.utils.logging_config import setup_logging

logger = logging.getLogger("water_quality_agent.download")


def parse_args():
    p = argparse.ArgumentParser(
        description="Auto-download water quality data from Indian government sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from data.gov.in (recommended — most reliable)
  python -m water_quality_agent.download_data --datagov-key YOUR_API_KEY

  # Download from CPCB portal + ENVIS PDFs
  python -m water_quality_agent.download_data --sources cpcb

  # Download everything
  python -m water_quality_agent.download_data --datagov-key KEY --sources all

  # Then run the pipeline on downloaded data
  python -m water_quality_agent.main --datagov-csv data/raw/datagov/*.csv --output-dir output
        """,
    )

    p.add_argument("--datagov-key", type=str,
                    help="data.gov.in API key (get free at https://data.gov.in)")
    p.add_argument("--kaggle", action="store_true",
                    help="Also download from Kaggle (needs KAGGLE_USERNAME + KAGGLE_KEY)")
    p.add_argument("--sources", nargs="+",
                    choices=["datagov", "cpcb", "kaggle", "all"],
                    default=["all"],
                    help="Which sources to download from (default: all available)")
    p.add_argument("--output-dir", type=Path, default=Path("data/raw"),
                    help="Base output directory (default: data/raw)")
    p.add_argument("--river", type=str, default="ganga",
                    help="Filter for specific river (default: ganga)")
    p.add_argument("--years", nargs="+", type=int,
                    default=list(range(2018, 2026)),
                    help="Years to download data for (default: 2018-2025)")
    p.add_argument("--extract-pdfs", action="store_true", default=True,
                    help="Extract tables from downloaded PDFs (default: True)")
    p.add_argument("--no-extract-pdfs", action="store_false", dest="extract_pdfs")
    p.add_argument("--consolidate", action="store_true", default=True,
                    help="Consolidate all downloads into pipeline-ready CSVs")
    p.add_argument("--verbose", "-v", action="store_true")

    return p.parse_args()


def download_from_datagov(output_dir: Path, api_key: str = None, river: str = "ganga") -> Dict[str, int]:
    """Download from data.gov.in API — no API key needed."""
    from water_quality_agent.downloaders.datagov import DataGovDownloader

    logger.info("=" * 60)
    logger.info("SOURCE: data.gov.in (no API key needed)")
    logger.info("=" * 60)

    dl = DataGovDownloader(api_key=api_key)
    return dl.download_all(output_dir=str(output_dir / "datagov"), river=river)


def download_from_cpcb(output_dir: Path, river: str, years: List[int]) -> Dict[str, Any]:
    """Download from CPCB portals."""
    from water_quality_agent.downloaders.cpcb_scraper import CPCBScraper

    logger.info("=" * 60)
    logger.info("SOURCE: CPCB (Real-time + ENVIS)")
    logger.info("=" * 60)

    scraper = CPCBScraper()
    return scraper.download_all(output_dir=str(output_dir / "cpcb"), river=river)


def download_from_kaggle(output_dir: Path) -> Dict[str, int]:
    """Download from Kaggle."""
    from water_quality_agent.downloaders.kaggle_dl import KaggleDownloader

    logger.info("=" * 60)
    logger.info("SOURCE: Kaggle")
    logger.info("=" * 60)

    dl = KaggleDownloader()
    return dl.download_all(output_dir=str(output_dir / "kaggle"))


def extract_pdfs(raw_dir: Path, output_dir: Path) -> Dict[str, int]:
    """Extract tables from all downloaded PDFs."""
    from water_quality_agent.downloaders.pdf_extractor import PDFExtractor

    logger.info("=" * 60)
    logger.info("STEP: Extracting tables from PDFs")
    logger.info("=" * 60)

    extractor = PDFExtractor()
    results = {}

    # Find all PDFs in the raw data directory
    pdf_files = list(raw_dir.rglob("*.pdf"))
    if not pdf_files:
        logger.info("No PDF files found to extract.")
        return results

    logger.info(f"Found {len(pdf_files)} PDFs to process")
    extracted_dir = output_dir / "pdf_extracted"

    for pdf_path in pdf_files:
        csvs = extractor.extract_to_csv(pdf_path, extracted_dir)
        results[pdf_path.name] = len(csvs)

    return results


def consolidate_downloads(raw_dir: Path, output_dir: Path, river: str) -> Path:
    """Consolidate all downloaded CSVs into a single pipeline-ready file.

    Only consolidates structured data (datagov CSVs). PDF-extracted tables
    are stored separately since they have different schemas.

    Returns path to consolidated CSV.
    """
    logger.info("=" * 60)
    logger.info("STEP: Consolidating all downloads")
    logger.info("=" * 60)

    # Only consolidate datagov CSVs — they have structured schemas
    # PDF-extracted tables have wildly different formats and are mostly
    # reference tables, not measurement data
    datagov_dir = raw_dir / "datagov"
    csv_files = list(datagov_dir.glob("*.csv")) if datagov_dir.exists() else []
    logger.info(f"Found {len(csv_files)} structured CSV files to consolidate")

    all_rows = []
    source_counts = {}

    for csv_path in csv_files:
        try:
            rows = _read_and_normalize_csv(csv_path, river)
            if rows:
                source_name = csv_path.parent.name + "/" + csv_path.name
                source_counts[source_name] = len(rows)
                all_rows.extend(rows)
        except Exception as e:
            logger.debug(f"Skipping {csv_path.name}: {e}")

    if not all_rows:
        logger.warning("No data found in any downloaded CSV!")
        return None

    # Determine all unique columns
    all_columns = set()
    for row in all_rows:
        all_columns.update(row.keys())

    # Priority columns first, then alphabetical
    priority = [
        "station", "station_name", "location", "river", "state",
        "latitude", "longitude", "lat", "lon",
        "date", "sample_date", "year", "month",
        "parameter", "value", "unit",
        "ph", "do", "bod", "cod", "tds", "conductivity", "turbidity",
        "temperature", "nitrate", "coliform", "hardness", "chloride",
        "source_file",
    ]
    ordered_cols = [c for c in priority if c in all_columns]
    ordered_cols += sorted(all_columns - set(ordered_cols))

    # Save consolidated CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    consolidated_path = output_dir / f"consolidated_{river}_downloads.csv"

    with open(consolidated_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info(f"Consolidated {len(all_rows)} rows from {len(source_counts)} files")
    logger.info(f"Saved to: {consolidated_path}")

    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {src}: {count} rows")

    return consolidated_path


def _read_and_normalize_csv(csv_path: Path, river: str) -> List[Dict]:
    """Read a CSV and normalize column names. Optionally filter for river."""
    rows = []

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        # Sniff delimiter
        sample = f.read(4096)
        f.seek(0)

        delimiter = ","
        if sample.count("\t") > sample.count(","):
            delimiter = "\t"
        elif sample.count(";") > sample.count(","):
            delimiter = ";"

        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            return []

        # Normalize column names
        col_map = {}
        for orig in reader.fieldnames:
            normalized = orig.strip().lower()
            normalized = normalized.replace(" ", "_").replace("-", "_")
            normalized = normalized.replace("(", "").replace(")", "")
            normalized = normalized.replace(".", "").replace("/", "_")
            col_map[orig] = normalized

        for row in reader:
            normalized_row = {col_map[k]: v.strip() for k, v in row.items()
                             if k and v and v.strip()}
            if normalized_row:
                normalized_row["source_file"] = csv_path.name
                rows.append(normalized_row)

    # Filter for river if data has a river column
    if river and rows:
        river_cols = [c for c in rows[0].keys()
                      if "river" in c or "water_body" in c or "source" in c]
        if river_cols:
            filtered = [
                r for r in rows
                if any(
                    river.lower() in r.get(col, "").lower()
                    for col in river_cols
                )
            ]
            if filtered:
                return filtered
            # If no matches, return all (might be Ganga but column named differently)

    return rows


def main():
    args = parse_args()

    setup_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    logger.info("=" * 60)
    logger.info("Water Quality Data Auto-Downloader")
    logger.info(f"River: {args.river}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)

    sources = set(args.sources)
    if "all" in sources:
        sources = {"datagov", "cpcb", "kaggle"}

    results = {}

    # 1. data.gov.in (NO API key needed — uses OGD default key)
    if "datagov" in sources:
        try:
            results["datagov"] = download_from_datagov(
                args.output_dir, api_key=args.datagov_key, river=args.river
            )
        except Exception as e:
            logger.error(f"data.gov.in download failed: {e}")
            results["datagov"] = {"error": str(e)}

    # 2. CPCB
    if "cpcb" in sources:
        try:
            results["cpcb"] = download_from_cpcb(
                args.output_dir, args.river, args.years
            )
        except Exception as e:
            logger.error(f"CPCB download failed: {e}")
            results["cpcb"] = {"error": str(e)}

    # 3. Kaggle
    if "kaggle" in sources and args.kaggle:
        try:
            results["kaggle"] = download_from_kaggle(args.output_dir)
        except Exception as e:
            logger.error(f"Kaggle download failed: {e}")
            results["kaggle"] = {"error": str(e)}

    # 4. Extract PDFs
    if args.extract_pdfs:
        try:
            results["pdf_extraction"] = extract_pdfs(
                args.output_dir, args.output_dir
            )
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")

    # 5. Consolidate
    if args.consolidate:
        try:
            consolidated = consolidate_downloads(
                args.output_dir, args.output_dir.parent / "consolidated",
                args.river,
            )
            if consolidated:
                results["consolidated"] = str(consolidated)
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")

    # Save download report
    report_path = args.output_dir / "download_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nDownload report saved to: {report_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    for source, result in results.items():
        if isinstance(result, dict) and "error" in result:
            logger.info(f"  {source}: FAILED — {result['error']}")
        else:
            logger.info(f"  {source}: {result}")

    logger.info("\nNext step: Run the pipeline on downloaded data:")
    logger.info(
        f"  python -m water_quality_agent.main "
        f"--datagov-csv data/consolidated/consolidated_{args.river}_downloads.csv "
        f"--output-dir water_quality_agent/output"
    )


if __name__ == "__main__":
    main()
