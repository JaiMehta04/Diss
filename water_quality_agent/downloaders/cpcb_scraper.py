"""
CPCB Water Quality Data Scraper.

Downloads water quality data from CPCB (Central Pollution Control Board):
  1. NWMP PDF reports from cpcb.nic.in/wqm/ (verified working)
  2. HTML table data from CPCB portal pages
  3. Discovers and downloads linked data files (PDF, Excel, CSV)

Verified working URLs (as of April 2026):
  - https://cpcb.nic.in/wqm/{YEAR}/NWMP_DATA_{YEAR}.pdf
    Available: 2015, 2018, 2019, 2020, 2021, 2022, 2024

Usage:
    scraper = CPCBScraper()
    scraper.download_all(output_dir="data/raw/cpcb")
"""

import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# ── Verified CPCB PDF report URLs ─────────────────────────────────────────
# Pattern discovered: https://cpcb.nic.in/wqm/{YEAR}/NWMP_DATA_{YEAR}.pdf
# Available years: 2015, 2018, 2019, 2020, 2021, 2022, 2024

NWMP_PDF_YEARS = [2015, 2018, 2019, 2020, 2021, 2022, 2024]
NWMP_PDF_URL = "https://cpcb.nic.in/wqm/{year}/NWMP_DATA_{year}.pdf"

# CPCB pages with downloadable links and data tables
CPCB_DATA_PAGES = [
    ("https://cpcb.nic.in/nwmp-data/", "nwmp_data"),
    ("https://cpcb.nic.in/wqm/", "wqm"),
    ("https://cpcb.nic.in/water-quality/", "water_quality"),
    ("https://cpcb.nic.in/real-time-water-quality-data/", "realtime_wq"),
]


class CPCBScraper:
    """Downloads water quality data from CPCB portals."""

    def __init__(self, rate_limit_delay: float = 1.5):
        self.delay = rate_limit_delay
        self.session = requests.Session()
        self.session.verify = False  # Indian gov sites have SSL issues
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/html, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })

    # ── 1. Download NWMP PDF Reports ──────────────────────────────────────

    def download_nwmp_pdfs(
        self, output_dir: Path, years: Optional[List[int]] = None
    ) -> List[Path]:
        """Download CPCB NWMP (National Water Monitoring Programme) PDF reports.

        These are the main annual water quality reports with station-wise
        data for rivers, lakes, ponds across India.

        Returns list of downloaded PDF paths.
        """
        years = years or NWMP_PDF_YEARS
        output_dir = Path(output_dir) / "nwmp_pdfs"
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded = []

        for year in years:
            url = NWMP_PDF_URL.format(year=year)
            filename = f"NWMP_DATA_{year}.pdf"
            filepath = output_dir / filename

            if filepath.exists() and filepath.stat().st_size > 10000:
                logger.info(f"Already downloaded: {filename}")
                downloaded.append(filepath)
                continue

            try:
                resp = self.session.get(url, timeout=30, stream=True)
                if (resp.status_code == 200 and
                        "pdf" in resp.headers.get("content-type", "").lower()):
                    with open(filepath, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    logger.info(f"Downloaded: {filename} ({size_mb:.1f} MB)")
                    downloaded.append(filepath)
                else:
                    logger.debug(f"Not available: {filename} (HTTP {resp.status_code})")
            except Exception as e:
                logger.warning(f"Failed to download {filename}: {e}")

            time.sleep(self.delay)

        logger.info(f"Downloaded {len(downloaded)} NWMP PDF reports")
        return downloaded

    # ── 2. Discover + Download linked files from CPCB pages ──────────────

    def discover_and_download_files(self, output_dir: Path) -> List[Path]:
        """Scrape CPCB portal pages for downloadable data files.

        Finds links to PDFs, Excel files, and CSVs from CPCB data pages.
        """
        from bs4 import BeautifulSoup

        output_dir = Path(output_dir) / "discovered_files"
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded = []

        for page_url, page_name in CPCB_DATA_PAGES:
            logger.info(f"Scanning: {page_name} ({page_url})")
            try:
                resp = self.session.get(page_url, timeout=20)
                if resp.status_code != 200:
                    continue
            except Exception as e:
                logger.debug(f"Failed to fetch {page_url}: {e}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Find all downloadable file links
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                lower_href = href.lower()

                # Only download data files
                if not any(ext in lower_href for ext in [".pdf", ".xlsx", ".xls", ".csv"]):
                    continue
                # Skip generic site files
                if any(skip in lower_href for skip in [
                    "stqc-cert", "gst_details", "it_technical", "e-governance",
                    "tender", "recruitment", "notification", "circular",
                ]):
                    continue

                # Make absolute URL
                if href.startswith("/"):
                    href = f"https://cpcb.nic.in{href}"
                elif not href.startswith("http"):
                    href = f"https://cpcb.nic.in/{href}"

                # Clean up ../
                href = re.sub(r'/[^/]+/\.\./', '/', href)

                link_text = a_tag.get_text(strip=True)[:50]
                filename = href.split("/")[-1].split("?")[0]
                if not filename:
                    continue

                # Prefix with page name for organization
                filepath = output_dir / f"{page_name}_{filename}"

                if filepath.exists() and filepath.stat().st_size > 1000:
                    downloaded.append(filepath)
                    continue

                try:
                    file_resp = self.session.get(href, timeout=30, stream=True)
                    content_type = file_resp.headers.get("content-type", "")
                    if file_resp.status_code == 200 and "html" not in content_type:
                        with open(filepath, "wb") as f:
                            for chunk in file_resp.iter_content(chunk_size=8192):
                                f.write(chunk)
                        size_kb = filepath.stat().st_size / 1024
                        logger.info(f"  Downloaded: {filename} ({size_kb:.0f} KB) [{link_text}]")
                        downloaded.append(filepath)
                        time.sleep(self.delay)
                except Exception as e:
                    logger.debug(f"  Failed: {filename}: {e}")

        logger.info(f"Downloaded {len(downloaded)} files from CPCB pages")
        return downloaded

    # ── 3. Scrape HTML tables ─────────────────────────────────────────────

    def scrape_html_tables(self, output_dir: Path) -> List[Path]:
        """Scrape data tables from CPCB HTML pages."""
        from bs4 import BeautifulSoup

        output_dir = Path(output_dir) / "html_tables"
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = []

        for page_url, page_name in CPCB_DATA_PAGES:
            try:
                resp = self.session.get(page_url, timeout=20)
                if resp.status_code != 200:
                    continue
            except Exception:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            for i, table in enumerate(soup.find_all("table")):
                rows = []
                for tr in table.find_all("tr"):
                    cells = [
                        td.get_text(strip=True)
                        for td in tr.find_all(["td", "th"])
                    ]
                    if any(c for c in cells):
                        rows.append(cells)

                # Only save non-trivial tables (>2 rows, >2 columns)
                if len(rows) > 2 and any(len(r) > 2 for r in rows):
                    filepath = output_dir / f"{page_name}_table{i+1}.csv"
                    with open(filepath, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerows(rows)
                    logger.info(f"Scraped table: {filepath.name} ({len(rows)} rows)")
                    saved.append(filepath)

        return saved

    # ── Master download ───────────────────────────────────────────────────

    def download_all(
        self, output_dir: str = "data/raw/cpcb", river: str = "ganga"
    ) -> Dict[str, Any]:
        """Download all available CPCB water quality data.

        Steps:
          1. Download NWMP annual PDF reports (verified working)
          2. Discover and download linked data files from CPCB pages
          3. Scrape HTML tables from CPCB portal

        Returns summary dict.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "nwmp_pdfs": 0,
            "discovered_files": 0,
            "html_tables": 0,
        }

        # 1. Download NWMP PDF reports (most reliable source)
        logger.info("=" * 50)
        logger.info("Step 1: Downloading NWMP PDF reports...")
        logger.info("  These contain year-wise water quality data for all")
        logger.info("  monitoring stations: rivers, lakes, ponds, ground water")
        pdfs = self.download_nwmp_pdfs(output_dir)
        results["nwmp_pdfs"] = len(pdfs)

        # 2. Discover and download additional files from CPCB pages
        logger.info("=" * 50)
        logger.info("Step 2: Discovering data files from CPCB pages...")
        try:
            files = self.discover_and_download_files(output_dir)
            results["discovered_files"] = len(files)
        except ImportError:
            logger.warning("beautifulsoup4 needed for page scraping. pip install beautifulsoup4")
        except Exception as e:
            logger.warning(f"File discovery failed: {e}")

        # 3. Scrape HTML tables
        logger.info("=" * 50)
        logger.info("Step 3: Scraping HTML tables from CPCB pages...")
        try:
            tables = self.scrape_html_tables(output_dir)
            results["html_tables"] = len(tables)
        except ImportError:
            logger.warning("beautifulsoup4 needed for HTML scraping.")
        except Exception as e:
            logger.warning(f"HTML table scraping failed: {e}")

        return results


def download_cpcb(output_dir: str = "data/raw/cpcb", river: str = "ganga") -> Dict:
    """Convenience function to download all CPCB data."""
    scraper = CPCBScraper()
    return scraper.download_all(output_dir=output_dir, river=river)
