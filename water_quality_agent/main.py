"""
Water Quality Data Integration Agent — Main Entry Point.

Usage:
  # Basic run with existing CPCB data (no LLM)
  python -m water_quality_agent.main --cpcb-csv data/combined_station_data_with_coords.csv

  # With station metadata
  python -m water_quality_agent.main \\
      --cpcb-csv data/combined_station_data_with_coords.csv \\
      --cpcb-stations data/cpcb_station_locations.csv

  # With CWC data
  python -m water_quality_agent.main \\
      --cpcb-csv data/combined_station_data_with_coords.csv \\
      --cwc-file data/cwc_water_levels.xlsx

  # With data.gov.in CSV
  python -m water_quality_agent.main \\
      --cpcb-csv data/combined_station_data_with_coords.csv \\
      --datagov-csv data/datagov_wq.csv

  # With GEMStat data
  python -m water_quality_agent.main \\
      --cpcb-csv data/combined_station_data_with_coords.csv \\
      --gemstat-csv data/gemstat_india.csv

  # Full run with LLM agents enabled
  python -m water_quality_agent.main \\
      --cpcb-csv data/combined_station_data_with_coords.csv \\
      --use-llm

  # data.gov.in via API
  python -m water_quality_agent.main \\
      --cpcb-csv data/combined_station_data_with_coords.csv \\
      --datagov-key YOUR_API_KEY
"""

import argparse
import sys
from pathlib import Path

from .adapters.cpcb import CPCBAdapter
from .adapters.cwc import CWCAdapter
from .adapters.data_gov import DataGovAdapter
from .adapters.gemstat import GEMStatAdapter
from .pipeline.orchestrator import Orchestrator
from .utils.logging_config import setup_logging


def parse_args():
    p = argparse.ArgumentParser(
        description="Water Quality Data Integration Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data sources
    p.add_argument("--cpcb-csv", type=Path,
                    help="Path to CPCB measurements CSV (e.g. combined_station_data_with_coords.csv)")
    p.add_argument("--cpcb-stations", type=Path,
                    help="Path to CPCB station locations CSV")
    p.add_argument("--cwc-file", type=Path,
                    help="Path to CWC data file (CSV or Excel)")
    p.add_argument("--datagov-csv", type=Path,
                    help="Path to data.gov.in downloaded CSV")
    p.add_argument("--datagov-key", type=str,
                    help="data.gov.in API key for live fetching")
    p.add_argument("--gemstat-csv", type=Path,
                    help="Path to GEMStat CSV file")

    # Pipeline options
    p.add_argument("--use-llm", action="store_true",
                    help="Enable LLM agents (uses Hugging Face models)")
    p.add_argument("--llm-mode", choices=["api", "local"], default="api",
                    help="LLM mode: 'api' (HF Inference API, no GPU needed) "
                         "or 'local' (download model, needs RAM/VRAM)")
    p.add_argument("--llm-model", type=str,
                    default="microsoft/Phi-3.5-mini-instruct",
                    help="Hugging Face model ID (default: microsoft/Phi-3.5-mini-instruct)")
    p.add_argument("--hf-token", type=str, default=None,
                    help="Hugging Face API token (or set HF_TOKEN env var)")
    p.add_argument("--output-dir", type=Path, default=Path("water_quality_agent/output"),
                    help="Output directory (default: water_quality_agent/output)")
    p.add_argument("--log-file", type=Path, default=None,
                    help="Log file path (default: console only)")
    p.add_argument("--verbose", "-v", action="store_true",
                    help="Verbose logging (DEBUG level)")

    return p.parse_args()


def main():
    args = parse_args()

    # Setup logging
    import logging
    setup_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=args.log_file,
    )

    logger = logging.getLogger("water_quality_agent")
    logger.info("=" * 60)
    logger.info("Water Quality Data Integration Agent")
    logger.info("=" * 60)

    # Validate at least one source provided
    has_source = any([
        args.cpcb_csv,
        args.cwc_file,
        args.datagov_csv,
        args.datagov_key,
        args.gemstat_csv,
    ])
    if not has_source:
        logger.error("No data sources specified! Use --help for options.")
        sys.exit(1)

    # Initialize LLM client if needed
    llm_client = None
    if args.use_llm:
        from water_quality_agent.utils.llm_client import HuggingFaceLLM
        llm_client = HuggingFaceLLM(
            mode=args.llm_mode,
            model_name=args.llm_model,
            hf_token=args.hf_token,
        )
        logger.info(f"LLM enabled: {args.llm_model} ({args.llm_mode} mode)")

    # Initialize orchestrator
    orchestrator = Orchestrator(
        output_dir=args.output_dir,
        use_llm=args.use_llm,
        llm_client=llm_client,
    )

    # ── Run adapters ──────────────────────────────────────────────────────

    # CPCB
    if args.cpcb_csv:
        adapter = CPCBAdapter(
            csv_path=args.cpcb_csv,
            station_locations_path=args.cpcb_stations,
            output_dir=args.output_dir,
        )
        orchestrator.add_adapter_records(adapter)

    # CWC
    if args.cwc_file:
        adapter = CWCAdapter(
            data_path=args.cwc_file,
            output_dir=args.output_dir,
        )
        orchestrator.add_adapter_records(adapter)

    # data.gov.in
    if args.datagov_csv or args.datagov_key:
        adapter = DataGovAdapter(
            api_key=args.datagov_key,
            csv_path=args.datagov_csv,
            output_dir=args.output_dir,
        )
        orchestrator.add_adapter_records(adapter)

    # GEMStat
    if args.gemstat_csv:
        adapter = GEMStatAdapter(
            csv_path=args.gemstat_csv,
            output_dir=args.output_dir,
        )
        orchestrator.add_adapter_records(adapter)

    # ── Run pipeline ──────────────────────────────────────────────────────
    records = orchestrator.run()

    # ── Save output ───────────────────────────────────────────────────────
    orchestrator.save("consolidated_data_lake.csv")

    logger.info("Done! Output files in: %s", args.output_dir)


if __name__ == "__main__":
    main()
