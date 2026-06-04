"""
Pipeline orchestrator.

Chains together: Adapters → Normalization → Deduplication → Validation
→ LLM Agents (optional) → Conflict Resolution → Output.
"""

import csv
import json
import logging
import time
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..config.schema import CANONICAL_SCHEMA, CanonicalRecord, QualityFlag
from ..pipeline.normalizer import Normalizer
from ..pipeline.validator import Validator
from ..pipeline.deduplicator import Deduplicator
from ..agents.schema_mapper import SchemaMapperAgent
from ..agents.quality_auditor import QualityAuditorAgent
from ..agents.conflict_resolver import ConflictResolverAgent
from ..adapters.base import BaseAdapter

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main pipeline orchestrator that chains all processing stages."""

    def __init__(
        self,
        output_dir: Path = Path("output"),
        use_llm: bool = False,
        llm_client=None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_llm = use_llm
        self.llm_client = llm_client

        # Pipeline components
        self.normalizer = Normalizer()
        self.validator = Validator()
        self.deduplicator = Deduplicator()
        self.conflict_resolver = ConflictResolverAgent(llm_client)

        if use_llm:
            self.schema_mapper = SchemaMapperAgent(llm_client)
            self.quality_auditor = QualityAuditorAgent(llm_client)
        else:
            self.schema_mapper = None
            self.quality_auditor = None

        # Tracking
        self.all_records: List[CanonicalRecord] = []
        self.adapters_run: List[str] = []
        self.timing: Dict[str, float] = {}

    def add_adapter_records(
        self, adapter: BaseAdapter, **kwargs
    ) -> int:
        """Run an adapter and add its records to the pipeline."""
        t0 = time.time()
        try:
            records = adapter.run(**kwargs)
            self.all_records.extend(records)
            self.adapters_run.append(adapter.source_id)
            elapsed = time.time() - t0
            self.timing[f"adapter_{adapter.source_id}"] = elapsed
            logger.info(
                f"[{adapter.source_id}] Added {len(records)} records "
                f"({elapsed:.1f}s)"
            )
            return len(records)
        except Exception as e:
            logger.error(f"[{adapter.source_id}] Failed: {e}")
            return 0

    def run(self) -> List[CanonicalRecord]:
        """Execute the full pipeline on all collected records."""
        if not self.all_records:
            logger.warning("No records to process!")
            return []

        logger.info(f"{'='*60}")
        logger.info(f"Starting pipeline with {len(self.all_records)} records")
        logger.info(f"Sources: {self.adapters_run}")
        logger.info(f"{'='*60}")

        # Stage 1: Normalization
        t0 = time.time()
        logger.info("[1/5] Normalizing parameter names and units...")
        records = self.normalizer.normalize(self.all_records)
        self.timing["normalization"] = time.time() - t0

        # Stage 1b: LLM schema mapping for unresolved parameters
        if self.use_llm and self.normalizer.unresolved_params:
            t0 = time.time()
            logger.info("[1b] LLM schema mapping for unresolved parameters...")
            unresolved = list(set(self.normalizer.unresolved_params))
            mappings = self.schema_mapper.map_batch(unresolved)
            # Apply LLM mappings to any records we skipped
            # (In the current flow, unresolved records were dropped.
            #  A production version would re-process them here.)
            self.timing["llm_schema_mapping"] = time.time() - t0

        # Stage 2: Deduplication
        t0 = time.time()
        logger.info("[2/5] Deduplicating stations...")
        self.deduplicator.build_station_registry(records)
        records = self.deduplicator.apply_ids(records)
        self.timing["deduplication"] = time.time() - t0

        # Stage 3: Validation
        t0 = time.time()
        logger.info("[3/5] Validating records...")
        records = self.validator.validate(records)
        self.timing["validation"] = time.time() - t0

        # Stage 4: LLM quality audit (optional)
        if self.use_llm and self.quality_auditor:
            t0 = time.time()
            logger.info("[4/5] LLM quality auditing suspect records...")
            records = self.quality_auditor.audit_batch(records, max_records=50)
            self.timing["llm_quality_audit"] = time.time() - t0
        else:
            logger.info("[4/5] Skipping LLM audit (disabled)")

        # Stage 5: Conflict resolution
        t0 = time.time()
        logger.info("[5/5] Resolving cross-source conflicts...")
        records = self.conflict_resolver.resolve(records)
        self.timing["conflict_resolution"] = time.time() - t0

        logger.info(f"{'='*60}")
        logger.info(f"Pipeline complete: {len(records)} records output")
        logger.info(f"{'='*60}")

        self.all_records = records
        return records

    def save(self, filename: str = "consolidated_data_lake.csv"):
        """Save the consolidated data lake to CSV."""
        path = self.output_dir / filename
        records = self.all_records

        if not records:
            logger.warning("No records to save!")
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CANONICAL_SCHEMA)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec.to_dict())

        logger.info(f"Saved {len(records)} records → {path}")

        # Also save station registry
        self._save_station_registry()

        # Save summary report
        self._save_report()

    def _save_station_registry(self):
        """Save the deduplicated station registry."""
        path = self.output_dir / "station_registry.csv"
        stations = self.deduplicator.stations

        if not stations:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "canonical_id", "name", "river", "state",
                "latitude", "longitude", "sources", "record_count",
            ])
            for sid, info in sorted(stations.items()):
                sources = "; ".join(
                    f"{src}:{sid}" for src, sid in info.source_ids.items()
                )
                writer.writerow([
                    info.canonical_id,
                    info.best_name,
                    info.river,
                    info.state,
                    f"{info.latitude:.6f}",
                    f"{info.longitude:.6f}",
                    sources,
                    info.record_count,
                ])

        logger.info(f"Saved {len(stations)} stations → {path}")

    def _save_report(self):
        """Save a JSON summary report of the pipeline run."""
        # Count by quality flag
        flag_counts = Counter(r.quality_flag for r in self.all_records)
        # Count by parameter
        param_counts = Counter(r.parameter for r in self.all_records)
        # Count by source
        source_counts = Counter(r.source for r in self.all_records)
        # Count by river
        river_counts = Counter(r.river for r in self.all_records if r.river)
        # Date range
        dates = [r.sample_date for r in self.all_records]
        min_date = min(dates).isoformat() if dates else None
        max_date = max(dates).isoformat() if dates else None

        report = {
            "run_timestamp": datetime.now().isoformat(),
            "total_records": len(self.all_records),
            "total_stations": len(self.deduplicator.stations),
            "date_range": {"min": min_date, "max": max_date},
            "sources_used": self.adapters_run,
            "quality_flags": dict(flag_counts),
            "records_by_parameter": dict(param_counts.most_common()),
            "records_by_source": dict(source_counts),
            "records_by_river": dict(river_counts.most_common(20)),
            "pipeline_timing_seconds": self.timing,
            "normalizer_stats": self.normalizer.stats,
            "validator_stats": self.validator.stats,
            "conflict_resolver_stats": self.conflict_resolver.stats,
        }

        path = self.output_dir / "pipeline_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Pipeline report → {path}")

        # Print summary to console
        logger.info(f"\n{'-'*50}")
        logger.info(f"PIPELINE SUMMARY")
        logger.info(f"{'-'*50}")
        logger.info(f"Total records:    {len(self.all_records):,}")
        logger.info(f"Unique stations:  {len(self.deduplicator.stations)}")
        logger.info(f"Date range:       {min_date} -> {max_date}")
        logger.info(f"Parameters:       {len(param_counts)}")
        logger.info(f"Quality breakdown:")
        for flag, count in sorted(flag_counts.items()):
            pct = count / len(self.all_records) * 100
            logger.info(f"  {flag:30s} {count:>7,} ({pct:.1f}%)")
        logger.info(f"{'-'*50}")
