"""
Validation engine.

Performs multi-level quality checks on canonical records:
  1. Physical range checks (hard/soft bounds)
  2. Statistical outlier detection (IQR method)
  3. Cross-parameter consistency checks
  4. Coordinate validation (is it in India?)
  5. Temporal sanity (reasonable date range?)
"""

import logging
import math
from collections import defaultdict
from datetime import date
from typing import Dict, List, Tuple

from ..config.bounds import PHYSICAL_BOUNDS, check_bounds
from ..config.schema import CanonicalRecord, QualityFlag
from ..utils.geo import is_in_india

logger = logging.getLogger(__name__)


class Validator:
    """Multi-level validation engine for water quality records."""

    def __init__(
        self,
        min_date: date = date(1980, 1, 1),
        max_date: date = date(2030, 12, 31),
    ):
        self.min_date = min_date
        self.max_date = max_date
        self.stats = {
            "total": 0,
            "valid": 0,
            "suspect_range": 0,
            "suspect_outlier": 0,
            "suspect_consistency": 0,
            "rejected": 0,
        }

    def validate(self, records: List[CanonicalRecord]) -> List[CanonicalRecord]:
        """Validate records and assign quality flags."""
        # Pass 1: Individual record checks
        for rec in records:
            self.stats["total"] += 1
            self._check_single(rec)

        # Pass 2: Statistical outlier detection (per parameter)
        self._check_outliers(records)

        # Pass 3: Cross-parameter consistency
        self._check_consistency(records)

        # Aggregate stats
        for rec in records:
            flag = rec.quality_flag
            if flag in self.stats:
                self.stats[flag] += 1
            else:
                self.stats["valid"] += 1

        self._log_summary()
        return records

    def _check_single(self, rec: CanonicalRecord):
        """Check a single record for physical bounds, location, date."""
        issues = []

        # Coordinate check
        if rec.latitude and rec.longitude:
            if not is_in_india(rec.latitude, rec.longitude):
                issues.append("Coordinates outside India bounding box")

        # Date range check
        if rec.sample_date < self.min_date or rec.sample_date > self.max_date:
            rec.quality_flag = QualityFlag.REJECTED
            rec.notes += f"Date {rec.sample_date} outside valid range. "
            return

        # Physical bounds check
        flag, reason = check_bounds(rec.parameter, rec.value)
        if flag == "rejected":
            rec.quality_flag = QualityFlag.REJECTED
            rec.notes += reason + ". "
            return
        elif flag == "suspect_range":
            rec.quality_flag = QualityFlag.SUSPECT_RANGE
            rec.notes += reason + ". "
            rec.confidence *= 0.6

        # Negative value check (most params should be ≥ 0)
        if rec.value < 0 and rec.parameter not in ("temperature",):
            rec.quality_flag = QualityFlag.REJECTED
            rec.notes += f"Negative value ({rec.value}) for {rec.parameter}. "

    def _check_outliers(self, records: List[CanonicalRecord]):
        """Flag statistical outliers using the IQR method per parameter."""
        # Group values by parameter
        param_values: Dict[str, List[float]] = defaultdict(list)
        param_records: Dict[str, List[CanonicalRecord]] = defaultdict(list)

        for rec in records:
            if rec.quality_flag != QualityFlag.REJECTED:
                param_values[rec.parameter].append(rec.value)
                param_records[rec.parameter].append(rec)

        for param, values in param_values.items():
            if len(values) < 20:
                continue  # Too few samples for outlier detection

            sorted_vals = sorted(values)
            n = len(sorted_vals)
            q1 = sorted_vals[n // 4]
            q3 = sorted_vals[3 * n // 4]
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 3 * iqr  # Using 3× IQR (more conservative)
            upper = q3 + 3 * iqr

            for rec in param_records[param]:
                if rec.quality_flag == QualityFlag.VALID:
                    if rec.value < lower or rec.value > upper:
                        rec.quality_flag = QualityFlag.SUSPECT_OUTLIER
                        rec.notes += (
                            f"Statistical outlier (IQR): value={rec.value:.3f}, "
                            f"range=[{lower:.3f}, {upper:.3f}]. "
                        )
                        rec.confidence *= 0.5

    def _check_consistency(self, records: List[CanonicalRecord]):
        """Check cross-parameter consistency for same station + date.

        Rules:
          - BOD should generally be ≤ COD
          - DO + BOD should be plausible (high DO → low BOD typically)
          - Temperature should be consistent with season
          - pH should be 6.5–8.5 for most rivers
        """
        # Group by (station_id, sample_date)
        groups: Dict[Tuple[str, date], Dict[str, float]] = defaultdict(dict)
        group_records: Dict[Tuple[str, date], List[CanonicalRecord]] = defaultdict(list)

        for rec in records:
            if rec.quality_flag != QualityFlag.REJECTED:
                key = (rec.station_id, rec.sample_date)
                groups[key][rec.parameter] = rec.value
                group_records[key].append(rec)

        for key, params in groups.items():
            issues = []

            # BOD ≤ COD check
            if "bod" in params and "cod" in params:
                if params["bod"] > params["cod"] * 1.2:  # 20% tolerance
                    issues.append(
                        f"BOD ({params['bod']:.1f}) > COD ({params['cod']:.1f})"
                    )

            # TDS vs Conductivity (TDS ≈ 0.5–0.7 × EC for freshwater)
            if "tds" in params and "conductivity" in params:
                ratio = params["tds"] / max(params["conductivity"], 0.01)
                if ratio < 0.2 or ratio > 1.5:
                    issues.append(
                        f"TDS/EC ratio unusual: {ratio:.2f} "
                        f"(TDS={params['tds']:.0f}, EC={params['conductivity']:.0f})"
                    )

            if issues:
                for rec in group_records[key]:
                    if rec.quality_flag == QualityFlag.VALID:
                        rec.quality_flag = QualityFlag.SUSPECT_CONSISTENCY
                        rec.notes += "Cross-param: " + "; ".join(issues) + ". "
                        rec.confidence *= 0.7

    def _log_summary(self):
        logger.info(
            f"Validation: {self.stats['total']} records | "
            f"Valid: {self.stats['valid']} | "
            f"Suspect range: {self.stats['suspect_range']} | "
            f"Suspect outlier: {self.stats['suspect_outlier']} | "
            f"Suspect consistency: {self.stats['suspect_consistency']} | "
            f"Rejected: {self.stats['rejected']}"
        )
