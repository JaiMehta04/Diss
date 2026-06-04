"""
Conflict resolver agent.

When multiple sources provide conflicting values for the same
station + date + parameter, this agent decides which to keep.
"""

import logging
from collections import defaultdict
from datetime import date
from typing import Dict, List, Tuple

from ..config.schema import CanonicalRecord, QualityFlag

logger = logging.getLogger(__name__)

# Source trust ranking (higher = more trusted)
SOURCE_TRUST = {
    "cpcb":        1.0,   # Central authority — highest trust
    "cwc":         0.95,  # Government hydrology data
    "gemstat":     0.85,  # UN-curated but international
    "data_gov_in": 0.80,  # Government open data
    "state_pcb":   0.75,  # State-level boards
}

# Maximum allowable relative difference before flagging as conflict
CONFLICT_THRESHOLD = 0.20  # 20% relative difference


class ConflictResolverAgent:
    """Resolve conflicting measurements from multiple sources."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.stats = {
            "groups_checked": 0,
            "conflicts_found": 0,
            "resolved_by_trust": 0,
            "resolved_by_llm": 0,
        }

    def resolve(self, records: List[CanonicalRecord]) -> List[CanonicalRecord]:
        """Find and resolve conflicts in records.

        Groups records by (station_id, sample_date, parameter) and
        checks for conflicting values from different sources.
        """
        # Group by (station_id, date, parameter)
        groups: Dict[Tuple[str, date, str], List[CanonicalRecord]] = defaultdict(list)

        for rec in records:
            if rec.quality_flag != QualityFlag.REJECTED:
                key = (rec.station_id, rec.sample_date, rec.parameter)
                groups[key].append(rec)

        resolved_records = []

        for key, group in groups.items():
            self.stats["groups_checked"] += 1

            if len(group) == 1:
                resolved_records.append(group[0])
                continue

            # Multiple sources for same station + date + parameter
            # Check if values conflict
            sources = {r.source for r in group}
            if len(sources) == 1:
                # Same source, multiple readings (e.g. hourly) — average them
                values = [r.value for r in group]
                avg_val = sum(values) / len(values)
                kept = group[0]
                kept.value = round(avg_val, 4)
                kept.notes += (
                    f"Averaged {len(group)} same-source readings "
                    f"(range: {min(values):.3f}–{max(values):.3f}). "
                )
                resolved_records.append(kept)
                continue

            # Different sources — check for conflict
            values = [r.value for r in group]
            mean_val = sum(values) / len(values)

            if mean_val == 0:
                # Can't compute relative diff meaningfully
                resolved_records.extend(group)
                continue

            max_rel_diff = max(abs(v - mean_val) / abs(mean_val) for v in values)

            if max_rel_diff <= CONFLICT_THRESHOLD:
                # Values agree within tolerance — keep highest-trust source
                best = max(group, key=lambda r: SOURCE_TRUST.get(r.source, 0.5))
                best.notes += (
                    f"Multiple sources agree (rel_diff={max_rel_diff:.1%}). "
                    f"Kept {best.source} value. "
                )
                resolved_records.append(best)
                for other in group:
                    if other is not best:
                        other.quality_flag = QualityFlag.SUSPECT_DUPLICATE
                        other.notes += f"Superseded by {best.source}. "
                        resolved_records.append(other)
            else:
                # Conflict detected
                self.stats["conflicts_found"] += 1
                resolved_records.extend(
                    self._resolve_conflict(key, group)
                )

        self._log_summary()
        return resolved_records

    def _resolve_conflict(
        self,
        key: Tuple[str, date, str],
        group: List[CanonicalRecord],
    ) -> List[CanonicalRecord]:
        """Resolve a conflict between sources.

        Strategy:
          1. Trust-based: pick highest-trust source
          2. If LLM available, ask for reasoning
        """
        station_id, sample_date, param = key
        values_desc = ", ".join(
            f"{r.source}={r.value:.3f}" for r in group
        )
        logger.warning(
            f"Conflict at {station_id}/{sample_date}/{param}: {values_desc}"
        )

        # Sort by trust score
        sorted_group = sorted(
            group,
            key=lambda r: SOURCE_TRUST.get(r.source, 0.5),
            reverse=True,
        )

        # Keep highest-trust source
        winner = sorted_group[0]
        winner.notes += (
            f"Conflict resolved: kept {winner.source} "
            f"(trust={SOURCE_TRUST.get(winner.source, 0.5):.2f}). "
            f"Alternatives: {values_desc}. "
        )
        winner.confidence *= 0.8  # Lower confidence due to conflict

        self.stats["resolved_by_trust"] += 1

        for loser in sorted_group[1:]:
            loser.quality_flag = QualityFlag.SUSPECT_CONSISTENCY
            loser.notes += (
                f"Conflict: value differs from {winner.source} "
                f"({winner.value:.3f} vs {loser.value:.3f}). "
            )
            loser.confidence *= 0.5

        return sorted_group

    def _log_summary(self):
        logger.info(
            f"Conflict resolution: {self.stats['groups_checked']} groups | "
            f"Conflicts: {self.stats['conflicts_found']} | "
            f"Resolved by trust: {self.stats['resolved_by_trust']} | "
            f"Resolved by LLM: {self.stats['resolved_by_llm']}"
        )
