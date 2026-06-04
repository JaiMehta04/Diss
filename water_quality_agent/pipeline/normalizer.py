"""
Normalization pipeline.

Converts source parameter names and units to canonical form.
Applies deterministic rules first, then flags unresolved records
for the LLM schema-mapper agent.
"""

import logging
from typing import List, Tuple

from ..config.parameters import CANONICAL_PARAMETERS, normalize_parameter_name
from ..config.units import normalize_unit, convert_value
from ..config.schema import CanonicalRecord

logger = logging.getLogger(__name__)


class Normalizer:
    """Normalize parameter names and units to canonical form."""

    def __init__(self):
        self.stats = {
            "total": 0,
            "param_resolved": 0,
            "param_unresolved": 0,
            "unit_converted": 0,
            "unit_already_canonical": 0,
            "unit_conversion_failed": 0,
        }
        self.unresolved_params: list[str] = []

    def normalize(self, records: List[CanonicalRecord]) -> List[CanonicalRecord]:
        """Normalize a batch of records in-place and return them."""
        normalized = []

        for rec in records:
            self.stats["total"] += 1

            # ── Parameter name normalization ──────────────────────────────
            if rec.parameter not in CANONICAL_PARAMETERS:
                # Try regex matching
                resolved = normalize_parameter_name(rec.source_parameter)
                if resolved:
                    rec.parameter = resolved
                    self.stats["param_resolved"] += 1
                else:
                    self.stats["param_unresolved"] += 1
                    self.unresolved_params.append(rec.source_parameter)
                    continue  # Skip record — LLM agent handles later

            # ── Unit normalization ────────────────────────────────────────
            target_unit = CANONICAL_PARAMETERS[rec.parameter][1]

            if not target_unit:
                # Dimensionless parameter (pH, etc.)
                rec.unit = ""
                self.stats["unit_already_canonical"] += 1
            elif not rec.unit:
                # Unit unknown from source — assume canonical
                rec.unit = target_unit
                self.stats["unit_already_canonical"] += 1
            else:
                # Normalize the source unit string
                source_unit_clean = normalize_unit(rec.unit)

                if source_unit_clean == target_unit:
                    rec.unit = target_unit
                    self.stats["unit_already_canonical"] += 1
                else:
                    try:
                        converted_val, confidence = convert_value(
                            rec.value, source_unit_clean, target_unit
                        )
                        rec.source_value = rec.value
                        rec.source_unit = source_unit_clean
                        rec.value = converted_val
                        rec.unit = target_unit
                        rec.confidence = min(rec.confidence, confidence)
                        self.stats["unit_converted"] += 1
                    except ValueError:
                        # Can't convert — keep original, flag it
                        rec.notes += f"Unit conversion failed: {source_unit_clean} → {target_unit}. "
                        rec.confidence *= 0.7
                        self.stats["unit_conversion_failed"] += 1

            normalized.append(rec)

        self._log_summary()
        return normalized

    def _log_summary(self):
        logger.info(
            f"Normalization: {self.stats['total']} records processed | "
            f"Params resolved: {self.stats['param_resolved']} | "
            f"Unresolved: {self.stats['param_unresolved']} | "
            f"Units converted: {self.stats['unit_converted']} | "
            f"Conversion failures: {self.stats['unit_conversion_failed']}"
        )
        if self.unresolved_params:
            unique = sorted(set(self.unresolved_params))
            logger.warning(
                f"Unresolved parameters ({len(unique)} unique): "
                f"{unique[:20]}{'...' if len(unique) > 20 else ''}"
            )
