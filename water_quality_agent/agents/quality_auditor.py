"""
Quality auditor agent.

Uses LLM reasoning to evaluate suspicious records that passed deterministic
validation but may still be erroneous. Provides human-readable explanations
for flagged records.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

from ..config.schema import CanonicalRecord, QualityFlag

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert water quality data auditor for Indian rivers.

You will be given a water quality measurement that has been flagged as suspicious.
Evaluate whether the measurement is likely valid, questionable, or erroneous.

Consider:
- Typical ranges for Indian rivers (Ganga, Yamuna, etc.)
- Seasonal variations (monsoon = higher turbidity/sediment, lower DO)
- Spatial context (upstream vs downstream, industrial areas)
- Cross-parameter relationships

Respond with a JSON object:
{
  "verdict": "valid" | "suspect" | "reject",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<detailed explanation in 1-2 sentences>"
}
"""


class QualityAuditorAgent:
    """LLM-powered quality auditor for suspicious records."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def _call_llm(self, prompt: str) -> str:
        if self.llm_client is None:
            try:
                from ..utils.llm_client import HuggingFaceLLM
                self.llm_client = HuggingFaceLLM(mode="api")
            except Exception as e:
                logger.error(f"Failed to create HF client: {e}")
                return '{"verdict": "suspect", "confidence": 0.5, "reasoning": "LLM unavailable"}'

        if hasattr(self.llm_client, "invoke"):
            response = self.llm_client.invoke([
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            return response.content if hasattr(response, "content") else str(response)

        if hasattr(self.llm_client, "chat"):
            return self.llm_client.chat(_SYSTEM_PROMPT, prompt)

        return '{"verdict": "suspect", "confidence": 0.5, "reasoning": "No LLM client"}'

    def audit_record(self, rec: CanonicalRecord) -> Tuple[str, float, str]:
        """Audit a single suspicious record.

        Returns:
            (verdict, confidence, reasoning)
        """
        prompt = (
            f"Evaluate this water quality measurement:\n"
            f"  Station: {rec.station_name} ({rec.river}, {rec.state})\n"
            f"  Coordinates: ({rec.latitude:.4f}, {rec.longitude:.4f})\n"
            f"  Date: {rec.sample_date}\n"
            f"  Parameter: {rec.parameter}\n"
            f"  Value: {rec.value} {rec.unit}\n"
            f"  Current flag: {rec.quality_flag}\n"
            f"  Notes: {rec.notes}\n"
        )

        try:
            response = self._call_llm(prompt)
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(text)
            return (
                result.get("verdict", "suspect"),
                float(result.get("confidence", 0.5)),
                result.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning(f"Audit failed for record: {e}")
            return "suspect", 0.5, f"Audit error: {e}"

    def audit_batch(
        self,
        records: List[CanonicalRecord],
        max_records: int = 100,
    ) -> List[CanonicalRecord]:
        """Audit suspicious records and update their flags.

        Only audits records with suspect flags, up to max_records.
        """
        suspect = [
            r for r in records
            if r.quality_flag in (
                QualityFlag.SUSPECT_RANGE,
                QualityFlag.SUSPECT_OUTLIER,
                QualityFlag.SUSPECT_CONSISTENCY,
            )
        ]

        if not suspect:
            logger.info("No suspect records to audit.")
            return records

        logger.info(f"Auditing {min(len(suspect), max_records)} suspect records...")

        for rec in suspect[:max_records]:
            verdict, conf, reasoning = self.audit_record(rec)

            if verdict == "valid":
                rec.quality_flag = QualityFlag.VALID
                rec.confidence = max(rec.confidence, conf)
                rec.notes += f"LLM audit: VALID — {reasoning} "
            elif verdict == "reject":
                rec.quality_flag = QualityFlag.REJECTED
                rec.confidence = min(rec.confidence, 1 - conf)
                rec.notes += f"LLM audit: REJECTED — {reasoning} "
            else:
                rec.notes += f"LLM audit: SUSPECT — {reasoning} "

        return records
