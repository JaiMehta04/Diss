"""
LLM-powered schema mapper agent.

Handles parameter names that the deterministic regex mapper couldn't resolve.
Uses an LLM to map unknown parameter names to canonical names, with
reasoning and confidence scores.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# System prompt for the schema mapper
_SYSTEM_PROMPT = """You are an expert water quality data scientist.

Your task is to map raw water quality parameter names from various Indian data
sources to a set of canonical parameter names.

Canonical parameters (use EXACTLY these names):
  temperature, turbidity, conductivity, tds, tss, colour, water_level,
  ph, do, bod, cod, toc, alkalinity, hardness,
  nitrate, nitrite, ammonia, ammoniacal_nitrogen, total_nitrogen,
  phosphate, total_phosphorus,
  chloride, fluoride, sulphate, sodium, potassium, calcium, magnesium, iron,
  arsenic, cadmium, chromium, copper, lead, mercury, nickel, zinc, manganese,
  boron, selenium,
  fecal_coliform, total_coliform, e_coli,
  oil_grease, phenol, detergent, pesticides,
  discharge, velocity, depth, sediment_load

For each raw parameter name, respond with a JSON object:
{
  "canonical_name": "<name from the list above, or null if unmappable>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<brief explanation>"
}

If the parameter doesn't map to any canonical name, set canonical_name to null.
Consider abbreviations, Hindi transliterations, and common misspellings.
"""


class SchemaMapperAgent:
    """LLM-based schema mapper for unresolved parameter names."""

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: An LLM client with a .chat() or .invoke() method.
                        If None, falls back to OpenAI API via environment variable.
        """
        self.llm_client = llm_client
        self._cache: Dict[str, Tuple[Optional[str], float]] = {}

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return the response text."""
        if self.llm_client is None:
            return self._call_default(prompt)

        # HuggingFaceLLM or LangChain-style client
        if hasattr(self.llm_client, "invoke"):
            response = self.llm_client.invoke([
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            return response.content if hasattr(response, "content") else str(response)

        # Generic .chat() interface
        if hasattr(self.llm_client, "chat"):
            return self.llm_client.chat(_SYSTEM_PROMPT, prompt)

        raise ValueError("LLM client must have .invoke() or .chat() method")

    def _call_default(self, prompt: str) -> str:
        """Create a default HuggingFace LLM client and call it."""
        try:
            from ..utils.llm_client import HuggingFaceLLM
            self.llm_client = HuggingFaceLLM(mode="api")
            return self._call_llm(prompt)
        except Exception as e:
            logger.error(f"HuggingFace LLM error: {e}")
            return '{"canonical_name": null, "confidence": 0, "reasoning": "LLM unavailable"}'

    def map_parameter(self, raw_name: str) -> Tuple[Optional[str], float, str]:
        """Map a single raw parameter name to canonical form.

        Returns:
            (canonical_name, confidence, reasoning)
        """
        if raw_name in self._cache:
            canon, conf = self._cache[raw_name]
            return canon, conf, "cached"

        prompt = f'Map this water quality parameter name to canonical form: "{raw_name}"'

        try:
            response = self._call_llm(prompt)
            # Parse JSON from response
            # Handle case where LLM wraps in ```json ... ```
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(text)

            canon = result.get("canonical_name")
            conf = float(result.get("confidence", 0.5))
            reason = result.get("reasoning", "")

            self._cache[raw_name] = (canon, conf)
            return canon, conf, reason

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response for '{raw_name}': {e}")
            return None, 0.0, f"Parse error: {e}"

    def map_batch(
        self, raw_names: List[str]
    ) -> Dict[str, Tuple[Optional[str], float, str]]:
        """Map multiple raw parameter names efficiently.

        Batches unique names and calls LLM once per unique name.
        """
        unique = sorted(set(raw_names))
        results = {}

        for name in unique:
            canon, conf, reason = self.map_parameter(name)
            results[name] = (canon, conf, reason)
            if canon:
                logger.info(
                    f"LLM mapped '{name}' → '{canon}' (conf={conf:.2f}): {reason}"
                )
            else:
                logger.warning(f"LLM could not map '{name}': {reason}")

        return results
