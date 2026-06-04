"""
Hugging Face LLM client for the water quality agent.

Supports two modes:
  1. **Local inference** — Downloads and runs a model locally via `transformers`
     (needs GPU for large models, CPU works for small ones like Phi-3-mini)
  2. **HF Inference API** — Free-tier API calls to Hugging Face servers
     (needs a HF token, no GPU required)

Recommended models (open-source, no Llama):
  - "microsoft/Phi-3.5-mini-instruct"   — 3.8B params, fast, good reasoning
  - "mistralai/Mistral-7B-Instruct-v0.3" — 7B params, strong general ability
  - "google/gemma-2-2b-it"              — 2B params, lightweight
  - "Qwen/Qwen2.5-7B-Instruct"         — 7B params, excellent at structured output

Usage:
  # API mode (recommended — no GPU needed)
  client = HuggingFaceLLM(mode="api", model_name="mistralai/Mistral-7B-Instruct-v0.3")

  # Local mode (needs sufficient RAM/VRAM)
  client = HuggingFaceLLM(mode="local", model_name="microsoft/Phi-3.5-mini-instruct")
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Default model — Phi-3.5-mini is small, fast, and great at structured output
DEFAULT_MODEL = "microsoft/Phi-3.5-mini-instruct"


class HuggingFaceLLM:
    """Unified Hugging Face LLM client supporting local and API inference."""

    def __init__(
        self,
        mode: str = "api",
        model_name: str = DEFAULT_MODEL,
        hf_token: Optional[str] = None,
        max_new_tokens: int = 300,
        temperature: float = 0.1,
    ):
        """
        Args:
            mode: "api" for HF Inference API, "local" for local transformers
            model_name: Hugging Face model ID
            hf_token: Hugging Face API token (reads HF_TOKEN env var if not provided)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.mode = mode
        self.model_name = model_name
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._pipeline = None  # Lazy-loaded for local mode
        self._api_client = None  # Lazy-loaded for API mode

        logger.info(f"HuggingFace LLM: mode={mode}, model={model_name}")

    # ── Public interface (compatible with agent expectations) ──────────────

    def invoke(self, messages: list[dict]) -> "LLMResponse":
        """LangChain-compatible invoke interface.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}

        Returns:
            Object with .content attribute containing the response text.
        """
        if self.mode == "api":
            text = self._call_api(messages)
        else:
            text = self._call_local(messages)

        return LLMResponse(content=text)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Simple chat interface."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.invoke(messages).content

    # ── API mode ──────────────────────────────────────────────────────────

    def _call_api(self, messages: list[dict]) -> str:
        """Call the HF Inference API."""
        if self._api_client is None:
            self._init_api_client()

        try:
            response = self._api_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"HF API error: {e}")
            return '{"canonical_name": null, "confidence": 0, "reasoning": "API error"}'

    def _init_api_client(self):
        """Initialize the HF Inference API client."""
        try:
            from huggingface_hub import InferenceClient
            self._api_client = InferenceClient(
                provider="hf-inference",
                api_key=self.hf_token,
            )
            logger.info(f"HF Inference API client initialized: {self.model_name}")
        except ImportError:
            raise ImportError(
                "huggingface_hub required for API mode: pip install huggingface-hub"
            )

    # ── Local mode ────────────────────────────────────────────────────────

    def _call_local(self, messages: list[dict]) -> str:
        """Run inference locally using transformers pipeline."""
        if self._pipeline is None:
            self._init_local_pipeline()

        try:
            outputs = self._pipeline(
                messages,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                return_full_text=False,
            )
            return outputs[0]["generated_text"]
        except Exception as e:
            logger.error(f"Local inference error: {e}")
            return '{"canonical_name": null, "confidence": 0, "reasoning": "Local inference error"}'

    def _init_local_pipeline(self):
        """Load model locally via transformers."""
        try:
            import torch
            from transformers import pipeline as hf_pipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            logger.info(
                f"Loading {self.model_name} locally on {device} ({dtype})..."
            )

            self._pipeline = hf_pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                token=self.hf_token if self.hf_token else None,
            )
            logger.info(f"Model loaded: {self.model_name}")

        except ImportError:
            raise ImportError(
                "transformers and torch required for local mode: "
                "pip install transformers torch"
            )


class LLMResponse:
    """Simple response object with .content attribute."""

    def __init__(self, content: str):
        self.content = content

    def __str__(self):
        return self.content
