"""
Темірадам — Ollama API client.

HTTP client for the Ollama REST API.
Supports generation, chat, model loading/unloading.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger("temiradam.llm.ollama")


class OllamaClient:
    """
    Client for the Ollama REST API.

    Handles model interaction, including explicit load/unload
    via the keep_alive parameter.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[dict[str, Any]]:
        """List all available models."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            return resp.json().get("models", [])
        except Exception:
            logger.exception("Failed to list models")
            return []

    def generate(
        self,
        prompt: str,
        model: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 256,
        keep_alive: str = "5m",
        raw: bool = False,
    ) -> str:
        """
        Generate text using the /api/generate endpoint.

        Args:
            prompt: User prompt.
            model: Model name (e.g., 'qwen3:4b').
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            keep_alive: How long to keep model loaded ('0' = unload immediately).
            raw: If True, don't apply chat template.

        Returns:
            Generated text.
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "keep_alive": keep_alive,
        }
        if system:
            payload["system"] = system
        if raw:
            payload["raw"] = True

        logger.debug("Ollama generate: model=%s, prompt_len=%d", model, len(prompt))

        try:
            resp = self._client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            response_text = data.get("response", "")
            logger.debug(
                "Ollama response: %d chars, eval_duration=%s",
                len(response_text),
                data.get("eval_duration"),
            )
            return response_text
        except httpx.TimeoutException:
            logger.error("Ollama request timed out (model=%s)", model)
            raise
        except Exception:
            logger.exception("Ollama generate failed")
            raise

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 256,
        keep_alive: str = "5m",
    ) -> str:
        """
        Chat using the /api/chat endpoint.

        Args:
            messages: List of {'role': 'system'|'user'|'assistant', 'content': '...'}.
            model: Model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            keep_alive: Model keep-alive duration.

        Returns:
            Assistant's response text.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "keep_alive": keep_alive,
        }

        logger.debug("Ollama chat: model=%s, messages=%d", model, len(messages))

        try:
            resp = self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            message = data.get("message", {})
            return message.get("content", "")
        except httpx.TimeoutException:
            logger.error("Ollama chat timed out (model=%s)", model)
            raise
        except Exception:
            logger.exception("Ollama chat failed")
            raise

    def unload_model(self, model: str) -> None:
        """
        Explicitly unload a model from memory.

        Sends a generate request with keep_alive=0 to trigger immediate unload.
        """
        logger.info("Unloading model from Ollama: %s", model)
        try:
            self._client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": "",
                    "keep_alive": "0",
                    "stream": False,
                },
                timeout=10.0,
            )
            logger.info("Model unloaded: %s", model)
        except Exception:
            logger.exception("Failed to unload model: %s", model)

    def pull_model(self, model: str) -> bool:
        """Pull a model if not already available."""
        logger.info("Pulling model: %s", model)
        try:
            resp = self._client.post(
                f"{self.base_url}/api/pull",
                json={"name": model, "stream": False},
                timeout=600.0,  # models can take a while to download
            )
            resp.raise_for_status()
            return True
        except Exception:
            logger.exception("Failed to pull model: %s", model)
            return False

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
