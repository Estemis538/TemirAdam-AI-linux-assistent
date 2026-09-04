"""
Темірадам — Model lifecycle manager.

Handles lazy loading and explicit unloading of heavy models
(STT, Speaker Verification, LLM, TTS) to minimize RAM/VRAM usage.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

logger = logging.getLogger("temiradam.core.lifecycle")


class ManagedModel(Protocol):
    """Protocol that any managed model must implement."""

    @property
    def is_loaded(self) -> bool: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    @property
    def name(self) -> str: ...


class ModelLifecycle:
    """
    Manages the lifecycle of heavy models.

    Ensures models are loaded only when needed and unloaded
    when no longer required to conserve RAM/VRAM.
    """

    def __init__(self) -> None:
        self._models: dict[str, ManagedModel] = {}
        self._load_times: dict[str, float] = {}

    def register(self, model: ManagedModel) -> None:
        """Register a model for lifecycle management."""
        self._models[model.name] = model
        logger.debug("Registered model: %s", model.name)

    def get(self, name: str) -> ManagedModel | None:
        """Get a registered model by name."""
        return self._models.get(name)

    def ensure_loaded(self, name: str) -> ManagedModel:
        """
        Ensure a model is loaded. Load it if not already loaded.

        Returns the model instance.
        Raises KeyError if model is not registered.
        """
        model = self._models.get(name)
        if model is None:
            raise KeyError(f"Model not registered: {name}")

        if not model.is_loaded:
            logger.info("Loading model: %s", name)
            start = time.monotonic()
            model.load()
            elapsed = time.monotonic() - start
            self._load_times[name] = elapsed
            logger.info("Model loaded: %s (%.2fs)", name, elapsed)

        return model

    def unload(self, name: str) -> None:
        """Unload a specific model if it is loaded."""
        model = self._models.get(name)
        if model is None:
            return

        if model.is_loaded:
            logger.info("Unloading model: %s", name)
            model.unload()
            logger.info("Model unloaded: %s", name)

    def unload_all(self, *, exclude: set[str] | None = None) -> None:
        """
        Unload all loaded models.

        Args:
            exclude: Set of model names to keep loaded (e.g., wake word detector).
        """
        exclude = exclude or set()
        for name, model in self._models.items():
            if name in exclude:
                continue
            if model.is_loaded:
                logger.info("Unloading model: %s", name)
                try:
                    model.unload()
                    logger.info("Model unloaded: %s", name)
                except Exception:
                    logger.exception("Failed to unload model: %s", name)

    def status(self) -> dict[str, dict[str, Any]]:
        """Get status of all registered models."""
        result = {}
        for name, model in self._models.items():
            result[name] = {
                "loaded": model.is_loaded,
                "load_time": self._load_times.get(name),
            }
        return result

    @property
    def loaded_models(self) -> list[str]:
        """List of currently loaded model names."""
        return [name for name, model in self._models.items() if model.is_loaded]
