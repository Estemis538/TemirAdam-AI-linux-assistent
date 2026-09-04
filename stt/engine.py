"""
Speech-to-Text engine using faster-whisper.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from utils.config import STTConfig
from utils.audio import audio_to_float32

logger = logging.getLogger("temiradam.stt.engine")


@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    segments: list[dict[str, Any]]


class STTEngine:
    """
    ManagedModel wrapper for faster-whisper STT engine.
    """

    def __init__(self, config: STTConfig) -> None:
        self.config = config
        self._model: Any | None = None

    @property
    def name(self) -> str:
        return "stt"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.is_loaded:
            logger.debug("STT model is already loaded.")
            return

        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed.")

        logger.info(
            "Loading STT model: %s on %s (compute_type: %s)",
            self.config.model,
            self.config.device,
            self.config.compute_type,
        )

        try:
            self._model = WhisperModel(
                self.config.model,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            logger.info("STT model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load STT model: %s", e)
            raise

    def unload(self) -> None:
        if not self.is_loaded:
            return

        logger.info("Unloading STT model...")
        self._model = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        gc.collect()
        logger.info("STT model unloaded and VRAM freed.")

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if not self.is_loaded or self._model is None:
            raise RuntimeError("STT model is not loaded. Call load() first.")

        # Convert audio from int16 to float32
        audio_float32 = audio_to_float32(audio)

        logger.debug("Transcribing audio of length %d", len(audio_float32))

        # Transcribe with auto-detect language
        segments_gen, info = self._model.transcribe(
            audio_float32,
            beam_size=self.config.beam_size,
        )

        segments = []
        texts = []

        for segment in segments_gen:
            texts.append(segment.text)
            segments.append(
                {
                    "id": segment.id,
                    "seek": segment.seek,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "tokens": segment.tokens,
                    "temperature": segment.temperature,
                    "avg_logprob": segment.avg_logprob,
                    "compression_ratio": segment.compression_ratio,
                    "no_speech_prob": segment.no_speech_prob,
                }
            )

        full_text = "".join(texts).strip()
        
        # Calculate average log probability from segments
        if segments:
            confidence = sum(s["avg_logprob"] for s in segments) / len(segments)
        else:
            confidence = 0.0

        result = TranscriptionResult(
            text=full_text,
            language=info.language,
            confidence=confidence,
            segments=segments,
        )

        logger.info(
            "Transcription completed: language='%s', confidence=%.4f, text='%s'",
            result.language,
            result.confidence,
            result.text,
        )

        return result
