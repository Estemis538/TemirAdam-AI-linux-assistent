from __future__ import annotations

import logging
import re
import string
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from utils.audio import AudioCapture, audio_to_float32
from utils.config import WakeWordConfig, AssistantConfig

logger = logging.getLogger("temiradam.wakeword.detector")


@dataclass
class ListenResult:
    """Result from the wake word detector."""
    detected: bool
    transcription: str = ""
    audio: np.ndarray | None = None


class WakeWordDetector:
    """
    Wake word detector using VAD + whisper keyword spotting.
    Also returns full transcription to skip the separate STT step.
    """
    
    def __init__(self, config: WakeWordConfig, assistant_config: AssistantConfig):
        self.config = config
        self.assistant_config = assistant_config
        self._model: Optional[WhisperModel] = None
        self._is_loaded = False
        
        # Build dynamic wake words set
        self.WAKE_WORDS = {assistant_config.wake_word.lower()}
        for variant in assistant_config.wake_word_variants:
            self.WAKE_WORDS.add(variant.lower())
        
    @property
    def name(self) -> str:
        return 'wakeword'
        
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
        
    def load(self) -> None:
        """Loads the whisper base model (fast + low RAM)."""
        if self._is_loaded:
            return
            
        model_name = getattr(self.config, 'whisper_model', 'small')
        if not model_name or model_name == 'tiny':
            model_name = 'small'
            
        logger.info(f"Loading wake word detector model (whisper-{model_name})...")
        if WhisperModel is None:
            logger.error("faster_whisper is not installed.")
            raise ImportError("faster_whisper is required for WakeWordDetector")
            
        try:
            self._model = WhisperModel(model_name, device="cpu", compute_type="int8")
            self._is_loaded = True
            logger.info(f"Wake word detector model whisper-{model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load wake word detector model: {e}", exc_info=True)
            raise

    def unload(self) -> None:
        """Frees the whisper model from memory."""
        if not self._is_loaded:
            return
            
        logger.info("Unloading wake word detector model...")
        self._model = None
        self._is_loaded = False
        logger.info("Wake word detector model unloaded.")

    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Transcribe raw audio using the loaded whisper model. Reusable for STT."""
        if not self.is_loaded or self._model is None:
            return ""
        
        audio_float = audio_to_float32(audio_data)
        
        try:
            initial_prompt = (
                "Тако, открой браузер, spotify, firefox, ютуб, телеграм, код, терминал, "
                "запусти, включи, громкость, дата, время, дата."
            )
            segments, info = self._model.transcribe(
                audio_float,
                beam_size=5,
                language="ru",
                initial_prompt=initial_prompt,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                condition_on_previous_text=False,
            )
            
            parts = []
            for segment in segments:
                if segment.no_speech_prob > 0.7:
                    continue
                parts.append(segment.text)
            
            transcription = " ".join(parts).strip()
            # Collapse multiple spaces
            transcription = re.sub(r'\s+', ' ', transcription).strip()
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return ""
        
    def listen(self, audio_capture: AudioCapture) -> ListenResult:
        """
        Listen for speech, transcribe it, check for wake word.
        Returns ListenResult with transcription included (no need for separate STT).
        """
        if not self.is_loaded:
            logger.error("Wake word detector model is not loaded.")
            return ListenResult(detected=False)
            
        # 1. Wait for speech
        pre_frames = audio_capture.wait_for_speech()
        if not pre_frames:
            return ListenResult(detected=False)
            
        # 2. Record until silence
        try:
            audio_data = audio_capture.record_until_silence(
                silence_threshold_ms=800,
                max_seconds=8.0,
                pre_frames=pre_frames
            )
        except Exception as e:
            logger.error(f"Error recording audio: {e}", exc_info=True)
            return ListenResult(detected=False)
            
        if audio_data is None or len(audio_data) == 0:
            return ListenResult(detected=False)
        
        # Minimum 0.5 seconds
        if len(audio_data) < audio_capture.sample_rate * 0.5:
            return ListenResult(detected=False)
            
        # 3. Transcribe ONCE (reused for both wakeword detection AND command parsing)
        transcription = self.transcribe_audio(audio_data)
        
        if not transcription:
            return ListenResult(detected=False)
        
        # Clean for wake word matching
        clean = transcription.lower()
        clean = clean.translate(str.maketrans('', '', string.punctuation))
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        logger.info(f"🎤 Услышал: '{clean}'")
        
        # 4. Check for wake word
        for wake_word in self.WAKE_WORDS:
            if wake_word in clean:
                logger.info(f"✅ Wake word '{wake_word}' detected in: '{clean}'")
                return ListenResult(
                    detected=True,
                    transcription=transcription,
                    audio=audio_data,
                )
        
        logger.info(f"❌ Имя '{self.assistant_config.name}' не названо, игнорирую.")
        return ListenResult(detected=False, transcription=transcription, audio=audio_data)
