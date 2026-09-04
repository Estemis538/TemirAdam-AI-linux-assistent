from __future__ import annotations

import logging
import re
import string
import numpy as np
from typing import Optional, Tuple

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from utils.audio import AudioCapture, audio_to_float32
from utils.config import WakeWordConfig

logger = logging.getLogger("temiradam.wakeword.detector")

class WakeWordDetector:
    """
    Wake word detector using VAD + whisper keyword spotting.
    Listens for the wake word 'Темірадам' or its variants.
    """
    
    WAKE_WORDS = {
        'темірадам', 'темір адам', 'темирадам', 'темир адам',
        'тимерадам', 'тимер адам', 'тимирадам', 'тимир адам',
        'темерадам', 'темер адам', 'демирадам', 'демир адам',
        'демірадам', 'демір адам', 'тәмірадам', 'тәмір адам',
        'temiradam', 'temir adam', 'timer adam', 'timeradam',
        'всем радам', 'тебе радам', 'темир', 'тимер',
    }
    
    def __init__(self, config: WakeWordConfig):
        self.config = config
        self._model: Optional[WhisperModel] = None
        self._is_loaded = False
        
    @property
    def name(self) -> str:
        return 'wakeword'
        
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
        
    def load(self) -> None:
        """Loads the whisper small model into memory."""
        if self._is_loaded:
            return
            
        logger.info("Loading wake word detector model (whisper-small)...")
        if WhisperModel is None:
            logger.error("faster_whisper is not installed.")
            raise ImportError("faster_whisper is required for WakeWordDetector")
            
        try:
            self._model = WhisperModel("small", device="cpu", compute_type="int8")
            self._is_loaded = True
            logger.info("Wake word detector model loaded successfully.")
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
        
    def listen(self, audio_capture: AudioCapture) -> tuple[bool, np.ndarray | None]:
        """
        Continuously listens to microphone using VAD, records a chunk when speech is detected,
        and uses whisper to detect the wake word.
        """
        if not self.is_loaded:
            logger.error("Wake word detector model is not loaded. Please call load() first.")
            return False, None
            
        logger.debug("Waiting for speech...")
        # 1. Wait for speech to start using VAD
        pre_frames = audio_capture.wait_for_speech()
        if not pre_frames:
            return False, None
            
        logger.debug("Speech detected, recording until silence...")
        # 2. Record until silence to capture the full phrase including wake word
        try:
            audio_data = audio_capture.record_until_silence(
                silence_threshold_ms=800,
                max_seconds=8.0,
                pre_frames=pre_frames
            )
        except Exception as e:
            logger.error(f"Error recording audio: {e}", exc_info=True)
            return False, None
            
        if audio_data is None or len(audio_data) == 0:
            return False, None
        
        # Check minimum audio length (at least 0.5 seconds of audio)
        if len(audio_data) < audio_capture.sample_rate * 0.5:
            logger.debug("Audio too short, skipping")
            return False, None
            
        logger.debug("Transcribing recorded segment for wake word detection...")
        audio_float = audio_to_float32(audio_data)
        
        try:
            segments, info = self._model.transcribe(
                audio_float, 
                beam_size=3,
                initial_prompt="Темірадам",
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                condition_on_previous_text=False,
            )
            
            parts = []
            for segment in segments:
                # Skip segments with very high no_speech_prob (hallucination filter)
                if segment.no_speech_prob > 0.7:
                    logger.debug(f"Skipping hallucinated segment: '{segment.text}' (no_speech={segment.no_speech_prob:.2f})")
                    continue
                parts.append(segment.text)
            
            transcription = " ".join(parts).lower().strip()
            
            # Clean up punctuation
            transcription = transcription.translate(str.maketrans('', '', string.punctuation))
            # Collapse multiple spaces
            transcription = re.sub(r'\s+', ' ', transcription).strip()
            
            if not transcription:
                logger.debug("Empty transcription after filtering")
                return False, None
            
            logger.info(f"🎤 Услышал: '{transcription}'")
            
            # 4. Check for wake word in transcription
            for wake_word in self.WAKE_WORDS:
                if wake_word in transcription:
                    logger.info(f"✅ Wake word '{wake_word}' detected in: '{transcription}'")
                    return True, audio_data
            
            logger.info("❌ Имя 'Темірадам' не названо, игнорирую фразу.")
                    
        except Exception as e:
            logger.error(f"Error during wake word transcription: {e}", exc_info=True)
            
        return False, audio_data
