from __future__ import annotations

import logging
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
    Wake word detector using VAD + whisper-tiny keyword spotting.
    Listens for the wake word 'Темірадам' or its variants.
    """
    
    WAKE_WORDS = {'темірадам', 'темір адам', 'темирадам', 'темир адам'}
    
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
        """Loads the whisper tiny model into memory."""
        if self._is_loaded:
            return
            
        logger.info("Loading wake word detector model (whisper-tiny)...")
        if WhisperModel is None:
            logger.error("faster_whisper is not installed.")
            raise ImportError("faster_whisper is required for WakeWordDetector")
            
        try:
            # Load tiny model with standard settings
            self._model = WhisperModel("tiny", device="cpu", compute_type="int8")
            self._is_loaded = True
            logger.info("Wake word detector model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load wake word detector model: {e}", exc_info=True)
            raise

    def unload(self) -> None:
        """Frees the whisper tiny model from memory."""
        if not self._is_loaded:
            return
            
        logger.info("Unloading wake word detector model...")
        self._model = None
        self._is_loaded = False
        logger.info("Wake word detector model unloaded.")
        
    def listen(self, audio_capture: AudioCapture) -> tuple[bool, np.ndarray | None]:
        """
        Continuously listens to microphone using VAD, records a chunk when speech is detected,
        and uses whisper-tiny to detect the wake word.
        
        Args:
            audio_capture: The audio capture instance to use.
            
        Returns:
            A tuple containing a boolean indicating if the wake word was detected,
            and the recorded audio segment as a numpy array (or None if no audio).
        """
        if not self.is_loaded:
            logger.error("Wake word detector model is not loaded. Please call load() first.")
            return False, None
            
        logger.debug("Waiting for speech...")
        # 1. Wait for speech to start using VAD
        has_speech = audio_capture.wait_for_speech()
        if not has_speech:
            return False, None
            
        logger.debug("Speech detected, recording a short segment...")
        # 2. Record a short segment (2-3 seconds)
        try:
            # Try to use record if it exists, otherwise read
            if hasattr(audio_capture, 'record'):
                audio_data = audio_capture.record(duration=3.0)
            else:
                # Fallback, read enough frames for 3 seconds assuming 16kHz sample rate
                audio_data = audio_capture.read(16000 * 3)
        except Exception as e:
            logger.error(f"Error recording audio: {e}", exc_info=True)
            return False, None
            
        if audio_data is None or len(audio_data) == 0:
            return False, None
            
        logger.debug("Transcribing recorded segment for wake word detection...")
        # 3. Run whisper tiny on it
        audio_float = audio_to_float32(audio_data)
        
        try:
            segments, _ = self._model.transcribe(audio_float, beam_size=1)
            
            transcription = " ".join([segment.text for segment in segments]).lower().strip()
            
            # Clean up punctuation that might be present
            transcription = transcription.translate(str.maketrans('', '', string.punctuation))
            
            logger.debug(f"Wake word transcription: '{transcription}'")
            
            # 4. Check for wake word in transcription
            for wake_word in self.WAKE_WORDS:
                if wake_word in transcription:
                    logger.info(f"Wake word '{wake_word}' detected!")
                    # 5. Return (detected: bool, audio_segment)
                    return True, audio_data
                    
        except Exception as e:
            logger.error(f"Error during wake word transcription: {e}", exc_info=True)
            
        return False, audio_data
