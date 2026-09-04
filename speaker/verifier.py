"""
Speaker verification module to authenticate users against a saved voice profile.
"""
from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

from utils.config import SpeakerVerificationConfig, AudioConfig
from utils.audio import audio_to_float32

logger = logging.getLogger("temiradam.speaker.verifier")


class SpeakerVerifier:
    """
    Verifies user identity by comparing spoken audio to a saved voice embedding.
    Implements the ManagedModel protocol.
    """
    def __init__(self, config: SpeakerVerificationConfig, audio_config: AudioConfig):
        self.config = config
        self.audio_config = audio_config
        self.encoder: VoiceEncoder | None = None
        self.saved_embedding: np.ndarray | None = None
        
        self.profile_dir = Path(getattr(config, 'profile_dir', 'data/speaker_profile'))
        self.embedding_path = self.profile_dir / 'embedding.npy'
        self.threshold = getattr(config, 'threshold', 0.75)
        
    @property
    def name(self) -> str:
        """Name of the managed model."""
        return "speaker_verifier"
        
    @property
    def is_loaded(self) -> bool:
        """Check if the model and embeddings are loaded."""
        return self.encoder is not None and self.saved_embedding is not None

    def load(self) -> None:
        """
        Load the Resemblyzer voice encoder and the saved user embedding.
        """
        if self.is_loaded:
            logger.info("SpeakerVerifier is already loaded.")
            return
            
        logger.info("Loading SpeakerVerifier model and saved profile...")
        try:
            self.encoder = VoiceEncoder()
            
            if not self.embedding_path.exists():
                logger.error(f"Saved embedding not found at {self.embedding_path}. Enrollment required.")
                raise FileNotFoundError(f"Missing speaker profile: {self.embedding_path}")
                
            self.saved_embedding = np.load(self.embedding_path)
            logger.info("SpeakerVerifier successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load SpeakerVerifier: {e}")
            self.unload()
            raise

    def unload(self) -> None:
        """
        Unload the voice encoder and embeddings to free memory.
        """
        logger.info("Unloading SpeakerVerifier...")
        self.encoder = None
        self.saved_embedding = None
        logger.info("SpeakerVerifier unloaded.")

    def verify(self, audio: np.ndarray) -> tuple[bool, float]:
        """
        Verify the speaker's identity from the provided audio sample.
        
        Args:
            audio (np.ndarray): The raw audio input (int16).
            
        Returns:
            tuple[bool, float]: A tuple of (is_authorized, similarity_score).
        """
        if not self.is_loaded or self.encoder is None or self.saved_embedding is None:
            logger.error("Attempted to verify but model is not loaded.")
            raise RuntimeError("SpeakerVerifier is not loaded. Call load() first.")
            
        try:
            # Convert int16 to float32
            audio_float = audio_to_float32(audio)
            
            # Preprocess audio for Resemblyzer
            processed_wav = preprocess_wav(audio_float, source_sr=self.audio_config.sample_rate)
            
            # Extract embedding from current audio
            current_embedding = self.encoder.embed_utterance(processed_wav)
            
            # Calculate cosine similarity
            # np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            similarity_score = float(np.dot(current_embedding, self.saved_embedding) / 
                                     (np.linalg.norm(current_embedding) * np.linalg.norm(self.saved_embedding)))
            
            is_authorized = similarity_score >= self.threshold
            
            logger.info(f"Verification complete: Score = {similarity_score:.4f}, Authorized = {is_authorized}")
            return is_authorized, similarity_score
            
        except Exception as e:
            logger.error(f"Error during speaker verification: {e}")
            raise
