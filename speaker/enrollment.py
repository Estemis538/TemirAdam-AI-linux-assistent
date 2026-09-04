"""
Speaker enrollment module for creating user voice profiles.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav

from utils.config import SpeakerVerificationConfig, AudioConfig
from utils.audio import AudioCapture, audio_to_float32

logger = logging.getLogger("temiradam.speaker.enrollment")


class SpeakerEnrollment:
    """
    Handles interactive voice enrollment to create a robust speaker profile.
    """
    def __init__(self, config: SpeakerVerificationConfig, audio_config: AudioConfig):
        self.config = config
        self.audio_config = audio_config
        self.encoder = VoiceEncoder()
        
        # Configuration for enrollment
        self.num_samples = getattr(config, 'enrollment_samples', 3)
        self.sample_duration = getattr(config, 'enrollment_duration', 5.0)
        
        self.profile_dir = Path(getattr(config, 'profile_dir', 'data/speaker_profile'))
        self.embedding_path = self.profile_dir / 'embedding.npy'

    def enroll(self) -> Path:
        """
        Interactively enroll the user by recording multiple samples, 
        averaging their embeddings, and saving the final profile.
        
        Returns:
            Path to the saved embedding.
        """
        logger.info("Starting speaker enrollment process...")
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        
        embeddings = []
        for i in range(self.num_samples):
            logger.info(f"Recording sample {i + 1}/{self.num_samples} for {self.sample_duration} seconds.")
            audio = self._record_sample(i)
            
            # Save the WAV file
            wav_path = self.profile_dir / f"sample_{i + 1}.wav"
            sf.write(str(wav_path), audio, self.audio_config.sample_rate)
            logger.info(f"Saved sample audio to {wav_path}")
            
            # Extract and store embedding
            embedding = self._extract_embedding(audio)
            embeddings.append(embedding)
            
            if i < self.num_samples - 1:
                logger.info("Waiting 2 seconds before the next recording...")
                time.sleep(2.0)
                
        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Normalize the averaged embedding
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        # Save to disk
        np.save(self.embedding_path, avg_embedding)
        logger.info(f"Successfully saved robust speaker embedding to {self.embedding_path}")
        
        return self.embedding_path

    def _record_sample(self, index: int) -> np.ndarray:
        """
        Record a single audio sample from the microphone.
        """
        capture = AudioCapture(self.audio_config)
        logger.info(f"Please speak now (Sample {index + 1})...")
        try:
            # Assumes AudioCapture has a record method that blocks for the given duration
            audio_data = capture.record(duration=self.sample_duration)
        except AttributeError:
            logger.warning("AudioCapture.record not found, using manual read loop.")
            # Fallback if record is not implemented
            capture.start()
            frames = []
            start_time = time.time()
            while time.time() - start_time < self.sample_duration:
                chunk = capture.read()
                if chunk is not None:
                    frames.append(chunk)
            capture.stop()
            audio_data = np.concatenate(frames) if frames else np.array([], dtype=np.int16)
            
        logger.info("Recording finished.")
        return audio_data

    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract speaker embedding from the audio sample.
        """
        try:
            # Convert int16 to float32
            audio_float = audio_to_float32(audio)
            
            # Preprocess for resemblyzer
            processed_wav = preprocess_wav(audio_float, source_sr=self.audio_config.sample_rate)
            
            # Extract embedding
            embedding = self.encoder.embed_utterance(processed_wav)
            return embedding
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            raise
