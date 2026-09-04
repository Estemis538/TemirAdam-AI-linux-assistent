"""
Темірадам — Audio capture utilities.

Provides microphone capture, Voice Activity Detection (VAD),
and end-of-speech detection using sounddevice and webrtcvad.
"""

from __future__ import annotations

import collections
import logging
import struct
import time
from typing import Generator

import numpy as np
import sounddevice as sd
import webrtcvad

logger = logging.getLogger("temiradam.utils.audio")


class AudioCapture:
    """
    Captures audio from the microphone with Voice Activity Detection.

    Uses sounddevice for capture and webrtcvad for VAD.
    Audio is returned as 16-bit PCM at 16kHz mono.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 30,
        vad_aggressiveness: int = 2,
        device: str | int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.device = device if device else None
        self.frame_size = int(sample_rate * chunk_duration_ms / 1000)

        self._vad = webrtcvad.Vad(vad_aggressiveness)
        self._stream: sd.InputStream | None = None

    def _create_stream(self) -> sd.InputStream:
        """Create a new audio input stream."""
        return sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.frame_size,
            device=self.device,
        )

    def read_frame(self, stream: sd.InputStream) -> bytes:
        """Read a single frame of audio from the stream."""
        data, _ = stream.read(self.frame_size)
        return data.tobytes()

    def is_speech(self, frame: bytes) -> bool:
        """Check if a frame contains speech using VAD."""
        try:
            return self._vad.is_speech(frame, self.sample_rate)
        except Exception:
            return False

    def stream_frames(self) -> Generator[tuple[bytes, bool], None, None]:
        """
        Generator that yields (frame_bytes, is_speech) tuples continuously.

        Opens the microphone and yields frames until the generator is closed.
        """
        stream = self._create_stream()
        stream.start()
        try:
            while True:
                frame = self.read_frame(stream)
                speech = self.is_speech(frame)
                yield frame, speech
        finally:
            stream.stop()
            stream.close()

    def wait_for_speech(
        self,
        min_speech_ms: int = 300,
        timeout_seconds: float | None = None,
    ) -> list[bytes] | None:
        """
        Wait until speech is detected, then return the speech frames.

        Returns the initial speech frames (including pre-roll buffer),
        or None if timeout is reached.
        """
        min_speech_frames = int(min_speech_ms / self.chunk_duration_ms)
        start_time = time.monotonic()

        # Keep a small ring buffer of recent frames for pre-roll
        pre_roll_count = 10
        ring_buffer: collections.deque[bytes] = collections.deque(
            maxlen=pre_roll_count
        )
        speech_count = 0

        stream = self._create_stream()
        stream.start()
        try:
            while True:
                if timeout_seconds and (time.monotonic() - start_time) > timeout_seconds:
                    return None

                frame = self.read_frame(stream)
                is_sp = self.is_speech(frame)

                if is_sp:
                    speech_count += 1
                    ring_buffer.append(frame)
                    if speech_count >= min_speech_frames:
                        return list(ring_buffer)
                else:
                    speech_count = 0
                    ring_buffer.append(frame)
        finally:
            stream.stop()
            stream.close()

    def record_until_silence(
        self,
        silence_threshold_ms: int = 800,
        max_seconds: float = 10.0,
        pre_frames: list[bytes] | None = None,
    ) -> np.ndarray:
        """
        Record audio until silence is detected or max duration is reached.

        Args:
            silence_threshold_ms: Duration of silence to stop recording.
            max_seconds: Maximum recording duration.
            pre_frames: Pre-recorded frames to include at the beginning.

        Returns:
            NumPy array of int16 audio samples.
        """
        silence_frames_needed = int(silence_threshold_ms / self.chunk_duration_ms)
        max_frames = int(max_seconds * 1000 / self.chunk_duration_ms)

        frames: list[bytes] = list(pre_frames) if pre_frames else []
        silence_count = 0
        frame_count = 0

        stream = self._create_stream()
        stream.start()
        try:
            while frame_count < max_frames:
                frame = self.read_frame(stream)
                frames.append(frame)
                frame_count += 1

                if self.is_speech(frame):
                    silence_count = 0
                else:
                    silence_count += 1
                    if silence_count >= silence_frames_needed:
                        logger.debug(
                            "End of speech detected after %d frames", frame_count
                        )
                        break
        finally:
            stream.stop()
            stream.close()

        if not frames:
            return np.array([], dtype=np.int16)

        # Concatenate all frames into a single numpy array
        raw = b"".join(frames)
        audio = np.frombuffer(raw, dtype=np.int16)

        duration = len(audio) / self.sample_rate
        logger.debug("Recorded %.2f seconds of audio (%d samples)", duration, len(audio))

        return audio

    def record_seconds(self, duration: float) -> np.ndarray:
        """Record a fixed number of seconds of audio."""
        total_frames = int(duration * 1000 / self.chunk_duration_ms)
        frames: list[bytes] = []

        stream = self._create_stream()
        stream.start()
        try:
            for _ in range(total_frames):
                frame = self.read_frame(stream)
                frames.append(frame)
        finally:
            stream.stop()
            stream.close()

        raw = b"".join(frames)
        return np.frombuffer(raw, dtype=np.int16)


def audio_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 audio to float32 normalized to [-1, 1]."""
    return audio.astype(np.float32) / 32768.0


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 audio to int16."""
    return (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
