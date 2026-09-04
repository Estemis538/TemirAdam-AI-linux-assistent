"""
Темірадам — Text-to-Speech engine.

Local TTS using Piper. Supports Russian and Kazakh voices.
Implements ManagedModel protocol for lazy loading/unloading.
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from utils.config import TTSConfig

logger = logging.getLogger("temiradam.tts.engine")


class TTSEngine:
    """
    Text-to-Speech engine using Piper.

    Supports multiple voices for different languages.
    Implements ManagedModel protocol.
    """

    def __init__(self, config: TTSConfig) -> None:
        self.config = config
        self._loaded = False
        self._piper_path: str | None = None
        self._voice_paths: dict[str, str] = {}  # language -> model path

    @property
    def name(self) -> str:
        return "tts"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Verify Piper is available and voice models are accessible."""
        if self._loaded:
            return

        # Find piper executable
        self._piper_path = shutil.which("piper")

        if self._piper_path is None:
            # Try common locations
            for path in [
                "/usr/bin/piper",
                "/usr/local/bin/piper",
                str(Path.home() / ".local/bin/piper"),
            ]:
                if Path(path).exists():
                    self._piper_path = path
                    break

        if self._piper_path is None:
            # Try using piper as a Python module
            try:
                result = subprocess.run(
                    ["python3", "-m", "piper", "--help"],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    self._piper_path = "python3 -m piper"
            except Exception:
                pass

        if self._piper_path is None:
            logger.error(
                "Piper TTS not found. Install via: pip install piper-tts "
                "or download from https://github.com/rhasspy/piper"
            )
            # Still mark as loaded so we can fall back to espeak
            self._loaded = True
            return

        logger.info("Piper TTS found at: %s", self._piper_path)
        self._loaded = True

    def unload(self) -> None:
        """Release TTS resources."""
        self._loaded = False
        self._voice_paths.clear()
        logger.info("TTS engine unloaded")

    def speak(self, text: str, language: str = "ru") -> None:
        """
        Synthesize speech and play it through speakers.

        Args:
            text: Text to speak.
            language: Language code ('ru' or 'kk').
        """
        if not text.strip():
            return

        if not self._loaded:
            self.load()

        logger.info("TTS speak [%s]: '%s'", language, text)

        # Select voice based on language
        voice = self._get_voice(language)

        if self._piper_path and self._piper_path != "python3 -m piper":
            self._speak_piper_cli(text, voice)
        elif self._piper_path == "python3 -m piper":
            self._speak_piper_module(text, voice)
        else:
            self._speak_espeak_fallback(text, language)

    def _get_voice(self, language: str) -> str:
        """Get the voice model name for a language."""
        if language == "kk":
            return self.config.voice_kk
        return self.config.voice_ru

    def _speak_piper_cli(self, text: str, voice: str) -> None:
        """Synthesize and play using Piper CLI."""
        try:
            # Piper outputs raw audio to stdout
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                cmd = [
                    self._piper_path,
                    "--model", voice,
                    "--output_file", tmp.name,
                ]

                logger.debug("Piper command: %s", " ".join(cmd))

                result = subprocess.run(
                    cmd,
                    input=text.encode("utf-8"),
                    capture_output=True,
                    timeout=30,
                )

                if result.returncode != 0:
                    stderr = result.stderr.decode("utf-8", errors="replace")
                    logger.error("Piper failed: %s", stderr)
                    self._speak_espeak_fallback(text, "ru")
                    return

                # Play the generated WAV
                self._play_wav(tmp.name)

        except subprocess.TimeoutExpired:
            logger.error("Piper TTS timed out")
        except Exception:
            logger.exception("Piper TTS failed")
            self._speak_espeak_fallback(text, "ru")

    def _speak_piper_module(self, text: str, voice: str) -> None:
        """Synthesize using Piper as Python module."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                cmd = [
                    "python3", "-m", "piper",
                    "--model", voice,
                    "--output_file", tmp.name,
                ]

                result = subprocess.run(
                    cmd,
                    input=text.encode("utf-8"),
                    capture_output=True,
                    timeout=30,
                )

                if result.returncode != 0:
                    stderr = result.stderr.decode("utf-8", errors="replace")
                    logger.error("Piper module failed: %s", stderr)
                    self._speak_espeak_fallback(text, "ru")
                    return

                self._play_wav(tmp.name)

        except Exception:
            logger.exception("Piper module TTS failed")
            self._speak_espeak_fallback(text, "ru")

    def _speak_espeak_fallback(self, text: str, language: str) -> None:
        """Fallback to espeak if Piper is unavailable."""
        espeak = shutil.which("espeak-ng") or shutil.which("espeak")
        if espeak is None:
            logger.error("No TTS engine available (neither Piper nor espeak)")
            return

        lang_code = "ru" if language == "ru" else "ru"  # espeak may not have kk
        try:
            subprocess.run(
                [espeak, "-v", lang_code, text],
                timeout=15,
                capture_output=True,
            )
        except Exception:
            logger.exception("espeak fallback failed")

    def _play_wav(self, wav_path: str) -> None:
        """Play a WAV file through the default audio output."""
        try:
            # Try pw-play first (PipeWire native)
            pw_play = shutil.which("pw-play")
            if pw_play:
                subprocess.run(
                    [pw_play, wav_path],
                    timeout=30,
                    capture_output=True,
                )
                return

            # Fallback to aplay
            aplay = shutil.which("aplay")
            if aplay:
                subprocess.run(
                    [aplay, wav_path],
                    timeout=30,
                    capture_output=True,
                )
                return

            # Fallback to sounddevice
            with wave.open(wav_path, "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                frames = wf.readframes(wf.getnframes())

            audio = np.frombuffer(frames, dtype=np.int16)
            if channels > 1:
                audio = audio.reshape(-1, channels)

            sd.play(audio, samplerate=sample_rate)
            sd.wait()

        except Exception:
            logger.exception("Failed to play audio: %s", wav_path)
