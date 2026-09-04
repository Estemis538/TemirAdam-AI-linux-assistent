"""
Темірадам — Main Orchestrator.

Ties together all components (audio, wakeword, stt, llm, tools, tts, quickshell)
and manages the main assistant loop.

Uses the three-tier LLM architecture:
  Fast Matcher → Router → EASY / MEDIUM / HARD → Tool Validator → TTS → IDLE
"""

from __future__ import annotations

import asyncio
import logging
import time

from core.lifecycle import ModelLifecycle
from core.state import AssistantState, StateMachine
from llm.intent import IntentClassifier, IntentResult
from llm.ollama import OllamaClient
from llm.prompts import get_response
from quickshell.ipc import QuickshellIPC
from speaker.verifier import SpeakerVerifier
from stt.engine import STTEngine
from tools.executor import ToolExecutor
from tools.registry import registry
from tools.schemas import ToolCall, ToolResult
from tools.validator import ToolValidator
from tts.engine import TTSEngine
from utils.audio import AudioCapture
from utils.config import TemirAdamConfig

from wakeword.detector import WakeWordDetector

logger = logging.getLogger("temiradam.core.assistant")


class AssistantOrchestrator:
    """
    Main assistant class that orchestrates the entire pipeline.
    """

    def __init__(self, config: TemirAdamConfig) -> None:
        self.config = config
        self.state_machine = StateMachine()
        self.lifecycle = ModelLifecycle()

        # Initialize core components
        self.audio = AudioCapture(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            chunk_duration_ms=config.audio.chunk_duration_ms,
            device=config.audio.device,
        )

        self.ui = QuickshellIPC(config.quickshell)
        self.ollama = OllamaClient(
            base_url=config.llm.base_url,
            timeout=config.llm.timeout_seconds,
        )

        # Initialize ML models & register them for lifecycle management
        self.wakeword = WakeWordDetector(config.wakeword)
        self.speaker = SpeakerVerifier(config.speaker_verification, config.audio)
        self.stt = STTEngine(config.stt)
        self.tts = TTSEngine(config.tts)

        self.lifecycle.register(self.wakeword)
        self.lifecycle.register(self.speaker)
        self.lifecycle.register(self.stt)
        self.lifecycle.register(self.tts)

        # Initialize tools & NLP
        self.tool_validator = ToolValidator(config.tools, registry)
        self.tool_executor = ToolExecutor(self.tool_validator, registry)
        self.intent_classifier = IntentClassifier(
            config=config.llm,
            ollama=self.ollama,
            tools_description=registry.get_schemas_for_llm(config.tools),
        )

        # Setup state change callback to update UI
        self.state_machine.on_state_change(self._on_state_change)

        self._running = False
        self._current_tier: str | None = None  # Track active LLM tier for UI

    def _on_state_change(
        self, old_state: AssistantState, new_state: AssistantState,
    ) -> None:
        """Callback fired when state machine transitions."""
        if self.config.quickshell.enabled:
            self.ui.set_status(new_state, model_tier=self._current_tier)

    def _unload_heavy_models(self) -> None:
        """Unload heavy models to free VRAM/RAM when going back to IDLE."""
        logger.info("Unloading heavy models (returning to IDLE)...")
        # Keep wakeword loaded as we need it in IDLE
        self.lifecycle.unload_all(exclude={self.wakeword.name})

        # Also unload any LLM tier models from Ollama if keep_alive == "0"
        if self.config.llm.keep_alive == "0":
            for tier_cfg in (
                self.config.llm.easy,
                self.config.llm.medium,
                self.config.llm.hard,
            ):
                if tier_cfg.enabled:
                    self.ollama.unload_model(tier_cfg.model)

        self._current_tier = None

    def run(self) -> None:
        """Run the main assistant loop."""
        self._running = True
        logger.info("Starting Темірадам Assistant (3-tier LLM)...")
        logger.info(
            "Models: EASY=%s  MEDIUM=%s  HARD=%s",
            self.config.llm.easy.model,
            self.config.llm.medium.model,
            self.config.llm.hard.model,
        )
        self.ui.show()

        # Load wake word model immediately
        self.lifecycle.ensure_loaded(self.wakeword.name)

        self.state_machine.transition(AssistantState.IDLE)

        try:
            while self._running:
                try:
                    self._run_once()
                except Exception:
                    logger.exception("Error in main loop")
                    self.state_machine.reset()
                    self._unload_heavy_models()
                    time.sleep(1.0)  # prevent tight error loops
        except KeyboardInterrupt:
            logger.info("Assistant stopped by user")
        finally:
            self._running = False
            self.ui.hide()
            self._unload_heavy_models()

    def stop(self) -> None:
        """Stop the assistant loop."""
        self._running = False

    def _run_once(self) -> None:
        """Execute one complete cycle of the assistant pipeline."""

        # 1. IDLE: Wait for wake word
        self.state_machine.transition(AssistantState.IDLE)
        logger.info("Listening for wake word 'Темірадам' (awaiting speech)...")

        detected, wake_audio = self.wakeword.listen(self.audio)
        if not detected or wake_audio is None:
            return

        logger.info("Wake word detected!")

        # 3. VERIFYING: Check if it's the owner's voice
        if self.config.speaker_verification.enabled:
            self.state_machine.transition(AssistantState.VERIFYING)
            self.lifecycle.ensure_loaded(self.speaker.name)

            is_owner, confidence = self.speaker.verify(wake_audio)
            logger.info(
                "Speaker verification: %s (confidence: %.2f)", is_owner, confidence,
            )

            if not is_owner:
                logger.info("Unauthorized voice, ignoring.")
                self.state_machine.transition(AssistantState.IDLE)
                return

        # 4. THINKING: STT + Intent parsing
        self.state_machine.transition(AssistantState.THINKING)

        # 4a. STT
        self.lifecycle.ensure_loaded(self.stt.name)
        transcription = self.stt.transcribe(wake_audio)
        logger.info(
            "STT: [%s] '%s' (conf: %.2f)",
            transcription.language, transcription.text, transcription.confidence,
        )

        text = transcription.text.strip()
        if not text:
            self.state_machine.transition(AssistantState.IDLE)
            return
            
        # Clean wake word from the start so fast matcher can work
        lower_text = text.lower()
        for wake in self.wakeword.WAKE_WORDS:
            if lower_text.startswith(wake):
                text = text[len(wake):].strip()
                # Remove any punctuation that might follow the wake word (e.g. "Темірадам, пауза")
                text = text.lstrip(" ,.!-?")
                break

        if not text:
            self.state_machine.transition(AssistantState.IDLE)
            return

        # 4b. Intent Classification (Fast Matcher → Router → Tiered LLM)
        intent = self.intent_classifier.classify(
            text,
            language=transcription.language,
        )

        # Update UI with tier info if an LLM was used
        if intent.tier:
            self._current_tier = intent.tier
            self.ui.set_status(AssistantState.THINKING, model_tier=intent.tier)

        logger.info(
            "Intent: source=%s tier=%s tool=%s conf=%.2f",
            intent.source, intent.tier, intent.tool, intent.confidence,
        )

        # 5. Handle Intent Result
        if intent.source == "qa":
            self._speak_and_idle(intent.response_text, intent.language)
        elif intent.source == "error":
            self._speak_and_idle(
                intent.response_text, intent.language, state=AssistantState.ERROR,
            )
        elif intent.tool is not None:
            self._handle_tool_call(intent)
        else:
            msg = get_response("not_understood", intent.language)
            self._speak_and_idle(msg, intent.language)

    def _handle_tool_call(self, intent: IntentResult) -> None:
        """Execute the parsed tool call and speak the result."""
        tool_call = ToolCall(
            tool=intent.tool,
            arguments=intent.arguments,
            confidence=intent.confidence,
        )

        # Validate tool
        is_valid, err_msg = self.tool_validator.validate(tool_call)
        if not is_valid:
            logger.warning("Tool validation failed: %s", err_msg)
            msg = get_response("error", intent.language)
            self._speak_and_idle(msg, intent.language, state=AssistantState.ERROR)
            return

        # Check confirmation requirement
        schema, _ = registry.get(tool_call.tool)
        if schema and schema.confirmation:
            # TODO: implement interactive confirmation
            logger.info(
                "Tool requires confirmation, skipping for now: %s", tool_call.tool,
            )
            msg = get_response("error", intent.language)
            self._speak_and_idle(msg, intent.language, state=AssistantState.ERROR)
            return

        # Execute
        self.state_machine.transition(AssistantState.EXECUTING)
        logger.info("Executing tool: %s(%s)", tool_call.tool, tool_call.arguments)

        try:
            result: ToolResult = asyncio.run(
                self.tool_executor.execute(tool_call, self.config.tools)
            )
        except Exception:
            logger.exception("Tool execution crashed")
            result = ToolResult(
                success=False, message=get_response("error", intent.language),
            )

        # Speak result
        if result.success:
            if tool_call.tool == "open_app":
                msg = get_response("launching", intent.language)
            elif tool_call.tool == "spotify_play":
                msg = get_response("playing", intent.language)
            elif tool_call.tool == "youtube_search":
                msg = get_response("searching", intent.language)
            elif tool_call.tool == "set_volume":
                msg = get_response(
                    "volume_set", intent.language,
                    volume=tool_call.arguments.get("volume", ""),
                )
            elif tool_call.tool == "mute":
                msg = get_response("muted", intent.language)
            elif tool_call.tool == "unmute":
                msg = get_response("unmuted", intent.language)
            elif tool_call.tool in ("get_time", "get_date"):
                msg = result.message
            else:
                msg = get_response("done", intent.language)

            self._speak_and_idle(msg, intent.language)
        else:
            self._speak_and_idle(
                result.message, intent.language, state=AssistantState.ERROR,
            )

    def _speak_and_idle(
        self,
        text: str,
        language: str = "ru",
        state: AssistantState = AssistantState.SPEAKING,
    ) -> None:
        """Speak the text and return to IDLE, unloading heavy models."""
        if not text:
            self.state_machine.transition(AssistantState.IDLE)
            self._unload_heavy_models()
            return

        self.state_machine.transition(state)

        if self.config.tts.enabled:
            self.lifecycle.ensure_loaded(self.tts.name)
            self.tts.speak(text, language=language)

        # Unload models and go back to IDLE
        self.state_machine.transition(AssistantState.IDLE)
        self._unload_heavy_models()
