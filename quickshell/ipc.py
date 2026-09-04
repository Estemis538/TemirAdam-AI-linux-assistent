"""
Темірадам — Quickshell IPC client.

Communicates with Quickshell to update the UI state.
Falls back to notify-send if Quickshell is not available.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from core.state import AssistantState
from utils.config import QuickshellConfig

logger = logging.getLogger("temiradam.quickshell.ipc")

# Map assistant states to display strings
STATE_DISPLAY = {
    AssistantState.IDLE: {"status": "idle", "text": ""},
    AssistantState.LISTENING: {"status": "listening", "text": "Слушаю..."},
    AssistantState.VERIFYING: {"status": "verifying", "text": "Проверяю..."},
    AssistantState.THINKING: {"status": "thinking", "text": "Думаю..."},
    AssistantState.EXECUTING: {"status": "executing", "text": "Выполняю..."},
    AssistantState.SPEAKING: {"status": "speaking", "text": ""},
    AssistantState.ERROR: {"status": "error", "text": "Ошибка"},
    AssistantState.CONFIRMING: {"status": "confirming", "text": "Подтвердите..."},
}


class QuickshellIPC:
    """
    IPC client for Quickshell UI.

    Uses `qs ipc call` to communicate with the Quickshell widget.
    Falls back to `notify-send` for notifications.
    """

    def __init__(self, config: QuickshellConfig) -> None:
        self.config = config
        self._qs_available: bool | None = None

    def _is_qs_available(self) -> bool:
        """Check if Quickshell IPC is available."""
        if self._qs_available is not None:
            return self._qs_available

        qs = shutil.which("qs")
        if qs is None:
            self._qs_available = False
            logger.info("Quickshell not found, using notify-send fallback")
            return False

        # Check if socket exists
        socket = Path(self.config.socket_path)
        if not socket.exists():
            self._qs_available = False
            logger.info("Quickshell socket not found: %s", socket)
            return False

        self._qs_available = True
        return True

    def _qs_call(self, method: str, *args: str) -> bool:
        """Make a Quickshell IPC call."""
        try:
            cmd = ["qs", "ipc", "call", "temiradam", method, *args]
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                logger.debug("qs ipc call failed: %s", stderr)
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.warning("qs ipc call timed out: %s", method)
            return False
        except Exception:
            logger.exception("qs ipc call failed: %s", method)
            return False

    def _notify(self, title: str, message: str, urgency: str = "normal") -> None:
        """Send a desktop notification via notify-send."""
        if not self.config.fallback_notify:
            return

        notify_send = shutil.which("notify-send")
        if notify_send is None:
            return

        try:
            cmd = [
                notify_send,
                "--app-name", "Темірадам",
                "--urgency", urgency,
                title,
                message,
            ]
            subprocess.run(cmd, capture_output=True, timeout=5)
        except Exception:
            logger.debug("notify-send failed")

    def set_status(
        self, state: AssistantState, model_tier: str | None = None,
    ) -> None:
        """
        Update the UI with current assistant state.

        Args:
            state: Current AssistantState.
            model_tier: Optional tier label ('easy', 'medium', 'hard')
                        shown during THINKING state.
        """
        display = STATE_DISPLAY.get(state, {"status": "idle", "text": ""})
        status_str = display["status"]
        display_text = display["text"]

        # Append model tier for THINKING state
        if model_tier and state == AssistantState.THINKING:
            tier_label = model_tier.upper()
            display_text = f"Думаю · {tier_label}"

        if self._is_qs_available():
            self._qs_call("setStatus", status_str)
            if model_tier and state == AssistantState.THINKING:
                self._qs_call("setModelTier", model_tier)
            if display_text:
                self._qs_call("setMessage", display_text)
        else:
            # Only notify on significant state changes
            if state in (
                AssistantState.LISTENING,
                AssistantState.ERROR,
            ):
                self._notify("Темірадам", display_text or status_str)

        logger.debug("UI status: %s (tier=%s)", status_str, model_tier)

    def show_message(self, message: str) -> None:
        """Display a message in the UI."""
        if self._is_qs_available():
            self._qs_call("showMessage", message)
        else:
            self._notify("Темірадам", message)

        logger.debug("UI message: %s", message)

    def show(self) -> None:
        """Show the assistant widget."""
        if self._is_qs_available():
            self._qs_call("show")

    def hide(self) -> None:
        """Hide the assistant widget."""
        if self._is_qs_available():
            self._qs_call("hide")

    def set_volume_display(self, volume: int) -> None:
        """Update volume display in UI."""
        if self._is_qs_available():
            self._qs_call("setVolume", str(volume))
