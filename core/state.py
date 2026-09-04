"""
Темірадам — Core state machine.

Defines all possible states the assistant can be in,
valid transitions, and state change callbacks.
"""

from __future__ import annotations

import enum
import logging
import time
from typing import Callable

logger = logging.getLogger("temiradam.core.state")


class AssistantState(enum.Enum):
    """All possible states of the assistant."""

    IDLE = "idle"
    LISTENING = "listening"
    VERIFYING = "verifying"
    THINKING = "thinking"
    EXECUTING = "executing"
    SPEAKING = "speaking"
    ERROR = "error"
    CONFIRMING = "confirming"


# Valid state transitions: from_state -> set of allowed to_states
VALID_TRANSITIONS: dict[AssistantState, set[AssistantState]] = {
    AssistantState.IDLE: {AssistantState.LISTENING, AssistantState.ERROR},
    AssistantState.LISTENING: {
        AssistantState.VERIFYING,
        AssistantState.THINKING,
        AssistantState.IDLE,
        AssistantState.ERROR,
    },
    AssistantState.VERIFYING: {
        AssistantState.LISTENING,  # verified -> continue listening for command
        AssistantState.IDLE,  # verification failed -> ignore
        AssistantState.ERROR,
    },
    AssistantState.THINKING: {
        AssistantState.EXECUTING,
        AssistantState.SPEAKING,  # general question -> direct LLM answer
        AssistantState.CONFIRMING,
        AssistantState.IDLE,
        AssistantState.ERROR,
    },
    AssistantState.CONFIRMING: {
        AssistantState.EXECUTING,
        AssistantState.SPEAKING,
        AssistantState.IDLE,
        AssistantState.ERROR,
    },
    AssistantState.EXECUTING: {
        AssistantState.SPEAKING,
        AssistantState.IDLE,
        AssistantState.ERROR,
    },
    AssistantState.SPEAKING: {AssistantState.IDLE, AssistantState.ERROR},
    AssistantState.ERROR: {AssistantState.IDLE},
}


class StateMachine:
    """
    Finite state machine for the assistant.

    Enforces valid transitions and notifies callbacks on state changes.
    """

    def __init__(self) -> None:
        self._state = AssistantState.IDLE
        self._callbacks: list[Callable[[AssistantState, AssistantState], None]] = []
        self._state_entered_at: float = time.monotonic()

    @property
    def state(self) -> AssistantState:
        """Current state."""
        return self._state

    @property
    def time_in_state(self) -> float:
        """Seconds elapsed since entering current state."""
        return time.monotonic() - self._state_entered_at

    def on_state_change(
        self, callback: Callable[[AssistantState, AssistantState], None]
    ) -> None:
        """Register a callback for state changes: callback(old_state, new_state)."""
        self._callbacks.append(callback)

    def transition(self, new_state: AssistantState) -> None:
        """
        Transition to a new state.

        Raises ValueError if the transition is not valid.
        """
        if new_state == self._state:
            return

        allowed = VALID_TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition: {self._state.value} -> {new_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )

        old_state = self._state
        self._state = new_state
        self._state_entered_at = time.monotonic()

        logger.info("State: %s -> %s", old_state.value, new_state.value)

        for callback in self._callbacks:
            try:
                callback(old_state, new_state)
            except Exception:
                logger.exception("Error in state change callback")

    def reset(self) -> None:
        """Force reset to IDLE (e.g., after error recovery)."""
        old_state = self._state
        self._state = AssistantState.IDLE
        self._state_entered_at = time.monotonic()
        logger.info("State RESET: %s -> IDLE", old_state.value)
        for callback in self._callbacks:
            try:
                callback(old_state, AssistantState.IDLE)
            except Exception:
                logger.exception("Error in state change callback")
