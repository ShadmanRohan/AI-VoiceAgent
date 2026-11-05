"""Explicit conversation state machine."""

from __future__ import annotations

from enum import Enum
from typing import Callable


class ConversationState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    USER_SPEAKING = "USER_SPEAKING"
    WAITING_FOR_TURN_END = "WAITING_FOR_TURN_END"
    PROCESSING = "PROCESSING"
    AI_SPEAKING = "AI_SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    ERROR = "ERROR"


ALLOWED_TRANSITIONS: dict[ConversationState, set[ConversationState]] = {
    ConversationState.IDLE: {
        ConversationState.LISTENING,
        ConversationState.ERROR,
    },
    ConversationState.LISTENING: {
        ConversationState.USER_SPEAKING,
        ConversationState.IDLE,
        ConversationState.ERROR,
    },
    ConversationState.USER_SPEAKING: {
        ConversationState.WAITING_FOR_TURN_END,
        ConversationState.LISTENING,
        ConversationState.PROCESSING,
        ConversationState.ERROR,
    },
    ConversationState.WAITING_FOR_TURN_END: {
        ConversationState.PROCESSING,
        ConversationState.USER_SPEAKING,
        ConversationState.LISTENING,
        ConversationState.ERROR,
    },
    ConversationState.PROCESSING: {
        ConversationState.AI_SPEAKING,
        ConversationState.LISTENING,
        ConversationState.INTERRUPTED,
        ConversationState.ERROR,
    },
    ConversationState.AI_SPEAKING: {
        ConversationState.LISTENING,
        ConversationState.INTERRUPTED,
        ConversationState.IDLE,
        ConversationState.ERROR,
    },
    ConversationState.INTERRUPTED: {
        ConversationState.USER_SPEAKING,
        ConversationState.LISTENING,
        ConversationState.PROCESSING,
        ConversationState.ERROR,
    },
    ConversationState.ERROR: {
        ConversationState.IDLE,
        ConversationState.LISTENING,
    },
}


class StateMachine:
    def __init__(
        self,
        initial: ConversationState = ConversationState.IDLE,
        on_change: Callable[[ConversationState, ConversationState], None] | None = None,
    ) -> None:
        self.state = initial
        self.on_change = on_change
        self.history: list[tuple[ConversationState, ConversationState]] = []

    def can_transition(self, new_state: ConversationState) -> bool:
        if new_state == self.state:
            return True
        return new_state in ALLOWED_TRANSITIONS.get(self.state, set())

    def transition(self, new_state: ConversationState, *, force: bool = False) -> ConversationState:
        if new_state == self.state:
            return self.state
        if not force and not self.can_transition(new_state):
            raise ValueError(f"Illegal transition {self.state} -> {new_state}")
        old = self.state
        self.state = new_state
        self.history.append((old, new_state))
        if self.on_change:
            self.on_change(old, new_state)
        return self.state
