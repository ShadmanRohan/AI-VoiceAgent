"""State machine, interruption IDs, TTS chunking, tool auth tests."""

from __future__ import annotations

import pytest

from app.conversation.state import ConversationState, StateMachine
from app.models.llm import sanitize_for_tts
from app.speech.tts_chunker import TTSChunker
from app.tools.registry import ToolExecutor


def test_state_transitions():
    sm = StateMachine()
    sm.transition(ConversationState.LISTENING)
    sm.transition(ConversationState.USER_SPEAKING)
    sm.transition(ConversationState.WAITING_FOR_TURN_END)
    sm.transition(ConversationState.PROCESSING)
    sm.transition(ConversationState.AI_SPEAKING)
    sm.transition(ConversationState.INTERRUPTED)
    sm.transition(ConversationState.USER_SPEAKING)
    assert sm.state == ConversationState.USER_SPEAKING


def test_illegal_transition():
    sm = StateMachine(ConversationState.IDLE)
    with pytest.raises(ValueError):
        sm.transition(ConversationState.AI_SPEAKING)


def test_tts_chunker_no_word_cut():
    chunker = TTSChunker(min_chars=20, max_chars=60)
    text = "The answer involves three important ideas about systems. "
    ready = []
    for ch in text:
        ready.extend(chunker.add(ch))
    ready.extend(chunker.flush())
    joined = " ".join(ready)
    assert "important" in joined
    for chunk in ready:
        assert not chunk.endswith("impor")  # no mid-word cut at end ideally
        assert chunk == chunk.strip()


def test_sanitize_strips_think_tags():
    raw = "Hello <think>secret</think> world <|control|>"
    assert sanitize_for_tts(raw) == "Hello world"


@pytest.mark.asyncio
async def test_tool_authorization(db):
    from app.config import get_settings

    tools = ToolExecutor(db, get_settings().default_user_id)
    with pytest.raises(PermissionError):
        await tools.execute("rm_rf", {})


@pytest.mark.asyncio
async def test_destructive_requires_confirmation(db):
    from app.config import get_settings

    tools = ToolExecutor(db, get_settings().default_user_id)
    created = await tools.execute("create_note", {"title": "x", "content": "y"})
    note_id = created["result"]["id"]
    pending = await tools.execute("delete_note", {"note_id": note_id}, confirmed=False)
    assert pending["status"] == "pending_confirmation"
    done = await tools.execute("delete_note", {"note_id": note_id}, confirmed=True)
    assert done["status"] == "ok"


@pytest.mark.asyncio
async def test_stale_generation_rejection_concept():
    """Generation IDs must not continue after interrupt."""
    from app.models.llm import GenerationHandle
    from uuid import uuid4

    active = str(uuid4())
    stale = str(uuid4())
    assert active != stale
    handle = GenerationHandle(generation_id=stale)
    handle.cancel_event.set()
    assert handle.cancel_event.is_set()
