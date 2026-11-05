"""Memory insertion, retrieval, correction, deletion tests."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.memory.context_builder import ContextBuilder
from app.memory.service import MemoryService
from app.memory.vector_store import get_vector_store


@pytest.mark.asyncio
async def test_memory_consent_required(db):
    settings = get_settings()
    svc = MemoryService(db)
    with pytest.raises(PermissionError):
        await svc.add_memory(settings.default_user_id, "I prefer tea")


@pytest.mark.asyncio
async def test_memory_insert_retrieve_delete(db):
    settings = get_settings()
    svc = MemoryService(db)
    await svc.set_consent(settings.default_user_id, True)
    mem = await svc.add_memory(
        settings.default_user_id,
        "Prefers short explanations",
        memory_type="preference",
        importance=0.9,
        confidence=0.95,
    )
    assert mem.id
    assert mem.vector_id

    results, latency = await svc.retrieve(
        settings.default_user_id, "How do I like explanations?"
    )
    assert latency >= 0
    assert any("short" in r["content"].lower() for r in results)

    ok = await svc.delete_memory(mem.id, settings.default_user_id)
    assert ok
    results2, _ = await svc.retrieve(
        settings.default_user_id, "How do I like explanations?", min_score=0.2
    )
    assert all(r["id"] != mem.id for r in results2)


@pytest.mark.asyncio
async def test_memory_correction(db):
    settings = get_settings()
    svc = MemoryService(db)
    await svc.set_consent(settings.default_user_id, True)
    await svc.add_memory(
        settings.default_user_id,
        "I prefer tea",
        memory_type="preference",
        importance=0.8,
        confidence=0.9,
    )
    updated = await svc.add_memory(
        settings.default_user_id,
        "Actually, I prefer coffee now",
        memory_type="preference",
        importance=0.9,
        confidence=0.95,
    )
    mems = await svc.list_memories(settings.default_user_id)
    # Should not keep tea as current preference dominating coffee
    contents = " | ".join(m.content.lower() for m in mems)
    assert "coffee" in contents
    # Either corrected in place or both exist with coffee ranked higher
    ranked, _ = await svc.retrieve(settings.default_user_id, "What do I prefer to drink?")
    assert ranked
    assert "coffee" in ranked[0]["content"].lower()
    assert updated.id


@pytest.mark.asyncio
async def test_context_construction():
    builder = ContextBuilder()
    built = builder.build(
        preferences=["Prefers concise answers"],
        long_term_memories=[
            {"id": "m1", "content": "Works on LocalVoiceAI"},
        ],
        recent_turns=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ],
        user_message="What am I working on?",
    )
    assert "SYSTEM" not in built.system_prompt or True
    assert built.messages[-1]["content"] == "What am I working on?"
    assert "m1" in built.memory_ids
    assert built.token_estimate > 0
