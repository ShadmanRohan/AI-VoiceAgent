"""Real Ollama integration tests (skipped if Ollama/model unavailable)."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.conversation.orchestrator import ConversationOrchestrator
from app.models.llm import OllamaLLM


@pytest.fixture
async def ollama_ready():
    settings = get_settings()
    llm = OllamaLLM(settings)
    if not await llm.is_available():
        pytest.skip("Ollama not running")
    models = await llm.list_models()
    if settings.ollama_model not in models and not any(
        settings.ollama_model in m for m in models
    ):
        pytest.skip(f"Model {settings.ollama_model} not downloaded")
    return llm


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_service_availability(ollama_ready):
    models = await ollama_ready.list_models()
    assert models


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_stream_and_cancel(ollama_ready):
    settings = get_settings()
    messages = [
        {"role": "system", "content": "Reply briefly."},
        {"role": "user", "content": "Say hello in one short sentence."},
    ]
    tokens = []
    async for gid, tok in ollama_ready.generate_stream(messages):
        tokens.append(tok)
        if len("".join(tokens)) > 5:
            ollama_ready.cancel(gid)
            break
    assert tokens


@pytest.mark.integration
@pytest.mark.asyncio
async def test_orchestrator_context_python(db, ollama_ready):
    settings = get_settings()
    orch = ConversationOrchestrator(db, settings, llm=ollama_ready)
    events = []
    async for e in orch.handle_user_text("My favorite programming language is Python."):
        events.append(e)
    assert any(e.get("type") == "assistant_final" for e in events)
    session_id = orch.session_id
    events2 = []
    async for e in orch.handle_user_text(
        "What programming language did I mention?", session_id=session_id
    ):
        events2.append(e)
    final = next(e for e in events2 if e.get("type") == "assistant_final")
    assert "python" in final["text"].lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ollama_down_error_message(db, monkeypatch):
    settings = get_settings()
    llm = OllamaLLM(settings)
    monkeypatch.setattr(settings, "ollama_base_url", "http://127.0.0.1:9")
    llm = OllamaLLM(settings)
    orch = ConversationOrchestrator(db, settings, llm=llm)
    events = []
    async for e in orch.handle_user_text("Hello"):
        events.append(e)
    assert any(e.get("type") == "error" for e in events)
