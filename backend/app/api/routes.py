"""REST API routes for LocalVoiceAI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    ChatRequest,
    ConsentUpdate,
    HealthResponse,
    InterruptRequest,
    MemoryCreate,
    MemoryUpdate,
    NoteCreate,
    SessionCreate,
    SettingsUpdate,
    SpokenAck,
    TaskCreate,
    TaskUpdate,
    ToolRequest,
    TTSRequest,
)
from app.config import PROFILE_PRESETS, get_settings, reload_settings
from app.conversation.orchestrator import ConversationOrchestrator
from app.conversation.sessions import SessionService
from app.database.session import get_db_session
from app.memory.service import MemoryService
from app.models.llm import OllamaLLM
from app.language.language_router import DetectedLanguage
from app.monitoring.hardware import build_hardware_report, resolve_profile
from app.monitoring.metrics import APP_METRICS
from app.monitoring.model_status import collect_model_status
from app.speech.tts_router import TTSRouter
from app.tools.registry import TOOL_SCHEMAS, ToolExecutor

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/system/status")
async def system_status(db: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    settings = get_settings()
    report = await build_hardware_report(settings)
    llm = OllamaLLM(settings)
    ollama_ok = await llm.is_available()
    profile = resolve_profile(settings, report)
    return {
        "backend": "ok",
        "ollama": {
            "ok": ollama_ok,
            "base_url": settings.ollama_base_url,
            "models": report.ollama.models,
            "error": report.ollama.error,
        },
        "hardware": report.to_dict(),
        "active_profile": profile,
        "metrics": APP_METRICS.snapshot(),
        "default_user_id": settings.default_user_id,
    }


@router.get("/models")
async def list_models() -> dict[str, Any]:
    settings = get_settings()
    llm = OllamaLLM(settings)
    try:
        models = await llm.list_models()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"Ollama unavailable: {exc}") from exc
    readiness = await collect_model_status()
    return {
        "ollama": models,
        "stt": ["whisper_turbo", "moonshine", "whisper"],
        "stt_models": ["large-v3", "large-v3-turbo", "small", "base", "tiny"],
        "llm_options": ["qwen3.5:4b", "qwen3:1.7b", "qwen3:4b-instruct"],
        "bengali_tts_voices": ["Aditi", "Arjun", "Tapan", "Rashmi"],
        "tts_voices": [
            "af_heart",
            "af_bella",
            "af_sarah",
            "am_adam",
            "am_michael",
            "bf_emma",
            "bf_isabella",
            "bm_george",
            "bm_lewis",
        ],
        "language_modes": ["auto", "bangla", "english"],
        "profiles": list(PROFILE_PRESETS.keys()),
        "readiness": readiness,
    }


@router.get("/models/status")
async def models_status() -> dict[str, Any]:
    return await collect_model_status()


@router.post("/tts/synthesize")
async def tts_synthesize(body: TTSRequest):
    """Synthesize speech via Indic Parler (Bengali) or Kokoro (English)."""
    import base64

    settings = get_settings()
    router = TTSRouter(settings)
    lang = body.language
    response_language = None
    if lang in {"bangla", "bn", "bn-BD"}:
        response_language = DetectedLanguage.BANGLA
    elif lang in {"english", "en"}:
        response_language = DetectedLanguage.ENGLISH
    try:
        chunks = []
        async for chunk in router.synthesize(
            body.text,
            voice=body.voice,
            generation_id=body.generation_id,
            response_language=response_language,
        ):
            chunks.append(chunk)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    if not chunks:
        raise HTTPException(500, "No audio generated")
    pcm = b"".join(c.pcm for c in chunks)
    sr = chunks[0].sample_rate
    provider = (chunks[0].meta or {}).get("provider")
    return {
        "provider": provider,
        "sample_rate": sr,
        "channels": 1,
        "format": "pcm_s16le",
        "audio_base64": base64.b64encode(pcm).decode("ascii"),
        "bytes": len(pcm),
    }


@router.post("/memory/reindex-e5")
async def reindex_e5() -> dict[str, Any]:
    """Rebuild multilingual E5 index from existing memory text (non-destructive)."""
    from app.memory.reindex import reindex_memories_to_e5

    try:
        return await reindex_memories_to_e5(activate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Reindex failed: {exc}") from exc


@router.get("/settings")
async def get_app_settings() -> dict[str, Any]:
    return get_settings().public_dict()


@router.put("/settings")
async def update_settings(body: SettingsUpdate) -> dict[str, Any]:
    settings = get_settings()
    overrides = {k: v for k, v in body.model_dump().items() if v is not None}
    if not overrides:
        return settings.public_dict()
    settings.save_overrides(overrides)
    reloaded = reload_settings()
    return reloaded.public_dict()


@router.post("/sessions")
async def create_session(
    body: SessionCreate,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    svc = SessionService(db)
    sess = await svc.create_session(settings.default_user_id, title=body.title)
    return _session_dict(sess)


@router.get("/sessions")
async def list_sessions(
    q: str | None = None,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    svc = SessionService(db)
    sessions = await svc.list_sessions(settings.default_user_id, q=q)
    return {"sessions": [_session_dict(s) for s in sessions]}


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    svc = SessionService(db)
    sess = await svc.get_session(session_id, settings.default_user_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    return {
        **_session_dict(sess),
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "spoken_status": m.spoken_status,
                "interrupted": m.interrupted,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in sorted(sess.messages, key=lambda x: x.created_at or datetime.min)
        ],
    }


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    ok = await SessionService(db).delete_session(session_id, settings.default_user_id)
    if not ok:
        raise HTTPException(404, "Session not found")
    return {"ok": True}


@router.get("/memories")
async def list_memories(
    q: str | None = None,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    mems = await MemoryService(db).list_memories(settings.default_user_id, q=q)
    user = await MemoryService(db).get_user(settings.default_user_id)
    return {
        "consent": bool(user and user.memory_consent),
        "memories": [_memory_dict(m) for m in mems],
    }


@router.post("/memories")
async def create_memory(
    body: MemoryCreate,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    svc = MemoryService(db)
    try:
        mem = await svc.add_memory(
            settings.default_user_id,
            body.content,
            memory_type=body.memory_type,
            importance=body.importance,
            confidence=body.confidence,
            require_consent=True,
        )
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _memory_dict(mem)


@router.patch("/memories/{memory_id}")
async def patch_memory(
    memory_id: str,
    body: MemoryUpdate,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    try:
        mem = await MemoryService(db).update_memory(
            memory_id,
            settings.default_user_id,
            **{k: v for k, v in body.model_dump().items() if v is not None},
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return _memory_dict(mem)


@router.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    ok = await MemoryService(db).delete_memory(memory_id, settings.default_user_id)
    if not ok:
        raise HTTPException(404, "Memory not found")
    return {"ok": True}


@router.put("/memories/consent")
async def update_consent(
    body: ConsentUpdate,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    user = await MemoryService(db).set_consent(settings.default_user_id, body.consent)
    return {"consent": user.memory_consent}


@router.get("/export/memories")
async def export_memories(db: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    settings = get_settings()
    data = await MemoryService(db).export_memories(settings.default_user_id)
    export_path = settings.data_dir / "exports" / f"memories-{int(datetime.now().timestamp())}.json"
    export_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    data["export_path"] = str(export_path)
    return data


@router.get("/notes")
async def list_notes(db: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    settings = get_settings()
    tools = ToolExecutor(db, settings.default_user_id)
    return await tools.execute("read_notes", {"limit": 100})


@router.post("/notes")
async def create_note(
    body: NoteCreate,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    tools = ToolExecutor(db, settings.default_user_id)
    result = await tools.execute(
        "create_note",
        {"title": body.title, "content": body.content, "tags": body.tags},
    )
    return result


@router.delete("/notes/{note_id}")
async def delete_note(
    note_id: str,
    confirmed: bool = Query(False),
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    tools = ToolExecutor(db, settings.default_user_id)
    return await tools.execute(
        "delete_note", {"note_id": note_id}, confirmed=confirmed
    )


@router.get("/tasks")
async def list_tasks(
    status: str | None = None,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    args = {"status": status} if status else {}
    return await ToolExecutor(db, settings.default_user_id).execute("list_tasks", args)


@router.post("/tasks")
async def create_task(
    body: TaskCreate,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    return await ToolExecutor(db, settings.default_user_id).execute(
        "create_task", {"title": body.title, "description": body.description}
    )


@router.patch("/tasks/{task_id}")
async def patch_task(
    task_id: str,
    body: TaskUpdate,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    if body.status == "completed":
        return await ToolExecutor(db, settings.default_user_id).execute(
            "complete_task", {"task_id": task_id}, confirmed=True
        )
    from app.database.models import Task

    task = await db.get(Task, task_id)
    if not task or task.user_id != settings.default_user_id:
        raise HTTPException(404, "Task not found")
    if body.title is not None:
        task.title = body.title
    if body.description is not None:
        task.description = body.description
    await db.flush()
    return {"id": task.id, "title": task.title, "status": task.status}


@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    confirmed: bool = Query(True),
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    return await ToolExecutor(db, settings.default_user_id).execute(
        "delete_task", {"task_id": task_id}, confirmed=confirmed
    )


@router.get("/tools")
async def list_tools() -> dict[str, Any]:
    return {"tools": TOOL_SCHEMAS}


@router.post("/tools/execute")
async def execute_tool(
    body: ToolRequest,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    try:
        return await ToolExecutor(
            db, settings.default_user_id, body.session_id
        ).execute(body.tool_name, body.arguments, confirmed=body.confirmed)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc


@router.post("/chat")
async def chat(
    body: ChatRequest,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Non-streaming chat turn (useful for tests and text fallback)."""
    orch = ConversationOrchestrator(db, get_settings())
    events: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    async for event in orch.handle_user_text(body.text, session_id=body.session_id):
        events.append(event)
        if event.get("type") == "assistant_final":
            final = event
        if event.get("type") == "error":
            raise HTTPException(503, event.get("message", "Chat failed"))
    if not final:
        raise HTTPException(500, "No assistant response generated")
    return {
        "session_id": orch.session_id,
        "message_id": final.get("message_id"),
        "text": final.get("text"),
        "generation_id": final.get("generation_id"),
        "events": events,
    }


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    db: AsyncSession = Depends(get_db_session),
):
    orch = ConversationOrchestrator(db, get_settings())

    async def gen():
        async for event in orch.handle_user_text(body.text, session_id=body.session_id):
            yield json.dumps(event) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@router.post("/conversation/interrupt")
async def interrupt_conversation(
    body: InterruptRequest,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    # Process-local active orchestrators are per-request; interruption for voice
    # is handled inside the Pipecat pipeline via InterruptionFrame.
    # For text mode we expose a no-op acknowledgement with guidance.
    APP_METRICS.record("interruption_to_silence_latency_ms", 0.0)
    return {
        "ok": True,
        "message": "Interrupt signal accepted",
        "session_id": body.session_id,
    }


@router.post("/playback/ack")
async def playback_ack(
    body: SpokenAck,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    svc = SessionService(db)
    msg = await svc.update_message(body.message_id, spoken_status=body.status)
    if not msg:
        raise HTTPException(404, "Message not found")
    return {"ok": True, "message_id": msg.id, "spoken_status": msg.spoken_status}


@router.get("/metrics")
async def metrics() -> dict[str, Any]:
    return APP_METRICS.snapshot()


@router.get("/search")
async def search(
    q: str = Query(min_length=1),
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    rows = await SessionService(db).search_messages(settings.default_user_id, q)
    return {"results": rows}


def _session_dict(sess) -> dict[str, Any]:
    return {
        "id": sess.id,
        "title": sess.title,
        "status": sess.status,
        "summary": sess.summary,
        "started_at": sess.started_at.isoformat() if sess.started_at else None,
        "ended_at": sess.ended_at.isoformat() if sess.ended_at else None,
        "updated_at": sess.updated_at.isoformat() if sess.updated_at else None,
    }


def _memory_dict(m) -> dict[str, Any]:
    return {
        "id": m.id,
        "type": m.memory_type,
        "content": m.content,
        "importance": m.importance,
        "confidence": m.confidence,
        "consent_status": m.consent_status,
        "active": m.active,
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "updated_at": m.updated_at.isoformat() if m.updated_at else None,
    }
