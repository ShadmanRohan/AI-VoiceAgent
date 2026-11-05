"""Text/voice conversation orchestrator with memory, tools, and interruption IDs."""

from __future__ import annotations

import time
from typing import Any, AsyncIterator
from uuid import uuid4

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.conversation.sessions import SessionService
from app.conversation.state import ConversationState, StateMachine
from app.memory.context_builder import ContextBuilder
from app.memory.extraction import detect_memory_intent, extract_candidates_with_llm
from app.memory.service import MemoryService
from app.models.llm import OllamaLLM, sanitize_for_tts
from app.monitoring.metrics import MetricsCollector
from app.tools.registry import ToolExecutor


class ConversationOrchestrator:
    def __init__(
        self,
        db: AsyncSession,
        settings: Settings | None = None,
        llm: OllamaLLM | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = db
        self.llm = llm or OllamaLLM(self.settings)
        self.metrics = metrics or MetricsCollector()
        self.sessions = SessionService(db)
        self.memories = MemoryService(db, self.settings)
        self.context_builder = ContextBuilder(self.settings)
        self.state = StateMachine(ConversationState.IDLE)
        self.active_generation_id: str | None = None
        self.session_id: str | None = None
        self.user_id = self.settings.default_user_id

    async def ensure_session(self, session_id: str | None = None) -> str:
        if session_id:
            sess = await self.sessions.get_session(session_id, self.user_id)
            if sess:
                self.session_id = sess.id
                return sess.id
        sess = await self.sessions.create_session(self.user_id)
        self.session_id = sess.id
        return sess.id

    def interrupt(self) -> str | None:
        """Cancel active generation; returns cancelled generation id."""
        gid = self.active_generation_id
        if gid:
            self.llm.cancel(gid)
            try:
                self.state.transition(ConversationState.INTERRUPTED)
            except ValueError:
                self.state.transition(ConversationState.INTERRUPTED, force=True)
        self.active_generation_id = None
        return gid

    async def handle_user_text(
        self,
        text: str,
        *,
        session_id: str | None = None,
        store_user: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """Process a finalized user turn. Yields event dicts for UI/WS."""
        text = (text or "").strip()
        if not text:
            yield {"type": "error", "message": "Empty transcript ignored"}
            return

        await self.ensure_session(session_id)
        assert self.session_id

        try:
            self.state.transition(ConversationState.PROCESSING)
        except ValueError:
            self.state.transition(ConversationState.PROCESSING, force=True)

        if store_user:
            await self.sessions.add_message(
                self.session_id, "USER", text, finalized=True, spoken_status="not_applicable"
            )

        # Memory intents
        intent = detect_memory_intent(text)
        user = await self.memories.get_user(self.user_id)
        consent = bool(user and user.memory_consent)

        if intent and intent["action"] == "list":
            mems = await self.memories.list_memories(self.user_id, limit=20)
            if not mems:
                reply = "I don't have any stored memories about you yet."
            else:
                preview = "; ".join(m.content for m in mems[:5])
                reply = f"Here's what I remember: {preview}"
            async for event in self._emit_assistant(reply):
                yield event
            return

        if intent and intent["action"] == "forget":
            if not consent:
                reply = "Persistent memory is off, so there is nothing to forget."
            else:
                deleted = await self.memories.forget_matching(self.user_id, text)
                reply = (
                    f"Okay, I forgot {len(deleted)} related memor{'y' if len(deleted)==1 else 'ies'}."
                    if deleted
                    else "I couldn't find a matching memory to forget."
                )
            async for event in self._emit_assistant(reply):
                yield event
            return

        if intent and intent["action"] == "store":
            if not consent:
                reply = (
                    "I can remember that after you enable persistent memory consent "
                    "in the Memory dashboard."
                )
                async for event in self._emit_assistant(reply):
                    yield event
                # Continue to also answer normally? Spec says store if permitted.
                # We'll still continue with normal response after informing.
            else:
                cand = intent["candidate"]
                try:
                    mem = await self.memories.add_memory(
                        self.user_id,
                        cand["content"],
                        memory_type=cand.get("type", "fact"),
                        importance=float(cand.get("importance", 0.8)),
                        confidence=float(cand.get("confidence", 0.9)),
                    )
                    yield {
                        "type": "memory_stored",
                        "memory_id": mem.id,
                        "content": mem.content,
                    }
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Memory store failed: {}", exc)

        # Retrieve memories
        retrieved, retrieval_ms = ([], 0.0)
        if consent:
            retrieved, retrieval_ms = await self.memories.retrieve(self.user_id, text)
            self.metrics.record("memory_retrieval_latency_ms", retrieval_ms)

        recent = await self.sessions.recent_messages(self.session_id, limit=16)
        recent_turns = [
            {"role": m.role, "content": m.content}
            for m in recent
            if m.role in {"USER", "ASSISTANT"} and m.content
        ][:-1]  # exclude the just-added user message duplicate if present

        sess = await self.sessions.get_session(self.session_id)
        built = self.context_builder.build(
            long_term_memories=retrieved,
            session_summary=sess.summary if sess else None,
            recent_turns=recent_turns,
            user_message=text,
        )
        yield {
            "type": "context",
            "memory_ids": built.memory_ids,
            "token_estimate": built.token_estimate,
        }

        # Tool shortcuts from simple phrases
        tool_reply = await self._maybe_run_tools(text)
        if tool_reply:
            async for event in self._emit_assistant(tool_reply):
                yield event
            return

        generation_id = str(uuid4())
        self.active_generation_id = generation_id
        yield {"type": "generation_started", "generation_id": generation_id}

        t0 = time.perf_counter()
        first_token_ms: float | None = None
        assembled: list[str] = []
        try:
            async for gid, token in self.llm.generate_stream(
                built.messages, generation_id=generation_id
            ):
                if gid != self.active_generation_id:
                    # Stale generation — discard
                    yield {"type": "stale_generation_rejected", "generation_id": gid}
                    return
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - t0) * 1000
                    self.metrics.record("llm_first_token_latency_ms", first_token_ms)
                assembled.append(token)
                yield {
                    "type": "assistant_token",
                    "generation_id": gid,
                    "token": token,
                }
        except Exception as exc:  # noqa: BLE001
            self.state.transition(ConversationState.ERROR, force=True)
            yield {"type": "error", "message": f"LLM failure: {exc}"}
            return

        if self.active_generation_id != generation_id:
            # Interrupted
            partial = sanitize_for_tts("".join(assembled))
            if partial:
                await self.sessions.add_message(
                    self.session_id,
                    "ASSISTANT",
                    partial,
                    spoken_status="interrupted",
                    interrupted=True,
                    generation_id=generation_id,
                )
            yield {"type": "interrupted", "generation_id": generation_id}
            return

        full = sanitize_for_tts("".join(assembled))
        msg = await self.sessions.add_message(
            self.session_id,
            "ASSISTANT",
            full,
            spoken_status="pending",
            generation_id=generation_id,
        )
        try:
            self.state.transition(ConversationState.AI_SPEAKING)
        except ValueError:
            self.state.transition(ConversationState.AI_SPEAKING, force=True)

        yield {
            "type": "assistant_final",
            "generation_id": generation_id,
            "message_id": msg.id,
            "text": full,
            "first_token_ms": first_token_ms,
        }

        # Background candidate extraction only when consent on and no explicit store
        if consent and not (intent and intent["action"] == "store"):
            try:
                candidates = await extract_candidates_with_llm(text, self.settings)
                for cand in candidates[:2]:
                    if float(cand.get("confidence", 0)) < 0.8:
                        continue
                    # Still require explicit-ish content; do not auto-store low value
                    # Spec: require consent; default opt-in already checked.
                    # Auto-store only high-confidence preferences/facts from extraction
                    # is aggressive — we emit candidates for UI confirmation instead.
                    yield {"type": "memory_candidate", "candidate": cand}
            except Exception as exc:  # noqa: BLE001
                logger.debug("Extraction skipped: {}", exc)

        self.active_generation_id = None
        try:
            self.state.transition(ConversationState.LISTENING)
        except ValueError:
            self.state.transition(ConversationState.LISTENING, force=True)

    async def _emit_assistant(self, text: str) -> AsyncIterator[dict[str, Any]]:
        assert self.session_id
        generation_id = str(uuid4())
        self.active_generation_id = generation_id
        msg = await self.sessions.add_message(
            self.session_id,
            "ASSISTANT",
            text,
            spoken_status="pending",
            generation_id=generation_id,
        )
        yield {
            "type": "assistant_final",
            "generation_id": generation_id,
            "message_id": msg.id,
            "text": text,
        }
        self.active_generation_id = None
        try:
            self.state.transition(ConversationState.LISTENING, force=True)
        except ValueError:
            pass

    async def _maybe_run_tools(self, text: str) -> str | None:
        lower = text.lower()
        tools = ToolExecutor(self.db, self.user_id, self.session_id)
        if "what time" in lower or "what's the date" in lower or "what date" in lower:
            result = await tools.execute("get_local_datetime", {})
            data = result.get("result") or {}
            return f"It's {data.get('time')} on {data.get('date')}."
        if lower.startswith("calculate ") or lower.startswith("what is "):
            expr = text.split(" ", 2)[-1] if lower.startswith("calculate ") else None
            if "calculate" in lower:
                expr = text.lower().split("calculate", 1)[-1].strip(" ?")
                try:
                    result = await tools.execute("calculate", {"expression": expr})
                    if result["status"] == "ok":
                        return f"The result is {result['result']['result']}."
                except Exception:
                    return None
        return None

    async def mark_spoken(self, message_id: str, status: str = "spoken") -> None:
        await self.sessions.update_message(message_id, spoken_status=status)
