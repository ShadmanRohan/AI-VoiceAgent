"""Long-term memory CRUD, retrieval, correction, and deletion."""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.database.models import Memory, MemoryEvent, User
from app.memory.embeddings import EmbeddingService, get_embedding_service
from app.memory.vector_store import get_vector_store


SENSITIVE_PATTERNS = [
    re.compile(r"password\s*[:=]", re.I),
    re.compile(r"api[_-]?key", re.I),
    re.compile(r"secret\s*[:=]", re.I),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-like
]


class MemoryService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings | None = None,
        embeddings: EmbeddingService | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.embeddings = embeddings or get_embedding_service()

    async def get_user(self, user_id: str) -> User | None:
        result = await self.session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def set_consent(self, user_id: str, consent: bool) -> User:
        user = await self.get_user(user_id)
        if user is None:
            raise ValueError(f"Unknown user {user_id}")
        user.memory_consent = consent
        await self.session.flush()
        await self._event(user_id, None, "consent_updated", {"consent": consent})
        return user

    def _is_sensitive(self, content: str) -> bool:
        return any(p.search(content) for p in SENSITIVE_PATTERNS)

    async def add_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.5,
        confidence: float = 0.5,
        source_message_id: str | None = None,
        consent_status: str | None = None,
        require_consent: bool = True,
        meta: dict[str, Any] | None = None,
    ) -> Memory:
        content = content.strip()
        if not content:
            raise ValueError("Memory content is empty")
        if self._is_sensitive(content):
            raise ValueError("Refusing to store potentially sensitive content")

        user = await self.get_user(user_id)
        if user is None:
            raise ValueError(f"Unknown user {user_id}")

        if require_consent and not user.memory_consent:
            raise PermissionError("Persistent memory requires user consent")

        # Duplicate / contradiction detection
        existing = await self.find_similar(user_id, content, top_k=3)
        for mem, score in existing:
            if score >= 0.92:
                # Near-duplicate: update instead
                mem.content = content
                mem.importance = max(mem.importance, importance)
                mem.confidence = max(mem.confidence, confidence)
                mem.updated_at = datetime.now(timezone.utc)
                mem.meta = {**(mem.meta or {}), **(meta or {}), "updated_from_duplicate": True}
                await self._reindex(mem)
                await self._event(user_id, mem.id, "updated_duplicate", {"score": score})
                return mem
            if score >= 0.75 and self._looks_like_contradiction(mem.content, content):
                mem.content = content
                mem.importance = importance
                mem.confidence = confidence
                mem.updated_at = datetime.now(timezone.utc)
                mem.meta = {
                    **(mem.meta or {}),
                    **(meta or {}),
                    "corrected_from": mem.content,
                    "contradiction": True,
                }
                await self._reindex(mem)
                await self._event(user_id, mem.id, "corrected", {"score": score})
                return mem

        vector_id = str(uuid.uuid4())
        memory = Memory(
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
            confidence=confidence,
            source_message_id=source_message_id,
            vector_id=vector_id,
            consent_status=consent_status
            or ("granted" if user.memory_consent else "pending"),
            meta=meta or {},
        )
        self.session.add(memory)
        await self.session.flush()
        await self._reindex(memory)
        await self._event(user_id, memory.id, "created", {})
        return memory

    def _looks_like_contradiction(self, old: str, new: str) -> bool:
        preference_words = ("prefer", "like", "favorite", "favourite")
        old_l, new_l = old.lower(), new.lower()
        if any(w in old_l for w in preference_words) and any(w in new_l for w in preference_words):
            return True
        if "prefer" in new_l and ("actually" in new_l or "now" in new_l or "instead" in new_l):
            return True
        return False

    async def _reindex(self, memory: Memory) -> None:
        vector = await self.embeddings.embed_one(memory.content)
        store = get_vector_store(memory.user_id, dimension=vector.shape[0])
        if not memory.vector_id:
            memory.vector_id = str(uuid.uuid4())
        store.add(memory.vector_id, vector)

    async def update_memory(
        self,
        memory_id: str,
        user_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        memory_type: str | None = None,
        active: bool | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Memory:
        memory = await self.get_memory(memory_id, user_id)
        if memory is None:
            raise ValueError("Memory not found")
        if content is not None:
            if self._is_sensitive(content):
                raise ValueError("Refusing to store potentially sensitive content")
            memory.content = content.strip()
        if importance is not None:
            memory.importance = importance
        if confidence is not None:
            memory.confidence = confidence
        if memory_type is not None:
            memory.memory_type = memory_type
        if active is not None:
            memory.active = active
        if meta is not None:
            memory.meta = {**(memory.meta or {}), **meta}
        memory.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        if content is not None and memory.active:
            await self._reindex(memory)
        await self._event(user_id, memory.id, "updated", {})
        return memory

    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        memory = await self.get_memory(memory_id, user_id)
        if memory is None:
            return False
        if memory.vector_id:
            store = get_vector_store(user_id)
            store.remove(memory.vector_id)
        await self._event(user_id, memory.id, "deleted", {"content_preview": memory.content[:80]})
        await self.session.delete(memory)
        await self.session.flush()
        return True

    async def get_memory(self, memory_id: str, user_id: str) -> Memory | None:
        result = await self.session.execute(
            select(Memory).where(Memory.id == memory_id, Memory.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def list_memories(
        self,
        user_id: str,
        *,
        active_only: bool = True,
        memory_type: str | None = None,
        q: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        stmt = select(Memory).where(Memory.user_id == user_id)
        if active_only:
            stmt = stmt.where(Memory.active.is_(True))
        if memory_type:
            stmt = stmt.where(Memory.memory_type == memory_type)
        stmt = stmt.order_by(Memory.updated_at.desc()).limit(limit)
        result = await self.session.execute(stmt)
        memories = list(result.scalars().all())
        if q:
            ql = q.lower()
            memories = [m for m in memories if ql in m.content.lower()]
        return memories

    async def find_similar(
        self, user_id: str, text: str, top_k: int = 5
    ) -> list[tuple[Memory, float]]:
        vector = await self.embeddings.embed_one(text)
        store = get_vector_store(user_id, dimension=vector.shape[0])
        hits = store.search(vector, top_k=top_k)
        if not hits:
            return []
        id_by_vector = {h[0]: h[1] for h in hits}
        result = await self.session.execute(
            select(Memory).where(
                Memory.user_id == user_id,
                Memory.active.is_(True),
                Memory.vector_id.in_(list(id_by_vector.keys())),
            )
        )
        memories = list(result.scalars().all())
        scored = [(m, id_by_vector.get(m.vector_id or "", 0.0)) for m in memories]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    async def retrieve(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float = 0.35,
    ) -> tuple[list[dict[str, Any]], float]:
        """Return ranked memories and retrieval latency in ms."""
        started = time.perf_counter()
        top_k = top_k or self.settings.memory_top_k
        similar = await self.find_similar(user_id, query, top_k=top_k * 2)
        ranked: list[dict[str, Any]] = []
        seen_content: set[str] = set()
        for memory, score in similar:
            if score < min_score:
                continue
            key = memory.content.strip().lower()
            if key in seen_content:
                continue
            seen_content.add(key)
            combined = (
                0.55 * score
                + 0.2 * memory.importance
                + 0.15 * memory.confidence
                + 0.1 * self._recency_score(memory.updated_at)
            )
            ranked.append(
                {
                    "id": memory.id,
                    "content": memory.content,
                    "memory_type": memory.memory_type,
                    "importance": memory.importance,
                    "confidence": memory.confidence,
                    "score": round(combined, 4),
                    "similarity": round(score, 4),
                }
            )
        ranked.sort(key=lambda x: x["score"], reverse=True)
        ranked = ranked[:top_k]
        latency_ms = (time.perf_counter() - started) * 1000
        logger.debug(
            "Retrieved {} memories for user {} in {:.1f}ms ids={}",
            len(ranked),
            user_id,
            latency_ms,
            [m["id"] for m in ranked],
        )
        return ranked, latency_ms

    def _recency_score(self, updated_at: datetime | None) -> float:
        if updated_at is None:
            return 0.0
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        age_days = max((datetime.now(timezone.utc) - updated_at).total_seconds() / 86400, 0)
        return max(0.0, 1.0 - (age_days / 90.0))

    async def forget_matching(self, user_id: str, query: str) -> list[str]:
        similar = await self.find_similar(user_id, query, top_k=5)
        deleted: list[str] = []
        for memory, score in similar:
            if score >= 0.45:
                mid = memory.id
                await self.delete_memory(mid, user_id)
                deleted.append(mid)
        return deleted

    async def export_memories(self, user_id: str) -> dict[str, Any]:
        memories = await self.list_memories(user_id, active_only=False, limit=10_000)
        return {
            "user_id": user_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "memories": [
                {
                    "id": m.id,
                    "type": m.memory_type,
                    "content": m.content,
                    "importance": m.importance,
                    "confidence": m.confidence,
                    "consent_status": m.consent_status,
                    "active": m.active,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "updated_at": m.updated_at.isoformat() if m.updated_at else None,
                    "meta": m.meta,
                }
                for m in memories
            ],
        }

    async def _event(
        self,
        user_id: str,
        memory_id: str | None,
        event_type: str,
        details: dict[str, Any],
    ) -> None:
        self.session.add(
            MemoryEvent(
                user_id=user_id,
                memory_id=memory_id,
                event_type=event_type,
                details=details,
            )
        )
