"""Session and message persistence helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database.models import Message, Session, SessionSummary


class SessionService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_session(self, user_id: str, title: str = "New conversation") -> Session:
        session = Session(user_id=user_id, title=title, status="active")
        self.db.add(session)
        await self.db.flush()
        return session

    async def get_session(self, session_id: str, user_id: str | None = None) -> Session | None:
        stmt = (
            select(Session)
            .where(Session.id == session_id)
            .options(selectinload(Session.messages), selectinload(Session.summaries))
        )
        if user_id:
            stmt = stmt.where(Session.user_id == user_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_sessions(
        self,
        user_id: str,
        *,
        q: str | None = None,
        limit: int = 50,
    ) -> list[Session]:
        stmt = (
            select(Session)
            .where(Session.user_id == user_id)
            .order_by(Session.updated_at.desc())
            .limit(limit)
        )
        if q:
            stmt = stmt.where(
                or_(Session.title.ilike(f"%{q}%"), Session.summary.ilike(f"%{q}%"))
            )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def delete_session(self, session_id: str, user_id: str) -> bool:
        session = await self.get_session(session_id, user_id)
        if session is None:
            return False
        await self.db.delete(session)
        await self.db.flush()
        return True

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        finalized: bool = True,
        generated: bool = True,
        spoken_status: str = "not_applicable",
        interrupted: bool = False,
        generation_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Message:
        msg = Message(
            session_id=session_id,
            role=role.upper(),
            content=content,
            finalized=finalized,
            generated=generated,
            spoken_status=spoken_status,
            interrupted=interrupted,
            generation_id=generation_id,
            meta=meta or {},
        )
        self.db.add(msg)
        sess = await self.db.get(Session, session_id)
        if sess:
            sess.updated_at = datetime.now(timezone.utc)
            if sess.title == "New conversation" and role.upper() == "USER" and content:
                sess.title = content[:80]
        await self.db.flush()
        return msg

    async def update_message(
        self,
        message_id: str,
        *,
        content: str | None = None,
        spoken_status: str | None = None,
        interrupted: bool | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Message | None:
        msg = await self.db.get(Message, message_id)
        if msg is None:
            return None
        if content is not None:
            msg.content = content
        if spoken_status is not None:
            msg.spoken_status = spoken_status
        if interrupted is not None:
            msg.interrupted = interrupted
        if meta is not None:
            msg.meta = {**(msg.meta or {}), **meta}
        await self.db.flush()
        return msg

    async def recent_messages(
        self, session_id: str, limit: int = 20
    ) -> list[Message]:
        result = await self.db.execute(
            select(Message)
            .where(Message.session_id == session_id, Message.finalized.is_(True))
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        msgs = list(result.scalars().all())
        msgs.reverse()
        return msgs

    async def end_session(self, session_id: str, summary: str | None = None) -> Session | None:
        sess = await self.db.get(Session, session_id)
        if sess is None:
            return None
        sess.status = "ended"
        sess.ended_at = datetime.now(timezone.utc)
        if summary:
            sess.summary = summary
            self.db.add(SessionSummary(session_id=session_id, summary=summary))
        await self.db.flush()
        return sess

    async def search_messages(self, user_id: str, q: str, limit: int = 50) -> list[dict[str, Any]]:
        result = await self.db.execute(
            select(Message, Session)
            .join(Session, Session.id == Message.session_id)
            .where(Session.user_id == user_id, Message.content.ilike(f"%{q}%"))
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        rows = []
        for msg, sess in result.all():
            rows.append(
                {
                    "message_id": msg.id,
                    "session_id": sess.id,
                    "session_title": sess.title,
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                }
            )
        return rows
