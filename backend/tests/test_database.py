"""Database initialization and session persistence tests."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.conversation.sessions import SessionService
from app.database.models import Message, Session, User


@pytest.mark.asyncio
async def test_database_initialization(db):
    settings = get_settings()
    result = await db.execute(select(User).where(User.id == settings.default_user_id))
    user = result.scalar_one()
    assert user.name


@pytest.mark.asyncio
async def test_session_creation(db):
    settings = get_settings()
    svc = SessionService(db)
    sess = await svc.create_session(settings.default_user_id, title="Test")
    assert sess.id
    assert sess.title == "Test"


@pytest.mark.asyncio
async def test_conversation_persistence(db):
    settings = get_settings()
    svc = SessionService(db)
    sess = await svc.create_session(settings.default_user_id)
    await svc.add_message(sess.id, "USER", "Hello")
    await svc.add_message(sess.id, "ASSISTANT", "Hi there", spoken_status="pending")
    loaded = await svc.get_session(sess.id, settings.default_user_id)
    assert loaded is not None
    assert len(loaded.messages) == 2
    roles = {m.role for m in loaded.messages}
    assert roles == {"USER", "ASSISTANT"}
