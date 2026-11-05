"""Database engine, session factory, and schema initialization."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings, get_settings
from app.database.migrations import apply_migrations
from app.database.models import User

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine(settings: Settings | None = None):
    global _engine, _session_factory
    settings = settings or get_settings()
    if _engine is None:
        settings.ensure_dirs()
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            future=True,
        )
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_session_factory(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    get_engine(settings)
    assert _session_factory is not None
    return _session_factory


async def init_db(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    engine = get_engine(settings)
    await apply_migrations(engine)

    factory = get_session_factory(settings)
    async with factory() as session:
        result = await session.execute(select(User).where(User.id == settings.default_user_id))
        user = result.scalar_one_or_none()
        if user is None:
            session.add(
                User(
                    id=settings.default_user_id,
                    name=settings.default_user_name,
                    memory_consent=settings.memory_consent_default,
                )
            )
            await session.commit()


async def reset_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


@asynccontextmanager
async def session_scope(
    settings: Settings | None = None,
) -> AsyncGenerator[AsyncSession, None]:
    factory = get_session_factory(settings)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
