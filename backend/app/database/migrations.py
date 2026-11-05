"""Lightweight schema migration tracker for SQLite.

Uses SQLAlchemy metadata.create_all for baseline schema and a
`schema_migrations` table for future ordered upgrades.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.database.models import Base

MIGRATIONS: list[tuple[str, str]] = [
    # (version, sql) — add ALTER statements here for upgrades
    # Example: ("2026_11_06_add_foo", "ALTER TABLE memories ADD COLUMN foo TEXT"),
]


async def apply_migrations(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        result = await conn.execute(text("SELECT version FROM schema_migrations"))
        applied = {row[0] for row in result.fetchall()}
        for version, sql in MIGRATIONS:
            if version in applied:
                continue
            await conn.execute(text(sql))
            await conn.execute(
                text("INSERT INTO schema_migrations(version) VALUES (:v)"),
                {"v": version},
            )
