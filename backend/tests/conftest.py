"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import pytest_asyncio

# Isolate test DB before importing app modules that cache settings
TEST_DIR = Path(__file__).resolve().parent / "_tmp"
TEST_DIR.mkdir(exist_ok=True)
TEST_DB = TEST_DIR / "test.db"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB}"
os.environ["DATA_DIR"] = str(TEST_DIR)
os.environ["SETTINGS_PATH"] = str(TEST_DIR / "settings.json")
os.environ["DEFAULT_USER_ID"] = "test-user"

from app.config import reload_settings  # noqa: E402
from app.database.session import init_db, reset_engine, session_scope  # noqa: E402
from app.memory.vector_store import reset_vector_stores  # noqa: E402


@pytest_asyncio.fixture(autouse=True)
async def clean_db():
    if TEST_DB.exists():
        TEST_DB.unlink()
    for p in TEST_DIR.glob("vectors/**/*"):
        if p.is_file():
            p.unlink()
    reset_vector_stores()
    await reset_engine()
    reload_settings()
    settings = reload_settings()
    await init_db(settings)
    yield
    await reset_engine()
    reset_vector_stores()


@pytest_asyncio.fixture
async def db():
    settings = reload_settings()
    async with session_scope(settings) as session:
        yield session
