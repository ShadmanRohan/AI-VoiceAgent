"""API validation and health tests."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.mark.asyncio
async def test_health_and_settings():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        r = await client.get("/api/settings")
        assert r.status_code == 200
        assert "ollama_model" in r.json()

        r = await client.put("/api/settings", json={"tts_speed": 1.1})
        assert r.status_code == 200
        assert r.json()["tts_speed"] == 1.1

        r = await client.post("/api/sessions", json={"title": "API Test"})
        assert r.status_code == 200
        sid = r.json()["id"]

        r = await client.get(f"/api/sessions/{sid}")
        assert r.status_code == 200
        assert r.json()["title"] == "API Test"

        r = await client.delete(f"/api/sessions/{sid}")
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_memory_api_consent_gate():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/api/memories", json={"content": "I like Python"})
        assert r.status_code == 403
        r = await client.put("/api/memories/consent", json={"consent": True})
        assert r.status_code == 200
        r = await client.post(
            "/api/memories",
            json={"content": "I like Python", "memory_type": "preference"},
        )
        assert r.status_code == 200
        mid = r.json()["id"]
        r = await client.delete(f"/api/memories/{mid}")
        assert r.status_code == 200
