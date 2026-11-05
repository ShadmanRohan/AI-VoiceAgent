"""LocalVoiceAI FastAPI entrypoint with REST + SmallWebRTC voice transport."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router as api_router
from app.config import get_settings, reload_settings
from app.database.session import init_db
from app.monitoring.hardware import build_hardware_report, resolve_profile

# Active WebRTC connections keyed by pc_id
_webrtc_connections: dict[str, Any] = {}
_bot_tasks: dict[str, asyncio.Task] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    await init_db(settings)
    report = await build_hardware_report(settings)
    profile = resolve_profile(settings, report)
    logger.info(
        "LocalVoiceAI starting on {}:{} | profile={} | gpu={} | ollama_running={}",
        settings.host,
        settings.port,
        profile.get("profile"),
        report.gpu.name or "none",
        report.ollama.running,
    )
    if report.missing_dependencies:
        logger.warning("Missing dependencies: {}", report.missing_dependencies)
    for note in report.notes:
        logger.info("Note: {}", note)

    # Apply auto profile defaults into runtime overrides without wiping user settings.
    # Always reconcile ollama_model against downloaded models to avoid hard failures.
    current = settings.runtime_overrides()
    desired_model = profile.get("ollama_model") or settings.ollama_model
    if desired_model != settings.ollama_model or (
        settings.performance_profile.value == "auto" and "resolved_profile" not in current
    ):
        auto_overrides = {
            k: v
            for k, v in profile.items()
            if k != "profile" and k in settings.public_dict()
        }
        fill = {k: v for k, v in auto_overrides.items() if k not in current}
        fill["ollama_model"] = desired_model
        fill["resolved_profile"] = profile["profile"]
        settings.save_overrides({**current, **fill})
        reload_settings()
        logger.info("Active Ollama model: {}", desired_model)

    yield

    # Graceful shutdown: cancel bot tasks
    for pc_id, task in list(_bot_tasks.items()):
        task.cancel()
        logger.info("Cancelled bot task for {}", pc_id)
    _bot_tasks.clear()
    for conn in list(_webrtc_connections.values()):
        try:
            await conn.disconnect()
        except Exception:  # noqa: BLE001
            pass
    _webrtc_connections.clear()
    logger.info("LocalVoiceAI shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="LocalVoiceAI",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            settings.frontend_origin,
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:7860",
            "http://localhost:7860",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api")

    @app.get("/")
    async def root():
        return {
            "name": "LocalVoiceAI",
            "docs": "/docs",
            "health": "/api/health",
            "voice_offer": "/api/offer",
        }

    @app.post("/api/offer")
    async def webrtc_offer(request: Request):
        """WebRTC signaling endpoint for SmallWebRTCTransport clients."""
        try:
            from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
            from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
            from pipecat.transports.base_transport import TransportParams
        except ImportError as exc:
            raise HTTPException(
                503,
                f"WebRTC dependencies unavailable: {exc}. Install pipecat-ai[webrtc].",
            ) from exc

        body = await request.json()
        sdp = body.get("sdp")
        typ = body.get("type")
        pc_id = body.get("pc_id")
        if not sdp or not typ:
            raise HTTPException(400, "sdp and type are required")

        settings = get_settings()

        if pc_id and pc_id in _webrtc_connections:
            connection = _webrtc_connections[pc_id]
            await connection.renegotiate(sdp=sdp, type=typ)
        else:
            connection = SmallWebRTCConnection()
            await connection.initialize(sdp=sdp, type=typ)
            pc_id = connection.pc_id
            _webrtc_connections[pc_id] = connection

            transport = SmallWebRTCTransport(
                webrtc_connection=connection,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    audio_in_sample_rate=settings.sample_rate,
                    audio_out_sample_rate=settings.sample_rate,
                ),
            )

            async def _run():
                try:
                    from app.conversation.pipeline import run_bot_with_transport

                    await run_bot_with_transport(transport, settings=settings)
                except asyncio.CancelledError:
                    logger.info("Bot cancelled for {}", pc_id)
                except Exception:
                    logger.exception("Bot failed for {}", pc_id)
                finally:
                    _webrtc_connections.pop(pc_id, None)
                    _bot_tasks.pop(pc_id, None)

            _bot_tasks[pc_id] = asyncio.create_task(_run())

            @connection.event_handler("closed")
            async def on_closed(conn):
                logger.info("Connection closed {}", conn.pc_id)
                task = _bot_tasks.pop(conn.pc_id, None)
                if task:
                    task.cancel()
                _webrtc_connections.pop(conn.pc_id, None)

        answer = connection.get_answer()
        return answer

    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
