"""LocalVoiceAI FastAPI entrypoint with REST + SmallWebRTC voice transport."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

# Cursor agent shells inherit log FDs; those + model IO wedge this host in D-state.
def _close_inherited_cursor_fds(limit: int = 1024) -> None:
    for fd in range(3, limit):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if "Cursor/logs" in target or "onnxruntime.db" in target:
            try:
                os.close(fd)
            except OSError:
                pass


_close_inherited_cursor_fds()

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
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
    # Make libcublas.so.12 visible before any Whisper/CTranslate2 import path runs.
    try:
        from app.speech.cuda_libs import ensure_cuda12_library_path

        ensure_cuda12_library_path()
    except Exception as exc:  # noqa: BLE001
        logger.debug("CUDA 12 path setup skipped: {}", exc)

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
            "http://192.168.10.146:5173",
            "http://192.168.10.146:7860",
        ],
        allow_origin_regex=r"https://.*\.(trycloudflare\.com|loca\.lt)",
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
            "voice_ws": "/ws",
        }

    @app.websocket("/ws")
    async def voice_websocket(websocket: WebSocket):
        """Tunnel-friendly voice path: mic/TTS audio over WebSocket (no WebRTC/TURN)."""
        await websocket.accept()
        settings = get_settings()
        ws_id = f"ws-{id(websocket)}"
        logger.info("Voice WebSocket connected {}", ws_id)
        try:
            from pipecat.serializers.protobuf import ProtobufFrameSerializer
            from pipecat.transports.websocket.fastapi import (
                FastAPIWebsocketParams,
                FastAPIWebsocketTransport,
            )
            from app.conversation.pipeline import run_bot_with_transport

            transport = FastAPIWebsocketTransport(
                websocket=websocket,
                params=FastAPIWebsocketParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    add_wav_header=False,
                    audio_in_sample_rate=settings.sample_rate,
                    audio_out_sample_rate=getattr(settings, "tts_sample_rate", None) or 24000,
                    serializer=ProtobufFrameSerializer(),
                    allowed_origins=[],  # allow Cloudflare tunnel + LAN origins
                ),
            )

            @transport.event_handler("on_client_disconnected")
            async def _on_ws_disconnected(_transport, _client):
                logger.info("Voice WebSocket disconnected {}", ws_id)

            await run_bot_with_transport(transport, settings=settings)
        except WebSocketDisconnect:
            logger.info("Voice WebSocket client left {}", ws_id)
        except Exception:
            logger.exception("Voice WebSocket session failed {}", ws_id)
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001
                pass

    async def _wait_ice_complete(connection, timeout_s: float = 8.0) -> None:
        """Block until local ICE candidates are in the SDP answer."""
        pc = getattr(connection, "pc", None) or getattr(connection, "_pc", None)
        if pc is None:
            return
        deadline = asyncio.get_event_loop().time() + timeout_s
        while pc.iceGatheringState != "complete":
            if asyncio.get_event_loop().time() >= deadline:
                logger.warning("ICE gathering timed out in state={}", pc.iceGatheringState)
                break
            await asyncio.sleep(0.05)

    @app.api_route("/api/offer", methods=["POST", "PATCH"])
    async def webrtc_offer(request: Request):
        """WebRTC signaling: POST=SDP offer/answer, PATCH=trickle ICE candidates."""
        try:
            from aiortc.sdp import candidate_from_sdp
            from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
            from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
            from pipecat.transports.base_transport import TransportParams
        except ImportError as exc:
            raise HTTPException(
                503,
                f"WebRTC dependencies unavailable: {exc}. Install pipecat-ai[webrtc].",
            ) from exc

        body = await request.json()
        settings = get_settings()

        # Remote browsers trickle ICE via PATCH after the answer. Without this,
        # connections stay stuck on "connecting" over the public internet.
        if request.method == "PATCH":
            pc_id = body.get("pc_id")
            candidates = body.get("candidates") or []
            connection = _webrtc_connections.get(pc_id) if pc_id else None
            if connection is None:
                raise HTTPException(404, "Peer connection not found")
            for raw in candidates:
                cand_str = (raw or {}).get("candidate") or ""
                if cand_str:
                    candidate = candidate_from_sdp(cand_str)
                    candidate.sdpMid = raw.get("sdp_mid")
                    candidate.sdpMLineIndex = raw.get("sdp_mline_index")
                else:
                    candidate = None
                await connection.add_ice_candidate(candidate)
            return {"ok": True}

        sdp = body.get("sdp")
        typ = body.get("type")
        pc_id = body.get("pc_id")
        if not sdp or not typ:
            raise HTTPException(400, "sdp and type are required")

        if pc_id and pc_id in _webrtc_connections:
            connection = _webrtc_connections[pc_id]
            await connection.renegotiate(
                sdp=sdp, type=typ, restart_pc=bool(body.get("restart_pc"))
            )
            await _wait_ice_complete(connection)
        else:
            from aiortc.rtcconfiguration import RTCIceServer

            ice_objs = [
                RTCIceServer(
                    urls=s["urls"],
                    username=s.get("username"),
                    credential=s.get("credential"),
                )
                for s in settings.webrtc_ice_servers()
            ]
            connection = SmallWebRTCConnection(
                ice_servers=ice_objs or None,
                connection_timeout_secs=90,
            )
            await connection.initialize(sdp=sdp, type=typ)
            await _wait_ice_complete(connection)
            pc_id = connection.pc_id
            _webrtc_connections[pc_id] = connection

            transport = SmallWebRTCTransport(
                webrtc_connection=connection,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    audio_in_sample_rate=settings.sample_rate,
                    # Kokoro is 24 kHz; keep mic path at 16 kHz for Whisper.
                    audio_out_sample_rate=getattr(settings, "tts_sample_rate", None)
                    or 24000,
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
