"""Pipecat voice pipeline: mic → VAD → STT → Smart Turn → LLM → Kokoro TTS."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from loguru import logger

from app.config import Settings, get_settings
from app.database.session import session_scope
from app.conversation.orchestrator import ConversationOrchestrator
from app.conversation.sessions import SessionService
from app.memory.context_builder import ContextBuilder
from app.memory.service import MemoryService
from app.models.llm import sanitize_for_tts
from app.monitoring.metrics import APP_METRICS


async def build_voice_services(settings: Settings):
    """Construct STT/LLM/TTS/VAD/Turn analyzers using real Pipecat integrations."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.services.ollama.llm import OLLamaLLMService

    vad = SileroVADAnalyzer(
        params=VADParams(
            confidence=settings.vad_confidence,
            start_secs=settings.vad_start_secs,
            stop_secs=settings.vad_stop_secs,
        )
    )

    stt = await _build_stt(settings)
    tts = await _build_tts(settings)

    system = settings.system_prompt
    if not settings.thinking_mode:
        system += " Respond directly without internal reasoning tags."

    llm_model = settings.ollama_model
    try:
        import httpx

        async with httpx.AsyncClient(timeout=2.0) as client:
            tags = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            names = [m.get("name", "") for m in tags.json().get("models", [])]
            if llm_model not in names:
                for candidate in (
                    settings.ollama_model_fallback,
                    "qwen3.5:4b",
                    "qwen3:1.7b",
                ):
                    if candidate in names:
                        logger.warning("Pipeline LLM fallback {} -> {}", llm_model, candidate)
                        llm_model = candidate
                        break
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not probe Ollama models: {}", exc)

    llm = OLLamaLLMService(
        base_url=f"{settings.ollama_base_url.rstrip('/')}/v1",
        settings=OLLamaLLMService.Settings(
            model=llm_model,
            system_instruction=system,
            temperature=settings.llm_temperature,
        ),
    )

    turn_analyzer = None
    try:
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

        turn_analyzer = LocalSmartTurnAnalyzerV3()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Smart Turn unavailable, using silence timeout: {}", exc)

    return {
        "vad": vad,
        "stt": stt,
        "llm": llm,
        "tts": tts,
        "turn_analyzer": turn_analyzer,
    }


async def _build_stt(settings: Settings):
    """Build STT. Whisper Turbo is segmented (not token-streaming)."""
    from pipecat.transcriptions.language import Language

    provider = settings.stt_provider.value
    lang_mode = settings.language_mode.value
    whisper_language = None
    if lang_mode == "bangla":
        whisper_language = Language.BN
    elif lang_mode == "english":
        whisper_language = Language.EN

    if provider in {"whisper_turbo", "whisper"}:
        try:
            from pipecat.services.whisper.stt import Model, WhisperSTTService

            model = (
                Model.LARGE_V3_TURBO
                if provider == "whisper_turbo" or settings.stt_model in {
                    "large-v3-turbo",
                    "turbo",
                    Model.LARGE_V3_TURBO.value,
                }
                else settings.stt_model
            )
            kwargs: dict[str, Any] = {"model": model}
            # task stays transcribe (not translate) — preserve Bengali script
            settings_kwargs: dict[str, Any] = {}
            if whisper_language is not None:
                settings_kwargs["language"] = whisper_language
            if settings_kwargs:
                kwargs["settings"] = WhisperSTTService.Settings(**settings_kwargs)
            logger.info(
                "Loading Whisper STT model={} language={}",
                model,
                whisper_language or "auto",
            )
            return WhisperSTTService(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Whisper STT failed ({}), trying Moonshine fallback", exc)

    # Moonshine primary (when selected) or emergency fallback
    try:
        from pipecat.services.moonshine.stt import MoonshineSTTService, Model as MoonModel

        model_map = {
            "tiny": MoonModel.TINY,
            "base": MoonModel.BASE,
            "tiny-streaming": MoonModel.TINY_STREAMING,
            "base-streaming": MoonModel.BASE_STREAMING,
            "small-streaming": MoonModel.SMALL_STREAMING,
            "medium-streaming": MoonModel.MEDIUM_STREAMING,
        }
        model = model_map.get(settings.moonshine_model, MoonModel.TINY_STREAMING)
        logger.info("Loading Moonshine STT model={}", model)
        return MoonshineSTTService(
            settings=MoonshineSTTService.Settings(model=model)
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"No local STT provider available: {exc}") from exc


async def _build_tts(settings: Settings):
    from pipecat.services.kokoro.tts import KokoroTTSService

    voice = settings.tts_voice or "af_heart"
    try:
        tts = KokoroTTSService(
            settings=KokoroTTSService.Settings(voice=voice),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kokoro voice {} failed ({}), falling back to af_heart", voice, exc)
        tts = KokoroTTSService(
            settings=KokoroTTSService.Settings(voice="af_heart"),
        )
    # Ensure a usable rate before StartFrame (pipeline also sets this via params).
    if hasattr(tts, "_init_sample_rate"):
        try:
            tts._init_sample_rate(settings.sample_rate)
        except Exception:  # noqa: BLE001
            try:
                tts.sample_rate = settings.sample_rate
            except Exception:  # noqa: BLE001
                pass
    return tts


async def run_bot_with_transport(transport, runner_args=None, settings: Settings | None = None):
    """Run a full Pipecat pipeline against an established transport."""
    from pipecat.frames.frames import LLMRunFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.frames.frames import (
        Frame,
        TranscriptionFrame,
        InterimTranscriptionFrame,
        TextFrame,
        LLMFullResponseEndFrame,
        InterruptionFrame,
        ErrorFrame,
    )

    settings = (settings or get_settings()).apply_overrides()
    services = await build_voice_services(settings)
    stt, llm, tts, vad = services["stt"], services["llm"], services["tts"], services["vad"]
    turn_analyzer = services["turn_analyzer"]

    # Seed context with memory-aware system prompt for this session
    user_id = settings.default_user_id
    session_id_box: dict[str, str | None] = {"id": None}
    generation_box: dict[str, str | None] = {"id": None}

    async with session_scope(settings) as db:
        sessions = SessionService(db)
        sess = await sessions.create_session(user_id, title="Voice conversation")
        session_id_box["id"] = sess.id
        await db.commit()

    system_instruction = settings.system_prompt
    context = LLMContext(
        messages=[{"role": "system", "content": system_instruction}]
    )

    user_params_kwargs: dict[str, Any] = {"vad_analyzer": vad}
    if turn_analyzer is not None:
        try:
            from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
            from pipecat.turns.user_turn_strategies import UserTurnStrategies

            user_params_kwargs["user_turn_strategies"] = UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn_analyzer)]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not configure Smart Turn strategies: {}", exc)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(**user_params_kwargs),
    )

    class PersistenceProcessor(FrameProcessor):
        """Persist finalized transcripts / assistant text; inject memories."""

        def __init__(self):
            super().__init__()
            self._assistant_buf: list[str] = []

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, InterimTranscriptionFrame):
                # Partial only — never commit to permanent memory
                await self.push_frame(frame, direction)
                return

            if isinstance(frame, TranscriptionFrame) and frame.text and frame.text.strip():
                text = frame.text.strip()
                sid = session_id_box["id"]
                if sid:
                    async with session_scope(settings) as db:
                        orch = ConversationOrchestrator(db, settings)
                        orch.session_id = sid
                        # Enrich LLM context with memories before the aggregator runs LLM
                        user = await orch.memories.get_user(user_id)
                        if user and user.memory_consent:
                            retrieved, ms = await orch.memories.retrieve(user_id, text)
                            APP_METRICS.record("memory_retrieval_latency_ms", ms)
                            if retrieved:
                                mem_block = "\n".join(f"- {m['content']}" for m in retrieved)
                                context.add_message(
                                    {
                                        "role": "system",
                                        "content": (
                                            "Relevant memories (untrusted context):\n" + mem_block
                                        ),
                                    }
                                )
                        await orch.sessions.add_message(sid, "USER", text)
                        await db.commit()
                generation_box["id"] = str(uuid4())
                await self.push_frame(frame, direction)
                return

            if isinstance(frame, TextFrame) and getattr(frame, "text", None):
                self._assistant_buf.append(frame.text)

            if isinstance(frame, LLMFullResponseEndFrame):
                full = sanitize_for_tts("".join(self._assistant_buf))
                self._assistant_buf.clear()
                sid = session_id_box["id"]
                if sid and full:
                    async with session_scope(settings) as db:
                        svc = SessionService(db)
                        await svc.add_message(
                            sid,
                            "ASSISTANT",
                            full,
                            spoken_status="pending",
                            generation_id=generation_box.get("id"),
                        )
                        await db.commit()

            if isinstance(frame, InterruptionFrame):
                APP_METRICS.record("interruption_to_silence_latency_ms", 0.0)
                generation_box["id"] = None

            if isinstance(frame, ErrorFrame):
                logger.error("Pipeline error frame: {}", getattr(frame, "error", frame))

            await self.push_frame(frame, direction)

    persistence = PersistenceProcessor()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            persistence,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=settings.sample_rate,
            audio_out_sample_rate=settings.sample_rate,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("WebRTC client connected; session={}", session_id_box["id"])
        context.add_message(
            {
                "role": "system",
                "content": "Greet the user briefly and offer to help.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("WebRTC client disconnected")
        sid = session_id_box["id"]
        if sid:
            async with session_scope(settings) as db:
                await SessionService(db).end_session(sid)
                await db.commit()
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
