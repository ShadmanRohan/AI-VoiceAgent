"""Pipecat voice pipeline: mic → VAD → STT → Smart Turn → LLM → Kokoro TTS."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from loguru import logger

import asyncio

from app.config import Settings, get_settings
from app.database.session import session_scope
from app.conversation.orchestrator import ConversationOrchestrator
from app.conversation.sessions import SessionService
from app.language.language_router import (
    ConversationLanguageState,
    LanguageMode,
    route_language,
)
from app.models.llm import sanitize_for_tts
from app.monitoring.metrics import APP_METRICS

async def build_voice_services(settings: Settings):
    """Construct STT/LLM/TTS/VAD for one pipeline session (fresh processors)."""
    from pipecat.audio.vad.vad_analyzer import VADParams

    from app.models.pipecat_ollama import NativeOllamaLLMService
    from app.speech.volume_vad import VolumeVADAnalyzer

    # Volume VAD avoids Silero/onnxruntime (wedged telemetry SQLite on this host).
    # Higher min_volume reduces speaker→mic echo triggering false barge-ins that
    # cancel TTS mid-sentence over the tunnel.
    vad = VolumeVADAnalyzer(
        params=VADParams(
            confidence=0.5,
            start_secs=0.25,
            stop_secs=0.45,
            min_volume=0.55,
        )
    )
    stt = await _build_stt(settings)
    tts = await _build_tts(settings)
    turn_analyzer = None

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
                    "qwen3.5:9b",
                    "qwen3.5:4b",
                    "qwen3:1.7b",
                ):
                    if candidate in names:
                        logger.warning("Pipeline LLM fallback {} -> {}", llm_model, candidate)
                        llm_model = candidate
                        break
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not probe Ollama models: {}", exc)

    # Native /api/chat — OpenAI /v1 ignores think=false and stalls ~25s on Qwen3.5.
    llm = NativeOllamaLLMService(
        native_base_url=settings.ollama_base_url.rstrip("/"),
        think=bool(settings.thinking_mode),
        num_ctx=settings.llm_context_tokens,
        base_url=f"{settings.ollama_base_url.rstrip('/')}/v1",
        settings=NativeOllamaLLMService.Settings(
            model=llm_model,
            system_instruction=system,
            temperature=settings.llm_temperature,
        ),
    )

    logger.info("Voice services built (fresh processors; Whisper weights cached)")
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

    from app.speech.cuda_libs import ensure_cuda12_library_path, resolve_whisper_device

    provider = getattr(settings.stt_provider, "value", settings.stt_provider)
    lang_mode = getattr(settings.language_mode, "value", settings.language_mode)
    # Always set language explicitly — Pipecat defaults to EN if omitted.
    # AUTO → None (true multilingual detect). Forced BN previously mangled English.
    if lang_mode == "bangla":
        whisper_language: Language | None = Language.BN
    elif lang_mode == "english":
        whisper_language = Language.EN
    else:
        whisper_language = None

    if provider in {"whisper_turbo", "whisper"}:
        try:
            from pipecat.services.whisper.stt import Model

            from app.speech.whisper_stt import RobustWhisperSTTService

            # faster-whisper needs libcublas.so.12; venv may only expose CUDA 13.
            ensure_cuda12_library_path()
            device, compute_type = resolve_whisper_device("auto")

            # large-v3-turbo is fast but poor on Bengali; prefer full large-v3 for quality.
            raw_model = (settings.stt_model or "").strip()
            if raw_model in {"large-v3-turbo", "turbo", Model.LARGE_V3_TURBO.value}:
                model = Model.LARGE_V3_TURBO
            elif raw_model in {"large-v3", "large", Model.LARGE.value} or not raw_model:
                model = Model.LARGE
            else:
                model = raw_model
            model_name = model if isinstance(model, str) else model.value
            from app.config import resolve_bengali_whisper_path

            bn_resolved = resolve_bengali_whisper_path()
            configured = getattr(settings, "bengali_stt_model", None) or ""
            if configured and __import__("pathlib").Path(configured).joinpath("model.bin").exists():
                bn_path = configured
            else:
                bn_path = str(bn_resolved) if bn_resolved else ""
            kwargs: dict[str, Any] = {
                "device": device,
                "compute_type": compute_type,
                "trailing_silence_secs": 0.75,
                "bengali_model_path": bn_path or None,
                "settings": RobustWhisperSTTService.Settings(
                    model=model_name,
                    language=whisper_language,
                    no_speech_prob=0.6,
                ),
            }
            logger.info(
                "Loading Whisper STT model={} bn_finetune={} device={} compute_type={} language={} (mode={})",
                model_name,
                bn_path if bn_path and __import__("pathlib").Path(bn_path).exists() else "none",
                device,
                compute_type,
                whisper_language if whisper_language is not None else "bn|en",
                lang_mode,
            )
            try:
                kw = dict(kwargs)
                return await asyncio.to_thread(lambda: RobustWhisperSTTService(**kw))
            except Exception as cuda_exc:  # noqa: BLE001
                if device != "cpu":
                    logger.warning(
                        "Whisper CUDA load failed ({}), retrying on CPU", cuda_exc
                    )
                    kwargs["device"] = "cpu"
                    kwargs["compute_type"] = "int8"
                    kw = dict(kwargs)
                    return await asyncio.to_thread(
                        lambda: RobustWhisperSTTService(**kw)
                    )
                raise
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
    """Bengali → Indic Parler or Edge bn-BD; English → Kokoro. Never BN via Kokoro."""
    from app.speech.pipecat_tts import RoutedTTSService

    tts = RoutedTTSService(app_settings=settings)
    out_rate = getattr(settings, "tts_sample_rate", None) or 24000
    if hasattr(tts, "_init_sample_rate"):
        try:
            tts._init_sample_rate(out_rate)
        except Exception:  # noqa: BLE001
            try:
                tts.sample_rate = out_rate
            except Exception:  # noqa: BLE001
                pass
    logger.info(
        "Voice TTS router ready (Bengali: Indic/Edge bn-BD, English: Kokoro {})",
        settings.tts_voice or "af_heart",
    )
    return tts


async def run_bot_with_transport(transport, runner_args=None, settings: Settings | None = None):
    """Run a full Pipecat pipeline against an established transport."""
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
    # Avoid LocalSmartTurn (ONNX). Disable interruption broadcasts so speaker
    # echo / tunnel jitter cannot cancel TTS mid-playback.
    try:
        from pipecat.turns.user_start import (
            TranscriptionUserTurnStartStrategy,
            VADUserTurnStartStrategy,
        )
        from pipecat.turns.user_stop import (
            SpeechTimeoutUserTurnStopStrategy,
            TurnAnalyzerUserTurnStopStrategy,
        )
        from pipecat.turns.user_turn_strategies import UserTurnStrategies

        if turn_analyzer is not None:
            stop = [TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn_analyzer)]
        else:
            stop = [SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=1.0)]
        user_params_kwargs["user_turn_strategies"] = UserTurnStrategies(
            start=[
                VADUserTurnStartStrategy(enable_interruptions=False),
                TranscriptionUserTurnStartStrategy(enable_interruptions=False),
            ],
            stop=stop,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not configure user-turn strategies: {}", exc)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(**user_params_kwargs),
    )

    lang_state = ConversationLanguageState(
        mode=LanguageMode(getattr(settings.language_mode, "value", settings.language_mode))
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
                # Push transcript immediately — never block the turn on memory I/O.
                generation_box["id"] = str(uuid4())
                decision = route_language(text, lang_state)
                lang_directive = {
                    "bangla": (
                        "Respond in Bangladeshi Standard Colloquial Bengali "
                        "(চলিত বাংলা). Keep it spoken and natural."
                    ),
                    "english": "Respond in clear, natural English for speech.",
                    "mixed": (
                        "Respond primarily in Bangladeshi Colloquial Bengali, "
                        "keeping technical English terms as spoken."
                    ),
                }.get(decision.response_language.value)
                if lang_directive:
                    context.add_message(
                        {
                            "role": "system",
                            "content": f"## CURRENT TURN LANGUAGE\n{lang_directive}",
                        }
                    )
                await self.push_frame(frame, direction)
                asyncio.create_task(
                    self._persist_user_turn(text),
                    name="persist-user-turn",
                )
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

        async def _persist_user_turn(self, text: str) -> None:
            sid = session_id_box["id"]
            if not sid:
                return
            try:
                async with session_scope(settings) as db:
                    orch = ConversationOrchestrator(db, settings)
                    orch.session_id = sid
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
            except Exception as exc:  # noqa: BLE001
                logger.warning("Background user-turn persist failed: {}", exc)

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
            audio_out_sample_rate=getattr(settings, "tts_sample_rate", None) or 24000,
        ),
        start_timeout_secs=60.0,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Voice client connected; session={}", session_id_box["id"])
        # Skip auto-greet LLMRun — it races VAD/mic open and cancels the first reply.
        context.add_message(
            {
                "role": "system",
                "content": (
                    "When the user first speaks, greet briefly in their language "
                    "and offer to help. Do not wait for a separate greeting cue."
                ),
            }
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Voice client disconnected")
        sid = session_id_box["id"]
        if sid:
            async with session_scope(settings) as db:
                await SessionService(db).end_session(sid)
                await db.commit()
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
