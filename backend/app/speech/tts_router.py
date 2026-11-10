"""Route synthesis to Indic Parler (Bengali) or Kokoro (English)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from loguru import logger

from app.config import Settings, get_settings
from app.language.bengali_normalizer import has_bengali, prepare_tts_text
from app.language.language_router import DetectedLanguage
from app.speech.interfaces import AudioChunk, TTSProvider
from app.speech.providers.edge_bengali_tts import EdgeBengaliTTSProvider
from app.speech.providers.indic_parler_tts import get_indic_parler_provider


class TTSRouter(TTSProvider):
    name = "tts_router"
    sample_rate = 24000

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._kokoro = None
        self._indic = get_indic_parler_provider(self.settings)
        self._edge = EdgeBengaliTTSProvider(self.settings)
        self._cancel: set[str] = set()

    async def _ensure_kokoro(self):
        if self._kokoro is not None:
            return self._kokoro
        from pipecat.services.kokoro.tts import KokoroTTSService

        voice = self.settings.tts_voice or "af_heart"
        try:
            self._kokoro = KokoroTTSService(
                settings=KokoroTTSService.Settings(voice=voice)
            )
        except Exception:
            self._kokoro = KokoroTTSService(
                settings=KokoroTTSService.Settings(voice="af_heart")
            )
        if hasattr(self._kokoro, "_init_sample_rate"):
            try:
                self._kokoro._init_sample_rate(self.settings.tts_sample_rate)
            except Exception:  # noqa: BLE001
                pass
        return self._kokoro

    async def cancel(self, generation_id: str | None = None) -> None:
        if generation_id:
            self._cancel.add(generation_id)
        await self._indic.cancel(generation_id)

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        generation_id: str | None = None,
        response_language: DetectedLanguage | str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        text = prepare_tts_text(text or "")
        if not text:
            return
            yield  # pragma: no cover
        gid = generation_id or "default"
        if gid in self._cancel:
            self._cancel.discard(gid)
            return
            yield  # pragma: no cover

        lang = response_language
        if isinstance(lang, str):
            try:
                lang = DetectedLanguage(lang)
            except ValueError:
                lang = None
        if lang is None:
            lang = (
                DetectedLanguage.BANGLA
                if has_bengali(text)
                else DetectedLanguage.ENGLISH
            )

        use_bengali = lang in {DetectedLanguage.BANGLA, DetectedLanguage.MIXED} or has_bengali(
            text
        )
        if use_bengali:
            bn_voice = voice or self.settings.bengali_tts_voice
            # Prefer local Indic Parler when HF access is configured.
            status = await self._indic.check_access()
            if status.available:
                async for chunk in self._indic.synthesize(
                    text, voice=bn_voice, speed=speed, generation_id=gid
                ):
                    if gid in self._cancel:
                        self._cancel.discard(gid)
                        return
                    yield chunk
                return

            # Interim: bn-BD Edge neural voice (never Kokoro — no Bengali phonemes).
            logger.debug(
                "Indic Parler unavailable ({}); using Edge bn-BD TTS fallback",
                status.error or "no access",
            )
            async for chunk in self._edge.synthesize(
                text, voice=bn_voice, speed=speed, generation_id=gid
            ):
                if gid in self._cancel:
                    self._cancel.discard(gid)
                    return
                yield chunk
            return

        # English / Latin → Kokoro (local)
        async for chunk in self._synthesize_kokoro(text, generation_id=gid):
            if gid in self._cancel:
                self._cancel.discard(gid)
                return
            yield chunk

    async def _synthesize_kokoro(
        self, text: str, *, generation_id: str
    ) -> AsyncIterator[AudioChunk]:
        import asyncio
        from pathlib import Path

        import numpy as np

        def _run():
            from kokoro_onnx import Kokoro

            cache = Path.home() / ".cache/pipecat/kokoro-onnx"
            model = cache / "kokoro-v1.0.onnx"
            voices = cache / "voices-v1.0.bin"
            if not model.exists() or not voices.exists():
                raise RuntimeError("Kokoro model files missing; start app once to download.")
            k = Kokoro(str(model), str(voices))
            voice = self.settings.tts_voice or "af_heart"
            try:
                samples, sr = k.create(text, voice=voice, speed=float(self.settings.tts_speed))
            except Exception:
                samples, sr = k.create(text, voice="af_heart", speed=1.0)
            pcm = (np.asarray(samples) * 32767).astype(np.int16).tobytes()
            return pcm, int(sr)

        try:
            pcm, sr = await asyncio.to_thread(_run)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Kokoro synthesis failed")
            raise RuntimeError(f"English TTS failed: {exc}") from exc
        yield AudioChunk(
            pcm=pcm,
            sample_rate=sr,
            generation_id=generation_id,
            index=0,
            meta={"provider": "kokoro"},
        )
