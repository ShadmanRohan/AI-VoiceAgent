"""Pipecat TTS service that routes Bengali → Indic/Edge and English → Kokoro.

Stops Kokoro from receiving বাংলা script (which produces unintelligible
"accents" because Kokoro has no Bengali phonemizer path).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

from app.config import Settings, get_settings
from app.language.bengali_normalizer import prepare_tts_text
from app.speech.tts_router import TTSRouter


class RoutedTTSService(TTSService):
    """Sentence-aggregating TTS that delegates to TTSRouter."""

    Settings = TTSSettings
    _settings: Settings

    def __init__(self, *, app_settings: Settings | None = None, **kwargs):
        self._app_settings = app_settings or get_settings()
        sample_rate = self._app_settings.tts_sample_rate or 24000
        default = self.Settings(
            model=None,
            voice=self._app_settings.tts_voice or "af_heart",
            language=None,
        )
        # Edge bn-BD often needs 3–6s before first PCM (network + decode). Pipecat's
        # default 3s stop timeout ends the audio context as "no audio" mid-wait.
        kwargs.setdefault("stop_frame_timeout_s", 20.0)
        # Keep pushing audio even if the client is briefly quiet / slow over tunnel.
        kwargs.setdefault("push_silence_after_stop", True)
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            settings=default,
            **kwargs,
        )
        self._router = TTSRouter(self._app_settings)

    def can_generate_metrics(self) -> bool:
        return True

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text = prepare_tts_text(text or "")
        if not text:
            return
            yield  # pragma: no cover
        try:
            await self.start_tts_usage_metrics(text)
            first = True
            async for chunk in self._router.synthesize(text, generation_id=context_id):
                if first:
                    await self.stop_ttfb_metrics()
                    first = False
                # Resample if provider SR differs from transport SR
                audio = chunk.pcm
                sr = chunk.sample_rate
                if sr != self.sample_rate and hasattr(self, "_resampler"):
                    audio = await self._resampler.resample(audio, sr, self.sample_rate)
                    sr = self.sample_rate
                elif sr != self.sample_rate:
                    from pipecat.audio.utils import create_stream_resampler

                    if not hasattr(self, "_lazy_resampler"):
                        self._lazy_resampler = create_stream_resampler()
                    audio = await self._lazy_resampler.resample(audio, sr, self.sample_rate)
                    sr = self.sample_rate
                yield TTSAudioRawFrame(
                    audio=audio,
                    sample_rate=sr,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Routed TTS failed")
            yield ErrorFrame(error=f"TTS failed: {exc}")
        finally:
            await self.stop_ttfb_metrics()
