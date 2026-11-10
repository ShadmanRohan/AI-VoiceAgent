"""Indic Parler-TTS provider for Bengali speech (gated Hugging Face model)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from loguru import logger

from app.config import Settings, get_settings
from app.speech.interfaces import AudioChunk, TTSProvider

BENGALI_VOICES = {
    "Aditi": (
        "Aditi speaks at a moderate pace and pitch, with a clear, neutral tone "
        "and no emotional emphasis. The recording is very high quality with no background noise."
    ),
    "Arjun": (
        "Arjun speaks at a moderate pace and pitch, with a clear, neutral tone "
        "and no emotional emphasis. The recording is very high quality with no background noise."
    ),
    "Tapan": (
        "Tapan speaks at a moderate pace and pitch, with a clear, neutral tone. "
        "The recording is very high quality with no background noise."
    ),
    "Rashmi": (
        "Rashmi speaks at a moderate pace and pitch, with a clear, neutral tone. "
        "The recording is very high quality with no background noise."
    ),
}


class IndicParlerAvailability:
    def __init__(self) -> None:
        self.available = False
        self.loaded = False
        self.gated = True
        self.error: str | None = None
        self.device: str | None = None
        self.model_id = "ai4bharat/indic-parler-tts"


class IndicParlerTTSProvider(TTSProvider):
    name = "indic_parler"
    sample_rate = 44100

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_id = self.settings.bengali_tts_model
        self._model = None
        self._desc_tok = None
        self._prompt_tok = None
        self._device = "cpu"
        self._lock = asyncio.Lock()
        self._cancel: set[str] = set()
        self._access_checked = False
        self.status = IndicParlerAvailability()
        self.status.model_id = self.model_id

    def voice_description(self, voice: str | None = None) -> str:
        voice = voice or self.settings.bengali_tts_voice or "Aditi"
        if voice in BENGALI_VOICES:
            return BENGALI_VOICES[voice]
        template = self.settings.bengali_tts_description
        return template.format(voice=voice)

    async def check_access(self) -> IndicParlerAvailability:
        # Cache result — probing HF on every sentence adds ~0.5s and can push Edge
        # past Pipecat's audio-context idle timeout.
        if self._access_checked:
            return self.status

        token = self.settings.hf_token()
        if not token:
            self.status.available = False
            self.status.gated = True
            self.status.error = (
                "Indic Parler-TTS is gated on Hugging Face. Accept the model terms at "
                "https://huggingface.co/ai4bharat/indic-parler-tts and set HF_TOKEN."
            )
            self._access_checked = True
            logger.warning(self.status.error)
            return self.status

        try:
            from huggingface_hub import hf_hub_download, model_info

            info = await asyncio.to_thread(model_info, self.model_id, token=token)
            self.status.gated = bool(getattr(info, "gated", False))
            # Probe a tiny metadata file / config
            await asyncio.to_thread(
                hf_hub_download,
                self.model_id,
                "config.json",
                token=token,
            )
            self.status.available = True
            self.status.error = None
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            self.status.available = False
            if "403" in msg or "gated" in msg.lower() or "authorized" in msg.lower():
                self.status.error = (
                    "Indic Parler-TTS is gated on Hugging Face. Accept the model terms at "
                    "https://huggingface.co/ai4bharat/indic-parler-tts and set HF_TOKEN."
                )
            elif "401" in msg:
                self.status.error = "Hugging Face authentication required (set HF_TOKEN)."
            else:
                self.status.error = f"Indic Parler unavailable: {msg[:300]}"
            logger.warning(self.status.error)
        self._access_checked = True
        return self.status

    async def load(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            status = await self.check_access()
            if not status.available:
                raise RuntimeError(status.error or "Indic Parler not available")

            def _load():
                import torch
                from transformers import AutoTokenizer

                try:
                    from parler_tts import ParlerTTSForConditionalGeneration
                except ImportError as exc:
                    raise RuntimeError(
                        "parler-tts package is not installed. "
                        "Install with: pip install git+https://github.com/huggingface/parler-tts.git"
                    ) from exc

                device = "cuda" if torch.cuda.is_available() else "cpu"
                token = self.settings.hf_token()
                model = ParlerTTSForConditionalGeneration.from_pretrained(
                    self.model_id, token=token
                ).to(device)
                desc_tok = AutoTokenizer.from_pretrained("google/flan-t5-large", token=token)
                prompt_tok = AutoTokenizer.from_pretrained(self.model_id, token=token)
                return model, desc_tok, prompt_tok, device

            logger.info("Loading Indic Parler-TTS {}", self.model_id)
            self._model, self._desc_tok, self._prompt_tok, self._device = await asyncio.to_thread(
                _load
            )
            self.status.loaded = True
            self.status.device = self._device
            self.sample_rate = int(getattr(self._model.config, "sampling_rate", 44100))

    async def cancel(self, generation_id: str | None = None) -> None:
        if generation_id:
            self._cancel.add(generation_id)

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        generation_id: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        text = (text or "").strip()
        if not text:
            return
            yield  # pragma: no cover
        gid = generation_id or "default"
        await self.load()
        if gid in self._cancel:
            self._cancel.discard(gid)
            return
            yield  # pragma: no cover

        description = self.voice_description(voice)
        started = time.perf_counter()

        def _generate() -> tuple[bytes, int]:
            import torch

            desc_inputs = self._desc_tok(description, return_tensors="pt").to(self._device)
            prompt_inputs = self._prompt_tok(text, return_tensors="pt").to(self._device)
            with torch.no_grad():
                generation = self._model.generate(
                    input_ids=desc_inputs.input_ids,
                    attention_mask=desc_inputs.attention_mask,
                    prompt_input_ids=prompt_inputs.input_ids,
                    prompt_attention_mask=prompt_inputs.attention_mask,
                )
            audio_arr = generation.cpu().numpy().squeeze()
            if audio_arr.dtype != np.float32:
                audio_arr = audio_arr.astype(np.float32)
            # Optional naive speed change via resampling factor is skipped; speed reserved
            pcm = np.clip(audio_arr, -1.0, 1.0)
            pcm_i16 = (pcm * 32767.0).astype(np.int16).tobytes()
            sr = int(getattr(self._model.config, "sampling_rate", self.sample_rate))
            return pcm_i16, sr

        try:
            pcm, sr = await asyncio.to_thread(_generate)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Indic Parler synthesis failed")
            raise RuntimeError(f"Bengali TTS failed: {exc}") from exc

        if gid in self._cancel:
            self._cancel.discard(gid)
            return
            yield  # pragma: no cover

        elapsed = (time.perf_counter() - started) * 1000
        yield AudioChunk(
            pcm=pcm,
            sample_rate=sr,
            generation_id=gid,
            index=0,
            meta={"provider": self.name, "latency_ms": elapsed, "voice": voice},
        )


_provider: IndicParlerTTSProvider | None = None


def get_indic_parler_provider(settings: Settings | None = None) -> IndicParlerTTSProvider:
    global _provider
    if _provider is None:
        _provider = IndicParlerTTSProvider(settings)
    return _provider
