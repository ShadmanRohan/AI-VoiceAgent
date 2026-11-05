"""STT/TTS adapter contract tests (mocked unit + optional real)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Optional

import pytest

from app.speech.interfaces import AudioChunk, STTProvider, TranscriptEvent, TTSProvider


class FakeSTT(STTProvider):
    name = "fake"

    async def transcribe(self, pcm: bytes, sample_rate: int) -> TranscriptEvent:
        if not pcm:
            return TranscriptEvent(text="", is_final=True, provider=self.name)
        return TranscriptEvent(
            text="hello",
            is_final=True,
            confidence=0.9,
            provider=self.name,
        )


class FakeTTS(TTSProvider):
    name = "fake"
    sample_rate = 24000

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        generation_id: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        if not text.strip():
            return
            yield  # pragma: no cover
        yield AudioChunk(
            pcm=b"\x00\x00" * 160,
            sample_rate=self.sample_rate,
            generation_id=generation_id,
            index=0,
        )


@pytest.mark.asyncio
async def test_stt_adapter_contract():
    stt = FakeSTT()
    empty = await stt.transcribe(b"", 16000)
    assert empty.text == ""
    result = await stt.transcribe(b"\x00\x00" * 100, 16000)
    assert result.is_final
    assert result.text == "hello"


@pytest.mark.asyncio
async def test_tts_adapter_contract():
    tts = FakeTTS()
    chunks = [c async for c in tts.synthesize("Hello there", generation_id="g1")]
    assert chunks
    assert chunks[0].generation_id == "g1"
    assert chunks[0].sample_rate == 24000


@pytest.mark.asyncio
async def test_vad_event_shape():
    # Lightweight stand-in for Silero event semantics used by state machine
    events = [
        {"type": "speech_start", "ts": 0.0},
        {"type": "speech_end", "ts": 1.2},
    ]
    assert events[0]["type"] == "speech_start"
    assert events[1]["ts"] > events[0]["ts"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kokoro_real_if_available():
    """Only runs when kokoro-onnx models can load."""
    try:
        from pipecat.services.kokoro.tts import KokoroTTSService
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Kokoro unavailable: {exc}")
    try:
        tts = KokoroTTSService(
            settings=KokoroTTSService.Settings(voice="af_heart"),
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Kokoro model load failed: {exc}")
    # Service construction success is the contract for local availability
    assert tts is not None
