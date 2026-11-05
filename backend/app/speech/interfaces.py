"""Provider interfaces for STT and TTS adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TranscriptEvent:
    text: str
    is_final: bool
    confidence: float | None = None
    start_ms: float | None = None
    end_ms: float | None = None
    provider: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioChunk:
    pcm: bytes
    sample_rate: int
    channels: int = 1
    generation_id: str | None = None
    index: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class STTProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def transcribe(self, pcm: bytes, sample_rate: int) -> TranscriptEvent:
        raise NotImplementedError

    async def stream_transcribe(
        self, pcm_iter: AsyncIterator[bytes], sample_rate: int
    ) -> AsyncIterator[TranscriptEvent]:
        """Default: buffer all audio then transcribe once."""
        chunks: list[bytes] = []
        async for chunk in pcm_iter:
            chunks.append(chunk)
        yield await self.transcribe(b"".join(chunks), sample_rate)

    async def cancel(self) -> None:
        return None


class TTSProvider(ABC):
    name: str = "base"
    sample_rate: int = 24000

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        generation_id: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        raise NotImplementedError
        yield  # pragma: no cover

    async def cancel(self, generation_id: str | None = None) -> None:
        return None
