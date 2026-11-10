"""Bengali TTS via Microsoft Edge neural voices (bn-BD).

Interim fallback while Indic Parler remains gated. Requires network.
Produces Bangladeshi Bengali speech (Nabanita / Pradeep). Never feed বাংলা
to Kokoro — it has no Bengali phonemes and sounds wrong.
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from loguru import logger

from app.config import Settings, get_settings
from app.speech.interfaces import AudioChunk, TTSProvider

VOICE_MAP = {
    "Aditi": "bn-BD-NabanitaNeural",
    "Rashmi": "bn-BD-NabanitaNeural",
    "Nabanita": "bn-BD-NabanitaNeural",
    "Arjun": "bn-BD-PradeepNeural",
    "Tapan": "bn-BD-PradeepNeural",
    "Pradeep": "bn-BD-PradeepNeural",
    "bn-BD-NabanitaNeural": "bn-BD-NabanitaNeural",
    "bn-BD-PradeepNeural": "bn-BD-PradeepNeural",
}


class EdgeBengaliTTSProvider(TTSProvider):
    name = "edge_bengali"
    sample_rate = 24000

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._cancel: set[str] = set()

    def resolve_voice(self, voice: str | None = None) -> str:
        raw = voice or self.settings.bengali_tts_voice or "Aditi"
        return VOICE_MAP.get(raw, "bn-BD-NabanitaNeural")

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
        if gid in self._cancel:
            self._cancel.discard(gid)
            return
            yield  # pragma: no cover

        edge_voice = self.resolve_voice(voice)
        rate = "+0%"
        spd = speed if speed is not None else self.settings.tts_speed
        if spd and float(spd) != 1.0:
            pct = int((float(spd) - 1.0) * 100)
            rate = f"{pct:+d}%"

        try:
            import edge_tts

            communicate = edge_tts.Communicate(text, voice=edge_voice, rate=rate)
            mp3 = bytearray()
            async for chunk in communicate.stream():
                if gid in self._cancel:
                    self._cancel.discard(gid)
                    return
                    yield  # pragma: no cover
                if chunk["type"] == "audio":
                    mp3.extend(chunk["data"])
            if not mp3:
                raise RuntimeError("Edge TTS returned empty audio")
            pcm = await asyncio.to_thread(self._mp3_to_pcm, bytes(mp3))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Edge Bengali TTS failed")
            raise RuntimeError(f"Bengali TTS failed: {exc}") from exc

        if gid in self._cancel:
            self._cancel.discard(gid)
            return
            yield  # pragma: no cover

        # 20ms frames keep the audio-context idle timer warm during playback.
        frame = int(self.sample_rate * 0.02) * 2
        index = 0
        for i in range(0, len(pcm), frame):
            if gid in self._cancel:
                self._cancel.discard(gid)
                return
            yield AudioChunk(
                pcm=pcm[i : i + frame],
                sample_rate=self.sample_rate,
                generation_id=gid,
                index=index,
                meta={"provider": self.name, "voice": edge_voice},
            )
            index += 1
            # Yield to the event loop so Pipecat can drain the queue promptly.
            if index % 5 == 0:
                await asyncio.sleep(0)

    def _mp3_to_pcm(self, mp3: bytes) -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.mp3"
            src.write_bytes(mp3)
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                str(self.sample_rate),
                "pipe:1",
            ]
            proc = subprocess.run(cmd, capture_output=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode("utf-8", errors="replace")[:300])
            return proc.stdout
