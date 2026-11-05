"""Split streaming LLM text into speakable TTS chunks without cutting words."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$")
CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s+")


@dataclass
class TTSChunker:
    min_chars: int = 40
    max_chars: int = 180
    _buffer: str = ""
    chunks: list[str] = field(default_factory=list)

    def add(self, token: str) -> list[str]:
        self._buffer += token
        ready: list[str] = []
        while True:
            chunk = self._extract_one()
            if not chunk:
                break
            ready.append(chunk)
            self.chunks.append(chunk)
        return ready

    def flush(self) -> list[str]:
        text = self._buffer.strip()
        self._buffer = ""
        if text:
            self.chunks.append(text)
            return [text]
        return []

    def reset(self) -> None:
        self._buffer = ""
        self.chunks.clear()

    def _extract_one(self) -> str | None:
        buf = self._buffer
        if len(buf.strip()) < self.min_chars and not SENTENCE_END.search(buf):
            # Wait for more unless we already have a clear sentence end after min
            if not re.search(r"[.!?]\s", buf):
                return None

        # Prefer sentence boundaries within max_chars
        window = buf[: self.max_chars + 1]
        matches = list(re.finditer(r"[.!?](?:\s+|$)", window))
        if matches:
            # Use the last sentence end that leaves at least min_chars when possible
            end = matches[-1].end()
            for m in matches:
                if m.end() >= self.min_chars:
                    end = m.end()
                    break
            chunk = buf[:end].strip()
            self._buffer = buf[end:]
            return chunk or None

        if len(buf) >= self.max_chars:
            # Break on clause or last whitespace to avoid cutting words
            window = buf[: self.max_chars]
            clause = list(CLAUSE_BREAK.finditer(window))
            if clause and clause[-1].end() >= self.min_chars:
                end = clause[-1].end()
            else:
                space = window.rfind(" ")
                end = space if space >= self.min_chars else self.max_chars
            chunk = buf[:end].strip()
            self._buffer = buf[end:]
            return chunk or None

        return None
