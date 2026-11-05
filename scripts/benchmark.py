#!/usr/bin/env python3
"""Measure local LLM / memory / TTS chunking latencies.

Definitions match docs/architecture.md and MetricsCollector.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.database.session import init_db, session_scope  # noqa: E402
from app.memory.service import MemoryService  # noqa: E402
from app.models.llm import OllamaLLM  # noqa: E402
from app.speech.tts_chunker import TTSChunker  # noqa: E402


async def bench_llm(n: int = 3) -> dict:
    settings = get_settings()
    llm = OllamaLLM(settings)
    if not await llm.is_available():
        return {"error": "Ollama unavailable"}
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        first = None
        async for _, tok in llm.generate_stream(
            [
                {"role": "system", "content": "Reply briefly."},
                {"role": "user", "content": "Say hello in one short sentence."},
            ]
        ):
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            if len(tok) > 0 and first is not None and (time.perf_counter() - t0) > 0.05:
                break
        if first is not None:
            samples.append(first)
            llm.cancel_all()
    return {
        "metric": "llm_first_token_latency_ms",
        "samples": samples,
        "mean": statistics.fmean(samples) if samples else None,
    }


async def bench_memory() -> dict:
    settings = get_settings()
    await init_db(settings)
    async with session_scope(settings) as db:
        svc = MemoryService(db, settings)
        await svc.set_consent(settings.default_user_id, True)
        await svc.add_memory(
            settings.default_user_id,
            "Prefers concise spoken answers",
            memory_type="preference",
        )
        _, ms = await svc.retrieve(settings.default_user_id, "How do I like answers?")
        return {"metric": "memory_retrieval_latency_ms", "value": ms}


def bench_chunker() -> dict:
    text = (
        "The answer involves three important ideas about systems. "
        "First, keep components modular. Second, stream early. Third, cancel stale work."
    )
    t0 = time.perf_counter()
    chunker = TTSChunker(min_chars=40, max_chars=120)
    ready = []
    for ch in text:
        ready.extend(chunker.add(ch))
    ready.extend(chunker.flush())
    ms = (time.perf_counter() - t0) * 1000
    return {
        "metric": "tts_chunking_overhead_ms",
        "chunks": ready,
        "value": ms,
    }


async def main() -> None:
    results = {
        "llm": await bench_llm(),
        "memory": await bench_memory(),
        "chunker": bench_chunker(),
    }
    out = ROOT / "data" / "benchmark-results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
