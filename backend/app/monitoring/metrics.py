"""In-process latency metrics with precise definitions."""

from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


METRIC_DEFINITIONS = {
    "speech_start_detection_latency_ms": (
        "Time from first audio frame containing speech energy until VAD emits speech-start."
    ),
    "end_of_turn_latency_ms": (
        "Time from last speech frame until Smart Turn / turn-end decision."
    ),
    "stt_finalization_latency_ms": (
        "Time from turn-end until final transcript is available."
    ),
    "llm_first_token_latency_ms": (
        "Time from LLM request start until first streamed token."
    ),
    "tts_first_audio_latency_ms": (
        "Time from first TTS chunk request until first PCM audio bytes."
    ),
    "first_audible_response_latency_ms": (
        "Time from finalized user turn until first AI audio frame is sent to the client."
    ),
    "interruption_to_silence_latency_ms": (
        "Time from interruption detection until AI audio output is cancelled/stopped."
    ),
    "memory_retrieval_latency_ms": (
        "Time to embed the query and return ranked long-term memories."
    ),
}


@dataclass
class MetricsCollector:
    window: int = 200
    _values: dict[str, deque[float]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=200))
    )
    _last: dict[str, float] = field(default_factory=dict)

    def record(self, name: str, value_ms: float, **labels: Any) -> None:
        self._values[name].append(float(value_ms))
        self._last[name] = float(value_ms)

    def timer(self, name: str):
        return _Timer(self, name)

    def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {"definitions": METRIC_DEFINITIONS, "metrics": {}}
        for name, values in self._values.items():
            if not values:
                continue
            data = list(values)
            out["metrics"][name] = {
                "count": len(data),
                "last": self._last.get(name),
                "mean": statistics.fmean(data),
                "p50": statistics.median(data),
                "min": min(data),
                "max": max(data),
            }
        return out


class _Timer:
    def __init__(self, collector: MetricsCollector, name: str) -> None:
        self.collector = collector
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed = (time.perf_counter() - self.start) * 1000
        self.collector.record(self.name, elapsed)


# Shared process metrics
APP_METRICS = MetricsCollector()
