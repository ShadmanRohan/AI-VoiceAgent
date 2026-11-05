"""Ollama LLM client with streaming, cancellation, and non-thinking mode."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import httpx
from loguru import logger

from app.config import Settings, get_settings

THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>", re.I)
CONTROL_TOKEN_RE = re.compile(r"<\|[^>]+\|>")


def sanitize_for_tts(text: str) -> str:
    text = THINK_TAG_RE.sub("", text)
    text = CONTROL_TOKEN_RE.sub("", text)
    # Remove leftover markdown that sounds bad when spoken
    text = re.sub(r"[#*_`]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class GenerationHandle:
    generation_id: str
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    started_at: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    completed: bool = False
    cancelled: bool = False
    text: str = ""


class OllamaLLM:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._active: dict[str, GenerationHandle] = {}

    @property
    def base_url(self) -> str:
        return self.settings.ollama_base_url.rstrip("/")

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{self.base_url}/api/tags")
            r.raise_for_status()
            return [m.get("name", "") for m in r.json().get("models", [])]

    def cancel(self, generation_id: str) -> bool:
        handle = self._active.get(generation_id)
        if not handle:
            return False
        handle.cancel_event.set()
        handle.cancelled = True
        return True

    def cancel_all(self) -> None:
        for gid in list(self._active.keys()):
            self.cancel(gid)

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        generation_id: str | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """Yield (generation_id, token_text). Stops on cancellation."""
        gid = generation_id or str(uuid4())
        handle = GenerationHandle(generation_id=gid)
        self._active[gid] = handle
        model = model or self.settings.ollama_model
        temperature = (
            self.settings.llm_temperature if temperature is None else temperature
        )

        # Prefer an installed model if the configured tag is missing.
        try:
            available = await self.list_models()
            if model not in available:
                for candidate in (
                    self.settings.ollama_model,
                    getattr(self.settings, "ollama_model_fallback", None) or "qwen3:1.7b",
                    "qwen3.5:4b",
                    "qwen3:1.7b",
                    self.settings.ollama_model_quality,
                ):
                    if candidate and candidate in available:
                        logger.warning(
                            "Model {} unavailable; falling back to {}", model, candidate
                        )
                        model = candidate
                        break
                else:
                    if available:
                        logger.warning(
                            "Model {} unavailable; falling back to {}",
                            model,
                            available[0],
                        )
                        model = available[0]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not list Ollama models before generate: {}", exc)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": self.settings.llm_context_tokens,
            },
        }
        # Qwen3 thinking control — Ollama supports think flag on recent versions
        if not self.settings.thinking_mode:
            payload["think"] = False

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as resp:
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        raise RuntimeError(
                            f"Ollama error {resp.status_code}: {body.decode()[:300]}"
                        )
                    async for line in resp.aiter_lines():
                        if handle.cancel_event.is_set():
                            handle.cancelled = True
                            break
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        msg = data.get("message") or {}
                        token = msg.get("content") or ""
                        # Skip thinking channel content if present
                        if msg.get("thinking") and not token:
                            continue
                        if token:
                            if handle.first_token_at is None:
                                handle.first_token_at = time.perf_counter()
                            handle.text += token
                            yield gid, token
                        if data.get("done"):
                            break
            handle.completed = not handle.cancelled
        except Exception:
            logger.exception("LLM generation failed for {}", gid)
            raise
        finally:
            self._active.pop(gid, None)

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        generation_id: str | None = None,
    ) -> tuple[str, GenerationHandle]:
        gid = generation_id or str(uuid4())
        parts: list[str] = []
        handle = GenerationHandle(generation_id=gid)
        async for _, token in self.generate_stream(
            messages, model=model, generation_id=gid
        ):
            parts.append(token)
        handle.text = "".join(parts)
        handle.completed = True
        return sanitize_for_tts(handle.text), handle
