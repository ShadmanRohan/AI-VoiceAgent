"""Pipecat Ollama LLM that uses the native /api/chat endpoint.

Ollama's OpenAI-compatible /v1/chat/completions path ignores ``think: false``
for Qwen3.5 and streams ~20–30s of reasoning before content. Native /api/chat
honors ``think`` and keeps voice replies snappy.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx
from loguru import logger
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.utils.types import assert_given


class NativeOllamaLLMService(OLLamaLLMService):
    """OLLamaLLMService backed by native Ollama streaming with think control."""

    def __init__(
        self,
        *,
        native_base_url: str = "http://127.0.0.1:11434",
        think: bool = False,
        num_ctx: int | None = None,
        **kwargs: Any,
    ):
        # Parent still wants an OpenAI-style base_url; we ignore it for generation.
        super().__init__(**kwargs)
        self._native_base_url = native_base_url.rstrip("/")
        self._think = think
        self._num_ctx = num_ctx

    async def get_chat_completions(self, context: LLMContext) -> AsyncIterator[ChatCompletionChunk]:
        adapter = self.get_llm_adapter()
        self._log_llm_response(context)

        params_from_context = await adapter.get_llm_invocation_params(
            context,
            system_instruction=assert_given(self._settings.system_instruction),
            convert_developer_to_user=not self.supports_developer_role,
        )
        messages = list(params_from_context.get("messages") or [])
        model = assert_given(self._settings.model)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": bool(self._think),
            "options": {
                "temperature": self._settings.temperature
                if self._settings.temperature is not None
                else 0.7,
            },
        }
        if self._num_ctx:
            payload["options"]["num_ctx"] = self._num_ctx

        return self._stream_native(payload)

    async def _stream_native(self, payload: dict[str, Any]) -> AsyncIterator[ChatCompletionChunk]:
        """Async generator of OpenAI-shaped chunks from native Ollama streaming."""
        url = f"{self._native_base_url}/api/chat"
        model = payload["model"]
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", errors="replace")[:400]
                        logger.error("Native Ollama error {}: {}", resp.status_code, body)
                        return
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        msg = data.get("message") or {}
                        # Skip thinking tokens entirely when present
                        content = msg.get("content") or ""
                        if content:
                            yield ChatCompletionChunk(
                                id="native-ollama",
                                choices=[
                                    Choice(
                                        index=0,
                                        delta=ChoiceDelta(content=content, role="assistant"),
                                        finish_reason=None,
                                    )
                                ],
                                created=0,
                                model=model,
                                object="chat.completion.chunk",
                            )
                        if data.get("done"):
                            yield ChatCompletionChunk(
                                id="native-ollama",
                                choices=[
                                    Choice(
                                        index=0,
                                        delta=ChoiceDelta(),
                                        finish_reason="stop",
                                    )
                                ],
                                created=0,
                                model=model,
                                object="chat.completion.chunk",
                            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Native Ollama stream failed: {}", exc)
