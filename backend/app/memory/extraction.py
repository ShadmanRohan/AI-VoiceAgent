"""Heuristic + optional LLM-assisted memory candidate extraction."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
from loguru import logger

from app.config import Settings, get_settings

REMEMBER_RE = re.compile(
    r"\b(remember(?:\s+that)?|don't forget|please note that)\b[:\s]*(.+)",
    re.I | re.S,
)
FORGET_RE = re.compile(
    r"\b(forget|delete|remove)\b.+\b(preference|memory|that|about)\b(.+)?",
    re.I,
)
PREFER_RE = re.compile(
    r"\b(?:i\s+)?(?:prefer|like|love)\b(.+)",
    re.I,
)
WHAT_REMEMBER_RE = re.compile(
    r"\bwhat\s+do\s+you\s+remember\b|\bdo\s+you\s+remember\b",
    re.I,
)


def detect_memory_intent(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    if WHAT_REMEMBER_RE.search(text):
        return {"action": "list", "query": text}
    forget = FORGET_RE.search(text)
    if forget:
        return {"action": "forget", "query": text}
    remember = REMEMBER_RE.search(text)
    if remember:
        content = remember.group(2).strip(" .")
        return {
            "action": "store",
            "candidate": {
                "type": "preference" if "prefer" in text.lower() else "fact",
                "content": content,
                "importance": 0.85,
                "confidence": 0.9,
            },
        }
    prefer = PREFER_RE.search(text)
    if prefer and ("prefer" in text.lower() or "favorite" in text.lower()):
        content = text.strip(" .")
        return {
            "action": "store",
            "candidate": {
                "type": "preference",
                "content": content,
                "importance": 0.8,
                "confidence": 0.85,
            },
        }
    return None


async def extract_candidates_with_llm(
    user_text: str,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Optional structured extraction via Ollama. Falls back to heuristics."""
    settings = settings or get_settings()
    heuristic = detect_memory_intent(user_text)
    if heuristic and heuristic.get("action") == "store":
        return [heuristic["candidate"]]

    prompt = (
        "Extract durable personal facts/preferences worth remembering from the user message. "
        "Return JSON list of objects with keys type, content, importance (0-1), confidence (0-1). "
        "If nothing should be stored, return []. Do not invent facts.\n\n"
        f"User message: {user_text}"
    )
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/chat",
                json={
                    "model": settings.ollama_model,
                    "stream": False,
                    "think": False,
                    "messages": [
                        {"role": "system", "content": "Return only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "[]")
            # Strip fences if present
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            data = json.loads(content)
            if isinstance(data, dict):
                data = data.get("memories") or data.get("candidates") or []
            if not isinstance(data, list):
                return []
            cleaned: list[dict[str, Any]] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                c = str(item.get("content", "")).strip()
                if not c:
                    continue
                cleaned.append(
                    {
                        "type": str(item.get("type", "fact")),
                        "content": c,
                        "importance": float(item.get("importance", 0.5)),
                        "confidence": float(item.get("confidence", 0.5)),
                    }
                )
            return cleaned
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM memory extraction failed: {}", exc)
        return []
