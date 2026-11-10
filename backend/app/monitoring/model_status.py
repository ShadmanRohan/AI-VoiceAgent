"""Model readiness reporting for bilingual stack."""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from app.config import get_settings
from app.speech.providers.indic_parler_tts import get_indic_parler_provider


def _val(x: Any) -> Any:
    return x.value if hasattr(x, "value") else x


async def collect_model_status() -> dict[str, Any]:
    settings = get_settings()
    out: dict[str, Any] = {"models": {}}

    # Ollama LLM
    ollama: dict[str, Any] = {
        "name": settings.ollama_model,
        "runtime": "ollama",
        "fallback": settings.ollama_model_fallback,
        "installed": False,
        "downloaded": False,
        "error": None,
    }
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            ollama["installed"] = True
            ollama["downloaded"] = settings.ollama_model in names or any(
                n.startswith(settings.ollama_model) for n in names
            )
            ollama["available_models"] = names
            if not ollama["downloaded"]:
                ollama["error"] = (
                    f"Run: ollama pull {settings.ollama_model}"
                )
    except Exception as exc:  # noqa: BLE001
        ollama["error"] = str(exc)
    out["models"]["llm"] = ollama

    # STT
    out["models"]["stt"] = {
        "name": settings.stt_model,
        "provider": _val(settings.stt_provider),
        "runtime": "faster-whisper / pipecat",
        "fallback": _val(settings.stt_fallback),
        "note": "Segmented (not token-streaming). Partial UI transcripts are provisional.",
    }

    # Bengali TTS
    indic = get_indic_parler_provider(settings)
    status = await indic.check_access()
    out["models"]["bengali_tts"] = {
        "name": settings.bengali_tts_model,
        "provider": _val(settings.bengali_tts_provider),
        "runtime": "parler-tts",
        "gated": status.gated,
        "available": True,  # Edge bn-BD fallback keeps voice usable without HF_TOKEN
        "indic_available": status.available,
        "loaded": status.loaded,
        "device": status.device,
        "voices": ["Aditi", "Arjun", "Tapan", "Rashmi"],
        "fallback": "edge_tts bn-BD-NabanitaNeural / bn-BD-PradeepNeural",
        "error": None,
        "indic_error": status.error,
        "hf_token_configured": bool(settings.hf_token()),
        "accent_note": (
            "Active Bengali voice uses Edge bn-BD neural TTS until Indic Parler "
            "HF access is configured. Native-speaker listening still required "
            "before claiming accent authenticity."
        ),
    }

    # English TTS
    out["models"]["english_tts"] = {
        "name": "hexgrad/Kokoro-82M (kokoro-onnx)",
        "provider": _val(settings.english_tts_provider),
        "voice": settings.tts_voice,
        "available": True,
    }

    # Embeddings
    out["models"]["embeddings"] = {
        "name": settings.embedding_model,
        "version": settings.embedding_index_version,
        "legacy": settings.legacy_embedding_model,
        "prefixes": "query: / passage: (E5)",
    }

    out["language"] = {
        "mode": _val(settings.language_mode),
        "primary": settings.primary_language,
        "secondary": settings.secondary_language,
    }
    return out
