"""Load the bilingual system prompt from the prompts package."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
BILINGUAL_PROMPT_PATH = PROMPTS_DIR / "bilingual_voice_assistant.txt"


@lru_cache
def load_bilingual_prompt() -> str:
    if BILINGUAL_PROMPT_PATH.exists():
        return BILINGUAL_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return (
        "You are a bilingual Bangladeshi Bengali / English voice assistant. "
        "Respond in the user's language with short spoken sentences."
    )
