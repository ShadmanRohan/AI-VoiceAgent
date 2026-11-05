"""Bengali / bilingual text normalization (non-destructive)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

BN_RANGE = re.compile(r"[\u0980-\u09FF]")
MULTI_SPACE = re.compile(r"[ \t\u00a0]+")
BN_PUNCT_FIX = {
    "\u0964": "।",  # Devanagari danda → Bengali danda if misencoded
}
DIGIT_BN_TO_ASCII = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
DIGIT_ASCII_TO_BN = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

# Lightweight TTS pronunciation helpers (validated, conservative)
ABBREV_TTS = {
    "LLM": "এল এল এম",
    "STT": "এস টি টি",
    "TTS": "টি টি এস",
    "AI": "এ আই",
    "API": "এ পি আই",
    "GPU": "জি পি ইউ",
    "CPU": "সি পি ইউ",
    "VAD": "ভি এ ডি",
}


@dataclass
class NormalizedText:
    original_transcript: str
    normalized_transcript: str
    tts_ready_text: str


def has_bengali(text: str) -> bool:
    return bool(BN_RANGE.search(text or ""))


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    for bad, good in BN_PUNCT_FIX.items():
        text = text.replace(bad, good)
    text = text.replace("\u200c", "").replace("\u200d", "")  # ZWNJ/ZWJ cleanup
    text = MULTI_SPACE.sub(" ", text)
    text = re.sub(r"\s+([।.!?,;:])", r"\1", text)
    return text.strip()


def normalize_whitespace_punct(text: str) -> str:
    text = normalize_unicode(text)
    # Unify quotes / dashes lightly
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return text.strip()


def prepare_tts_text(text: str, *, numerals: str = "keep") -> str:
    """Prepare speech-friendly text without destroying the stored transcript."""
    out = normalize_whitespace_punct(text)
    # Expand a few technical acronyms only when surrounded by word boundaries
    for abbr, spoken in ABBREV_TTS.items():
        out = re.sub(rf"\b{re.escape(abbr)}\b", spoken, out)
    if numerals == "bengali" and has_bengali(out):
        out = out.translate(DIGIT_ASCII_TO_BN)
    elif numerals == "ascii":
        out = out.translate(DIGIT_BN_TO_ASCII)
    return out


def process_transcript(text: str) -> NormalizedText:
    original = text or ""
    normalized = normalize_whitespace_punct(original)
    tts_ready = prepare_tts_text(normalized)
    return NormalizedText(
        original_transcript=original,
        normalized_transcript=normalized,
        tts_ready_text=tts_ready,
    )
