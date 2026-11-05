"""Conversation language detection and response routing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from app.language.bengali_normalizer import has_bengali, normalize_whitespace_punct

BN_SCRIPT = re.compile(r"[\u0980-\u09FF]")
LATIN = re.compile(r"[A-Za-z]")
SWITCH_TO_EN = re.compile(
    r"(english(?:\s*এ)?\s*(?:বলো|বলুন|উত্তর)|এবার\s*ইংরেজ|speak\s+english|answer\s+in\s+english|switch\s+to\s+english)",
    re.I,
)
SWITCH_TO_BN = re.compile(
    r"(বাংলা(?:য়|তে)?\s*(?:বলো|বলুন|উত্তর)|bangla(?:y)?\s*(?:bolo|bolun)|speak\s+bengali|answer\s+in\s+bengali|switch\s+to\s+(?:bangla|bengali))",
    re.I,
)


class LanguageMode(str, Enum):
    AUTO = "auto"
    BANGLA = "bangla"
    ENGLISH = "english"


class DetectedLanguage(str, Enum):
    BANGLA = "bangla"
    ENGLISH = "english"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class LanguageDecision:
    mode: LanguageMode
    detected: DetectedLanguage
    response_language: DetectedLanguage
    stt_language: str | None  # whisper language code or None for auto
    tts_provider: str  # indic_parler | kokoro
    reason: str = ""
    sticky_preference: DetectedLanguage | None = None


@dataclass
class ConversationLanguageState:
    mode: LanguageMode = LanguageMode.AUTO
    sticky_preference: DetectedLanguage | None = None
    last_user_language: DetectedLanguage | None = None
    last_response_language: DetectedLanguage | None = None
    history: list[str] = field(default_factory=list)

    def set_mode(self, mode: LanguageMode | str) -> None:
        self.mode = LanguageMode(mode)


def detect_text_language(text: str) -> DetectedLanguage:
    text = normalize_whitespace_punct(text or "")
    if not text:
        return DetectedLanguage.UNKNOWN
    bn = len(BN_SCRIPT.findall(text))
    lat = len(LATIN.findall(text))
    total = bn + lat
    if total == 0:
        return DetectedLanguage.UNKNOWN
    bn_ratio = bn / total
    lat_ratio = lat / total
    # One isolated English technical term inside Bengali should stay Bangla
    if bn_ratio >= 0.35 and lat_ratio >= 0.15:
        return DetectedLanguage.MIXED
    if bn_ratio >= 0.2:
        return DetectedLanguage.BANGLA
    if lat_ratio >= 0.5:
        return DetectedLanguage.ENGLISH
    return DetectedLanguage.UNKNOWN


def apply_explicit_switch(text: str, state: ConversationLanguageState) -> DetectedLanguage | None:
    if SWITCH_TO_EN.search(text or ""):
        state.sticky_preference = DetectedLanguage.ENGLISH
        return DetectedLanguage.ENGLISH
    if SWITCH_TO_BN.search(text or ""):
        state.sticky_preference = DetectedLanguage.BANGLA
        return DetectedLanguage.BANGLA
    return None


def route_language(
    text: str,
    state: ConversationLanguageState,
    *,
    stt_language_hint: str | None = None,
) -> LanguageDecision:
    detected = detect_text_language(text)
    if stt_language_hint:
        hint = stt_language_hint.lower()
        if hint.startswith("bn") and detected == DetectedLanguage.UNKNOWN:
            detected = DetectedLanguage.BANGLA
        elif hint.startswith("en") and detected == DetectedLanguage.UNKNOWN:
            detected = DetectedLanguage.ENGLISH

    explicit = apply_explicit_switch(text, state)
    if explicit:
        detected = DetectedLanguage.MIXED if detected == DetectedLanguage.MIXED else detected

    mode = state.mode
    if mode == LanguageMode.BANGLA:
        response = DetectedLanguage.BANGLA
        stt_lang = "bn"
        reason = "manual_bangla"
    elif mode == LanguageMode.ENGLISH:
        response = DetectedLanguage.ENGLISH
        stt_lang = "en"
        reason = "manual_english"
    else:
        stt_lang = None  # auto / multilingual whisper
        if explicit:
            response = explicit
            reason = "explicit_switch"
        elif state.sticky_preference and detected in {
            DetectedLanguage.UNKNOWN,
            DetectedLanguage.MIXED,
        }:
            # Don't flip language for one English tech word in Bangla context
            response = state.sticky_preference
            reason = "sticky_preference"
        elif detected == DetectedLanguage.ENGLISH:
            response = DetectedLanguage.ENGLISH
            reason = "detected_english"
        elif detected in {DetectedLanguage.BANGLA, DetectedLanguage.MIXED}:
            response = DetectedLanguage.BANGLA
            reason = "detected_bangla_or_mixed"
        elif state.last_user_language:
            response = (
                DetectedLanguage.BANGLA
                if state.last_user_language != DetectedLanguage.ENGLISH
                else DetectedLanguage.ENGLISH
            )
            reason = "continuity"
        else:
            response = DetectedLanguage.BANGLA  # primary language default
            reason = "default_bangla"

    if detected in {DetectedLanguage.BANGLA, DetectedLanguage.ENGLISH, DetectedLanguage.MIXED}:
        state.last_user_language = detected
    state.last_response_language = response
    state.history.append(f"{detected.value}->{response.value}:{reason}")

    tts_provider = "kokoro" if response == DetectedLanguage.ENGLISH else "indic_parler"
    return LanguageDecision(
        mode=mode,
        detected=detected,
        response_language=response,
        stt_language=stt_lang,
        tts_provider=tts_provider,
        reason=reason,
        sticky_preference=state.sticky_preference,
    )


def whisper_language_for_mode(mode: LanguageMode) -> str | None:
    if mode == LanguageMode.BANGLA:
        return "bn"
    if mode == LanguageMode.ENGLISH:
        return "en"
    return None  # auto / multilingual
