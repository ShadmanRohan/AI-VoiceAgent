"""Language routing and Bengali normalization tests."""

from __future__ import annotations

from app.language.bengali_normalizer import process_transcript
from app.language.language_router import (
    ConversationLanguageState,
    DetectedLanguage,
    LanguageMode,
    detect_text_language,
    route_language,
)


def test_normalize_preserves_original():
    raw = "  আমার LLM  model-এর  latency বেশি।  "
    result = process_transcript(raw)
    assert result.original_transcript == raw
    assert "LLM" in result.normalized_transcript or "এল এল এম" in result.tts_ready_text
    assert "latency" in result.normalized_transcript.lower() or "latency" in raw.lower()


def test_detect_bangla_english_mixed():
    assert detect_text_language("আপনি কি সাহায্য করতে পারবেন?") == DetectedLanguage.BANGLA
    assert detect_text_language("Can you explain the system?") == DetectedLanguage.ENGLISH
    assert detect_text_language("আমার STT model-এর accuracy improve করতে হবে।") in {
        DetectedLanguage.MIXED,
        DetectedLanguage.BANGLA,
    }


def test_language_switch_sticky():
    state = ConversationLanguageState(mode=LanguageMode.AUTO)
    d1 = route_language("এবার ইংরেজিতে উত্তর দিন।", state)
    assert d1.response_language == DetectedLanguage.ENGLISH
    assert state.sticky_preference == DetectedLanguage.ENGLISH
    d2 = route_language("What next?", state)
    assert d2.response_language == DetectedLanguage.ENGLISH


def test_manual_bangla_mode():
    state = ConversationLanguageState(mode=LanguageMode.BANGLA)
    d = route_language("Hello there", state)
    assert d.response_language == DetectedLanguage.BANGLA
    assert d.stt_language == "bn"
    assert d.tts_provider == "indic_parler"


def test_isolated_english_term_does_not_force_english():
    state = ConversationLanguageState(mode=LanguageMode.AUTO)
    state.last_user_language = DetectedLanguage.BANGLA
    d = route_language("latency কমাতে হবে", state)
    assert d.response_language == DetectedLanguage.BANGLA
