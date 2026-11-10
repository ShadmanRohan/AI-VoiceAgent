"""Post-corrections for Bengali ASR artifacts from Whisper.

Unicode cleanup uses bnunicodenormalizer when installed; lexical repairs cover
common phonetic confusions that Unicode normalization cannot fix.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Bengali dependent vowel signs (কার) that cannot start a word alone.
_LEADING_VOWEL_SIGN = re.compile(
    r"^[\u09be\u09bf\u09c0\u09c1\u09c2\u09c3\u09c4\u09c7\u09c8\u09cb\u09cc]+\s*"
)

# High-confidence phonetic / domain repairs from live logs.
_REPLACEMENTS = (
    (re.compile(r"বুজতে"), "বুঝতে"),
    (re.compile(r"কছা"), "কথা"),
    (re.compile(r"কথাথা"), "কথা"),
    (re.compile(r"পরাও"), "পারো"),
    (re.compile(r"পারারো"), "পারো"),
    (re.compile(r"আনেক\s*কিছু"), "অনেক কিছু"),
    (re.compile(r"আনেকিছু"), "অনেক কিছু"),
    (re.compile(r"অনেকিছু"), "অনেক কিছু"),
    (re.compile(r"মুন\s*ভারো"), "মন ভালো"),
    (re.compile(r"মুনভারো"), "মন ভালো"),
    (re.compile(r"মনভারো"), "মন ভালো"),
    (re.compile(r"মনভাল[োো]"), "মন ভালো"),
    (re.compile(r"ভাগ\s+(?=খে)"), "ভাত "),
    (re.compile(r"আতিকে"), "আজকে"),
    (re.compile(r"ইনিভার্সিটি"), "ইউনিভার্সিটি"),
    (re.compile(r"ইউননিভার্সিটি"), "ইউনিভার্সিটি"),
    (re.compile(r"ইউনিভারসিটি"), "ইউনিভার্সিটি"),
    (re.compile(r"ঢাকাকা"), "ঢাকা"),
    (re.compile(r"প্রক্রাম"), "প্রোগ্রাম"),
    (re.compile(r"অ্যাড্মিশন"), "এডমিশন"),
    (re.compile(r"এড্মিশন"), "এডমিশন"),
    (re.compile(r"পুরোপুরী"), "পুরোপুরি"),
    # After orphan-কার repair these often become আনি/আদে (\b is weak for BN)
    (re.compile(r"^আনি(?=\s|$|[।,?\!])"), "আপনি"),
    (re.compile(r"^আদে(?=\s|$|[।,?\!])"), "আপনার"),
    (re.compile(r"বলছেছে"), "বলছি"),
    (re.compile(r"\s+"), " "),
)

# Full-token repeats only (never single-letter bridges like অনেক+কিছু).
_DUP_TOKEN = re.compile(r"(^|[\s,।])([\u0980-\u09FF]{2,12})(?:\s+\2)+(?=[\s,।!?]|$)")
_BN_LETTERS = re.compile(r"[\u0980-\u09FF]")


@lru_cache(maxsize=1)
def _bn_unicode_normalizer():
    try:
        from bnunicodenormalizer import Normalizer

        return Normalizer(allow_english=True)
    except Exception:  # noqa: BLE001
        return None


def _normalize_unicode_tokens(text: str) -> str:
    """Apply bnunicodenormalizer per whitespace token when available."""
    norm = _bn_unicode_normalizer()
    if norm is None:
        return text
    out: list[str] = []
    for tok in text.split(" "):
        if not tok:
            continue
        if not _BN_LETTERS.search(tok):
            out.append(tok)
            continue
        # Keep trailing Latin/punct attached after normalize of the BN core.
        m = re.match(r"^([\u0980-\u09FF]+)(.*)$", tok)
        if not m:
            out.append(tok)
            continue
        core, tail = m.group(1), m.group(2)
        try:
            result = norm(core)
            fixed = (result or {}).get("normalized") or core
        except Exception:  # noqa: BLE001
            fixed = core
        out.append(fixed + tail)
    return " ".join(out)


def fix_bengali_asr(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text

    # Orphan leading কার (often after VAD chops "আ"): "া মার" → "আমার"
    if _LEADING_VOWEL_SIGN.match(text):
        rest = _LEADING_VOWEL_SIGN.sub("", text, count=1).lstrip()
        if rest:
            text = "আ" + rest

    text = _normalize_unicode_tokens(text)

    for pat, repl in _REPLACEMENTS:
        text = pat.sub(repl, text)

    prev = None
    while prev != text:
        prev = text
        text = _DUP_TOKEN.sub(r"\1\2", text)

    return text.strip()
