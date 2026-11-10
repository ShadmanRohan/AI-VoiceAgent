"""Whisper STT tuned for Bengali + English.

Uses a Bengali-finetuned CT2 model when available, multilingual large-v3 for
English. Pads audio and keeps extra preroll so VAD does not eat the first
Bangla syllable.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast

import numpy as np
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.types import assert_given

from app.language.bengali_asr_fix import fix_bengali_asr

_EN_PROMPT = (
    "Clear conversational English. How are you? "
    "I want to know about Dhaka University admission process."
)

_REPEAT_RE = re.compile(r"(.{2,12})\1{3,}")

# Classic Whisper silence / YouTube-outro hallucinations.
_EN_HALLUCINATIONS = re.compile(
    r"^\s*("
    r"thank(s| you)( for watching)?"
    r"|thanks for watching"
    r"|subscribe( now)?"
    r"|please subscribe"
    r"|see you (next time|later)"
    r"|bye[!. ]*"
    r"|you"
    r"|okay"
    r"|ok"
    r"|hmm+"
    r"|uh+"
    r"|um+"
    r"|ah+"
    r"|i'?m going to ask you a little question"
    r")[.!?…\s]*$",
    re.IGNORECASE,
)


# Processors cannot be reused across PipelineTasks; weights can.
_WHISPER_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_BN_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


class RobustWhisperSTTService(WhisperSTTService):
    """Dual-model Whisper with BN finetune + English multilingual."""

    def __init__(self, *args: Any, bengali_model_path: str | None = None, **kwargs: Any):
        # Prefer a longer trailing pad so last BN syllables aren't clipped.
        kwargs.setdefault("trailing_silence_secs", 0.75)
        super().__init__(*args, **kwargs)
        self._bn_model_path = bengali_model_path
        self._bn_model = None

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        # Keep ~1.6s preroll (field name is historical); VAD often fires late on BN.
        self._audio_buffer_size_1s = int(self.sample_rate * 2 * 1.6)

    def _load(self):
        """Reuse cached faster-whisper weights across voice sessions."""
        from faster_whisper import WhisperModel
        from pipecat.utils.types import require_given

        model_name = require_given(self._settings.model, "Whisper model")
        key = (str(model_name), str(self._device), str(self._compute_type))
        cached = _WHISPER_MODEL_CACHE.get(key)
        if cached is not None:
            logger.debug("Reusing cached Whisper model {}", model_name)
            self._model = cached
            unsupported = self._unsupported_language()
            if unsupported:
                raise ValueError(unsupported)
            return
        logger.debug("Loading Whisper model...")
        self._model = WhisperModel(model_name, device=self._device, compute_type=self._compute_type)
        _WHISPER_MODEL_CACHE[key] = self._model
        logger.debug("Loaded Whisper model")
        unsupported = self._unsupported_language()
        if unsupported:
            raise ValueError(unsupported)

    def _ensure_bn_model(self):
        if self._bn_model is not None:
            return self._bn_model
        path = self._bn_model_path
        if not path:
            return None
        p = Path(path)
        if not p.exists() or not ((p / "model.bin").exists() or list(p.glob("model*.bin"))):
            logger.warning("Bengali CT2 model missing/incomplete: {}", path)
            return None
        from faster_whisper import WhisperModel

        key = (str(path), str(self._device), str(self._compute_type))
        cached = _BN_MODEL_CACHE.get(key)
        if cached is not None:
            self._bn_model = cached
            return self._bn_model
        logger.info("Loading Bengali-finetuned Whisper from {}", path)
        self._bn_model = WhisperModel(
            str(path),
            device=self._device,
            compute_type=self._compute_type
            if self._compute_type not in {None, "default"}
            else "float16",
        )
        _BN_MODEL_CACHE[key] = self._bn_model
        return self._bn_model

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not self._model:
            yield ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()

        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        peak = float(np.max(np.abs(audio_float))) if audio_float.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio_float)))) if audio_float.size else 0.0
        # Drop near-silent / click segments before Whisper can invent text.
        if audio_float.size < int(0.55 * 16000) or peak < 0.03 or rms < 0.008:
            await self.stop_processing_metrics()
            return

        # Lead/trail pad: Whisper often drops the first BN consonant without it.
        pad = np.zeros(int(0.4 * 16000), dtype=np.float32)
        audio_float = np.concatenate([pad, audio_float, pad])

        language = cast("Language | str | None", assert_given(self._settings.language))
        forced = None if language in (None, "", "auto") else str(language)
        if forced and "-" in forced:
            forced = forced.split("-", 1)[0]
        if forced not in {"bn", "en", None}:
            forced = None

        multi = self._model

        def detect_probs() -> tuple[float, float]:
            try:
                _lang, _prob, all_probs = multi.detect_language(
                    audio=audio_float,
                    vad_filter=False,
                    language_detection_segments=2,
                    language_detection_threshold=0.3,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("detect_language failed ({}), defaulting bn", exc)
                return 1.0, 0.0
            by_code = {code: float(p) for code, p in (all_probs or [])}
            return by_code.get("bn", 0.0), by_code.get("en", 0.0)

        def transcribe(lang: str) -> tuple[str, float] | None:
            model = multi
            use_finetune = False
            if lang == "bn":
                bn = self._ensure_bn_model()
                if bn is not None:
                    model = bn
                    use_finetune = True

            kwargs: dict[str, Any] = dict(
                language=lang,
                task="transcribe",
                beam_size=5,
                best_of=5,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
                compression_ratio_threshold=2.2,
                log_prob_threshold=-0.45,
                no_speech_threshold=0.45,
            )
            # Finetuned BN models often degrade with English-biased prompts/hotwords.
            if lang == "en":
                kwargs["initial_prompt"] = _EN_PROMPT
            elif not use_finetune:
                kwargs["initial_prompt"] = (
                    "বাংলাদেশের প্রমিত চলিত বাংলা। ঢাকা বিশ্ববিদ্যালয় এডমিশন।"
                )

            segments, _info = model.transcribe(audio_float, **kwargs)
            parts: list[str] = []
            logprobs: list[float] = []
            no_speech: list[float] = []
            for segment in segments:
                text = (segment.text or "").strip()
                if not text or segment.no_speech_prob >= 0.5:
                    continue
                if float(segment.avg_logprob) < -0.7:
                    continue
                parts.append(text)
                logprobs.append(float(segment.avg_logprob))
                no_speech.append(float(segment.no_speech_prob))
            if not parts:
                return None
            text = " ".join(parts).strip()
            if lang == "bn":
                text = fix_bengali_asr(text)
            text = _scrub(text)
            if not text or _looks_garbage(text) or _is_hallucination(text):
                return None
            avg = sum(logprobs) / len(logprobs)
            avg_ns = sum(no_speech) / len(no_speech)
            if avg_ns >= 0.4:
                logger.debug("Rejecting high no-speech {} ({:.2f}): {!r}", lang, avg_ns, text)
                return None
            # Short scraps need some confidence, but tunnel/WebSocket mic audio
            # often lands around -0.2..-0.3 for real greetings like "Hello".
            letters = len(re.findall(r"[\u0980-\u09FFa-zA-Z]", text))
            if letters < 8 and avg < -0.35:
                logger.debug("Rejecting short weak {} ({:.2f}): {!r}", lang, avg, text)
                return None
            if letters < 16 and avg < -0.40:
                logger.debug("Rejecting weak {} ({:.2f}): {!r}", lang, avg, text)
                return None
            if letters < 28 and avg < -0.45:
                logger.debug("Rejecting middling weak {} ({:.2f}): {!r}", lang, avg, text)
                return None
            if avg < -0.7:
                logger.debug("Rejecting low-quality {} ({:.2f}): {!r}", lang, avg, text)
                return None
            return text, avg

        def pick_better(
            primary: tuple[str, float] | None,
            primary_lang: str,
            secondary: tuple[str, float] | None,
            secondary_lang: str,
        ) -> tuple[str, str, float] | None:
            if primary is None and secondary is None:
                return None
            if primary is None:
                # Don't accept a weak fallback after the other language failed —
                # that path produced phantoms like "সে জান্য কারে?".
                if secondary[1] < -0.12:  # type: ignore[index]
                    logger.debug(
                        "Rejecting weak {} fallback ({:.2f}): {!r}",
                        secondary_lang,
                        secondary[1],  # type: ignore[index]
                        secondary[0],  # type: ignore[index]
                    )
                    return None
                return secondary[0], secondary_lang, secondary[1]  # type: ignore[index]
            if secondary is None:
                return primary[0], primary_lang, primary[1]
            # Prefer clearer score; small ties keep the language detector's pick.
            if secondary[1] > primary[1] + 0.08:
                return secondary[0], secondary_lang, secondary[1]
            return primary[0], primary_lang, primary[1]

        def run() -> tuple[str, str, float] | None:
            if forced in {"bn", "en"}:
                result = transcribe(forced)
                if result is None:
                    return None
                return result[0], forced, result[1]

            bn_p, en_p = detect_probs()
            # Bias to Bengali — product primary language.
            primary_lang = "bn" if bn_p + 0.05 >= en_p else "en"
            secondary_lang = "en" if primary_lang == "bn" else "bn"
            primary = transcribe(primary_lang)

            # Cross-check when detector is unsure, or the first pass is weak/missing.
            margin = abs(bn_p - en_p)
            need_second = primary is None or margin < 0.18 or primary[1] < -0.18
            secondary = transcribe(secondary_lang) if need_second else None
            return pick_better(primary, primary_lang, secondary, secondary_lang)

        try:
            chosen = await asyncio.to_thread(run)
        except Exception as exc:  # noqa: BLE001
            await self.stop_processing_metrics()
            logger.exception("Whisper transcription failed")
            yield ErrorFrame(f"{self}: transcription failed: {exc}")
            return

        await self.stop_processing_metrics()
        if chosen is None:
            return

        text, lang, score = chosen
        try:
            lang_out = Language(lang)
        except Exception:  # noqa: BLE001
            lang_out = None

        await self._handle_transcription(text, True, lang_out)
        logger.debug("Transcription: [{}] (lang={} logprob={:.2f})", text, lang, score)
        yield TranscriptionFrame(
            text,
            self._user_id,
            time_now_iso8601(),
            lang_out,
        )


def _scrub(text: str) -> str:
    text = text.strip()
    if _REPEAT_RE.search(text.replace(" ", "")):
        text = _REPEAT_RE.sub(r"\1\1", text.replace(" ", ""))
    tokens = text.split()
    if len(tokens) >= 3:
        cleaned: list[str] = []
        for tok in tokens:
            if len(cleaned) >= 2 and cleaned[-1] == tok and cleaned[-2] == tok:
                continue
            cleaned.append(tok)
        text = " ".join(cleaned)
    return text.strip()


def _is_hallucination(text: str) -> bool:
    if _EN_HALLUCINATIONS.match(text or ""):
        return True
    return False


def _looks_garbage(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) >= 8 and len(set(compact)) <= 3:
        return True
    if re.search(r"(.{2,6})\1{3,}", compact):
        return True
    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    # One-token Bangla exclamations that aren't common affirmations.
    if len(tokens) == 1 and re.search(r"[!?]{1,}$", tokens[0]):
        core = re.sub(r"[!?।.]+$", "", tokens[0])
        if re.fullmatch(r"(হ্যাঁ?|হুঁ?|আচ্ছা|ঠিক|না|জি)", core):
            return False
        bn = re.findall(r"[\u0980-\u09FF]", core)
        if 0 < len(bn) <= 5:
            return True
    # Very short 1–2 token BN scraps with no clear function words.
    if len(tokens) <= 2:
        letters = re.findall(r"[\u0980-\u09FF]", text)
        if 0 < len(letters) <= 12 and not re.search(
            r"(?:^|[\s,।])(আমি|আপনি|তুমি|তোর|তোমার|কী|কি|কেমন|নাম|হ্যাঁ?|না|ঠিক|আচ্ছা|ভালো|বলো?|চাই|পারি)(?:[\s,।!?]|$)",
            text,
        ):
            return True
    return False
