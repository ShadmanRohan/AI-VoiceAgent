# Bengali Bilingual Upgrade Report

Branch: `feature/bengali-bilingual-upgrade`  
Date: 2026-11-06

## Original configuration

| Component | Value |
|-----------|--------|
| STT | Moonshine `tiny-streaming` |
| LLM | `qwen3:1.7b` |
| TTS | Kokoro `af_heart` |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Prompt | English-only LocalVoiceAI prompt |

Baseline unit tests: **19 passed**  
DB backup: `data/backups/localvoiceai.pre-bilingual.*.db`  
E5 reindex backup: `data/backups/localvoiceai.pre-e5-reindex.*.db`

## New configuration

| Component | Value |
|-----------|--------|
| Primary language | `bn-BD` (Bangladeshi Colloquial Bengali) |
| Language modes | `auto` / `bangla` / `english` |
| STT | Whisper Large V3 Turbo via Pipecat (`Model.LARGE_V3_TURBO`) |
| STT fallback | Moonshine |
| LLM | `qwen3.5:4b` (fallback `qwen3:1.7b`) |
| Bengali TTS | `ai4bharat/indic-parler-tts` (Aditi / Arjun) — **gated** |
| English TTS | Kokoro (unchanged) |
| Embeddings | `intfloat/multilingual-e5-small` index `e5-small-v1` |
| Prompt | `backend/app/prompts/bilingual_voice_assistant.txt` |

## Code changes (high level)

- `app/config.py` — bilingual settings, new defaults, enum-safe overrides
- `app/language/*` — router, Bengali normalizer, prompt loader
- `app/prompts/bilingual_voice_assistant.txt` — system prompt
- `app/models/llm.py` — Qwen3.5 + safer fallbacks
- `app/conversation/orchestrator.py` — language events + response directives
- `app/conversation/pipeline.py` — Whisper Turbo STT + LLM fallback
- `app/speech/providers/indic_parler_tts.py` — Indic Parler provider
- `app/speech/tts_router.py` — Bengali→Indic / English→Kokoro
- `app/memory/embeddings.py` — E5 query/passage prefixes
- `app/memory/vector_store.py` — versioned FAISS namespaces
- `app/memory/reindex.py` — non-destructive reindex
- `app/api/routes.py` — `/api/models/status`, `/api/tts/synthesize`, `/api/memory/reindex-e5`
- Frontend Settings + Voice language indicator

## Dependencies

- Existing: `pipecat-ai[whisper,...]`, `faster-whisper` (already present)
- New optional for Bengali TTS: `parler-tts` (+ HF access to gated model)
- Embeddings: `sentence-transformers` loads `intfloat/multilingual-e5-small`

### Install / download

```bash
# LLM
ollama pull qwen3.5:4b

# Optional Bengali TTS (after accepting HF terms)
# 1) https://huggingface.co/ai4bharat/indic-parler-tts  (accept access)
# 2) export HF_TOKEN=hf_...
pip install git+https://github.com/huggingface/parler-tts.git

# Memory reindex (safe; keeps SQL rows)
./scripts/reindex-memory.sh
```

Whisper Turbo CTranslate2 weights download on first STT use (`deepdml/faster-whisper-large-v3-turbo-ct2`).

## Database / memory migration

- **No tables dropped.** Conversation history retained (sessions still listed).
- New FAISS namespace: `data/vectors/e5-small-v1/`
- Legacy vectors remain under previous paths for rollback
- Reindex result: 1→N memories re-embedded; smoke BN+EN retrieval hits verified
- Activated embedding model: `intfloat/multilingual-e5-small`

## Test evidence

| Test | Result |
|------|--------|
| Unit tests after upgrade | **24 passed** |
| BN chat (Qwen3.5) | Natural Bangla reply |
| EN chat | Natural English reply |
| Code-switch STT text | Bangla response with technical EN terms |
| Explicit switch to English | Sticky English responses |
| Cross-lingual memory | BN preference → EN question answered correctly |
| EN Kokoro TTS `/api/tts/synthesize` | 86KB PCM @ 24kHz |
| BN Indic Parler TTS | Correct **503** gated error (no silent EN fake) |
| Model status API | Reports llm/stt/tts/embeddings |
| English regression hello | Works |
| Data preserved | Sessions + memories present |

## Performance notes (this machine, RTX 3090)

- LLM first replies with `qwen3.5:4b`: typically a few seconds cold, faster warm
- E5 retrieval: ~5–7 ms after load
- Indic Parler not loaded (gated) — expected to be the largest latency source when enabled
- Whisper Turbo: segmented recognition (not token-streaming); first load downloads CT2 weights

## Remaining limitations / UNVERIFIED

1. **Indic Parler gated** — needs HF terms + `HF_TOKEN`; Bengali audible TTS not generated in this environment.
2. **Bangladeshi accent authenticity** — not claimed; needs native-speaker listening once Indic works.
3. **Live mic Bengali STT** — Whisper Turbo wired; real BN microphone audio not exercised here.
4. **WebRTC pipeline TTS** still uses Kokoro FrameProcessor; Bengali synthesis is via `/api/tts/synthesize` router until a Pipecat Indic processor is added.
5. Offline NIC-down test not repeated in this pass.
6. SraVaani Live intentionally not integrated (optional, non-blocking).

## Rollback

```bash
git checkout master   # or previous commit
# restore DB if needed
cp data/backups/localvoiceai.pre-bilingual.*.db data/localvoiceai.db
# point settings embedding back to BGE / moonshine / qwen3:1.7b in data/settings.json
```

## Launch

```bash
./scripts/start-ollama.sh
./scripts/start-backend.sh   # http://127.0.0.1:7860
./scripts/start-frontend.sh  # http://127.0.0.1:5173
```

Open Settings → choose Language mode (Auto / বাংলা / English), LLM `qwen3.5:4b`, STT Whisper Turbo, Bengali voice Aditi/Arjun (after HF access).
