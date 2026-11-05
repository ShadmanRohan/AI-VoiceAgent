# Bengali Bilingual Upgrade — Migration Plan

## Detected architecture (source of truth)

| Area | Location |
|------|----------|
| Backend | FastAPI `backend/app/main.py` |
| Frontend | React/Vite `frontend/` |
| STT | Pipecat Moonshine (+ Whisper fallback) `conversation/pipeline.py` |
| LLM | Ollama via `models/llm.py` + Pipecat `OLLamaLLMService` |
| TTS | Pipecat Kokoro `pipeline.py` |
| VAD / Turn | Silero + LocalSmartTurnV3 |
| Memory | SQLite + FAISS + `BAAI/bge-small-en-v1.5` |
| Config | `app/config.py` + `data/settings.json` |

## Original model configuration

- STT: `moonshine` / `tiny-streaming`
- LLM: `qwen3:1.7b` (fallback quality: `qwen3:4b-instruct`)
- TTS: Kokoro `af_heart`
- Embeddings: `BAAI/bge-small-en-v1.5`
- Profile: balanced

## Baseline

- Git branch: `feature/bengali-bilingual-upgrade`
- DB backup: `data/backups/localvoiceai.pre-bilingual.*.db`
- Unit tests before change: **19 passed**

## Target

- STT: Whisper Large V3 Turbo (Pipecat `Model.LARGE_V3_TURBO`)
- LLM: `qwen3.5:4b` (keep `qwen3:1.7b` fallback)
- Bengali TTS: Indic Parler (gated HF) — Arjun/Aditi
- English TTS: Kokoro (unchanged)
- Embeddings: `intfloat/multilingual-e5-small` (versioned index, no data wipe)
