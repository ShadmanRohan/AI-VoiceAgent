# LocalVoiceAI

Fully local, privacy-focused, real-time conversational voice AI.

Stack: **FastAPI · Pipecat · React/Vite · Ollama (Qwen3) · Moonshine STT · Kokoro TTS · Silero VAD · Smart Turn v3 · SQLite + FAISS memory**.

## Quick start

```bash
cd local-voice-ai
chmod +x scripts/*.sh
./scripts/setup.sh
./scripts/start-ollama.sh
./scripts/download-models.sh   # pulls qwen3:1.7b + prefetches speech/embedding models
./scripts/start-backend.sh     # http://127.0.0.1:7860
# other terminal
./scripts/start-frontend.sh    # http://127.0.0.1:5173
```

Open **http://127.0.0.1:5173**, allow microphone access, click **Start voice**.

Text fallback: type in the Voice panel Send box (exercises LLM + memory without mic).

## Health & tests

```bash
./scripts/check-health.sh
./scripts/run-tests.sh
python scripts/benchmark.py
```

## Configuration

Copy `.env.example` → `.env`. Runtime overrides are also stored in `data/settings.json` via the Settings UI.

Performance profiles: `auto` | `low` | `balanced` | `quality`.

## Features

- Continuous listening with Silero VAD + Smart Turn end-of-turn detection
- Streaming LLM tokens → phrase-chunked Kokoro TTS
- Mid-speech interruption (barge-in) with generation cancellation
- Working / session / long-term / episodic memory (opt-in consent)
- Local notes & tasks tools with destructive-action confirmation
- Conversation history, memory dashboard, system status

## Project layout

See `docs/architecture.md` and `docs/implementation-status.md`.

## Privacy

Binds to `127.0.0.1` by default. No cloud APIs required after model download. Persistent personal memory is **opt-in**.
