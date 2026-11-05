# Architecture

## Pipeline

```
Microphone (browser WebRTC)
  → SmallWebRTCTransport
  → Silero VAD
  → Moonshine STT (Whisper fallback)
  → Smart Turn v3 (end-of-turn)
  → Conversation orchestrator / LLM context aggregator
      + working / session / long-term / episodic memory
      + local tools
  → Ollama Qwen3
  → Streaming text buffer / TTS chunker
  → Kokoro TTS
  → WebRTC audio playback
```

Parallel path: user speech during `AI_SPEAKING` raises an interruption, cancels LLM + TTS, clears stale audio, and starts a new user turn.

## Modules

| Module | Role |
|--------|------|
| `app/api` | REST validation & routes |
| `app/conversation` | State machine, sessions, Pipecat pipeline, orchestrator |
| `app/speech` | STT/TTS interfaces, TTS chunker |
| `app/memory` | Embeddings, FAISS vectors, retrieval, extraction, context builder |
| `app/models` | Ollama LLM client |
| `app/tools` | Allowlisted local tools |
| `app/database` | SQLAlchemy models + async SQLite |
| `app/monitoring` | Hardware detection + latency metrics |

## Latency metrics (definitions)

| Metric | Definition |
|--------|------------|
| `speech_start_detection_latency_ms` | First speech-energy frame → VAD speech-start |
| `end_of_turn_latency_ms` | Last speech frame → turn-end decision |
| `stt_finalization_latency_ms` | Turn-end → final transcript |
| `llm_first_token_latency_ms` | LLM request start → first token |
| `tts_first_audio_latency_ms` | First TTS request → first PCM |
| `first_audible_response_latency_ms` | Finalized user turn → first AI audio to client |
| `interruption_to_silence_latency_ms` | Interrupt detect → AI audio stopped |
| `memory_retrieval_latency_ms` | Query embed + ranked recall |

Values are recorded only when measured; never invent targets.

## Provider swap

STT/TTS/LLM are constructed behind factory helpers in `conversation/pipeline.py` and `models/llm.py`. Change settings (`stt_provider`, `ollama_model`, `tts_voice`) without rewriting the app.
