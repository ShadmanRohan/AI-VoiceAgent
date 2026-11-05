# Troubleshooting

## Ollama unavailable

- Start with `./scripts/start-ollama.sh`
- Ensure `PATH` includes `~/.local/bin` and `LD_LIBRARY_PATH` includes `~/.local/ollama/lib/ollama`
- Pull model: `ollama pull qwen3:1.7b`
- UI Status tab should show Ollama running; chat returns a clear error if down (does not fake a reply)

## Microphone denied / disconnected

- Use HTTPS or localhost origins
- Re-click Start voice after granting permission
- Select the correct input device in the OS / browser settings
- Enable push-to-talk in Settings if echo cancellation is weak

## STT empty / wrong

- Moonshine expects 16 kHz mono PCM (pipeline default)
- Fallback: set `stt_provider=whisper` in Settings
- First Moonshine/Whisper load downloads models — wait for cache

## TTS silent / wrong voice

- First Kokoro run downloads ONNX + voices to `~/.cache/pipecat/kokoro-onnx/`
- Unavailable voices fall back to `af_heart`
- Check browser audio element and system volume

## Interruptions not working

- Ensure continuous listening is active (mic not muted)
- VAD must hear barge-in speech; raise `vad_confidence` if false triggers
- Generation IDs reject stale audio after cancel

## Memory not recalling

- Enable consent on the Memory tab
- Only opted-in / high-confidence items are stored
- Deletion removes FAISS vectors as well as SQL rows

## Out of memory

- Switch profile to `low`
- Use `qwen3:1.7b` instead of 4B
- Close other GPU apps (`nvidia-smi`)
