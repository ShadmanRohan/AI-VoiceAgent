#!/usr/bin/env bash
set -euo pipefail
echo "Stopping LocalVoiceAI listeners on 7860 / 5173 (if any)"
fuser -k 7860/tcp 2>/dev/null || true
fuser -k 5173/tcp 2>/dev/null || true
echo "Done. Ollama left running (stop manually if desired)."
