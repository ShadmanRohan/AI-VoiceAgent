#!/usr/bin/env bash
set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${HOME}/.local/ollama/lib/ollama:${LD_LIBRARY_PATH:-}"

if ! command -v ollama >/dev/null; then
  echo "Ollama binary not found. Install to ~/.local/ollama or system PATH."
  exit 1
fi

if curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "Ollama already running"
  exit 0
fi

OLLAMA_HOST=127.0.0.1:11434 nohup ollama serve > /tmp/localvoiceai-ollama.log 2>&1 &
echo "Started ollama (pid $!). Log: /tmp/localvoiceai-ollama.log"
sleep 2
curl -sf http://127.0.0.1:11434/api/tags >/dev/null && echo "Ollama healthy"
