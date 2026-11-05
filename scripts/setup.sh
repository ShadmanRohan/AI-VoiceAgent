#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> LocalVoiceAI setup"
mkdir -p data models data/exports data/vectors
cp -n .env.example .env 2>/dev/null || true

echo "==> Backend venv + deps"
cd "$ROOT/backend"
if [[ ! -d .venv ]]; then
  uv venv .venv --python 3.11
fi
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ".[dev]"

echo "==> Frontend deps"
cd "$ROOT/frontend"
npm install

echo "==> Ensure Ollama is reachable"
export PATH="${HOME}/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${HOME}/.local/ollama/lib/ollama:${LD_LIBRARY_PATH:-}"
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "Ollama is not running. Start with: scripts/start-ollama.sh"
fi

echo "Setup complete. Next: scripts/download-models.sh && scripts/start-backend.sh"
