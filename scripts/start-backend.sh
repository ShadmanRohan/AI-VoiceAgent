#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$ROOT/backend:${PYTHONPATH:-}"
export PATH="${HOME}/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${HOME}/.local/ollama/lib/ollama:${LD_LIBRARY_PATH:-}"
"$ROOT/scripts/start-ollama.sh" || true
exec python -m app.main
