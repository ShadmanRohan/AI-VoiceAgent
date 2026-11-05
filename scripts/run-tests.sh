#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$ROOT/backend"
export PATH="${HOME}/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${HOME}/.local/ollama/lib/ollama:${LD_LIBRARY_PATH:-}"

echo "==> Unit / mocked tests"
pytest -m "not integration" -q

echo "==> Integration tests (Ollama / local models)"
pytest -m integration -q || true

cd "$ROOT/frontend"
if [[ -f package.json ]]; then
  echo "==> Frontend typecheck"
  npx tsc -b --pretty false || true
fi
