#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$ROOT/backend:${PYTHONPATH:-}"
export PATH="${HOME}/.local/bin:${PATH}"
# Ollama libs + CUDA 12 cublas for faster-whisper/CTranslate2 (venv may ship CUDA 13 only)
export LD_LIBRARY_PATH="${HOME}/.local/ollama/lib/ollama/cuda_v12:${HOME}/.local/ollama/lib/ollama:${HOME}/miniconda3/lib/python3.11/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"
"$ROOT/scripts/start-ollama.sh" || true
exec python -m app.main
