#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${HOME}/.local/ollama/lib/ollama:${LD_LIBRARY_PATH:-}"

"$ROOT/scripts/start-ollama.sh"

echo "==> Pulling Ollama LLM models"
ollama pull qwen3:1.7b
# Optional higher-quality model (comment out if low VRAM/disk)
if [[ "${PULL_QUALITY:-0}" == "1" ]]; then
  ollama pull qwen3:4b-instruct
fi

echo "==> Prefetching embedding model via Python"
cd "$ROOT/backend"
# shellcheck disable=SC1091
source .venv/bin/activate
python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("BAAI/bge-small-en-v1.5")
print("bge-small-en-v1.5 ready")
PY

echo "==> Prefetching Kokoro TTS (first construct downloads ONNX)"
python - <<'PY'
try:
    from pipecat.services.kokoro.tts import KokoroTTSService
    KokoroTTSService(settings=KokoroTTSService.Settings(voice="af_heart"))
    print("Kokoro ready")
except Exception as e:
    print("Kokoro prefetch note:", e)
PY

echo "==> Prefetching Moonshine STT"
python - <<'PY'
try:
    from pipecat.services.moonshine.stt import MoonshineSTTService, Model
    MoonshineSTTService(settings=MoonshineSTTService.Settings(model=Model.TINY_STREAMING))
    print("Moonshine ready")
except Exception as e:
    print("Moonshine prefetch note:", e)
PY

echo "Model download step finished."
