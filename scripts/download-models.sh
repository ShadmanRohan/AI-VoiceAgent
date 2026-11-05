#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${HOME}/.local/ollama/lib/ollama:${LD_LIBRARY_PATH:-}"

"$ROOT/scripts/start-ollama.sh"

echo "==> Pulling Ollama LLM models"
ollama pull qwen3.5:4b
ollama pull qwen3:1.7b   # fallback

echo "==> Prefetching embedding model via Python"
cd "$ROOT/backend"
# shellcheck disable=SC1091
source .venv/bin/activate
python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("intfloat/multilingual-e5-small")
print("multilingual-e5-small ready")
# Legacy rollback model (optional)
if __import__("os").environ.get("PULL_LEGACY_BGE") == "1":
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
