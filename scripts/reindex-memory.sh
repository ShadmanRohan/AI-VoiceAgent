#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$ROOT/backend"
python - <<'PY'
import asyncio
from app.memory.reindex import reindex_memories_to_e5
print(asyncio.run(reindex_memories_to_e5(activate=True)))
PY
