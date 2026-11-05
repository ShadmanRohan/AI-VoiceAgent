#!/usr/bin/env bash
set -euo pipefail
BASE="${1:-http://127.0.0.1:7860}"
echo "Health: $BASE/api/health"
curl -sf "$BASE/api/health" | python3 -m json.tool
echo
echo "System status:"
curl -sf "$BASE/api/system/status" | python3 -m json.tool | head -80
