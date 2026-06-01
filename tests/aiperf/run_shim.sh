#!/usr/bin/env bash
# Run the OpenAI-compatibility shim. Reads BACKEND_URL from the environment
# (default http://localhost:8000) and listens on SHIM_PORT (default 8001).

set -euo pipefail

cd "$(dirname "$0")"

export BACKEND_URL=${BACKEND_URL:-http://localhost:8000}
export SHIM_PORT=${SHIM_PORT:-8001}

# Wait until the backend is ready before binding the shim — otherwise AIPerf
# may hit /v1/models before the LoRA adapter has loaded.
echo "Waiting for backend at $BACKEND_URL/health ..."
for _ in $(seq 1 120); do
    if curl -sf "$BACKEND_URL/health" \
         | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin).get('loaded_models') else 1)" \
         2>/dev/null; then
        break
    fi
    sleep 1
done

exec python3 shim.py
