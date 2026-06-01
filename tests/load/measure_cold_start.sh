#!/usr/bin/env bash
# Measure cold-start time: from `docker compose up` to /health reporting
# loaded_models. Maps onto §3.5 Table 22 ("Время холодного старта").
#
# Usage:
#   ./measure_cold_start.sh                                   # uses docker-compose.yml
#   ./measure_cold_start.sh docker-compose.standalone.yml     # alternative compose
#
# Output: appends one line to results/cold_start.csv
#         writes /health response to results/cold_start_health.json

set -euo pipefail

COMPOSE=${1:-docker-compose.standalone.yml}
URL=${SAYIT_GENERATOR_URL:-http://localhost:8000}
RESULTS_DIR=${RESULTS_DIR:-results}
mkdir -p "$RESULTS_DIR"
CSV="$RESULTS_DIR/cold_start.csv"

if [[ ! -f "$CSV" ]]; then
  echo "iteration,compose_file,cold_start_ms,first_request_ms,timestamp" > "$CSV"
fi

ITERATION=$(($(wc -l < "$CSV") - 1 + 1))

echo "[$(date -Iseconds)] iteration #$ITERATION — bringing stack down..."
docker compose -f "$COMPOSE" down >/dev/null 2>&1 || true

# --- cold start ------------------------------------------------------------
T0=$(date +%s%3N)
docker compose -f "$COMPOSE" up -d --build >/dev/null

# Poll /health until loaded_models is non-empty.
while :; do
  if curl -sf "$URL/health" 2>/dev/null \
      | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('loaded_models') else 1)" \
      ; then
    break
  fi
  sleep 0.5
done
T1=$(date +%s%3N)
COLD_START_MS=$((T1 - T0))

# Save the /health snapshot for the appendix.
curl -s "$URL/health" > "$RESULTS_DIR/cold_start_health.json"

# --- first-request latency ------------------------------------------------
T2=$(date +%s%3N)
curl -sf -X POST "$URL/api/v1/generate" \
  -H 'Content-Type: application/json' \
  -d "$(cat data/single_payloads.json | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)[0]))')" \
  > /dev/null
T3=$(date +%s%3N)
FIRST_REQ_MS=$((T3 - T2))

echo "$ITERATION,$COMPOSE,$COLD_START_MS,$FIRST_REQ_MS,$(date -Iseconds)" >> "$CSV"

echo "Cold start:    ${COLD_START_MS} ms"
echo "First request: ${FIRST_REQ_MS} ms"
echo "Logged to $CSV"
