#!/usr/bin/env bash
# Run 1: latency distribution under single-client load.
# Populates Table 19 (min/median/p90/p95/p99/max/mean/std of response time).
#
# Usage: ./run_latency.sh
# Outputs:
#   results/latency_*           Locust CSVs and HTML
#   results/latency.gpu.csv     nvidia-smi dmon trace

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results

DURATION=${DURATION:-10m}
WARMUP=${WARMUP:-20}
URL=${SAYIT_GENERATOR_URL:-http://localhost:8000}

echo "[$(date -Iseconds)] warming up ($WARMUP requests)..."
for _ in $(seq 1 "$WARMUP"); do
  curl -sf -X POST "$URL/api/v1/generate" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json; print(json.dumps(json.load(open("data/single_payloads.json"))[0]))')" \
    > /dev/null
done

# Convert Locust's "10m" to seconds for nvidia-smi -c.
DURATION_S=$(python3 -c "
s = '$DURATION'.strip()
mult = {'s': 1, 'm': 60, 'h': 3600}.get(s[-1], 1)
print(int(float(s[:-1] if s[-1] in 'smh' else s)) * mult)
")

echo "[$(date -Iseconds)] starting latency run for $DURATION ($DURATION_S s)..."

# Start GPU telemetry in the background.
./monitor_gpu.sh results/latency "$DURATION_S" &
GPU_PID=$!

# Single virtual user, hammering POST /generate.
locust -f locustfile.py \
  --host="$URL" \
  --headless \
  --users 1 --spawn-rate 1 \
  --run-time "$DURATION" \
  --csv=results/latency \
  --csv-full-history \
  --html=results/latency.html \
  --logfile=results/latency.log \
  --loglevel=INFO \
  SingleGenerationUser

wait "$GPU_PID" || true
echo "[$(date -Iseconds)] done. Artefacts in results/"
