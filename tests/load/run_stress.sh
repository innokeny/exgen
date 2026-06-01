#!/usr/bin/env bash
# Run 3: stress test. Steps the load through 1x, 2x, 4x, 8x baseline and
# returns to 1x, holding each step for 5 minutes. Populates Table 21.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results

URL=${SAYIT_GENERATOR_URL:-http://localhost:8000}
TOTAL_S=1500

./monitor_gpu.sh results/stress "$TOTAL_S" &
GPU_PID=$!

locust -f locustfile.py \
  --host="$URL" \
  --headless \
  --csv=results/stress \
  --csv-full-history \
  --html=results/stress.html \
  --logfile=results/stress.log \
  --loglevel=INFO \
  StressShape

wait "$GPU_PID" || true
echo "Done. analyze.py --mode=stress will compute the 4-row table."
