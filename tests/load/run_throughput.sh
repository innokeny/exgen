#!/usr/bin/env bash
# Run 2: throughput vs. concurrency sweep (1, 2, 4, 8, 16 users).
# Populates Table 20 and Figure 16 in §3.4.
#
# Implemented as a Locust LoadTestShape (ThroughputShape in locustfile.py)
# so Locust drives the steps and emits a single CSV with the full timeline.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results

URL=${SAYIT_GENERATOR_URL:-http://localhost:8000}
# 5 stages × 5 minutes each = 25 minutes total.
TOTAL_S=1500

./monitor_gpu.sh results/throughput "$TOTAL_S" &
GPU_PID=$!

locust -f locustfile.py \
  --host="$URL" \
  --headless \
  --csv=results/throughput \
  --csv-full-history \
  --html=results/throughput.html \
  --logfile=results/throughput.log \
  --loglevel=INFO \
  ThroughputShape

wait "$GPU_PID" || true
echo "Done. Use analyze.py to extract per-step metrics."
