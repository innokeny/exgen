#!/usr/bin/env bash
# Run 4: endurance / soak test. 8 hours under steady baseline load.
# Used in §3.4 to claim "no failures, no VRAM leaks, no latency drift over X hours".

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results

URL=${SAYIT_GENERATOR_URL:-http://localhost:8000}
DURATION=${DURATION:-8h}

# 8h * 3600 = 28800 seconds for nvidia-smi.
DURATION_S=$(python3 -c "
s = '$DURATION'.strip()
mult = {'s': 1, 'm': 60, 'h': 3600}.get(s[-1], 1)
print(int(float(s[:-1] if s[-1] in 'smh' else s)) * mult)
")

./monitor_gpu.sh results/endurance "$DURATION_S" &
GPU_PID=$!

locust -f locustfile.py \
  --host="$URL" \
  --headless \
  --users 1 --spawn-rate 1 \
  --run-time "$DURATION" \
  --csv=results/endurance \
  --csv-full-history \
  --html=results/endurance.html \
  --logfile=results/endurance.log \
  --loglevel=INFO \
  BatchGenerationUser

wait "$GPU_PID" || true
echo "Endurance run finished. Use analyze.py --mode=endurance for drift report."
