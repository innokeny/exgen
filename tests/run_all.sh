#!/usr/bin/env bash
# End-to-end benchmark suite. Run on the SAYIT host with the generator
# service already up and /health reporting OK.
#
# Order:
#   1. cold start  -> Table 22 (cold-start time, first-request latency)
#   2. functional  -> Table 17 (test cases) + Table 18 (valid JSON share)
#   3. latency     -> Table 19 (latency distribution)
#   4. throughput  -> Table 20 + Figure 16 (throughput vs concurrency)
#   5. stress      -> Table 21 (behaviour at 1x..8x load)
#   6. endurance   -> §3.4 long-run claim (no leaks / drift)
#   7. AIPerf      -> independent validation of (3) and (4)
#   8. analyze     -> all tables, ready to paste into the thesis

set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
RESULTS="$ROOT/load/results"
mkdir -p "$RESULTS"

RUN_AIPERF=${RUN_AIPERF:-true}

echo "=== 1/8 cold start ==="
( cd "$ROOT/load" && ./measure_cold_start.sh )

echo
echo "=== 2/8 functional & contract tests ==="
( cd "$ROOT/functional" && ./run_functional.sh ) || true

echo
echo "=== 3/8 latency profile (10 min, single user) ==="
( cd "$ROOT/load" && ./run_latency.sh )

echo
echo "=== 4/8 throughput sweep (25 min, 1->16 users) ==="
( cd "$ROOT/load" && ./run_throughput.sh )

echo
echo "=== 5/8 stress test (25 min, 1x..8x) ==="
( cd "$ROOT/load" && ./run_stress.sh )

echo
echo "=== 6/8 endurance (default 8h; override with DURATION) ==="
DURATION=${DURATION:-8h}
( cd "$ROOT/load" && DURATION="$DURATION" ./run_endurance.sh )

if [[ "$RUN_AIPERF" == "true" ]]; then
    echo
    echo "=== 7/8 AIPerf cross-validation ==="
    ( cd "$ROOT/aiperf" && ./run_shim.sh ) &
    SHIM_PID=$!
    sleep 5

    ( cd "$ROOT/aiperf" && ./run_aiperf.sh ) || true
    ( cd "$ROOT/aiperf" && python3 cross_validate.py ) || true

    kill "$SHIM_PID" 2>/dev/null || true
else
    echo "=== 7/8 AIPerf — skipped (RUN_AIPERF=false) ==="
fi

echo
echo "=== 8/8 generate filled thesis tables ==="
( cd "$ROOT/load" && python3 analyze.py results )

echo
echo "All done."
echo "  Locust tables:      $RESULTS/tables/"
echo "  AIPerf comparison:  $ROOT/aiperf/reports/cross_validation.md"
echo "  Functional reports: $ROOT/functional/reports/"
