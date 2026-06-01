#!/usr/bin/env bash
# Drive a concurrency sweep against a running exercise-generator instance.
#
# For each level in CONCURRENCY_LEVELS the script runs a headless Locust load
# generator for RUN_DURATION, spawning all virtual users immediately (spawn
# rate equals concurrency), then cools off COOLDOWN_SECONDS before the next
# level so any GPU thermal / KV-cache state from the previous run does not
# contaminate the next measurement.
#
# Per-run CSV reports and a fallback-statistics JSON are written under
# results/locust/concurrency_N/ at the repository root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

TARGET_HOST="${TARGET_HOST:-http://localhost:8000}"
RUN_DURATION="${RUN_DURATION:-5m}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-30}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 2 4 8 16}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/locust}"

export TARGET_HOST

if ! command -v locust >/dev/null 2>&1; then
    echo "ERROR: locust is not on PATH. Install with:" >&2
    echo "    pip install -r ${SCRIPT_DIR}/requirements.txt" >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

echo "============================================================"
echo "Exercise generator load test"
echo "  target          : ${TARGET_HOST}"
echo "  run duration    : ${RUN_DURATION} per level"
echo "  cooldown        : ${COOLDOWN_SECONDS}s between levels"
echo "  concurrency set : ${CONCURRENCY_LEVELS}"
echo "  results dir     : ${RESULTS_DIR}"
echo "============================================================"

# Sanity-check the service is reachable before burning a 5-minute slot on a
# misconfigured target.
if ! curl -fsS --max-time 5 "${TARGET_HOST}/health" >/dev/null; then
    echo "ERROR: ${TARGET_HOST}/health is not reachable. Is the service up?" >&2
    exit 1
fi

first=1
for n in ${CONCURRENCY_LEVELS}; do
    if [[ "${first}" -eq 0 ]]; then
        echo ">>> cooling off for ${COOLDOWN_SECONDS}s..."
        sleep "${COOLDOWN_SECONDS}"
    fi
    first=0

    run_dir="${RESULTS_DIR}/concurrency_${n}"
    mkdir -p "${run_dir}"
    csv_prefix="${run_dir}/locust"

    echo ">>> concurrency=${n}  duration=${RUN_DURATION}"
    LOCUST_CSV_PREFIX="${csv_prefix}" locust \
        --locustfile "${SCRIPT_DIR}/locustfile.py" \
        --host "${TARGET_HOST}" \
        --users "${n}" \
        --spawn-rate "${n}" \
        --run-time "${RUN_DURATION}" \
        --headless \
        --only-summary \
        --csv "${csv_prefix}" \
        --csv-full-history \
        --reset-stats \
        --html "${run_dir}/report.html" \
        --logfile "${run_dir}/locust.log"
done

echo ""
echo "All runs complete. Build the summary table with:"
echo "    python ${SCRIPT_DIR}/parse_results.py"
