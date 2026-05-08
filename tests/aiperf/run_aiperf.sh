#!/usr/bin/env bash
# Run NVIDIA AIPerf against the SAYIT shim. Produces an INDEPENDENT
# measurement of request latency and throughput against the same backend
# that Locust hammered in §3.4. Discrepancies between the two should be
# small (≤3% on median, ≤10% on p99) — significant divergence indicates
# a measurement artefact in one of the tools.
#
# Concurrency sweep matches load/run_throughput.sh so the two runs can be
# placed side-by-side in Table 21.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p reports

SHIM_URL=${SHIM_URL:-http://localhost:8001}
MODEL=${MODEL:-qwen2.5-3b}
TOKENIZER=${TOKENIZER:-Qwen/Qwen2.5-3B-Instruct}
REQUEST_COUNT=${REQUEST_COUNT:-100}

# Wait for the shim to come up.
echo "Waiting for shim at $SHIM_URL/v1/models ..."
for _ in $(seq 1 60); do
    if curl -sf "$SHIM_URL/v1/models" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

run_aiperf() {
    local concurrency=$1
    local outdir="reports/aiperf-c${concurrency}"
    mkdir -p "$outdir"

    echo
    echo "=== AIPerf @ concurrency=$concurrency, $REQUEST_COUNT requests ==="

    # Prefer the new `aiperf` package; fall back to `genai-perf` (legacy name).
    local cli
    if command -v aiperf >/dev/null 2>&1; then
        cli=aiperf
    elif command -v genai-perf >/dev/null 2>&1; then
        cli=genai-perf
    else
        echo "ERROR: neither aiperf nor genai-perf found on PATH"
        echo "Install: pip install aiperf"
        exit 1
    fi

    "$cli" profile \
        --model "$MODEL" \
        --tokenizer "$TOKENIZER" \
        --endpoint-type chat \
        --service-kind openai \
        --url "$SHIM_URL" \
        --concurrency "$concurrency" \
        --request-count "$REQUEST_COUNT" \
        --synthetic-input-tokens-mean 200 \
        --synthetic-input-tokens-stddev 50 \
        --output-tokens-mean 1000 \
        --output-tokens-stddev 200 \
        --artifact-dir "$outdir" \
        --profile-export-file aiperf.json \
    || echo "WARNING: concurrency=$concurrency failed (continuing)"
}

# Single-user run = comparison baseline for Locust's run_latency.sh.
run_aiperf 1

# Concurrency sweep matches Locust's ThroughputShape stages.
for c in 2 4 8 16; do
    run_aiperf "$c"
done

echo
echo "Done. AIPerf artefacts in $(pwd)/reports/"
echo "Compare with Locust:"
echo "  load/results/latency_stats.csv     <-> aiperf-c1/aiperf.json"
echo "  load/results/throughput_stats.csv  <-> aiperf-c{2,4,8,16}/aiperf.json"
