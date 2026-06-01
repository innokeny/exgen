#!/usr/bin/env bash
# Sample GPU telemetry once per second into a CSV file.
#
# Captures the columns needed for §3.5 of the thesis:
#   - power     (W)        -> energy / exercise
#   - sm util   (%)        -> GPU busy fraction
#   - mem util  (%)        -> memory bandwidth pressure
#   - mem used  (MiB)      -> VRAM occupancy
#   - temp      (C)        -> thermal headroom
#
# Usage:
#   ./monitor_gpu.sh results/run_baseline 1500   # capture 1500 s
#   ./monitor_gpu.sh results/run_stress  1500    # second run, second file
#
# The script writes a real CSV with a header (nvidia-smi dmon's native format
# uses spaces); analyze.py parses this CSV directly.

set -euo pipefail

OUT_PREFIX=${1:?"usage: $0 <out-prefix> <duration-s>"}
DURATION=${2:?"usage: $0 <out-prefix> <duration-s>"}

mkdir -p "$(dirname "$OUT_PREFIX")"
CSV="${OUT_PREFIX}.gpu.csv"

# `dmon` columns selected:
#   p = power, u = util (sm, mem, enc, dec), m = memory (used, free), c = clocks
# We strip dmon's "#" header, prepend a clean CSV header, and write one row/sec.
echo "ts,gpu_idx,power_w,sm_util_pct,mem_util_pct,enc_util_pct,dec_util_pct,mem_used_mib,mem_free_mib,sm_clock_mhz,mem_clock_mhz" \
    > "$CSV"

# `-c $DURATION` exits cleanly after N samples (1 sample/sec by default).
nvidia-smi dmon -s pumc -d 1 -c "$DURATION" -o T \
  | awk 'NR>2 { printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", \
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11 }' \
  >> "$CSV"

echo "GPU telemetry written to $CSV"
