from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import httpx

from scripts.benchmarks._common import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    SAMPLE_SINGLE_PAYLOAD,
    post_generate,
    save_results,
    warmup,
)

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]


def _send_one(base_url: str, timeout: float) -> tuple[int, float]:
    with httpx.Client(timeout=timeout) as client:
        status, _, dt_ms = post_generate(client, base_url, SAMPLE_SINGLE_PAYLOAD)
    return status, dt_ms


def measure_one_level(base_url: str, concurrency: int, total_requests: int, timeout: float) -> Dict[str, Any]:
    latencies_ms: List[float] = []
    successes = 0
    failures = 0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_send_one, base_url, timeout) for _ in range(total_requests)]
        for fut in as_completed(futures):
            status, dt_ms = fut.result()
            latencies_ms.append(dt_ms)
            if status == 200:
                successes += 1
            else:
                failures += 1
    elapsed_s = time.perf_counter() - t0

    avg_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    rps = successes / elapsed_s if elapsed_s > 0 else 0.0
    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successes": successes,
        "failures": failures,
        "elapsed_s": round(elapsed_s, 2),
        "rps": round(rps, 3),
        "mean_latency_ms": round(avg_ms, 1),
        "success_rate_pct": round(100.0 * successes / total_requests, 2),
    }


def find_saturation(rows: List[Dict[str, Any]], threshold_pct: float = 5.0) -> int:
    for i in range(1, len(rows)):
        prev_rps = rows[i - 1]["rps"]
        cur_rps = rows[i]["rps"]
        if prev_rps == 0:
            continue
        gain_pct = 100.0 * (cur_rps - prev_rps) / prev_rps
        if gain_pct < threshold_pct:
            return rows[i - 1]["concurrency"]
    return rows[-1]["concurrency"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_BASE_URL)
    parser.add_argument("--requests", type=int, default=60,
                        help="total requests sent at each concurrency level")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    if args.warmup:
        warmup(args.url, n=args.warmup, timeout=args.timeout)

    rows: List[Dict[str, Any]] = []
    for c in CONCURRENCY_LEVELS:
        row = measure_one_level(args.url, c, args.requests, args.timeout)
        print(json.dumps(row, ensure_ascii=False))
        rows.append(row)

    saturation = find_saturation(rows)
    print(json.dumps({"saturation_concurrency": saturation}, ensure_ascii=False))

    save_results("throughput", {"levels": rows, "saturation_concurrency": saturation})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
