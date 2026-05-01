"""Latency benchmark for /api/v1/generate.

Sends N sequential requests to a running container and prints p50/p95 latency.
The service is single-instance and synchronous, so concurrency > 1 just queues.
"""

from __future__ import annotations

import argparse
import statistics
import time

import httpx

DEFAULT_PAYLOAD = {
    "user_id": "u_bench",
    "message_content": "It has a high - density population because for its small territory.",
    "grammar_error": "Preposition",
    "explanation": "Wrong preposition: 'for' should be 'of'.",
    "llm_confidence": 0.95,
    "task_type": "grammar_choice",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/api/v1/generate")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    latencies: list[float] = []
    with httpx.Client(timeout=300.0) as client:
        for i in range(args.n):
            t0 = time.perf_counter()
            r = client.post(args.url, json=DEFAULT_PAYLOAD)
            dt = time.perf_counter() - t0
            r.raise_for_status()
            latencies.append(dt)
            print(f"[{i+1}/{args.n}] {dt*1000:.0f} ms — status={r.json().get('status')}")

    print()
    print(f"n      = {len(latencies)}")
    print(f"mean   = {statistics.mean(latencies)*1000:.0f} ms")
    print(f"median = {statistics.median(latencies)*1000:.0f} ms")
    if len(latencies) >= 2:
        p95 = statistics.quantiles(latencies, n=20)[18]
        print(f"p95    = {p95*1000:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
