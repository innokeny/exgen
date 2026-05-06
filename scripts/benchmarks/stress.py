"""Bench: stepwise stress test.

Fills in таблицу 21 (3.4) — поведение сервиса при стресс-нагрузке: для уровней
1×, 2×, 4×, 8× от расчётного фиксируется RPS, p95 задержки, доля ошибок и
расход видеопамяти. После снятия избыточной нагрузки измеряется время
возврата времени отклика к исходному уровню.

Usage:
    python -m scripts.benchmarks.stress --baseline 2 --duration 60 --recovery 30
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import httpx

from scripts.benchmarks._common import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    SAMPLE_SINGLE_PAYLOAD,
    post_generate,
    save_results,
    summarize,
    warmup,
)

STEPS = [1, 2, 4, 8]


def _vram_used_pct(base_url: str, timeout: float) -> float:
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.get(f"{base_url}/health")
            r.raise_for_status()
            gpu = r.json().get("gpu", {})
            total = gpu.get("vram_total_mb")
            used = gpu.get("vram_used_mb")
            if not total or used is None:
                return 0.0
            return round(100.0 * used / total, 2)
    except Exception:
        return 0.0


def _send_loop(base_url: str, deadline: float, timeout: float) -> List[tuple[int, float]]:
    out: List[tuple[int, float]] = []
    with httpx.Client(timeout=timeout) as client:
        while time.perf_counter() < deadline:
            status, _, dt_ms = post_generate(client, base_url, SAMPLE_SINGLE_PAYLOAD)
            out.append((status, dt_ms))
    return out


def run_step(base_url: str, concurrency: int, duration_s: float, timeout: float) -> Dict[str, Any]:
    deadline = time.perf_counter() + duration_s
    latencies_ms: List[float] = []
    successes = 0
    failures = 0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_send_loop, base_url, deadline, timeout) for _ in range(concurrency)]
        for fut in futures:
            for status, dt_ms in fut.result():
                latencies_ms.append(dt_ms)
                if status == 200:
                    successes += 1
                else:
                    failures += 1
    elapsed_s = time.perf_counter() - t0

    summary = summarize(latencies_ms) if latencies_ms else None
    total = successes + failures
    return {
        "concurrency": concurrency,
        "duration_s": round(elapsed_s, 1),
        "total_requests": total,
        "rps": round(successes / elapsed_s, 3) if elapsed_s > 0 else 0.0,
        "p95_latency_ms": round(summary.p95_ms, 1) if summary else 0.0,
        "error_rate_pct": round(100.0 * failures / max(1, total), 2),
        "vram_used_pct": _vram_used_pct(base_url, timeout),
    }


def measure_recovery(base_url: str, baseline_p95_ms: float, max_wait_s: float, timeout: float) -> Dict[str, Any]:
    """Probe the service every second until p95 over a sliding window matches baseline."""
    window: List[float] = []
    started = time.perf_counter()
    recovered_at: float | None = None

    with httpx.Client(timeout=timeout) as client:
        while time.perf_counter() - started < max_wait_s:
            _, _, dt_ms = post_generate(client, base_url, SAMPLE_SINGLE_PAYLOAD)
            window.append(dt_ms)
            if len(window) >= 5:
                window = window[-5:]
                cur_max = max(window)
                if cur_max <= baseline_p95_ms * 1.2:
                    recovered_at = time.perf_counter() - started
                    break

    return {
        "recovered_in_s": round(recovered_at, 1) if recovered_at is not None else None,
        "max_wait_s": max_wait_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_BASE_URL)
    parser.add_argument("--baseline", type=int, default=2,
                        help="concurrency that corresponds to штатный уровень")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="seconds per step")
    parser.add_argument("--recovery", type=float, default=30.0,
                        help="seconds to wait for recovery")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    if args.warmup:
        warmup(args.url, n=args.warmup, timeout=args.timeout)

    rows: List[Dict[str, Any]] = []
    for mult in STEPS:
        concurrency = mult * args.baseline
        row = run_step(args.url, concurrency, args.duration, args.timeout)
        row["multiplier"] = f"{mult}x"
        print(json.dumps(row, ensure_ascii=False))
        rows.append(row)

    baseline_p95_ms = rows[0]["p95_latency_ms"] if rows else 0.0
    recovery = measure_recovery(args.url, baseline_p95_ms, args.recovery, args.timeout)
    print(json.dumps({"recovery": recovery}, ensure_ascii=False))

    save_results("stress", {"steps": rows, "recovery": recovery})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
