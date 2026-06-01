from __future__ import annotations

import argparse
import json
import statistics
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
    warmup,
)


def _vram_used_mb(base_url: str, timeout: float) -> int | None:
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.get(f"{base_url}/health")
            r.raise_for_status()
            return r.json().get("gpu", {}).get("vram_used_mb")
    except Exception:
        return None


def _send_loop(base_url: str, deadline: float, timeout: float) -> List[tuple[int, float]]:
    out: List[tuple[int, float]] = []
    with httpx.Client(timeout=timeout) as client:
        while time.perf_counter() < deadline:
            status, _, dt_ms = post_generate(client, base_url, SAMPLE_SINGLE_PAYLOAD)
            out.append((status, dt_ms))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_BASE_URL)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--baseline", type=int, default=2,
                        help="concurrency that corresponds to штатный уровень")
    parser.add_argument("--probe-interval", type=float, default=300.0,
                        help="seconds between VRAM/health probes")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    if args.warmup:
        warmup(args.url, n=args.warmup, timeout=args.timeout)

    duration_s = args.hours * 3600.0
    started = time.perf_counter()
    deadline = started + duration_s

    vram_samples_mb: List[int] = []
    initial_vram = _vram_used_mb(args.url, args.timeout)
    if initial_vram is not None:
        vram_samples_mb.append(initial_vram)

    latencies_ms: List[float] = []
    successes = 0
    failures = 0

    print(f"long-run: {args.hours} h at concurrency {args.baseline}")

    with ThreadPoolExecutor(max_workers=args.baseline + 1) as pool:
        worker_futures = [
            pool.submit(_send_loop, args.url, deadline, args.timeout)
            for _ in range(args.baseline)
        ]
        next_probe = started + args.probe_interval
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if now >= next_probe:
                vram = _vram_used_mb(args.url, args.timeout)
                if vram is not None:
                    vram_samples_mb.append(vram)
                next_probe += args.probe_interval
                hours_done = (now - started) / 3600.0
                print(f"[{hours_done:.2f} h] vram_used_mb={vram}")
            time.sleep(min(10.0, max(1.0, next_probe - now)))

        for fut in worker_futures:
            for status, dt_ms in fut.result():
                latencies_ms.append(dt_ms)
                if status == 200:
                    successes += 1
                else:
                    failures += 1

    elapsed_s = time.perf_counter() - started

    if latencies_ms:
        median_ms = statistics.median(latencies_ms)
        p95_ms = (statistics.quantiles(latencies_ms, n=20)[18]
                  if len(latencies_ms) >= 20 else max(latencies_ms))
    else:
        median_ms = p95_ms = 0.0

    out: Dict[str, Any] = {
        "duration_hours": round(elapsed_s / 3600.0, 2),
        "baseline_concurrency": args.baseline,
        "successes": successes,
        "failures": failures,
        "error_rate_pct": round(100.0 * failures / max(1, successes + failures), 4),
        "median_latency_ms": round(median_ms, 1),
        "p95_latency_ms": round(p95_ms, 1),
        "vram_initial_mb": vram_samples_mb[0] if vram_samples_mb else None,
        "vram_final_mb": vram_samples_mb[-1] if vram_samples_mb else None,
        "vram_max_mb": max(vram_samples_mb) if vram_samples_mb else None,
        "vram_growth_mb": (
            (vram_samples_mb[-1] - vram_samples_mb[0]) if len(vram_samples_mb) >= 2 else 0
        ),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    save_results("long_run", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
