"""Bench: VRAM consumption and initialization timing.

Fills in таблицу 22 (3.5):
    — VRAM after container start (before model load)
    — VRAM after model load
    — VRAM in steady state under load
    — cold-start time (container start → ready)
    — first-request latency after restart
    — steady-state latency

Usage:
    python -m scripts.benchmarks.init_resources --container sayit-exgen
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from typing import Any, Dict, List

import httpx

from scripts.benchmarks._common import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    SAMPLE_SINGLE_PAYLOAD,
    post_generate,
    save_results,
)


def _vram_used_mb(base_url: str, timeout: float) -> int | None:
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.get(f"{base_url}/health")
            r.raise_for_status()
            return r.json().get("gpu", {}).get("vram_used_mb")
    except Exception:
        return None


def _wait_until_ready(base_url: str, timeout_s: float) -> float | None:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout_s:
        try:
            with httpx.Client(timeout=10.0) as c:
                r = c.get(f"{base_url}/health")
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return time.perf_counter() - started
        except Exception:
            pass
        time.sleep(1.0)
    return None


def _restart_container(name: str) -> None:
    subprocess.run(["docker", "restart", name], check=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_BASE_URL)
    parser.add_argument("--container", default="sayit-exgen",
                        help="docker container name to restart for cold-start measurement")
    parser.add_argument("--ready-timeout", type=float, default=300.0)
    parser.add_argument("--load-requests", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--skip-restart", action="store_true",
                        help="skip docker restart and only measure steady state")
    args = parser.parse_args()

    out: Dict[str, Any] = {}

    if not args.skip_restart:
        _restart_container(args.container)
        time.sleep(2.0)
        out["vram_after_container_start_mb"] = _vram_used_mb(args.url, args.timeout)

        cold_start_s = _wait_until_ready(args.url, args.ready_timeout)
        out["cold_start_s"] = round(cold_start_s, 2) if cold_start_s is not None else None

        out["vram_after_model_load_mb"] = _vram_used_mb(args.url, args.timeout)

        with httpx.Client(timeout=args.timeout) as client:
            _, _, first_ms = post_generate(client, args.url, SAMPLE_SINGLE_PAYLOAD)
        out["first_request_after_restart_ms"] = round(first_ms, 1)

    steady_latencies: List[float] = []
    with httpx.Client(timeout=args.timeout) as client:
        for _ in range(args.load_requests):
            _, _, dt_ms = post_generate(client, args.url, SAMPLE_SINGLE_PAYLOAD)
            steady_latencies.append(dt_ms)

    if steady_latencies:
        steady_latencies.sort()
        out["steady_state_latency_ms"] = round(
            steady_latencies[len(steady_latencies) // 2], 1
        )

    out["vram_under_load_mb"] = _vram_used_mb(args.url, args.timeout)

    for key in ("vram_after_container_start_mb", "vram_after_model_load_mb", "vram_under_load_mb"):
        if out.get(key) is not None:
            out[key.replace("_mb", "_gb")] = round(out[key] / 1024.0, 2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    save_results("init_resources", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
