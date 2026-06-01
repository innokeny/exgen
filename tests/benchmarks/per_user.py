"""Bench: per-user resource projection.

Fills in таблицу 23 (3.5) — расчётные показатели на одного активного
пользователя в сутки. Берёт измеренное среднее время инференса одного
пакетного запроса и среднее энергопотребление графического ускорителя
во время генерации, после чего масштабирует их на типовой профиль
использования платформы (число пакетных запросов в сутки × число
упражнений в пакете).

Usage:
    python -m scripts.benchmarks.per_user --requests-per-day 4 --batch-size 9 --runs 20
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from typing import Any, Dict, List

import httpx

from scripts.benchmarks._common import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    SAMPLE_BATCH_PAYLOAD,
    post_batch,
    save_results,
    warmup,
)


def _gpu_power_w() -> float | None:
    """Read instantaneous GPU power draw via nvidia-smi (single GPU)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5.0,
        ).stdout.strip().splitlines()
        if not out:
            return None
        return float(out[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_BASE_URL)
    parser.add_argument("--runs", type=int, default=20,
                        help="number of batch requests to sample")
    parser.add_argument("--requests-per-day", type=int, default=4,
                        help="typical number of batch requests per active user per day")
    parser.add_argument("--batch-size", type=int, default=9,
                        help="typical questions per batch")
    parser.add_argument("--seconds-per-day", type=int, default=86_400)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    if args.warmup:
        warmup(args.url, n=args.warmup, timeout=args.timeout)

    batch_durations_s: List[float] = []
    questions_produced: List[int] = []
    powers_w: List[float] = []

    payload = dict(SAMPLE_BATCH_PAYLOAD, max_questions=args.batch_size)

    with httpx.Client(timeout=args.timeout) as client:
        for i in range(args.runs):
            t0 = time.perf_counter()
            status, body, _ = post_batch(client, args.url, payload)
            elapsed_s = time.perf_counter() - t0

            mid_power = _gpu_power_w()
            if mid_power is not None:
                powers_w.append(mid_power)

            if status != 200:
                continue
            batch_durations_s.append(elapsed_s)
            questions_produced.append(len(body.get("questions", []) or []))
            if (i + 1) % 5 == 0:
                print(f"[{i+1}/{args.runs}] {elapsed_s:.1f} s, "
                      f"{questions_produced[-1]} questions")

    if not batch_durations_s:
        print(json.dumps({"error": "no successful batches recorded"}, ensure_ascii=False))
        return 1

    avg_batch_s = statistics.mean(batch_durations_s)
    avg_questions = statistics.mean(questions_produced) if questions_produced else args.batch_size
    avg_power_w = statistics.mean(powers_w) if powers_w else 0.0

    daily_inference_s = args.requests_per_day * avg_batch_s
    daily_energy_wh = (daily_inference_s / 3600.0) * avg_power_w
    daily_questions = args.requests_per_day * avg_questions
    gpu_utilization_pct = round(100.0 * daily_inference_s / args.seconds_per_day, 4)

    max_users = int(args.seconds_per_day / max(daily_inference_s, 1e-6))

    out: Dict[str, Any] = {
        "samples": {
            "runs": len(batch_durations_s),
            "avg_batch_seconds": round(avg_batch_s, 2),
            "avg_questions_per_batch": round(avg_questions, 2),
            "avg_gpu_power_w": round(avg_power_w, 1) if avg_power_w else None,
        },
        "profile": {
            "requests_per_day": args.requests_per_day,
            "batch_size": args.batch_size,
        },
        "per_user_per_day": {
            "exercises": round(daily_questions, 1),
            "inference_seconds": round(daily_inference_s, 2),
            "energy_wh": round(daily_energy_wh, 3) if avg_power_w else None,
            "gpu_busy_pct": gpu_utilization_pct,
        },
        "users_per_gpu_per_day": max_users,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    save_results("per_user", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
