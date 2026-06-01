from __future__ import annotations

import argparse
import json
from dataclasses import asdict
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


def _count_tokens_in_exercise(body: Dict[str, Any]) -> int:
    try:
        text = json.dumps(body.get("exercise", {}), ensure_ascii=False)
    except (TypeError, ValueError):
        return 0
    return max(1, len(text.split()))


def _slope(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_BASE_URL)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    if args.warmup:
        warmup(args.url, n=args.warmup, timeout=args.timeout)

    latencies_ms: List[float] = []
    token_lengths: List[int] = []
    failures = 0

    with httpx.Client(timeout=args.timeout) as client:
        for i in range(args.n):
            status, body, dt_ms = post_generate(client, args.url, SAMPLE_SINGLE_PAYLOAD)
            if status != 200:
                failures += 1
                continue
            latencies_ms.append(dt_ms)
            token_lengths.append(_count_tokens_in_exercise(body))
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{args.n}] last={dt_ms:.0f} ms")

    summary = summarize(latencies_ms)
    slope_ms_per_token = _slope([float(t) for t in token_lengths], latencies_ms)

    out = {
        "n": args.n,
        "failures": failures,
        "summary": asdict(summary),
        "ms_per_token_estimate": round(slope_ms_per_token, 3),
    }
    print(json.dumps({k: v for k, v in out.items() if k != "summary"}, ensure_ascii=False))
    print(json.dumps({k: round(v, 1) if isinstance(v, float) else v
                      for k, v in asdict(summary).items() if k != "raw_ms"},
                     ensure_ascii=False))

    save_results("latency", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
