"""Run every benchmark and write a single summary file aligned with the X
placeholders in chapter 3.

The output file `results/bench/summary.json` lists each placeholder by its
table/paragraph reference and the measured value.

Usage:
    python -m scripts.benchmarks.run_all
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results" / "bench"


def _run(module: str, *extra: str) -> int:
    cmd = [sys.executable, "-m", module, *extra]
    print(f">>> {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT))


def _load(name: str) -> Dict[str, Any] | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _build_summary() -> Dict[str, Any]:
    from scripts.benchmarks import stand_info as stand_info_mod

    stand = stand_info_mod.collect()
    validity = _load("validity_rate") or {}
    latency = _load("latency") or {}
    throughput = _load("throughput") or {}
    stress = _load("stress") or {}
    init_res = _load("init_resources") or {}
    per_user = _load("per_user") or {}
    long_run = _load("long_run") or {}

    summary: Dict[str, Any] = {
        "section_3_4": {
            "stand": {
                "vram_total_gb": stand.get("gpu", {}).get("vram_total_gb"),
                "cpu_logical_cores": stand.get("cpu", {}).get("logical_cores"),
                "ram_total_gb": stand.get("ram", {}).get("total_gb"),
            },
            "warmup_requests": 3,
            "table_18_validity_rate": validity,
            "table_19_latency": latency.get("summary"),
            "ms_per_token": latency.get("ms_per_token_estimate"),
            "table_20_throughput": throughput.get("levels"),
            "saturation_concurrency": throughput.get("saturation_concurrency"),
            "table_21_stress": stress.get("steps"),
            "recovery_seconds": (stress.get("recovery") or {}).get("recovered_in_s"),
            "long_run_hours": long_run.get("duration_hours"),
            "long_run_error_rate_pct": long_run.get("error_rate_pct"),
        },
        "section_3_5": {
            "table_22_init_resources": init_res,
            "table_23_per_user": per_user.get("per_user_per_day"),
            "users_per_gpu_per_day": per_user.get("users_per_gpu_per_day"),
        },
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--throughput-requests", type=int, default=60)
    parser.add_argument("--stress-duration", type=float, default=60.0)
    parser.add_argument("--long-run-hours", type=float, default=8.0)
    parser.add_argument("--skip-long-run", action="store_true")
    parser.add_argument("--skip-init", action="store_true",
                        help="skip the cold-start measurement (which restarts the container)")
    args = parser.parse_args()

    _run("scripts.benchmarks.validity_rate", "--url", args.url, "--n", str(args.n))
    _run("scripts.benchmarks.latency", "--url", args.url, "--n", str(args.n))
    _run("scripts.benchmarks.throughput", "--url", args.url, "--requests", str(args.throughput_requests))
    _run("scripts.benchmarks.stress", "--url", args.url, "--duration", str(args.stress_duration))
    _run("scripts.benchmarks.per_user", "--url", args.url)
    if not args.skip_init:
        _run("scripts.benchmarks.init_resources", "--url", args.url)
    if not args.skip_long_run:
        _run("scripts.benchmarks.long_run", "--url", args.url, "--hours", str(args.long_run_hours))

    summary = _build_summary()
    out_path = RESULTS_DIR / "summary.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nsummary written to {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
