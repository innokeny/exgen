"""
Cross-validate Locust and AIPerf measurements.

Reads:
    ../load/results/latency_stats.csv         single-user Locust run
    ../load/results/throughput_stats_history.csv  Locust concurrency sweep
    reports/aiperf-c{1,2,4,8,16}/aiperf.json  AIPerf runs

Writes:
    reports/cross_validation.md               for the thesis appendix
    reports/cross_validation.csv              raw side-by-side numbers

The thesis claim that the two tools agree should hold: Locust measures
end-to-end HTTP latency from a Python client, AIPerf measures the same
through a Triton-derived async benchmark. If they diverge by more than
~10% on p99 latency, something is wrong (most likely the shim adds
non-trivial overhead).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
LOCUST_DIR = ROOT.parent / "load" / "results"
AIPERF_DIR = ROOT / "reports"


def _read_aiperf(concurrency: int) -> dict | None:
    path = AIPERF_DIR / f"aiperf-c{concurrency}" / "aiperf.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    # AIPerf JSON layout has a top-level "metrics" or "results" mapping
    # depending on the version. Handle both.
    metrics = data.get("metrics") or data.get("results") or data
    return metrics


def _aiperf_metric(metrics: dict | None, key: str, stat: str = "p50") -> float | None:
    """Best-effort extraction across AIPerf / GenAI-Perf layout variants."""
    if not metrics:
        return None
    candidates = [
        (key, stat),
        (key.replace("_", " "), stat),
        (key.title().replace("_", " "), stat),
    ]
    for k, s in candidates:
        node = metrics.get(k)
        if isinstance(node, dict):
            value = node.get(s) or node.get(s.upper())
            if value is not None:
                return float(value)
    return None


def _locust_single_user() -> dict:
    """Latency stats from the single-user Locust run."""
    f = LOCUST_DIR / "latency_stats.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    agg = df[df["Name"] == "Aggregated"]
    if agg.empty:
        return {}
    row = agg.iloc[0]
    return {
        "p50_ms": float(row["Median Response Time"]),
        "p95_ms": float(row["95%"]),
        "p99_ms": float(row["99%"]),
        "mean_ms": float(row["Average Response Time"]),
        "rps": float(row["Requests/s"]),
    }


def _locust_throughput_sweep() -> dict[int, dict]:
    """Per-stage stats for the concurrency sweep."""
    f = LOCUST_DIR / "throughput_stats_history.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    df = df[df["Name"] == "Aggregated"]
    if df.empty:
        return {}

    t0 = df["Timestamp"].iloc[0]
    df["t"] = df["Timestamp"] - t0

    stages = [
        (1,    0,  300),
        (2,  300,  600),
        (4,  600,  900),
        (8,  900, 1200),
        (16, 1200, 1500),
    ]
    out: dict[int, dict] = {}
    for users, lo, hi in stages:
        win = df[(df["t"] >= lo) & (df["t"] < hi)]
        if win.empty:
            continue
        out[users] = {
            "p50_ms": float(win["Total Median Response Time"].mean()) if "Total Median Response Time" in win else float("nan"),
            "p95_ms": float(win["95%"].mean()) if "95%" in win else float("nan"),
            "p99_ms": float(win["99%"].mean()) if "99%" in win else float("nan"),
            "mean_ms": float(win["Total Average Response Time"].mean()),
            "rps": float(win["Requests/s"].mean()),
        }
    return out


def _row(label: str, locust: dict, aiperf: dict | None) -> list[str]:
    def fmt(value: float | None) -> str:
        return "—" if value is None or value != value else f"{value:.0f}"

    a_p50 = _aiperf_metric(aiperf, "request_latency", "p50")
    a_p95 = _aiperf_metric(aiperf, "request_latency", "p95")
    a_p99 = _aiperf_metric(aiperf, "request_latency", "p99")
    a_rps = _aiperf_metric(aiperf, "request_throughput", "avg")

    def diff_pct(a: float | None, b: float | None) -> str:
        if a is None or b is None or b == 0:
            return "—"
        return f"{(a - b) / b * 100:+.1f}%"

    return [
        label,
        fmt(locust.get("p50_ms")),
        fmt(a_p50),
        diff_pct(a_p50, locust.get("p50_ms")),
        fmt(locust.get("p95_ms")),
        fmt(a_p95),
        diff_pct(a_p95, locust.get("p95_ms")),
        fmt(locust.get("p99_ms")),
        fmt(a_p99),
        diff_pct(a_p99, locust.get("p99_ms")),
        f"{locust.get('rps', 0):.2f}",
        f"{(a_rps or 0):.2f}",
    ]


def main() -> int:
    AIPERF_DIR.mkdir(parents=True, exist_ok=True)

    headers = [
        "Сценарий",
        "Locust p50, мс", "AIPerf p50, мс", "Δ p50",
        "Locust p95, мс", "AIPerf p95, мс", "Δ p95",
        "Locust p99, мс", "AIPerf p99, мс", "Δ p99",
        "Locust RPS", "AIPerf RPS",
    ]

    rows: list[list[str]] = []
    rows.append(_row("1 клиент (latency)", _locust_single_user(), _read_aiperf(1)))

    sweep = _locust_throughput_sweep()
    for c in (2, 4, 8, 16):
        rows.append(_row(f"{c} клиентов", sweep.get(c, {}), _read_aiperf(c)))

    # Markdown.
    md = ["# Кросс-валидация Locust ↔ AIPerf",
          "",
          "Независимые замеры одной и той же системы двумя инструментами. ",
          "Расхождение ≤10 % по p99 подтверждает точность измерений.",
          "",
          "| " + " | ".join(headers) + " |",
          "| " + " | ".join("---" for _ in headers) + " |"]
    md.extend("| " + " | ".join(r) + " |" for r in rows)
    (AIPERF_DIR / "cross_validation.md").write_text("\n".join(md) + "\n",
                                                    encoding="utf-8")

    # CSV.
    with open(AIPERF_DIR / "cross_validation.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

    print(f"Wrote {AIPERF_DIR / 'cross_validation.md'}")
    print(f"Wrote {AIPERF_DIR / 'cross_validation.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
