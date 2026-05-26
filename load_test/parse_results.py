"""Build a summary table from the per-level Locust CSV reports.

Reads ``results/concurrency_<N>/locust_stats.csv`` for every concurrency level
that has been run and emits a single Markdown table (to stdout and to
``results/summary.md``) plus a CSV equivalent (``results/summary.csv``).

The summary holds, per concurrency level:

* total_requests   — sum of successful and failed requests
* success_rate     — share of requests that locust counted as successful
                     (locust's "success" already accounts for the HTTP-200 +
                     non-empty-exercise check enforced in locustfile.py)
* RPS              — measured throughput (requests / second)
* latency_p50/p95/p99 — milliseconds; sourced from locust's percentile columns
* fallback_rate    — share of *successful* responses that came from the
                     template-engine safety net (LLM output unparseable);
                     this is loaded from the JSON sidecar dumped by the
                     locustfile's ``quitting`` hook.

Locust 2.x writes the per-endpoint rows and one aggregate row named
``Aggregated``. This script reads the aggregate row only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# Locust 2.x stats CSV columns we depend on. The 50/95/99 columns are written
# as floats in milliseconds for the response-time percentiles.
COL_NAME = "Name"
COL_REQS = "Request Count"
COL_FAILS = "Failure Count"
COL_RPS = "Requests/s"
COL_P50 = "50%"
COL_P95 = "95%"
COL_P99 = "99%"


def _safe_float(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "").strip()
    if not raw or raw in {"N/A", "-"}:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _safe_int(row: dict[str, str], key: str) -> int:
    return int(_safe_float(row, key))


def _read_aggregate_row(stats_csv: Path) -> dict[str, str] | None:
    with stats_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        aggregate: dict[str, str] | None = None
        for row in reader:
            if row.get(COL_NAME, "").strip().lower() == "aggregated":
                aggregate = row
                break
    return aggregate


def _read_fallback_stats(run_dir: Path) -> tuple[int, int, float]:
    """Return (ok, fallback, fallback_rate). All zeros if sidecar is missing."""
    sidecar = run_dir / "locust_fallbacks.json"
    if not sidecar.exists():
        return 0, 0, 0.0
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, 0, 0.0
    return (
        int(data.get("ok", 0)),
        int(data.get("fallback", 0)),
        float(data.get("fallback_rate", 0.0)),
    )


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not RESULTS_DIR.exists():
        return rows

    run_dirs = sorted(
        (p for p in RESULTS_DIR.iterdir() if p.is_dir() and p.name.startswith("concurrency_")),
        key=lambda p: int(p.name.split("_", 1)[1]),
    )
    for run_dir in run_dirs:
        concurrency = int(run_dir.name.split("_", 1)[1])
        stats_csv = run_dir / "locust_stats.csv"
        if not stats_csv.exists():
            continue

        aggregate = _read_aggregate_row(stats_csv)
        if aggregate is None:
            continue

        total = _safe_int(aggregate, COL_REQS)
        fails = _safe_int(aggregate, COL_FAILS)
        success_rate = ((total - fails) / total) if total else 0.0

        _, _, fallback_rate = _read_fallback_stats(run_dir)

        rows.append(
            {
                "concurrency": concurrency,
                "total_requests": total,
                "success_rate": success_rate,
                "RPS": _safe_float(aggregate, COL_RPS),
                "latency_p50": _safe_float(aggregate, COL_P50),
                "latency_p95": _safe_float(aggregate, COL_P95),
                "latency_p99": _safe_float(aggregate, COL_P99),
                "fallback_rate": fallback_rate,
            }
        )
    return rows


def render_markdown(rows: list[dict[str, object]]) -> str:
    header = (
        "| concurrency | total_requests | success_rate | RPS | "
        "latency_p50 (ms) | latency_p95 (ms) | latency_p99 (ms) | fallback_rate |"
    )
    sep = "|" + "|".join(["---"] * 8) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            "| {concurrency} | {total_requests} | {success_rate:.2%} | "
            "{RPS:.2f} | {p50:.0f} | {p95:.0f} | {p99:.0f} | "
            "{fallback_rate:.2%} |".format(
                concurrency=r["concurrency"],
                total_requests=r["total_requests"],
                success_rate=r["success_rate"],
                RPS=r["RPS"],
                p50=r["latency_p50"],
                p95=r["latency_p95"],
                p99=r["latency_p99"],
                fallback_rate=r["fallback_rate"],
            )
        )
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "concurrency",
        "total_requests",
        "success_rate",
        "RPS",
        "latency_p50",
        "latency_p95",
        "latency_p99",
        "fallback_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows = collect_rows()
    if not rows:
        print(
            f"No per-level results found under {RESULTS_DIR}. "
            "Run ./run_benchmark.sh first.",
        )
        return 1

    md = render_markdown(rows)
    print(md, end="")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "summary.md").write_text(md, encoding="utf-8")
    write_csv(rows, RESULTS_DIR / "summary.csv")
    print(f"\nWrote {RESULTS_DIR / 'summary.md'} and {RESULTS_DIR / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
