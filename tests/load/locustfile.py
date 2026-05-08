"""
Locust load testing scenarios for SAYIT Exercise Generator.

Three user classes correspond to the three production endpoints:
- SingleGenerationUser  -> POST /api/v1/generate
- BatchGenerationUser   -> POST /api/v1/generate/batch
- TemplateUser          -> POST /api/v1/generate/template

Run examples:

    # Latency profile (single user, 200 requests on /generate)
    locust -f locustfile.py --host=http://localhost:8000 \
        --headless -u 1 -r 1 --run-time 10m \
        --csv=results/latency --html=results/latency.html \
        SingleGenerationUser

    # Concurrency sweep (1, 2, 4, 8, 16 users via shape)
    locust -f locustfile.py --host=http://localhost:8000 \
        --headless --csv=results/throughput --html=results/throughput.html \
        ThroughputShape

    # Stress test (steps of 2x, 4x, 8x baseline)
    locust -f locustfile.py --host=http://localhost:8000 \
        --headless --csv=results/stress --html=results/stress.html \
        StressShape

    # Endurance (8 hours steady)
    locust -f locustfile.py --host=http://localhost:8000 \
        --headless -u 1 -r 1 --run-time 8h \
        --csv=results/endurance --html=results/endurance.html \
        BatchGenerationUser
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from locust import HttpUser, between, events, task

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"

with open(DATA_DIR / "single_payloads.json", encoding="utf-8") as f:
    SINGLE_PAYLOADS: list[dict] = json.load(f)

with open(DATA_DIR / "error_profiles.json", encoding="utf-8") as f:
    ERROR_PROFILES: list[dict] = json.load(f)

with open(DATA_DIR / "template_payloads.json", encoding="utf-8") as f:
    TEMPLATE_PAYLOADS: list[dict] = json.load(f)


# ---------------------------------------------------------------------------
# User classes
# ---------------------------------------------------------------------------


class SingleGenerationUser(HttpUser):
    """Hits POST /api/v1/generate with realistic single-exercise payloads."""

    wait_time = between(0.0, 0.0)  # closed-loop: keep pressure on

    @task
    def generate(self) -> None:
        payload = random.choice(SINGLE_PAYLOADS)
        with self.client.post(
            "/api/v1/generate",
            json=payload,
            name="POST /api/v1/generate",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
                return
            try:
                body = response.json()
            except json.JSONDecodeError:
                response.failure("Body is not valid JSON")
                return
            if body.get("status") not in {"ok", "fallback"}:
                response.failure(f"Unexpected status: {body.get('status')}")
                return
            # Track tokens/sec for §3.5 calculations.
            items = (
                body.get("exercise", {})
                .get("task", {})
                .get("content_en", {})
                .get("items", [])
            )
            output_chars = len(json.dumps(body, ensure_ascii=False))
            response.request_meta["context"] = {
                "items": len(items),
                "output_chars": output_chars,
                "fallback": body.get("status") == "fallback",
            }


class BatchGenerationUser(HttpUser):
    """Hits POST /api/v1/generate/batch — production-shaped traffic."""

    wait_time = between(0.0, 0.0)

    @task
    def batch(self) -> None:
        profile = random.choice(ERROR_PROFILES)
        with self.client.post(
            "/api/v1/generate/batch",
            json=profile,
            name="POST /api/v1/generate/batch",
            catch_response=True,
            timeout=120,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
                return
            try:
                body = response.json()
            except json.JSONDecodeError:
                response.failure("Body is not valid JSON")
                return
            if body.get("status") not in {"ok", "partial"}:
                response.failure(f"Unexpected status: {body.get('status')}")
                return
            response.request_meta["context"] = {
                "questions": len(body.get("questions", [])),
                "fallback_categories": len(body.get("fallback_categories") or []),
                "partial": body.get("status") == "partial",
            }


class TemplateUser(HttpUser):
    """Hits POST /api/v1/generate/template — CPU-only fallback path."""

    wait_time = between(0.0, 0.0)

    @task
    def template(self) -> None:
        payload = random.choice(TEMPLATE_PAYLOADS)
        with self.client.post(
            "/api/v1/generate/template",
            json=payload,
            name="POST /api/v1/generate/template",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
            elif response.json().get("status") != "ok":
                response.failure("status != ok")


# ---------------------------------------------------------------------------
# Load shapes — drive the throughput sweep, stress test, endurance run
# ---------------------------------------------------------------------------

from locust import LoadTestShape


class ThroughputShape(LoadTestShape):
    """Concurrency sweep: 1 -> 2 -> 4 -> 8 -> 16 users, 5 min per step.

    Maps directly onto Table 20 (throughput vs. parallel clients) and
    Figure 16 (throughput-vs-concurrency curve) in §3.4 of the thesis.
    """

    use_common_options = True
    user_classes = [BatchGenerationUser]

    stages = [
        {"duration": 300,  "users": 1,  "spawn_rate": 1},
        {"duration": 600,  "users": 2,  "spawn_rate": 1},
        {"duration": 900,  "users": 4,  "spawn_rate": 1},
        {"duration": 1200, "users": 8,  "spawn_rate": 1},
        {"duration": 1500, "users": 16, "spawn_rate": 2},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None


class StressShape(LoadTestShape):
    """Step load: 1x -> 2x -> 4x -> 8x baseline, 5 min each, then ramp down.

    Maps onto Table 21 (stress-load behaviour) in §3.4.
    """

    use_common_options = True
    user_classes = [BatchGenerationUser]

    stages = [
        {"duration": 300,  "users": 1,  "spawn_rate": 1},   # 1x baseline
        {"duration": 600,  "users": 2,  "spawn_rate": 1},   # 2x
        {"duration": 900,  "users": 4,  "spawn_rate": 1},   # 4x
        {"duration": 1200, "users": 8,  "spawn_rate": 2},   # 8x
        {"duration": 1500, "users": 1,  "spawn_rate": 1},   # recovery
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None


# ---------------------------------------------------------------------------
# Telemetry hooks
# ---------------------------------------------------------------------------


@events.request.add_listener
def _record_extra_metrics(
    request_type, name, response_time, response_length, exception, context, **kwargs
):
    """Push tokens/items into a per-process aggregator for analyze.py."""
    if exception is not None or not context:
        return
    _aggregator.add(name, response_time, context)


class _MetricsAggregator:
    """Minimal in-memory aggregator. analyze.py reads the JSONL on shutdown."""

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def add(self, name: str, response_time_ms: float, ctx: dict) -> None:
        self.entries.append(
            {"endpoint": name, "rt_ms": response_time_ms, **ctx}
        )

    def dump(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


_aggregator = _MetricsAggregator()


@events.test_stop.add_listener
def _on_test_stop(environment, **kwargs):
    out = Path(environment.parsed_options.csv_prefix or "results/run") if hasattr(
        environment.parsed_options, "csv_prefix"
    ) else Path("results/run")
    out.parent.mkdir(parents=True, exist_ok=True)
    _aggregator.dump(out.with_suffix(".extra.jsonl"))
