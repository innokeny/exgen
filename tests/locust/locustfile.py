from __future__ import annotations

import json
import os
import random
from pathlib import Path
from threading import Lock
from typing import Any

from locust import HttpUser, between, events, task

PAYLOADS_PATH = Path(__file__).parent / "payloads.json"
REQUEST_TIMEOUT_S = 120.0


def _load_payloads() -> list[dict[str, Any]]:
    with PAYLOADS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"{PAYLOADS_PATH} must contain a non-empty JSON array")
    return data


PAYLOADS: list[dict[str, Any]] = _load_payloads()


class _FallbackCounter:

    def __init__(self) -> None:
        self._lock = Lock()
        self._total_ok = 0
        self._total_fallback = 0
        self._reasons: dict[str, int] = {}

    def record_ok(self) -> None:
        with self._lock:
            self._total_ok += 1

    def record_fallback(self, reason: str | None) -> None:
        key = reason or "unknown"
        with self._lock:
            self._total_fallback += 1
            self._reasons[key] = self._reasons.get(key, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = self._total_ok + self._total_fallback
            rate = (self._total_fallback / total) if total else 0.0
            return {
                "successful_responses": total,
                "ok": self._total_ok,
                "fallback": self._total_fallback,
                "fallback_rate": rate,
                "fallback_reasons": dict(self._reasons),
            }


FALLBACKS = _FallbackCounter()


class GenerateUser(HttpUser):
    wait_time = between(0.0, 0.2)
    host = os.environ.get("TARGET_HOST", "http://localhost:8000")

    @task
    def generate(self) -> None:
        payload = random.choice(PAYLOADS)
        with self.client.post(
            "/api/v1/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT_S,
            name="POST /api/v1/generate",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
                return

            try:
                body = response.json()
            except ValueError as e:
                response.failure(f"invalid JSON: {e}")
                return

            exercise = body.get("exercise")
            if not exercise:
                response.failure("response missing non-empty 'exercise' field")
                return

            fallback_reason = body.get("fallback_reason")
            if body.get("status") == "fallback" or fallback_reason:
                FALLBACKS.record_fallback(fallback_reason)
            else:
                FALLBACKS.record_ok()

            response.success()


@events.quitting.add_listener
def _report_fallbacks(environment, **_kwargs) -> None:  # noqa: ANN001
    snap = FALLBACKS.snapshot()
    print("")
    print("=== Fallback statistics (HTTP 200 responses only) ===")
    print(f"  successful responses : {snap['successful_responses']}")
    print(f"  ok (LLM parsed)      : {snap['ok']}")
    print(f"  fallback (template)  : {snap['fallback']}")
    print(f"  fallback rate        : {snap['fallback_rate']:.3%}")
    if snap["fallback_reasons"]:
        print("  reasons:")
        for reason, count in sorted(
            snap["fallback_reasons"].items(), key=lambda kv: -kv[1]
        ):
            print(f"    - {reason}: {count}")

    csv_prefix = os.environ.get("LOCUST_CSV_PREFIX")
    if csv_prefix:
        out_path = Path(f"{csv_prefix}_fallbacks.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
