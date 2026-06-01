"""
Functional smoke tests for the SAYIT generator service.

Maps directly onto the test-case table in §3.4 of the thesis (ФТ-01 ... ФТ-14).
Run after `docker compose up -d` and a successful /health check.

    pytest -v --junit-xml=reports/pytest-junit.xml test_smoke.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import pytest

BASE_URL = os.environ.get("SAYIT_GENERATOR_URL", "http://localhost:8000")
TIMEOUT = httpx.Timeout(120.0)

DATA_DIR = Path(__file__).parent.parent / "load" / "data"
SINGLE_PAYLOADS = json.loads((DATA_DIR / "single_payloads.json").read_text())
ERROR_PROFILES = json.loads((DATA_DIR / "error_profiles.json").read_text())
TEMPLATE_PAYLOADS = json.loads((DATA_DIR / "template_payloads.json").read_text())


@pytest.fixture(scope="session")
def client() -> httpx.Client:
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as c:
        # Wait until the service reports loaded models (ФТ-13: cold-start).
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            try:
                r = c.get("/health")
                if r.status_code == 200 and r.json().get("loaded_models"):
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
        else:
            pytest.fail("Service did not become ready within 5 minutes")
        yield c


# ---------- ФТ-01: single happy path -----------------------------------------


def test_ft01_generate_happy_path(client: httpx.Client) -> None:
    payload = SINGLE_PAYLOADS[0]
    r = client.post("/api/v1/generate", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] in {"ok", "fallback"}
    assert body["model_used"]
    assert "exercise" in body
    assert body["exercise"]["task"]["type"] == payload["task_type"]


# ---------- ФТ-02..ФТ-05: each task_type produces a valid exercise ----------


@pytest.mark.parametrize(
    "task_type",
    ["grammar_choice", "transformation", "vocabulary_fill", "matching"],
)
def test_ft02_05_task_types(client: httpx.Client, task_type: str) -> None:
    payload = dict(SINGLE_PAYLOADS[0], task_type=task_type)
    r = client.post("/api/v1/generate", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["exercise"]["task"]["type"] == task_type
    items = body["exercise"]["task"]["content_en"]["items"]
    assert isinstance(items, list) and len(items) >= 1


# ---------- ФТ-06: coverage of error categories ------------------------------


@pytest.mark.parametrize("payload", SINGLE_PAYLOADS, ids=lambda p: p["grammar_error"])
def test_ft06_error_category_coverage(client: httpx.Client, payload: dict) -> None:
    r = client.post("/api/v1/generate", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["exercise"]["target_error_category"]


# ---------- ФТ-07: unsupported model id --------------------------------------


def test_ft07_unsupported_model_returns_4xx(client: httpx.Client) -> None:
    payload = dict(SINGLE_PAYLOADS[0], model="nonexistent-7b")
    r = client.post("/api/v1/generate", json=payload)
    assert r.status_code in {400, 404, 422, 503}, r.text


# ---------- ФТ-08: missing required fields ----------------------------------


def test_ft08_missing_required_fields(client: httpx.Client) -> None:
    r = client.post("/api/v1/generate", json={"user_id": "x"})
    assert r.status_code == 422, r.text
    detail = r.json().get("detail", [])
    assert any(d.get("loc") for d in detail), "validation must point at offending field"


# ---------- ФТ-09: batch with multi-category profile ------------------------


def test_ft09_batch_distribution(client: httpx.Client) -> None:
    profile = ERROR_PROFILES[0]
    r = client.post("/api/v1/generate/batch", json=profile)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] in {"ok", "partial"}
    questions = body["questions"]
    assert len(questions) <= profile["max_questions"]
    # Every category that had examples should appear at least once.
    seen = {q["error_type"] for q in questions}
    expected = {c["error_type"] for c in profile["error_profile"] if c["examples"]}
    assert seen == expected, f"missing categories: {expected - seen}"


# ---------- ФТ-10: empty profile is rejected --------------------------------


def test_ft10_empty_profile_rejected(client: httpx.Client) -> None:
    r = client.post(
        "/api/v1/generate/batch",
        json={"user_id": "u", "language_level": "B1", "error_profile": []},
    )
    assert r.status_code == 400, r.text


# ---------- ФТ-11: fallback path is exercised by template endpoint ----------


def test_ft11_template_fallback_path(client: httpx.Client) -> None:
    r = client.post("/api/v1/generate/template", json=TEMPLATE_PAYLOADS[0])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body.get("model_used") in {None, "template"}


# ---------- ФТ-12: missing adapter -> 503 (retryable) -----------------------


def test_ft12_missing_adapter_returns_503(client: httpx.Client) -> None:
    payload = dict(SINGLE_PAYLOADS[0], model="qwen2.5-3b-not-loaded")
    r = client.post("/api/v1/generate", json=payload)
    # Either rejected at validation (422) or treated as transient (503) — both OK.
    assert r.status_code in {404, 422, 503}, r.text


# ---------- ФТ-13: repeated calls — no VRAM leak, stable latency ------------


def test_ft13_no_latency_drift_over_n_calls(client: httpx.Client) -> None:
    payload = SINGLE_PAYLOADS[0]
    timings = []
    for _ in range(20):
        t0 = time.perf_counter()
        r = client.post("/api/v1/generate", json=payload)
        timings.append(time.perf_counter() - t0)
        assert r.status_code == 200
    # First request gets warm-up overhead; ignore it.
    warm = timings[1:]
    drift = max(warm) / min(warm)
    assert drift < 5.0, f"latency drift too large: {drift:.1f}x"


# ---------- ФТ-14: end-to-end batch shape matches SAYIT contract ------------


def test_ft14_questions_match_sayit_contract(client: httpx.Client) -> None:
    r = client.post("/api/v1/generate/batch", json=ERROR_PROFILES[2])
    assert r.status_code == 200
    for q in r.json()["questions"]:
        assert {"id", "error_type", "text", "options", "correct_answer", "explanation"} <= set(q)
        assert isinstance(q["options"], list) and len(q["options"]) == 4
        assert q["correct_answer"] in q["options"]


# ---------- /health ---------------------------------------------------------


def test_health_reports_gpu(client: httpx.Client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "gpu" in body
    assert body["loaded_models"], "no models loaded — service not ready"
