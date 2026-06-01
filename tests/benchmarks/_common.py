from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_BASE_URL = os.environ.get("EXGEN_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = float(os.environ.get("EXGEN_TIMEOUT", "300"))

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "bench"


SAMPLE_SINGLE_PAYLOAD: Dict[str, Any] = {
    "user_id": "bench-user",
    "message_content": "It has a high-density population because for its small territory.",
    "grammar_error": "Preposition",
    "explanation": "Wrong preposition: 'for' should be 'of'.",
    "llm_confidence": 0.95,
    "task_type": "grammar_choice",
}


SAMPLE_BATCH_PAYLOAD: Dict[str, Any] = {
    "user_id": "bench-user",
    "language_level": "B1",
    "max_questions": 9,
    "error_profile": [
        {
            "error_type": "Preposition",
            "occurrences": 6,
            "examples": [{
                "original": "It has a high-density population because for its small territory.",
                "corrected": "It has a high-density population because of its small territory.",
                "explanation": "Use 'because of' before a noun phrase.",
            }],
        },
        {
            "error_type": "Tense",
            "occurrences": 3,
            "examples": [{
                "original": "She have finished her homework.",
                "corrected": "She has finished her homework.",
                "explanation": "Subject-verb agreement in present perfect.",
            }],
        },
        {
            "error_type": "Article",
            "occurrences": 1,
            "examples": [{
                "original": "I saw elephant at the zoo.",
                "corrected": "I saw an elephant at the zoo.",
                "explanation": "Use the indefinite article before a singular countable noun.",
            }],
        },
    ],
}


@dataclass
class LatencySummary:
    n: int
    min_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float
    stdev_ms: float
    raw_ms: List[float] = field(default_factory=list)


def summarize(latencies_ms: List[float]) -> LatencySummary:
    if not latencies_ms:
        raise ValueError("no samples to summarize")
    sorted_ms = sorted(latencies_ms)
    n = len(sorted_ms)

    def percentile(p: float) -> float:
        if n == 1:
            return sorted_ms[0]
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return sorted_ms[idx]

    return LatencySummary(
        n=n,
        min_ms=min(sorted_ms),
        median_ms=statistics.median(sorted_ms),
        p90_ms=percentile(0.90),
        p95_ms=percentile(0.95),
        p99_ms=percentile(0.99),
        max_ms=max(sorted_ms),
        mean_ms=statistics.mean(sorted_ms),
        stdev_ms=statistics.stdev(sorted_ms) if n >= 2 else 0.0,
        raw_ms=list(sorted_ms),
    )


def post_generate(client: httpx.Client, base_url: str, payload: Optional[Dict[str, Any]] = None) -> tuple[int, Dict[str, Any], float]:
    body = payload if payload is not None else SAMPLE_SINGLE_PAYLOAD
    t0 = time.perf_counter()
    r = client.post(f"{base_url}/api/v1/generate", json=body)
    dt_ms = (time.perf_counter() - t0) * 1000
    try:
        return r.status_code, r.json(), dt_ms
    except Exception:
        return r.status_code, {"_raw": r.text}, dt_ms


def post_batch(client: httpx.Client, base_url: str, payload: Optional[Dict[str, Any]] = None) -> tuple[int, Dict[str, Any], float]:
    body = payload if payload is not None else SAMPLE_BATCH_PAYLOAD
    t0 = time.perf_counter()
    r = client.post(f"{base_url}/api/v1/generate/batch", json=body)
    dt_ms = (time.perf_counter() - t0) * 1000
    try:
        return r.status_code, r.json(), dt_ms
    except Exception:
        return r.status_code, {"_raw": r.text}, dt_ms


def warmup(base_url: str, n: int = 3, timeout: float = DEFAULT_TIMEOUT) -> int:
    with httpx.Client(timeout=timeout) as c:
        ok = 0
        for _ in range(n):
            status, _, _ = post_generate(c, base_url)
            if status == 200:
                ok += 1
    return ok


def save_results(name: str, payload: Dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_to_serializable)
    return out


def _to_serializable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"not serializable: {type(obj).__name__}")
