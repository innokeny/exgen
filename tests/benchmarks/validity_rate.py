"""Bench: structural validity rate by task type.

Fills in таблицу 18 (3.4) — доля структурно валидных JSON-ответов.
For each of the four supported task types, sends N generations and counts
those that pass schema validation without falling back to the template path.

Usage:
    python -m scripts.benchmarks.validity_rate --n 100
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import httpx

from scripts.benchmarks._common import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    post_generate,
    save_results,
    warmup,
)

TASK_TYPES: List[str] = ["grammar_choice", "transformation", "vocabulary_fill", "matching"]

SAMPLE_INPUTS: List[Dict[str, str]] = [
    {
        "message_content": "It has a high-density population because for its small territory.",
        "grammar_error": "Preposition",
        "explanation": "Wrong preposition: 'for' should be 'of'.",
    },
    {
        "message_content": "She have finished her homework.",
        "grammar_error": "Tense",
        "explanation": "Subject-verb agreement in present perfect.",
    },
    {
        "message_content": "I saw elephant at the zoo.",
        "grammar_error": "Article",
        "explanation": "Indefinite article required before a singular countable noun.",
    },
    {
        "message_content": "The students is studying for the exam.",
        "grammar_error": "SubjectVerbAgreement",
        "explanation": "Plural subject takes a plural verb.",
    },
    {
        "message_content": "He bought two book at the store.",
        "grammar_error": "Pluralization",
        "explanation": "Plural noun required after a numeral greater than one.",
    },
]


def measure_one_task_type(base_url: str, task_type: str, n: int, timeout: float) -> Dict[str, Any]:
    valid = 0
    fallback = 0
    errors = 0
    with httpx.Client(timeout=timeout) as client:
        for i in range(n):
            sample = SAMPLE_INPUTS[i % len(SAMPLE_INPUTS)]
            payload = {
                "user_id": f"valid-{task_type}-{i}",
                "message_content": sample["message_content"],
                "grammar_error": sample["grammar_error"],
                "explanation": sample["explanation"],
                "llm_confidence": 0.9,
                "task_type": task_type,
            }
            status, body, _ = post_generate(client, base_url, payload)
            if status != 200:
                errors += 1
                continue
            if body.get("status") == "ok":
                valid += 1
            elif body.get("status") == "fallback":
                fallback += 1
            else:
                errors += 1

    total = max(1, n - errors)
    return {
        "task_type": task_type,
        "runs": n,
        "valid": valid,
        "fallback": fallback,
        "errors": errors,
        "rate_pct": round(100.0 * valid / total, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_BASE_URL)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    if args.warmup:
        warmup(args.url, n=args.warmup, timeout=args.timeout)

    results: List[Dict[str, Any]] = []
    for tt in TASK_TYPES:
        row = measure_one_task_type(args.url, tt, args.n, args.timeout)
        print(json.dumps(row, ensure_ascii=False))
        results.append(row)

    total_runs = sum(r["runs"] for r in results)
    total_valid = sum(r["valid"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    aggregate = {
        "task_type": "aggregate",
        "runs": total_runs,
        "valid": total_valid,
        "errors": total_errors,
        "rate_pct": round(100.0 * total_valid / max(1, total_runs - total_errors), 2),
    }
    print(json.dumps(aggregate, ensure_ascii=False))

    save_results("validity_rate", {"per_task": results, "aggregate": aggregate})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
