"""Parses raw LLM output into a validated Exercise.

The fine-tuned model is trained to emit JSON only, but in practice it sometimes
adds a code-fence wrapper or trailing commentary. This module:
  1. Strips fences / locates the first `{...}` block.
  2. Validates the parsed dict against `Exercise`.
  3. On failure, returns a structured error so the API can fall back.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog
from pydantic import ValidationError

from app.api.schemas import Exercise, TestQuestion

log = structlog.get_logger(__name__)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class ParseResult:
    exercise: Optional[Exercise]
    error: Optional[str]

    @property
    def ok(self) -> bool:
        return self.exercise is not None


def _extract_json_blob(raw: str) -> Optional[str]:
    """Pull the most likely JSON document out of a model response."""
    if not raw:
        return None

    # 1. Fenced ```json ... ``` block.
    m = _FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()

    # 2. First balanced { ... } block.
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        ch = raw[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return None


def parse_exercise(raw: str) -> ParseResult:
    """Try hard to recover a valid `Exercise` from `raw`. Never raises."""
    blob = _extract_json_blob(raw)
    if blob is None:
        return ParseResult(None, "no_json_found")

    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        log.warning("postprocess.json_decode_failed", error=str(e))
        return ParseResult(None, f"json_decode_error: {e.msg}")

    try:
        exercise = Exercise.model_validate(data)
    except ValidationError as e:
        log.warning("postprocess.validation_failed", errors=e.errors())
        return ParseResult(None, "schema_validation_error")

    if not exercise.task.content_en.items:
        return ParseResult(None, "empty_items")

    return ParseResult(exercise, None)


# ---------- exercise → flat MCQ conversion (SAYIT batch endpoint) ------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(s: str) -> str:
    s = _SLUG_RE.sub("_", s.lower()).strip("_")
    return s or "x"


def _ensure_four_options(correct: str, options: List[str], extra_pool: List[str]) -> List[str]:
    """Return exactly 4 options that include `correct`, padded from `extra_pool` then dashes."""
    seen = set()
    out: List[str] = []
    for o in [correct, *options]:
        if not o or o in seen:
            continue
        seen.add(o)
        out.append(o)
    for o in extra_pool:
        if len(out) >= 4:
            break
        if o and o not in seen:
            seen.add(o)
            out.append(o)
    while len(out) < 4:
        out.append("—")
    return out[:4]


def exercise_to_questions(
    exercise: Dict[str, Any],
    error_type: str,
    explanation_fallback: str = "",
    *,
    start_idx: int = 0,
    id_prefix: str = "q",
    max_items: Optional[int] = None,
) -> List[TestQuestion]:
    """Flatten a structured exercise into MCQ-style questions for the SAYIT UI.

    Each `item` in the exercise becomes one question. Options are coerced to
    exactly 4 entries with the correct answer included. Items that can't be
    turned into a valid MCQ (no question text, no answer) are skipped.
    """
    task = exercise.get("task", {}) or {}
    instruction = task.get("instruction_en") or ""
    content = task.get("content_en", {}) or {}
    items = content.get("items") or []
    word_bank: List[str] = content.get("word_bank") or []
    context_text = content.get("context_text")

    slug = _slug(error_type)
    questions: List[TestQuestion] = []

    for i, item in enumerate(items):
        if max_items is not None and len(questions) >= max_items:
            break

        question_en = (item.get("question_en") or "").strip()
        correct = (item.get("student_answer_en") or "").strip()
        if not question_en and not context_text:
            continue
        if not correct:
            continue

        text = question_en
        if context_text and (not text or "____" in text or len(text) < 20):
            # Vocab-fill items often carry only the local cloze; prepend context.
            text = f"{context_text.strip()}\n\n{text}".strip()

        raw_options = list(item.get("options_en") or [])
        options = _ensure_four_options(correct, raw_options, word_bank)
        random.shuffle(options)

        explanation = instruction or explanation_fallback or f"Correct answer: {correct}"

        questions.append(
            TestQuestion(
                id=f"{id_prefix}_{slug}_{(start_idx + len(questions)):03d}",
                error_type=error_type,
                text=text,
                options=options,
                correct_answer=correct,
                explanation=explanation.strip(),
            )
        )

    return questions
