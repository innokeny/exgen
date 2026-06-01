from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

import structlog

from app.api.schemas import (
    ErrorProfileEntry,
    Exercise,
    TestQuestion,
)
from app.core.postprocessor import exercise_to_questions, parse_exercise
from app.core.template_engine import GECPair, fill_in_blanks, fill_in_blanks_exercise

log = structlog.get_logger(__name__)


class _GeneratorProto(Protocol):
    def generate_raw(
        self,
        *,
        user_id: str,
        message_content: str,
        grammar_error: str,
        explanation: str,
        llm_confidence: float,
        task_type: str,
        model_key: Optional[str] = None,
    ) -> tuple[str, str, int]: ...


@dataclass
class BatchResult:
    questions: List[TestQuestion]
    exercises_full: List[dict]
    fallback_categories: List[str]
    model_used: str


def _allocate(entries: List[ErrorProfileEntry], max_questions: int) -> List[int]:
    n = len(entries)
    if n == 0:
        return []
    if max_questions <= n:
        order = sorted(range(n), key=lambda i: entries[i].occurrences, reverse=True)
        alloc = [0] * n
        for i in order[:max_questions]:
            alloc[i] = 1
        return alloc

    total = sum(max(1, e.occurrences) for e in entries)
    raw = [max(1, e.occurrences) / total * max_questions for e in entries]
    floor = [max(1, int(r)) for r in raw]

    diff = max_questions - sum(floor)
    if diff > 0:
        order = sorted(range(n), key=lambda i: raw[i] - int(raw[i]), reverse=True)
        for i in order:
            if diff == 0:
                break
            floor[i] += 1
            diff -= 1
    elif diff < 0:
        order = sorted(range(n), key=lambda i: floor[i], reverse=True)
        for i in order:
            if diff == 0:
                break
            if floor[i] > 1:
                floor[i] -= 1
                diff += 1
    return floor


def _fallback_questions(
    entry: ErrorProfileEntry,
    count: int,
    *,
    counter_start: int,
) -> tuple[List[TestQuestion], dict]:
    if not entry.examples:
        return [], {}

    pairs = [GECPair(source=e.original, corrected=e.corrected) for e in entry.examples if e.original and e.corrected]
    explanation = entry.examples[0].explanation if entry.examples else ""

    ex = fill_in_blanks_exercise(
        error_type=entry.error_type,
        profile=pairs,
        n_items=max(1, count),
    )
    if ex is None:
        best = entry.examples[0]
        ex = fill_in_blanks(
            source_sentence=best.original,
            corrected_sentence=best.corrected,
            error_type=entry.error_type,
        )

    questions = exercise_to_questions(
        ex,
        entry.error_type,
        explanation,
        start_idx=counter_start,
        max_items=count,
    )
    return questions, ex


def generate_batch(
    *,
    generator: _GeneratorProto,
    user_id: str,
    error_profile: List[ErrorProfileEntry],
    max_questions: int,
    model_key: Optional[str],
    default_model: str,
) -> BatchResult:
    profile = [p for p in error_profile if p.examples]
    profile.sort(key=lambda p: p.occurrences, reverse=True)
    if not profile:
        return BatchResult(
            questions=[],
            exercises_full=[],
            fallback_categories=[],
            model_used=model_key or default_model,
        )

    allocations = _allocate(profile, max_questions)
    questions: List[TestQuestion] = []
    exercises_full: List[dict] = []
    fallback_categories: List[str] = []
    model_used = model_key or default_model

    for entry, alloc in zip(profile, allocations):
        if alloc <= 0:
            continue
        best = entry.examples[0]
        produced: List[TestQuestion] = []

        try:
            raw, used, _ms = generator.generate_raw(
                user_id=user_id,
                message_content=best.original,
                grammar_error=entry.error_type,
                explanation=best.explanation,
                llm_confidence=1.0,
                task_type="grammar_choice",
                model_key=model_key,
            )
            model_used = used
            parsed = parse_exercise(raw)
            if parsed.ok:
                ex_dict = parsed.exercise.model_dump()
                exercises_full.append(ex_dict)
                produced = exercise_to_questions(
                    ex_dict,
                    entry.error_type,
                    best.explanation,
                    start_idx=len(questions),
                    max_items=alloc,
                )
            else:
                log.warning(
                    "batch.parse_failed",
                    error_type=entry.error_type,
                    reason=parsed.error,
                )
        except FileNotFoundError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("batch.generation_failed", error_type=entry.error_type, error=str(e))

        shortfall = alloc - len(produced)
        if shortfall > 0:
            fb_qs, fb_ex = _fallback_questions(
                entry, shortfall, counter_start=len(questions) + len(produced)
            )
            if fb_qs:
                produced.extend(fb_qs)
                if fb_ex:
                    exercises_full.append(fb_ex)
                if entry.error_type not in fallback_categories:
                    fallback_categories.append(entry.error_type)

        questions.extend(produced)

    return BatchResult(
        questions=questions[:max_questions],
        exercises_full=exercises_full,
        fallback_categories=fallback_categories,
        model_used=model_used,
    )