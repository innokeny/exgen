from __future__ import annotations

import json

from app.api.schemas import ErrorExample, ErrorProfileEntry
from app.core.batch import _allocate, generate_batch
from app.core.postprocessor import exercise_to_questions


def _multi_item_exercise(error_type: str, n_items: int = 6) -> str:
    items = [
        {
            "question_en": f"({error_type}) Choose the correct word #{i+1}: she ____ home.",
            "options_en": ["of", "for", "to", "with"],
            "student_answer_en": "to",
        }
        for i in range(n_items)
    ]
    payload = {
        "target_error_category": error_type,
        "corrected_sentence": "She went to home.",
        "task": {
            "type": "grammar_choice",
            "instruction_en": f"Pick the correct {error_type.lower()}.",
            "content_en": {
                "context_text": None,
                "items": items,
                "word_bank": None,
            },
        },
    }
    return json.dumps(payload)


SAMPLE_PROFILE = [
    ErrorProfileEntry(
        error_type="Preposition",
        occurrences=12,
        examples=[
            ErrorExample(
                original="I go to school by walk",
                corrected="I go to school on foot",
                explanation="Use 'on foot' instead of 'by walk'",
            )
        ],
    ),
    ErrorProfileEntry(
        error_type="Article",
        occurrences=8,
        examples=[
            ErrorExample(
                original="I saw cat in garden",
                corrected="I saw a cat in the garden",
                explanation="Use articles 'a' and 'the' before nouns",
            )
        ],
    ),
]


def test_allocate_proportional_to_occurrences():
    alloc = _allocate(SAMPLE_PROFILE, max_questions=10)
    assert sum(alloc) == 10
    assert alloc[0] >= alloc[1]
    assert all(a >= 1 for a in alloc)


def test_allocate_handles_tiny_budget():
    alloc = _allocate(SAMPLE_PROFILE, max_questions=1)
    assert sum(alloc) == 1
    assert alloc[0] == 1 and alloc[1] == 0


def test_allocate_empty_profile():
    assert _allocate([], max_questions=10) == []


def test_exercise_to_questions_skips_items_without_answer():
    exercise = {
        "task": {
            "type": "grammar_choice",
            "instruction_en": "Pick one.",
            "content_en": {
                "context_text": None,
                "word_bank": None,
                "items": [
                    {"question_en": "Q1", "options_en": ["a", "b"], "student_answer_en": "a"},
                    {"question_en": "Q2", "options_en": [], "student_answer_en": ""},
                ],
            },
        }
    }
    qs = exercise_to_questions(exercise, "Preposition", "expl")
    assert len(qs) == 1
    assert qs[0].correct_answer == "a"
    assert len(qs[0].options) == 4


def test_exercise_to_questions_pads_options_from_word_bank():
    exercise = {
        "task": {
            "type": "vocabulary_fill",
            "instruction_en": "Fill the gap.",
            "content_en": {
                "context_text": "The cat sat on the mat.",
                "word_bank": ["the", "a", "an", "this", "that"],
                "items": [
                    {"question_en": "____ cat sat.", "options_en": [], "student_answer_en": "the"},
                ],
            },
        }
    }
    qs = exercise_to_questions(exercise, "Article", "expl")
    assert len(qs) == 1
    assert "the" in qs[0].options
    assert len(qs[0].options) == 4


def test_exercise_to_questions_unique_ids():
    exercise = json.loads(_multi_item_exercise("Preposition", n_items=5))
    qs = exercise_to_questions(exercise, "Preposition", "expl", start_idx=0)
    ids = [q.id for q in qs]
    assert len(set(ids)) == len(ids)


class _FakeGen:
    def __init__(self, by_error_type: dict[str, str]):
        self.by_error_type = by_error_type
        self.calls = 0

    def generate_raw(self, **kwargs):
        self.calls += 1
        et = kwargs["grammar_error"]
        return self.by_error_type.get(et, "garbage"), "qwen2.5-3b", 1


def test_generate_batch_distributes_questions():
    gen = _FakeGen({
        "Preposition": _multi_item_exercise("Preposition", 10),
        "Article": _multi_item_exercise("Article", 10),
    })
    result = generate_batch(
        generator=gen,
        user_id="u_x",
        error_profile=SAMPLE_PROFILE,
        max_questions=15,
        model_key=None,
        default_model="qwen2.5-3b",
    )
    assert len(result.questions) == 15
    by_type = {}
    for q in result.questions:
        by_type[q.error_type] = by_type.get(q.error_type, 0) + 1
    assert by_type["Preposition"] >= by_type["Article"]
    assert result.fallback_categories == []
    assert result.model_used == "qwen2.5-3b"


def test_generate_batch_falls_back_when_llm_returns_garbage():
    gen = _FakeGen({})
    result = generate_batch(
        generator=gen,
        user_id="u_x",
        error_profile=SAMPLE_PROFILE,
        max_questions=6,
        model_key=None,
        default_model="qwen2.5-3b",
    )
    assert len(result.questions) == 6
    assert set(result.fallback_categories) == {"Preposition", "Article"}


BATCH_PAYLOAD = {
    "user_id": "u_123",
    "language_level": "B1",
    "max_questions": 10,
    "error_profile": [
        {
            "error_type": "Preposition",
            "occurrences": 12,
            "examples": [
                {
                    "original": "I go to school by walk",
                    "corrected": "I go to school on foot",
                    "explanation": "Use 'on foot' instead of 'by walk'",
                }
            ],
        },
        {
            "error_type": "Article",
            "occurrences": 8,
            "examples": [
                {
                    "original": "I saw cat in garden",
                    "corrected": "I saw a cat in the garden",
                    "explanation": "Use articles before nouns",
                }
            ],
        },
    ],
}


def test_batch_endpoint_happy_path(make_client):
    def _by_call(**kwargs):
        return _multi_item_exercise(kwargs["grammar_error"], n_items=8)

    client = make_client(raw_output=_by_call)
    r = client.post("/api/v1/generate/batch", json=BATCH_PAYLOAD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_used"] == "qwen2.5-3b"
    assert len(body["questions"]) == 10
    for q in body["questions"]:
        assert len(q["options"]) == 4
        assert q["correct_answer"] in q["options"]
        assert q["error_type"] in {"Preposition", "Article"}


def test_batch_endpoint_partial_status_on_garbage(make_client):
    client = make_client(raw_output="not json")
    r = client.post("/api/v1/generate/batch", json=BATCH_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "partial"
    assert body["fallback_categories"]
    assert len(body["questions"]) > 0


def test_batch_endpoint_rejects_empty_profile(client):
    r = client.post(
        "/api/v1/generate/batch",
        json={**BATCH_PAYLOAD, "error_profile": []},
    )
    assert r.status_code == 400


def test_batch_endpoint_rejects_unsupported_model(client):
    r = client.post(
        "/api/v1/generate/batch",
        json={**BATCH_PAYLOAD, "model": "gpt-99"},
    )
    assert r.status_code == 400


def test_models_endpoint_lists_qwen(client):
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "exercise-generator"
    assert "qwen2.5-3b" in body["models"]
    qwen = body["models"]["qwen2.5-3b"]
    assert qwen["status"] in {"loaded", "available"}
    assert qwen["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert qwen["metrics"]["tr_composite"] == 0.938
    assert "grammar_choice" in body["supported_task_types"]
    assert "fill_in_blanks" in body["fallback_methods"]
