"""ФТ-11: Активация резервного шаблонного пути при некорректном выходе модели.

Ожидаемый результат: ответ со статусом частичного успеха (для пакетной генерации)
или статусом «fallback» (для одиночной генерации) и указанием категории, для
которой задействован резерв.
"""
from __future__ import annotations

GARBAGE_OUTPUT = "this is not valid JSON, just plain text from a misbehaving model"


def test_ft_11_single_generate_fallback(make_client):
    client = make_client(GARBAGE_OUTPUT)
    payload = {
        "user_id": "ft11-user",
        "message_content": "He go to school every day.",
        "grammar_error": "SubjectVerbAgreement",
        "explanation": "Subject-verb agreement is required.",
        "llm_confidence": 0.9,
        "task_type": "grammar_choice",
    }

    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] == "fallback"
    assert body["fallback_reason"], "fallback responses must include a reason"
    assert body["exercise"]["task"]["content_en"]["items"], (
        "fallback exercise must still contain items"
    )


def test_ft_11_batch_partial_status(make_client):
    client = make_client(GARBAGE_OUTPUT)
    payload = {
        "user_id": "ft11-user",
        "language_level": "B1",
        "max_questions": 4,
        "error_profile": [
            {
                "error_type": "Preposition",
                "occurrences": 4,
                "examples": [{
                    "original": "It has a high-density population because for its small territory.",
                    "corrected": "It has a high-density population because of its small territory.",
                    "explanation": "Use 'because of' before a noun phrase.",
                }],
            },
        ],
    }

    response = client.post("/api/v1/generate/batch", json=payload)
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] == "partial"
    assert "Preposition" in body["fallback_categories"], (
        "category for which fallback was used must be listed"
    )
    assert body["questions"], "questions must still be produced via the fallback path"
