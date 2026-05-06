"""ФТ-12: Запрос на отсутствующий адаптер модели.

Ожидаемый результат: ответ с кодом 503 без падения сервиса.
"""
from __future__ import annotations


def _raise_missing_adapter(**_kwargs):
    raise FileNotFoundError(
        "Adapter directory not found for 'qwen2.5-3b': /app/adapters/qwen2.5-3b"
    )


def test_ft_12_missing_adapter_returns_503(make_client):
    client = make_client(_raise_missing_adapter)
    payload = {
        "user_id": "ft12-user",
        "message_content": "He go to school every day.",
        "grammar_error": "SubjectVerbAgreement",
        "explanation": "Subject-verb agreement is required.",
        "llm_confidence": 0.9,
        "task_type": "grammar_choice",
    }

    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 503, response.text

    body = response.json()
    detail = body.get("detail", "")
    assert "Adapter" in detail or "adapter" in detail.lower()

    health = client.get("/health")
    assert health.status_code == 200, "service must remain reachable after the failure"


def test_ft_12_missing_adapter_in_batch(make_client):
    client = make_client(_raise_missing_adapter)
    payload = {
        "user_id": "ft12-user",
        "language_level": "B1",
        "max_questions": 3,
        "error_profile": [
            {
                "error_type": "Preposition",
                "occurrences": 3,
                "examples": [{
                    "original": "It has a high-density population because for its small territory.",
                    "corrected": "It has a high-density population because of its small territory.",
                    "explanation": "Use 'because of' before a noun phrase.",
                }],
            },
        ],
    }

    response = client.post("/api/v1/generate/batch", json=payload)
    assert response.status_code == 503, response.text
