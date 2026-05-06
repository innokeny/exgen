"""ФТ-08: Запрос с отсутствующими обязательными полями.

Ожидаемый результат: ответ с кодом 422 и указанием некорректных полей.
"""
from __future__ import annotations

import pytest

REQUIRED_FIELDS = [
    "user_id",
    "message_content",
    "grammar_error",
    "explanation",
    "llm_confidence",
]

FULL_PAYLOAD = {
    "user_id": "ft08-user",
    "message_content": "He go to school every day.",
    "grammar_error": "SubjectVerbAgreement",
    "explanation": "Subject-verb agreement.",
    "llm_confidence": 0.9,
    "task_type": "grammar_choice",
}


@pytest.mark.parametrize("missing_field", REQUIRED_FIELDS)
def test_ft_08_missing_required_field(client, missing_field: str):
    payload = {k: v for k, v in FULL_PAYLOAD.items() if k != missing_field}

    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 422, (
        f"missing '{missing_field}' produced {response.status_code}: {response.text}"
    )

    body = response.json()
    detail = body.get("detail", [])
    assert detail, "validation error response must include details"
    assert any(missing_field in str(item) for item in detail), (
        f"validation error must mention the missing field '{missing_field}'"
    )


def test_ft_08_invalid_confidence_range(client):
    payload = dict(FULL_PAYLOAD, llm_confidence=1.5)

    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 422
