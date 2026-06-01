from __future__ import annotations

import pytest

ERROR_CATEGORIES = [
    ("Preposition", "It has a high-density population because for its small territory."),
    ("Tense", "She have finished her homework."),
    ("Article", "I saw elephant at the zoo."),
    ("SubjectVerbAgreement", "The students is studying for the exam."),
    ("WordOrder", "I yesterday went to the store."),
    ("Pluralization", "He bought two book at the store."),
    ("WordChoice", "She made a significant donation to the project."),
    ("Pronoun", "Me and him went to the park."),
]


@pytest.mark.parametrize("category,sentence", ERROR_CATEGORIES)
def test_ft_06_error_coverage(client, category: str, sentence: str):
    payload = {
        "user_id": f"ft06-{category}",
        "message_content": sentence,
        "grammar_error": category,
        "explanation": f"Test coverage for the '{category}' error category.",
        "llm_confidence": 0.9,
        "task_type": "grammar_choice",
    }

    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 200, (
        f"category '{category}' produced status {response.status_code}: {response.text}"
    )

    body = response.json()
    assert body["status"] == "ok"
    assert body["exercise"]["task"]["content_en"]["items"]
