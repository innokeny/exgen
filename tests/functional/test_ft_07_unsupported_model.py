from __future__ import annotations


def test_ft_07_unsupported_model(client):
    payload = {
        "user_id": "ft07-user",
        "message_content": "He go to school every day.",
        "grammar_error": "SubjectVerbAgreement",
        "explanation": "Verb form must agree with subject.",
        "llm_confidence": 0.9,
        "task_type": "grammar_choice",
        "model": "nonexistent-model-7b",
    }

    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 400, response.text

    body = response.json()
    detail = body.get("detail", "")
    assert "Unsupported" in detail or "unsupported" in detail.lower()
    assert "nonexistent-model-7b" in detail
