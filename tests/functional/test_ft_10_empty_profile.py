from __future__ import annotations


def test_ft_10_empty_profile_rejected(client):
    payload = {
        "user_id": "ft10-user",
        "language_level": "B1",
        "max_questions": 5,
        "error_profile": [],
    }

    response = client.post("/api/v1/generate/batch", json=payload)
    assert response.status_code == 400, response.text

    body = response.json()
    detail = body.get("detail", "")
    assert "error_profile" in detail or "non-empty" in detail.lower()


def test_ft_10_profile_with_no_examples_rejected(client):
    payload = {
        "user_id": "ft10-user",
        "language_level": "B1",
        "max_questions": 5,
        "error_profile": [
            {"error_type": "Preposition", "occurrences": 3, "examples": []},
            {"error_type": "Tense", "occurrences": 2, "examples": []},
        ],
    }

    response = client.post("/api/v1/generate/batch", json=payload)
    assert response.status_code == 400, response.text

    body = response.json()
    detail = body.get("detail", "")
    assert "no usable examples" in detail.lower() or "no questions" in detail.lower()
