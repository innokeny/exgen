from __future__ import annotations


def test_ft_09_batch_generation_distributes_questions(client):
    payload = {
        "user_id": "ft09-user",
        "language_level": "B1",
        "max_questions": 9,
        "error_profile": [
            {
                "error_type": "Preposition",
                "occurrences": 6,
                "examples": [{
                    "original": "It has a high-density population because for its small territory.",
                    "corrected": "It has a high-density population because of its small territory.",
                    "explanation": "Use 'because of' before a noun phrase.",
                }],
            },
            {
                "error_type": "Tense",
                "occurrences": 3,
                "examples": [{
                    "original": "She have finished her homework.",
                    "corrected": "She has finished her homework.",
                    "explanation": "Subject-verb agreement in present perfect.",
                }],
            },
            {
                "error_type": "Article",
                "occurrences": 1,
                "examples": [{
                    "original": "I saw elephant at the zoo.",
                    "corrected": "I saw an elephant at the zoo.",
                    "explanation": "Use the indefinite article before a singular countable noun.",
                }],
            },
        ],
    }

    response = client.post("/api/v1/generate/batch", json=payload)
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] in {"ok", "partial"}
    assert body["model_used"] == "qwen2.5-3b"
    assert body["generation_time_ms"] >= 0

    questions = body["questions"]
    assert questions, "batch must produce at least one question"
    assert len(questions) <= payload["max_questions"]

    counts: dict[str, int] = {}
    for q in questions:
        counts[q["error_type"]] = counts.get(q["error_type"], 0) + 1

    assert counts.get("Preposition", 0) >= counts.get("Tense", 0), (
        "more frequent error must receive at least as many questions"
    )
    assert counts.get("Tense", 0) >= counts.get("Article", 0)

    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "question identifiers must be unique"
