from __future__ import annotations

VOCABULARY_FILL_JSON = """
{
  "target_error_category": "WordChoice",
  "corrected_sentence": "She made a significant contribution to the project.",
  "task": {
    "type": "vocabulary_fill",
    "instruction_en": "Fill in the gap with the most appropriate word from the word bank.",
    "content_en": {
      "context_text": null,
      "items": [
        {
          "question_en": "She made a significant ____ to the project.",
          "options_en": null,
          "student_answer_en": "contribution"
        }
      ],
      "word_bank": ["contribution", "donation", "addition", "submission"]
    }
  }
}
"""


def test_ft_04_vocabulary_fill(make_client):
    client = make_client(VOCABULARY_FILL_JSON)
    payload = {
        "user_id": "ft04-user",
        "message_content": "She made a significant donation to the project.",
        "grammar_error": "WordChoice",
        "explanation": "Use 'contribution' instead of 'donation' in this academic context.",
        "llm_confidence": 0.85,
        "task_type": "vocabulary_fill",
    }

    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] == "ok"
    task = body["exercise"]["task"]
    assert task["type"] == "vocabulary_fill"

    content = task["content_en"]
    assert content["word_bank"], "word_bank must be provided"
    assert len(content["word_bank"]) >= 2, "word_bank must contain alternatives"

    item = content["items"][0]
    assert "____" in item["question_en"] or "_" in item["question_en"], (
        "question must contain a gap marker"
    )
    assert item["student_answer_en"] in content["word_bank"], (
        "correct answer must be present in the word bank"
    )
