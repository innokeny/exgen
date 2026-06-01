from __future__ import annotations

GRAMMAR_CHOICE_JSON = """
{
  "target_error_category": "Tense",
  "corrected_sentence": "She has finished her homework.",
  "task": {
    "type": "grammar_choice",
    "instruction_en": "Choose the correct verb form to complete the sentence.",
    "content_en": {
      "context_text": null,
      "items": [
        {
          "question_en": "She ____ her homework.",
          "options_en": ["has finished", "have finished", "had finished", "is finishing"],
          "student_answer_en": "has finished"
        }
      ],
      "word_bank": null
    }
  }
}
"""


def test_ft_02_grammar_choice(make_client):
    client = make_client(GRAMMAR_CHOICE_JSON)
    payload = {
        "user_id": "ft02-user",
        "message_content": "She have finished her homework.",
        "grammar_error": "Tense",
        "explanation": "Use present perfect for completed actions with relevance to now.",
        "llm_confidence": 0.92,
        "task_type": "grammar_choice",
    }

    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] == "ok"
    task = body["exercise"]["task"]
    assert task["type"] == "grammar_choice"

    item = task["content_en"]["items"][0]
    assert item["options_en"] is not None
    assert len(item["options_en"]) >= 2, "must offer at least two options"
    assert item["student_answer_en"] in item["options_en"], (
        "correct answer must be among the offered options"
    )
    assert len(set(item["options_en"])) == len(item["options_en"]), (
        "options must be unique"
    )
