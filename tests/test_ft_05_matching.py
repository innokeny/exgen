"""ФТ-05: Генерация задания типа «сопоставление функциональных элементов».

Ожидаемый результат: упражнение со списком пар для сопоставления.
"""
from __future__ import annotations

MATCHING_JSON = """
{
  "target_error_category": "Preposition",
  "corrected_sentence": "He depends on his parents.",
  "task": {
    "type": "matching",
    "instruction_en": "Match each verb with the preposition it most commonly takes.",
    "content_en": {
      "context_text": null,
      "items": [
        {"question_en": "depend", "options_en": ["on", "of", "to", "with"], "student_answer_en": "on"},
        {"question_en": "belong", "options_en": ["on", "of", "to", "with"], "student_answer_en": "to"},
        {"question_en": "agree", "options_en": ["on", "of", "to", "with"], "student_answer_en": "with"},
        {"question_en": "consist", "options_en": ["on", "of", "to", "with"], "student_answer_en": "of"}
      ],
      "word_bank": ["on", "to", "with", "of"]
    }
  }
}
"""


def test_ft_05_matching(make_client):
    client = make_client(MATCHING_JSON)
    payload = {
        "user_id": "ft05-user",
        "message_content": "He depends of his parents.",
        "grammar_error": "Preposition",
        "explanation": "'depend' takes the preposition 'on'.",
        "llm_confidence": 0.91,
        "task_type": "matching",
    }

    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] == "ok"
    task = body["exercise"]["task"]
    assert task["type"] == "matching"

    items = task["content_en"]["items"]
    assert len(items) >= 2, "matching must contain at least two pairs"

    for item in items:
        assert item["question_en"], "left side of the pair must be provided"
        assert item["student_answer_en"], "right side of the pair must be provided"
