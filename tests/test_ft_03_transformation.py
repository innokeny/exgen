"""ФТ-03: Генерация задания типа «трансформация предложения».

Ожидаемый результат: упражнение с инструкцией на преобразование и эталонным ответом.
"""
from __future__ import annotations

TRANSFORMATION_JSON = """
{
  "target_error_category": "ActiveToPassive",
  "corrected_sentence": "The book was read by the student.",
  "task": {
    "type": "transformation",
    "instruction_en": "Rewrite the sentence in the passive voice without changing its meaning.",
    "content_en": {
      "context_text": "Source sentence: The student read the book.",
      "items": [
        {
          "question_en": "The student read the book.",
          "options_en": null,
          "student_answer_en": "The book was read by the student."
        }
      ],
      "word_bank": null
    }
  }
}
"""


def test_ft_03_transformation(make_client):
    client = make_client(TRANSFORMATION_JSON)
    payload = {
        "user_id": "ft03-user",
        "message_content": "The student read the book.",
        "grammar_error": "Voice",
        "explanation": "Convert active voice to passive voice.",
        "llm_confidence": 0.88,
        "task_type": "transformation",
    }

    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] == "ok"
    task = body["exercise"]["task"]
    assert task["type"] == "transformation"
    assert "passive" in task["instruction_en"].lower() or task["instruction_en"]

    item = task["content_en"]["items"][0]
    assert item["question_en"], "source sentence must be provided"
    assert item["student_answer_en"], "reference transformation must be provided"
    assert item["question_en"] != item["student_answer_en"], (
        "transformation must change the source"
    )
