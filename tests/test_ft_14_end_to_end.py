"""ФТ-14: Сценарий сквозного встраивания в платформу.

Цикл «ответ обучающегося → профиль ошибок → набор вопросов» завершается успешно.
Тест воспроизводит полный путь: исходный текст обучающегося приводит к
формированию профиля ошибок, на основании которого микросервис формирует
плоский набор тестовых вопросов в формате, ожидаемом интерфейсом платформы.
"""
from __future__ import annotations


STUDENT_RESPONSE = (
    "Yesterday I go to the store and buy two book. "
    "Then me and him played in park because for the rain stopped."
)


def _build_error_profile_from_response(_text: str) -> list[dict]:
    """Имитация компонента анализа ответов на стороне платформы.

    В реальной платформе этот шаг выполняется отдельным сервисом анализа
    ошибок, который возвращает структурированный профиль допущенных
    обучающимся грамматических ошибок.
    """
    return [
        {
            "error_type": "Tense",
            "occurrences": 2,
            "examples": [{
                "original": "Yesterday I go to the store and buy two book.",
                "corrected": "Yesterday I went to the store and bought two books.",
                "explanation": "Past actions require the simple past tense.",
            }],
        },
        {
            "error_type": "Pluralization",
            "occurrences": 1,
            "examples": [{
                "original": "buy two book",
                "corrected": "bought two books",
                "explanation": "Countable nouns take the plural form after numerals greater than one.",
            }],
        },
        {
            "error_type": "Preposition",
            "occurrences": 1,
            "examples": [{
                "original": "because for the rain stopped",
                "corrected": "because the rain stopped",
                "explanation": "'because' is followed directly by a clause without a preposition.",
            }],
        },
    ]


def test_ft_14_end_to_end_embedding(client):
    profile = _build_error_profile_from_response(STUDENT_RESPONSE)

    batch_payload = {
        "user_id": "ft14-user",
        "language_level": "B1",
        "max_questions": 6,
        "error_profile": profile,
    }

    response = client.post("/api/v1/generate/batch", json=batch_payload)
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] in {"ok", "partial"}

    questions = body["questions"]
    assert questions, "end-to-end scenario must produce questions"
    assert len(questions) <= batch_payload["max_questions"]

    for q in questions:
        assert q["id"]
        assert q["error_type"] in {"Tense", "Pluralization", "Preposition"}
        assert q["text"], "question text must be present"
        assert q["options"], "answer options must be present"
        assert q["correct_answer"] in q["options"], (
            "correct answer must be among the options"
        )
        assert q["explanation"], "explanation must be present"

    covered_categories = {q["error_type"] for q in questions}
    assert covered_categories, "at least one error category must be covered"
