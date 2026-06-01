"""ФТ-01: Генерация одного упражнения по корректному запросу.

Ожидаемый результат: успешный ответ, разобранное упражнение,
время отклика в пределах ожидаемого.
"""
from __future__ import annotations

import time


def test_ft_01_single_generation(client):
    payload = {
        "user_id": "ft01-user",
        "message_content": "It has a high-density population because for its small territory.",
        "grammar_error": "Preposition",
        "explanation": "Wrong preposition: 'for' should be 'of'.",
        "llm_confidence": 0.9,
        "task_type": "grammar_choice",
    }

    t0 = time.perf_counter()
    response = client.post("/api/v1/generate", json=payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_used"] == "qwen2.5-3b"
    assert body["generation_time_ms"] >= 0
    assert "exercise" in body

    exercise = body["exercise"]
    assert exercise["target_error_category"]
    assert exercise["corrected_sentence"]
    assert exercise["task"]["type"]
    assert exercise["task"]["instruction_en"]
    assert exercise["task"]["content_en"]["items"], "items must not be empty"

    assert elapsed_ms < 5000, f"response took {elapsed_ms:.0f} ms"
