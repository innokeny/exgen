"""ФТ-13: Многократное последовательное обращение.

Ожидаемый результат: стабильность времени отклика, отсутствие утечек видеопамяти.
В тесте проверяется стабильность контракта и отсутствие накопления состояния
на уровне приложения; контроль видеопамяти и времени инференса под реальной
нагрузкой выполняется отдельным бенчмарком.
"""
from __future__ import annotations

import statistics
import time

ITERATIONS = 50


def test_ft_13_repeated_calls_remain_stable(client):
    base_payload = {
        "user_id": "ft13-user",
        "message_content": "It has a high-density population because for its small territory.",
        "grammar_error": "Preposition",
        "explanation": "Wrong preposition.",
        "llm_confidence": 0.9,
        "task_type": "grammar_choice",
    }

    latencies_ms: list[float] = []

    for i in range(ITERATIONS):
        t0 = time.perf_counter()
        response = client.post("/api/v1/generate", json=base_payload)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        assert response.status_code == 200, (
            f"iteration {i} failed: {response.status_code} {response.text}"
        )
        body = response.json()
        assert body["status"] == "ok"

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] in {"ok", "loading"}

    median_ms = statistics.median(latencies_ms)
    p95_ms = statistics.quantiles(latencies_ms, n=20)[18]
    assert p95_ms < median_ms * 10 + 1000, (
        f"95th percentile ({p95_ms:.1f} ms) is unreasonably far from the median "
        f"({median_ms:.1f} ms) — possible degradation"
    )
