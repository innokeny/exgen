VALID_PAYLOAD = {
    "user_id": "u_123",
    "message_content": "It has a high - density population because for its small territory.",
    "grammar_error": "Preposition",
    "explanation": "The student wrote 'because for'. The correct form is 'because of'.",
    "llm_confidence": 0.95,
    "task_type": "grammar_choice",
}


def test_generate_happy_path(client):
    r = client.post("/api/v1/generate", json=VALID_PAYLOAD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_used"] == "qwen2.5-3b"
    assert body["exercise"]["task"]["type"] == "grammar_choice"
    assert len(body["exercise"]["task"]["content_en"]["items"]) >= 1


def test_generate_rejects_unsupported_model(client):
    payload = {**VALID_PAYLOAD, "model": "gpt-99"}
    r = client.post("/api/v1/generate", json=payload)
    assert r.status_code == 400


def test_generate_falls_back_when_llm_emits_garbage(make_client):
    client = make_client(raw_output="this is not json at all, sorry")
    r = client.post("/api/v1/generate", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "fallback"
    assert body["fallback_reason"]
    assert body["exercise"]["task"]["type"] == "grammar_choice"


def test_generate_handles_fenced_json(make_client):
    raw = """Here is the exercise:
```json
{
  "target_error_category": "Preposition",
  "corrected_sentence": "It has X because of Y.",
  "task": {
    "type": "grammar_choice",
    "instruction_en": "Pick one.",
    "content_en": {
      "context_text": null,
      "items": [
        {"question_en": "X ____ Y", "options_en": ["of","to"], "student_answer_en": "of"}
      ],
      "word_bank": null
    }
  }
}
```
"""
    client = make_client(raw_output=raw)
    r = client.post("/api/v1/generate", json=VALID_PAYLOAD)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_template_endpoint_fill_in_blanks(client):
    payload = {
        "source_sentence": "It has a high-density population because for its small territory.",
        "corrected_sentence": "It has a high-density population because of its small territory.",
        "error_type": "Preposition",
        "method": "fill_in_blanks",
    }
    r = client.post("/api/v1/generate/template", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["method"] == "fill_in_blanks"
    items = body["exercise"]["task"]["content_en"]["items"]
    assert len(items) == 1
    assert "____" in items[0]["question_en"]


def test_template_endpoint_reconstruction(client):
    payload = {
        "source_sentence": "It has a high-density population because for its small territory.",
        "corrected_sentence": "It has a high-density population because of its small territory.",
        "error_type": "Preposition",
        "method": "reconstruction",
    }
    r = client.post("/api/v1/generate/template", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["exercise"]["task"]["type"] == "transformation"
    assert body["exercise"]["task"]["content_en"]["word_bank"]


def test_validation_error_on_missing_fields(client):
    r = client.post("/api/v1/generate", json={"user_id": "x"})
    assert r.status_code == 422
