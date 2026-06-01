"""Unit tests for postprocessor + template engine. No GPU/model required."""

from app.core.postprocessor import parse_exercise
from app.core.template_engine import (
    build_template_exercise,
    fill_in_blanks,
    sentence_reconstruction,
)


VALID_RAW = """
{
  "target_error_category": "Preposition",
  "corrected_sentence": "It has a high-density population because of its small territory.",
  "task": {
    "type": "grammar_choice",
    "instruction_en": "Pick the correct preposition.",
    "content_en": {
      "context_text": null,
      "items": [
        {"question_en": "X ____ Y", "options_en": ["of","for"], "student_answer_en": "of"}
      ],
      "word_bank": null
    }
  }
}
"""


def test_parse_valid_exercise():
    res = parse_exercise(VALID_RAW)
    assert res.ok
    assert res.exercise.target_error_category == "Preposition"
    assert res.exercise.task.type == "grammar_choice"


def test_parse_strips_code_fence():
    raw = f"```json\n{VALID_RAW}\n```\nsome trailing junk"
    res = parse_exercise(raw)
    assert res.ok


def test_parse_handles_leading_text():
    raw = f"Sure, here is the JSON:\n{VALID_RAW}\nLet me know if you need changes."
    res = parse_exercise(raw)
    assert res.ok


def test_parse_returns_error_for_garbage():
    res = parse_exercise("just some text without braces")
    assert not res.ok
    assert res.error == "no_json_found"


def test_parse_returns_error_for_invalid_schema():
    raw = '{"foo": "bar"}'
    res = parse_exercise(raw)
    assert not res.ok
    assert res.error == "schema_validation_error"


def test_parse_rejects_empty_items():
    raw = """
    {
      "target_error_category": "X",
      "corrected_sentence": "Y",
      "task": {
        "type": "grammar_choice",
        "instruction_en": "Z",
        "content_en": {"context_text": null, "items": [], "word_bank": null}
      }
    }
    """
    res = parse_exercise(raw)
    assert not res.ok
    assert res.error == "empty_items"


def test_fill_in_blanks_creates_gap():
    ex = fill_in_blanks(
        source_sentence="It has a high-density population because for its small territory.",
        corrected_sentence="It has a high-density population because of its small territory.",
        error_type="Preposition",
    )
    item = ex["task"]["content_en"]["items"][0]
    assert "____" in item["question_en"]
    assert item["student_answer_en"] in item["options_en"]


def test_reconstruction_produces_word_bank():
    ex = sentence_reconstruction(
        source_sentence="A b c.",
        corrected_sentence="The cat sat on the mat.",
        error_type="WordOrder",
    )
    assert ex["task"]["type"] == "transformation"
    assert ex["task"]["content_en"]["word_bank"]


def test_template_dispatch_unknown_method_raises():
    import pytest

    with pytest.raises(ValueError):
        build_template_exercise(
            method="nope",
            source_sentence="x",
            corrected_sentence="x",
            error_type="X",
        )
