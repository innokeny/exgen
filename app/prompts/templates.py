"""Prompt templates for the exercise-generation LLM.

Reproduces the SYSTEM/USER prompts used during fine-tuning so the LoRA
adapter receives input in the format it was trained on.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are an expert English-language exercise designer for an adaptive learning platform.

Your job:
Given a learner's writing sample and the grammar error found in it, generate ONE exercise that:
1. Directly targets the specific error category.
2. Uses the corrected sentence as seed material.
3. Follows the exact JSON schema below.
4. Contains 8-14 items per exercise (depending on task type).

Output requirements:
- JSON only.
- Strict adherence to the requested "task_type".
- NO bundles. Just a single exercise task.
- Instructions and content in English.

Safety:
- No PII.
- Correct English grammar in examples.
"""


USER_PROMPT_TEMPLATE = """
Generate exactly one exercise task matching the requested type.

INPUT DATA:
user_id: "{user_id}"
message_content: "{message_content}"
grammar_error: "{grammar_error}"
explanation: "{explanation}"
llm_confidence: {llm_confidence}
task_type: "{task_type}"

OUTPUT JSON SCHEMA (strict; JSON only):
{{
  "target_error_category": "string",
  "corrected_sentence": "string",
  "task": {{
      "type": "string",
      "instruction_en": "string",
      "content_en": {{
        "context_text": "string" | null,
        "items": [
          {{
            "question_en": "string",
            "options_en": ["string"] | null,
            "student_answer_en": "string"
          }}
        ],
        "word_bank": ["string"] | null
      }}
  }}
}}

EXERCISE RULES (Strict Formatting):
1. Vocabulary Fill (Long text): 12-20 gaps, word_bank with UNIQUE words.
2. Matching / Categories: 8-12 pairs or 10-16 category items.
3. Grammar Choice (10-14 items): Each item has 3-4 options.
4. Transformation (10-14 items): Rewrite sentence with given word.
"""


def build_messages(
    *,
    user_id: str,
    message_content: str,
    grammar_error: str,
    explanation: str,
    llm_confidence: float,
    task_type: str,
) -> list[dict]:
    """Render chat-template messages for the model."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        user_id=user_id,
        message_content=message_content,
        grammar_error=grammar_error,
        explanation=explanation,
        llm_confidence=llm_confidence,
        task_type=task_type,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
