from __future__ import annotations

SYSTEM_PROMPT = """You are an expert English exercise generator.

Your task:
Generate exactly ONE exercise based on the user's grammar error 
and the requested task type.

TAXONOMY:
- Adjective Form
- Noun Inflection
- Noun Number
- Tense Usage
- Verb Form
- Verb Inflection
- Verb Agreement
- Verb Tense
- Part of Speech
- Adverb
- Determiner
- Particle
- Preposition
- Spelling
- Word Order
- No Error
(Use only these labels.)

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
1. Vocabulary Fill (Long text):
   - Generate a coherent text with 12-20 gaps.
   - Provide a "word_bank" list.
   - word_bank must contain UNIQUE words only.

2. Matching / Categories:
   - Provide 8-12 pairs or 10-16 category items.
   - Format: "**_1–D_**" or "**_Furniture: Sofa_**".

3. Grammar Choice (10-14 items):
   - MINIMUM 10 items, MAXIMUM 14 items.
   - Provide exactly 2-3 options per item.
   - student_answer_en: full sentence with bold answer 
     AND options.

4. Transformations (8-12 items):
   - MINIMUM 8 items.
   - student_answer_en: full rewritten sentence with 
     bolded changes.

5. Reading (True/False):
   - Provide text (120-180 words) in context_text.
   - 8 statements in items.

6. Functional Matching:
   - 8-10 items with 4 labelled choices each.
   - Format: "**_1–C_**".

7. Writing Sample:
   - Provide context_text (prompt).
   - Student answer: 60-90 words.

GENERAL RULES:
- Target the specific grammar_error primarily, but expand 
  context to the general topic of the user's message.
- If llm_confidence < 0.75, generate a "Verification/Choice" 
  task first.
- Return ONLY valid JSON.
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
