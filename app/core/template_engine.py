"""Rule-based exercise generators (Fill-in-the-Blanks + Sentence Reconstruction).

Used both as a public endpoint (`/api/v1/generate/template`) and as a fallback
when the LLM output cannot be parsed into a valid exercise.

spaCy is loaded lazily — if the model is unavailable, dist­ractor generation
gracefully degrades to a small static pool.
"""

from __future__ import annotations

import random
import re
from typing import List, Optional

import structlog

log = structlog.get_logger(__name__)


_SPACY_NLP = None
_SPACY_TRIED = False


def _get_spacy():
    """Load `en_core_web_sm` once. Return None if not installed."""
    global _SPACY_NLP, _SPACY_TRIED
    if _SPACY_NLP is not None or _SPACY_TRIED:
        return _SPACY_NLP
    _SPACY_TRIED = True
    try:
        import spacy

        _SPACY_NLP = spacy.load("en_core_web_sm")
    except Exception as e:  # noqa: BLE001
        log.warning("spacy.load.failed", error=str(e))
        _SPACY_NLP = None
    return _SPACY_NLP


# Static fallback pools, used when spaCy is unavailable or no candidates match.
_POS_FALLBACK_POOL = {
    "ADP": ["of", "in", "on", "at", "for", "with", "about", "by"],
    "DET": ["the", "a", "an", "this", "that"],
    "AUX": ["is", "was", "are", "were", "has", "have", "had"],
    "VERB": ["make", "take", "give", "go", "have", "do"],
    "NOUN": ["thing", "person", "place", "time", "way"],
    "ADJ": ["good", "small", "large", "high", "low"],
    "ADV": ["very", "quite", "really", "rather"],
    "PRON": ["he", "she", "it", "they", "we"],
}


def _tokenize_simple(sentence: str) -> List[str]:
    """Split a sentence into surface tokens preserving punctuation as separate items."""
    return re.findall(r"\w+|[^\w\s]", sentence, flags=re.UNICODE)


def _diff_tokens(source: str, corrected: str) -> List[tuple[int, str, str]]:
    """Find positions where corrected differs from source.

    Returns list of (corrected_index, source_token, corrected_token).
    Naive alignment — works well for the single-edit GEC errors this system targets.
    """
    src = _tokenize_simple(source)
    cor = _tokenize_simple(corrected)
    diffs: List[tuple[int, str, str]] = []
    n = min(len(src), len(cor))
    for i in range(n):
        if src[i].lower() != cor[i].lower():
            diffs.append((i, src[i], cor[i]))
    if len(cor) > len(src):
        for i in range(n, len(cor)):
            diffs.append((i, "", cor[i]))
    return diffs


def _generate_distractors(target: str, error_type: str, k: int = 3) -> List[str]:
    """Produce `k` plausible wrong-answer options using spaCy POS or a fallback pool."""
    nlp = _get_spacy()
    candidates: List[str] = []

    if nlp is not None:
        doc = nlp(target)
        if len(doc) > 0:
            pos = doc[0].pos_
            pool = _POS_FALLBACK_POOL.get(pos, [])
            candidates = [w for w in pool if w.lower() != target.lower()]

    if not candidates:
        # Heuristic: prepositions are the most common error class in the dataset.
        if "prep" in error_type.lower():
            pool = _POS_FALLBACK_POOL["ADP"]
        else:
            pool = _POS_FALLBACK_POOL["NOUN"]
        candidates = [w for w in pool if w.lower() != target.lower()]

    random.shuffle(candidates)
    return candidates[:k]


def fill_in_blanks(
    *,
    source_sentence: str,
    corrected_sentence: str,
    error_type: str,
) -> dict:
    """Build a single-item Grammar Choice exercise from a GEC pair.

    The token that differs between `source` and `corrected` becomes the blank;
    the correct option is the corrected token, distractors come from a POS pool.
    """
    diffs = _diff_tokens(source_sentence, corrected_sentence)
    if not diffs:
        # Nothing to gap — emit a degenerate item that still validates.
        target = "of"
        question = corrected_sentence.replace(target, "____", 1) if target in corrected_sentence else corrected_sentence + " ____"
    else:
        idx, _src_tok, target = diffs[0]
        cor_tokens = _tokenize_simple(corrected_sentence)
        cor_tokens[idx] = "____"
        question = _detokenize(cor_tokens)

    distractors = _generate_distractors(target, error_type, k=3)
    options = list({target, *distractors})
    random.shuffle(options)

    return {
        "target_error_category": error_type,
        "corrected_sentence": corrected_sentence,
        "task": {
            "type": "grammar_choice",
            "instruction_en": "Choose the word that correctly completes the sentence.",
            "content_en": {
                "context_text": None,
                "items": [
                    {
                        "question_en": question,
                        "options_en": options,
                        "student_answer_en": target,
                    }
                ],
                "word_bank": None,
            },
        },
    }


def sentence_reconstruction(
    *,
    source_sentence: str,
    corrected_sentence: str,
    error_type: str,
) -> dict:
    """Produce a Transformation-style exercise: reorder shuffled tokens of the corrected sentence."""
    tokens = _tokenize_simple(corrected_sentence)
    shuffled = tokens.copy()
    if len(shuffled) > 1:
        # Make sure the shuffle actually changes order.
        for _ in range(5):
            random.shuffle(shuffled)
            if shuffled != tokens:
                break

    return {
        "target_error_category": error_type,
        "corrected_sentence": corrected_sentence,
        "task": {
            "type": "transformation",
            "instruction_en": "Reorder the words to form a correct sentence.",
            "content_en": {
                "context_text": None,
                "items": [
                    {
                        "question_en": " / ".join(shuffled),
                        "options_en": None,
                        "student_answer_en": corrected_sentence,
                    }
                ],
                "word_bank": shuffled,
            },
        },
    }


def _detokenize(tokens: List[str]) -> str:
    """Join tokens back, gluing punctuation to the preceding word."""
    out: List[str] = []
    for tok in tokens:
        if out and re.match(r"^[^\w\s]$", tok):
            out[-1] = out[-1] + tok
        else:
            out.append(tok)
    return " ".join(out)


def build_template_exercise(
    *,
    method: str,
    source_sentence: str,
    corrected_sentence: str,
    error_type: str,
) -> dict:
    if method == "fill_in_blanks":
        return fill_in_blanks(
            source_sentence=source_sentence,
            corrected_sentence=corrected_sentence,
            error_type=error_type,
        )
    if method == "reconstruction":
        return sentence_reconstruction(
            source_sentence=source_sentence,
            corrected_sentence=corrected_sentence,
            error_type=error_type,
        )
    raise ValueError(f"Unknown template method: {method}")
