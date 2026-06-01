from __future__ import annotations

import functools
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import structlog

log = structlog.get_logger(__name__)


_SPACY_NLP = None
_SPACY_TRIED = False


def _get_spacy():
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


PUNCT_NO_SPACE_BEFORE = {
    ".", ",", ":", ";", "!", "?", ")", "]", "}",
    "'s", "n't", "'re", "'ve", "'ll", "'d", "'m",
}
PUNCT_NO_SPACE_AFTER = {"(", "[", "{", "$", "#"}
PUNCT_CHUNK_SPLIT = {",", ";", ":"}
PUNCT_DASH = {"--", "---", "–", "—"}
PUNCT_TERMINAL = {".", "!", "?"}

LONG_CHUNK_THRESHOLD = 12
SUB_CHUNK_SIZE = 6
MIN_ITEMS = 8
MAX_ITEMS = 12
TARGET_ITEMS = 10


POS_MAP = {
    "ADP": "PREP",
    "DET": "DET",
    "PRON": "PRON",
    "AUX": "AUX",
    "VERB": "VERB",
    "NOUN": "NOUN",
    "PROPN": "NOUN",
    "ADJ": "ADJ",
    "ADV": "ADV",
    "CCONJ": "CONJ",
    "SCONJ": "CONJ",
    "PART": "PART",
    "NUM": "NUM",
    "INTJ": "INTJ",
    "X": "OTHER",
    "SYM": "OTHER",
    "PUNCT": "PUNCT",
    "SPACE": "OTHER",
}


ERROR_TYPE_CONFIG: Dict[str, Dict[str, str]] = {
    "Preposition":              {"fib_type": "grammar_choice",  "instruction": "Choose the correct preposition to complete each sentence."},
    "Article":                  {"fib_type": "grammar_choice",  "instruction": "Choose the correct article (a / an / the / —) for each sentence."},
    "Verb Tense":               {"fib_type": "grammar_choice",  "instruction": "Choose the correct verb tense to complete each sentence."},
    "Subject-Verb Agreement":   {"fib_type": "grammar_choice",  "instruction": "Choose the verb form that agrees with the subject."},
    "Auxiliary Verb":           {"fib_type": "grammar_choice",  "instruction": "Choose the correct auxiliary verb for each sentence."},
    "Pronoun-Antecedent Agreement": {"fib_type": "grammar_choice", "instruction": "Choose the pronoun that agrees with its antecedent."},
    "Transitive Verb":          {"fib_type": "grammar_choice",  "instruction": "Choose the correct transitive verb for each sentence."},
    "Gerund":                   {"fib_type": "grammar_choice",  "instruction": "Choose the correct gerund form for each sentence."},
    "Infinitives":              {"fib_type": "grammar_choice",  "instruction": "Choose the correct infinitive form for each sentence."},
    "Participle":               {"fib_type": "grammar_choice",  "instruction": "Choose the correct participle form for each sentence."},
    "Number":                   {"fib_type": "grammar_choice",  "instruction": "Choose the correct number form (singular / plural) for each sentence."},
    "Possessive":               {"fib_type": "grammar_choice",  "instruction": "Choose the correct possessive form for each sentence."},
    "POS Confusion":            {"fib_type": "vocabulary_fill", "instruction": "Choose the word with the correct part of speech to complete each sentence."},
    "Collocation":              {"fib_type": "vocabulary_fill", "instruction": "Choose the word that collocates correctly in each sentence."},
    "Others":                   {"fib_type": "grammar_choice",  "instruction": "Choose the correct word to complete each sentence."},
}


def config_for(error_type: str) -> Dict[str, str]:
    return ERROR_TYPE_CONFIG.get(error_type, ERROR_TYPE_CONFIG["Others"])


_TOKEN_RE = re.compile(r"\w+(?:'[a-zA-Z]+)?|[^\w\s]", flags=re.UNICODE)


def tokenize(sentence: str) -> List[str]:
    if not sentence:
        return []
    return _TOKEN_RE.findall(sentence)


def detokenize(tokens: Sequence[str]) -> str:
    tokens = [t for t in tokens if t]
    if not tokens:
        return ""
    out: List[str] = []
    for i, tok in enumerate(tokens):
        if i == 0:
            out.append(tok)
        elif tok in PUNCT_NO_SPACE_BEFORE:
            out.append(tok)
        elif out and out[-1] in PUNCT_NO_SPACE_AFTER:
            out.append(tok)
        else:
            out.append(" " + tok)
    return "".join(out)


def _is_meaningful(token: str) -> bool:
    return token not in PUNCT_NO_SPACE_BEFORE and token not in PUNCT_NO_SPACE_AFTER and token not in PUNCT_DASH


@functools.lru_cache(maxsize=50_000)
def _cached_doc_pos(sentence_text: str) -> Tuple[Tuple[str, str], ...]:
    nlp = _get_spacy()
    if nlp is None or not sentence_text:
        return tuple()
    try:
        doc = nlp(sentence_text)
    except Exception:  # noqa: BLE001
        return tuple()
    return tuple((tok.text.lower(), POS_MAP.get(tok.pos_, "OTHER")) for tok in doc)


def _pos_in_context(sentence_text: str, word: str, occurrence: int = 0) -> str:
    if not sentence_text or not word:
        return "OTHER"
    target_lower = word.lower()
    matches = [pos for text, pos in _cached_doc_pos(sentence_text) if text == target_lower]
    if not matches:
        return "OTHER"
    return matches[min(occurrence, len(matches) - 1)]


@functools.lru_cache(maxsize=200_000)
def _pos_isolated(word: str) -> str:
    nlp = _get_spacy()
    if nlp is None or not word:
        return "OTHER"
    try:
        doc = nlp(word)
    except Exception:  # noqa: BLE001
        return "OTHER"
    if len(doc) == 0:
        return "OTHER"
    return POS_MAP.get(doc[0].pos_, "OTHER")


@dataclass
class _Correction:
    applicable: bool
    target_word: str = ""
    target_pos: int = -1
    wrong_word: Optional[str] = None
    kind: str = ""


def _diff_correction(source: str, corrected: str) -> _Correction:
    src = tokenize(source)
    cor = tokenize(corrected)
    if not cor:
        return _Correction(False)

    n_pref = 0
    while n_pref < min(len(src), len(cor)) and src[n_pref].lower() == cor[n_pref].lower():
        n_pref += 1
    n_suf = 0
    while (
        n_suf < min(len(src) - n_pref, len(cor) - n_pref)
        and src[len(src) - 1 - n_suf].lower() == cor[len(cor) - 1 - n_suf].lower()
    ):
        n_suf += 1

    src_mid = src[n_pref : len(src) - n_suf]
    cor_mid = cor[n_pref : len(cor) - n_suf]

    if not cor_mid:
        return _Correction(False)

    target_word = cor_mid[0]
    target_pos = n_pref
    if target_word in PUNCT_NO_SPACE_BEFORE or target_word in PUNCT_NO_SPACE_AFTER:
        return _Correction(False)

    if not src_mid:
        return _Correction(
            applicable=True,
            target_word=target_word,
            target_pos=target_pos,
            wrong_word="",
            kind="insertion",
        )

    wrong = src_mid[0]
    if wrong == target_word:
        return _Correction(False)
    return _Correction(
        applicable=True,
        target_word=target_word,
        target_pos=target_pos,
        wrong_word=wrong,
        kind="replacement",
    )


DistractorPool = Dict[Tuple[str, str], Dict[str, int]]


def build_distractor_pool(pairs: Iterable[Tuple[str, str, str]]) -> DistractorPool:
    pool: DistractorPool = {}
    for source, corrected, error_type in pairs:
        diff = _diff_correction(source, corrected)
        if not diff.applicable or not diff.wrong_word:
            continue
        wrong = diff.wrong_word
        if wrong in PUNCT_NO_SPACE_BEFORE or wrong in PUNCT_NO_SPACE_AFTER:
            continue
        src_tokens = tokenize(source)
        try:
            wrong_pos_idx = src_tokens.index(wrong)
        except ValueError:
            wrong_pos_idx = 0
        occurrence = sum(1 for t in src_tokens[:wrong_pos_idx] if t.lower() == wrong.lower())
        pos = _pos_in_context(source, wrong, occurrence)
        if pos == "OTHER":
            pos = _pos_isolated(wrong)
        if pos in {"PUNCT", "OTHER"}:
            continue
        bucket = pool.setdefault((error_type, pos), {})
        bucket[wrong] = bucket.get(wrong, 0) + 1
    return pool


def _match_case(distractor: str, target: str, is_sentence_start: bool) -> str:
    if not distractor:
        return distractor
    if is_sentence_start:
        return distractor[0].upper() + distractor[1:]
    if target and target[0].islower():
        return distractor[0].lower() + distractor[1:]
    return distractor


_POS_FALLBACK_POOL: Dict[str, List[str]] = {
    "PREP": ["of", "in", "on", "at", "for", "with", "about", "by", "from", "to"],
    "DET": ["the", "a", "an", "this", "that", "some", "any"],
    "AUX": ["is", "was", "are", "were", "has", "have", "had", "do", "does", "did"],
    "VERB": ["make", "take", "give", "go", "have", "do", "see", "say"],
    "NOUN": ["thing", "person", "place", "time", "way", "people", "year"],
    "ADJ": ["good", "small", "large", "high", "low", "new", "old"],
    "ADV": ["very", "quite", "really", "rather", "always", "often"],
    "PRON": ["he", "she", "it", "they", "we", "him", "her", "them"],
    "PART": ["to", "not", "up", "out"],
    "CONJ": ["and", "but", "or", "so", "because", "while"],
}


def _pick_distractors(
    *,
    target_word: str,
    error_type: str,
    target_pos_context: str,
    wrong_word: Optional[str],
    is_sentence_start: bool,
    pool: DistractorPool,
    n: int = 3,
    forbidden_lower: Optional[set] = None,
) -> List[str]:
    chosen: List[str] = []
    seen = {target_word.lower()}
    forbidden = set(forbidden_lower) if forbidden_lower else set()

    if isinstance(wrong_word, str) and wrong_word and wrong_word.lower() not in seen and wrong_word.lower() not in forbidden:
        chosen.append(_match_case(wrong_word, target_word, is_sentence_start))
        seen.add(wrong_word.lower())

    primary = pool.get((error_type, target_pos_context), {})
    for word, _freq in sorted(primary.items(), key=lambda x: -x[1]):
        if len(chosen) >= n:
            break
        wl = word.lower()
        if wl in seen or wl in forbidden:
            continue
        chosen.append(_match_case(word, target_word, is_sentence_start))
        seen.add(wl)

    if len(chosen) < n:
        for (et, pos), words in pool.items():
            if pos != target_pos_context or et == error_type:
                continue
            for word, _freq in sorted(words.items(), key=lambda x: -x[1]):
                if len(chosen) >= n:
                    break
                wl = word.lower()
                if wl in seen or wl in forbidden:
                    continue
                chosen.append(_match_case(word, target_word, is_sentence_start))
                seen.add(wl)
            if len(chosen) >= n:
                break

    if len(chosen) < n:
        for word in _POS_FALLBACK_POOL.get(target_pos_context, []):
            if len(chosen) >= n:
                break
            wl = word.lower()
            if wl in seen or wl in forbidden:
                continue
            chosen.append(_match_case(word, target_word, is_sentence_start))
            seen.add(wl)

    if not chosen and wrong_word == "":
        chosen = ["(no word needed)"]

    return chosen[:n]


def _build_template(target_tokens: List[str], blank_pos: int) -> str:
    tokens = list(target_tokens)
    if blank_pos < 0 or blank_pos >= len(tokens):
        return ""
    tokens[blank_pos] = "___"
    return detokenize(tokens)


def _build_student_answer(target_tokens: List[str], blank_pos: int) -> str:
    tokens = list(target_tokens)
    if blank_pos < 0 or blank_pos >= len(tokens):
        return ""
    tokens[blank_pos] = f"**{tokens[blank_pos]}**"
    return detokenize(tokens)


def _build_fib_item(
    *,
    source: str,
    corrected: str,
    error_type: str,
    pool: DistractorPool,
) -> Optional[Dict[str, Any]]:
    diff = _diff_correction(source, corrected)
    if not diff.applicable:
        return None
    target_tokens = tokenize(corrected)
    if diff.target_pos >= len(target_tokens):
        return None

    blank_pos = diff.target_pos
    target_word = diff.target_word

    template = _build_template(target_tokens, blank_pos)
    if not template:
        return None

    occurrence = sum(1 for t in target_tokens[:blank_pos] if t.lower() == target_word.lower())
    pos_ctx = _pos_in_context(corrected, target_word, occurrence)
    if pos_ctx in {"OTHER", "PUNCT"}:
        pos_ctx = _pos_isolated(target_word)

    is_start = blank_pos == 0
    forbidden = {t.lower() for t in target_tokens if t}
    forbidden.discard(target_word.lower())

    distractors = _pick_distractors(
        target_word=target_word,
        error_type=error_type,
        target_pos_context=pos_ctx,
        wrong_word=diff.wrong_word,
        is_sentence_start=is_start,
        pool=pool,
        n=3,
        forbidden_lower=forbidden,
    )
    if len(distractors) < 2:
        return None

    correct_display = target_word
    if is_start and correct_display:
        correct_display = correct_display[0].upper() + correct_display[1:]
    options = [correct_display, *distractors]
    random.shuffle(options)

    student_answer = _build_student_answer(target_tokens, blank_pos)
    return {
        "question_en": template,
        "options_en": options,
        "student_answer_en": student_answer,
    }


def _split_into_chunks(tokens: List[str]) -> List[List[str]]:
    chunks: List[List[str]] = []
    current: List[str] = []
    for tok in tokens:
        if tok in PUNCT_CHUNK_SPLIT or tok in PUNCT_DASH:
            if current:
                current.append(tok)
                chunks.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        chunks.append(current)
    return [c for c in chunks if any(_is_meaningful(t) for t in c)]


def _chunk_by_length(tokens: List[str], chunk_size: int = SUB_CHUNK_SIZE) -> List[List[str]]:
    chunks: List[List[str]] = []
    i = 0
    n = len(tokens)
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(tokens[i:end])
        i = end
    return chunks


def _merge_punct_only_chunks(chunks: List[List[str]]) -> List[List[str]]:
    out: List[List[str]] = []
    for c in chunks:
        if any(_is_meaningful(t) for t in c):
            out.append(list(c))
        elif out:
            out[-1].extend(c)
        else:
            out.append(list(c))
    return out


def _split_long_chunks(
    chunks: List[List[str]],
    threshold: int = LONG_CHUNK_THRESHOLD,
    chunk_size: int = SUB_CHUNK_SIZE,
) -> List[List[str]]:
    out: List[List[str]] = []
    for c in chunks:
        if len(c) > threshold:
            out.extend(_chunk_by_length(c, chunk_size=chunk_size))
        else:
            out.append(c)
    return _merge_punct_only_chunks(out)


def _shuffle_chunks(chunks: List[List[str]], seed: int) -> List[List[str]]:
    if len(chunks) <= 1:
        return chunks
    rng = random.Random(seed)
    order = list(range(len(chunks)))
    for _ in range(10):
        rng.shuffle(order)
        if order != list(range(len(chunks))):
            break
    return [chunks[i] for i in order]


def _shuffle_short_sentence(tokens: List[str], seed: int) -> List[str]:
    meaningful_idx = [i for i, t in enumerate(tokens) if _is_meaningful(t)]
    if len(meaningful_idx) < 3:
        return list(tokens)
    rng = random.Random(seed)
    inner = meaningful_idx[1:-1]
    inner_words = [tokens[i] for i in inner]
    shuffled_inner = inner_words[:]
    for _ in range(10):
        rng.shuffle(shuffled_inner)
        if shuffled_inner != inner_words:
            break
    out = list(tokens)
    for idx, new_word in zip(inner, shuffled_inner):
        out[idx] = new_word
    return out


def _build_reconstruction_item(corrected: str, seed: int) -> Optional[Dict[str, Any]]:
    target_tokens = tokenize(corrected)
    if len(target_tokens) < 4:
        return None
    chunks = _split_into_chunks(target_tokens)
    if not chunks:
        return None

    if len(chunks) == 1 and len(chunks[0]) <= 14:
        only = chunks[0]
        shuffled_tokens = _shuffle_short_sentence(only, seed)
        if shuffled_tokens == only:
            return None
        question = " / ".join(shuffled_tokens)
        answer = detokenize(only)
        return {
            "question_en": f"Restore the correct word order: {question}",
            "options_en": [],
            "student_answer_en": f"**{answer}**",
        }

    chunks = _split_long_chunks(chunks)
    if len(chunks) < 2:
        return None
    shuffled = _shuffle_chunks(chunks, seed)
    if shuffled == chunks:
        return None
    question_fragments = [detokenize(c) for c in shuffled]
    question = " / ".join(question_fragments)
    flat: List[str] = []
    for c in chunks:
        flat.extend(c)
    answer = detokenize(flat)
    return {
        "question_en": f"Restore the correct order of fragments: {question}",
        "options_en": [],
        "student_answer_en": f"**{answer}**",
    }


@dataclass
class GECPair:
    source: str
    corrected: str


def _coerce_pairs(profile: Iterable[Any]) -> List[GECPair]:
    pairs: List[GECPair] = []
    for p in profile:
        if isinstance(p, GECPair):
            pairs.append(p)
        elif isinstance(p, dict):
            src = p.get("source") or p.get("original") or ""
            cor = p.get("corrected") or p.get("target") or ""
            if src and cor:
                pairs.append(GECPair(source=src, corrected=cor))
        elif isinstance(p, (tuple, list)) and len(p) >= 2:
            pairs.append(GECPair(source=str(p[0]), corrected=str(p[1])))
    return pairs


def fill_in_blanks_exercise(
    *,
    error_type: str,
    profile: Iterable[Any],
    n_items: int = TARGET_ITEMS,
    pool: Optional[DistractorPool] = None,
) -> Optional[Dict[str, Any]]:
    pairs = _coerce_pairs(profile)
    if len(pairs) < MIN_ITEMS:
        if not pairs:
            return None

    if pool is None:
        pool = build_distractor_pool((p.source, p.corrected, error_type) for p in pairs)

    n_items = max(1, min(MAX_ITEMS, n_items))
    cfg = config_for(error_type)
    items: List[Dict[str, Any]] = []
    corrected_ref: Optional[str] = None

    indices = list(range(len(pairs)))
    random.shuffle(indices)
    for i in indices:
        if len(items) >= n_items:
            break
        pair = pairs[i]
        item = _build_fib_item(
            source=pair.source,
            corrected=pair.corrected,
            error_type=error_type,
            pool=pool,
        )
        if item is None:
            continue
        items.append(item)
        if corrected_ref is None:
            corrected_ref = pair.corrected

    if not items:
        return None

    return {
        "target_error_category": error_type,
        "corrected_sentence": corrected_ref or "",
        "task": {
            "type": cfg["fib_type"],
            "instruction_en": cfg["instruction"],
            "content_en": {
                "context_text": None,
                "items": items,
                "word_bank": None,
            },
        },
    }


def sentence_reconstruction_exercise(
    *,
    error_type: str,
    profile: Iterable[Any],
    n_items: int = TARGET_ITEMS,
) -> Optional[Dict[str, Any]]:
    pairs = _coerce_pairs(profile)
    if not pairs:
        return None

    n_items = max(1, min(MAX_ITEMS, n_items))
    items: List[Dict[str, Any]] = []
    corrected_ref: Optional[str] = None
    seed_base = random.randint(0, 100_000)

    indices = list(range(len(pairs)))
    random.shuffle(indices)
    for k, i in enumerate(indices):
        if len(items) >= n_items:
            break
        pair = pairs[i]
        item = _build_reconstruction_item(pair.corrected, seed=seed_base + k)
        if item is None:
            continue
        items.append(item)
        if corrected_ref is None:
            corrected_ref = pair.corrected

    if not items:
        return None

    return {
        "target_error_category": error_type,
        "corrected_sentence": corrected_ref or "",
        "task": {
            "type": "transformation",
            "instruction_en": (
                f"Restore the correct order in each sentence. "
                f"Focus on the structure typical for {error_type.lower()} corrections."
            ),
            "content_en": {
                "context_text": None,
                "items": items,
                "word_bank": None,
            },
        },
    }


def fill_in_blanks(
    *,
    source_sentence: str,
    corrected_sentence: str,
    error_type: str,
) -> dict:
    ex = fill_in_blanks_exercise(
        error_type=error_type,
        profile=[GECPair(source=source_sentence, corrected=corrected_sentence)],
        n_items=1,
    )
    if ex is not None:
        return ex
    cfg = config_for(error_type)
    tokens = tokenize(corrected_sentence)
    if not tokens:
        question = corrected_sentence + " ___"
        answer = corrected_sentence
    else:
        blank_pos = 0
        question = _build_template(tokens, blank_pos)
        answer = _build_student_answer(tokens, blank_pos)
    return {
        "target_error_category": error_type,
        "corrected_sentence": corrected_sentence,
        "task": {
            "type": cfg["fib_type"],
            "instruction_en": cfg["instruction"],
            "content_en": {
                "context_text": None,
                "items": [
                    {
                        "question_en": question,
                        "options_en": ["—", "a", "the", "of"],
                        "student_answer_en": answer,
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
    ex = sentence_reconstruction_exercise(
        error_type=error_type,
        profile=[GECPair(source=source_sentence, corrected=corrected_sentence)],
        n_items=1,
    )
    if ex is not None:
        return ex
    tokens = tokenize(corrected_sentence) or [corrected_sentence]
    shuffled = list(tokens)
    if len(shuffled) > 1:
        for _ in range(5):
            random.shuffle(shuffled)
            if shuffled != tokens:
                break
    return {
        "target_error_category": error_type,
        "corrected_sentence": corrected_sentence,
        "task": {
            "type": "transformation",
            "instruction_en": (
                f"Restore the correct order in each sentence. "
                f"Focus on the structure typical for {error_type.lower()} corrections."
            ),
            "content_en": {
                "context_text": None,
                "items": [
                    {
                        "question_en": f"Restore the correct word order: {' / '.join(shuffled)}",
                        "options_en": [],
                        "student_answer_en": f"**{corrected_sentence}**",
                    }
                ],
                "word_bank": None,
            },
        },
    }


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


def build_template_exercise_from_profile(
    *,
    method: str,
    error_type: str,
    profile: Iterable[Any],
    n_items: int = TARGET_ITEMS,
) -> Optional[Dict[str, Any]]:
    if method == "fill_in_blanks":
        return fill_in_blanks_exercise(error_type=error_type, profile=profile, n_items=n_items)
    if method == "reconstruction":
        return sentence_reconstruction_exercise(error_type=error_type, profile=profile, n_items=n_items)
    raise ValueError(f"Unknown template method: {method}")
