# SAYIT Exercise Generator

Production-ready GPU service that generates personalized English grammar
exercises from a learner's writing sample and an identified error.

- **Primary model:** Qwen2.5-3B-Instruct + LoRA adapter (TR Composite ≈ 93.8%)
- **Fallback:** rule-based Fill-in-the-Blanks / Sentence Reconstruction
- **Stack:** FastAPI · PyTorch (CUDA 12.1) · Transformers · PEFT · spaCy
- **Deployment:** single Docker container with the NVIDIA runtime

The API is intended to be consumed by the SAYIT learning platform; this
service exposes only inference endpoints (no auth, no DB).

---

## 1. Prerequisites

- NVIDIA GPU with ≥ 8 GB VRAM (Qwen2.5-3B in fp16 ≈ 6 GB)
- Docker ≥ 24 with the NVIDIA Container Toolkit installed
  (`nvidia-ctk runtime configure --runtime=docker` once on the host)
- LoRA adapter present at [adapters/qwen2.5-3b/](adapters/qwen2.5-3b/) — must
  contain `adapter_config.json`, `adapter_model.safetensors`, and tokenizer files
- Outbound network access on first run (HuggingFace Hub download of the base model)

## 2. Quick start

Two deployment modes — pick one:

| Compose file                    | When to use                                  |
|---------------------------------|----------------------------------------------|
| `docker-compose.yml`            | Production / staging next to SAYIT (joins the SAYIT docker network) |
| `docker-compose.standalone.yml` | Local dev, demos, or running independently   |

```bash
cp .env.example .env

# Standalone (no SAYIT around)
docker compose -f docker-compose.standalone.yml up -d --build

# Or, alongside SAYIT (assumes its compose stack is already up)
docker compose up -d --build
```

First start downloads `Qwen/Qwen2.5-3B-Instruct` (~6 GB) into the named volume
`hf_cache` — give it 2–5 min. Subsequent starts are instant. Check progress:

```bash
docker compose logs -f generator
```

When `app.startup` and `model.load.done` appear, the service is ready:

```bash
curl http://localhost:8000/health | jq .
```

To pre-populate the cache before starting the service:

```bash
docker compose run --rm generator python scripts/download_models.py
```

## 3. API reference

### `POST /api/v1/generate` — LLM-driven generation

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u_123",
    "message_content": "It has a high - density population because for its small territory.",
    "grammar_error": "Preposition",
    "explanation": "Wrong preposition: '\''for'\'' should be '\''of'\''.",
    "llm_confidence": 0.95,
    "task_type": "grammar_choice"
  }'
```

Response:

```json
{
  "status": "ok",
  "model_used": "qwen2.5-3b",
  "generation_time_ms": 1234,
  "exercise": {
    "target_error_category": "Preposition",
    "corrected_sentence": "...",
    "task": {
      "type": "grammar_choice",
      "instruction_en": "...",
      "content_en": {
        "context_text": "...",
        "items": [
          {
            "question_en": "...",
            "options_en": ["of", "for", "to", "with"],
            "student_answer_en": "of"
          }
        ],
        "word_bank": null
      }
    }
  }
}
```

If the model emits unparseable output, the API still returns 200 with
`status: "fallback"` and a template-generated exercise — the caller should
treat both shapes as success.

`task_type` accepts: `grammar_choice`, `transformation`, `vocabulary_fill`,
`matching`, `categories`. The optional `model` field selects an adapter
(default: `qwen2.5-3b`).

### `POST /api/v1/generate/batch` — SAYIT integration (Grok replacement)

Accepts a `UserErrorProfile`-shaped payload and returns a flat list of MCQ
questions in the exact shape SAYIT's `/tests/personalized/start` already
forwards to the frontend. This is the **only endpoint the SAYIT backend needs
to call** to retire the Grok dependency.

```bash
curl -X POST http://localhost:8000/api/v1/generate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u_123",
    "language_level": "B1",
    "max_questions": 15,
    "error_profile": [
      {
        "error_type": "Preposition",
        "occurrences": 12,
        "examples": [
          {
            "original": "I go to school by walk",
            "corrected": "I go to school on foot",
            "explanation": "Use '\''on foot'\'' instead of '\''by walk'\''"
          }
        ]
      },
      {
        "error_type": "Article",
        "occurrences": 8,
        "examples": [
          {
            "original": "I saw cat in garden",
            "corrected": "I saw a cat in the garden",
            "explanation": "Use articles '\''a'\'' and '\''the'\'' before nouns"
          }
        ]
      }
    ]
  }'
```

Response:

```json
{
  "status": "ok",
  "model_used": "qwen2.5-3b",
  "generation_time_ms": 4200,
  "questions": [
    {
      "id": "q_preposition_000",
      "error_type": "Preposition",
      "text": "Choose the correct preposition: I usually go to work ___ bus.",
      "options": ["by", "on", "with", "in"],
      "correct_answer": "by",
      "explanation": "We use 'by' + means of transport (by bus, by car)."
    }
  ],
  "exercises_full": [],
  "fallback_categories": []
}
```

Behavior:
- Allocates `max_questions` proportionally to each category's `occurrences`
  (every category with examples gets ≥1 question while budget allows).
- Calls the LLM once per category with `task_type=grammar_choice`, then flattens
  the resulting items into MCQs.
- If the LLM fails or returns fewer items than allocated, missing slots are
  filled with template-engine questions and the category is listed in
  `fallback_categories`. `status` becomes `"partial"`.
- Returns `400` only when nothing usable was produced (empty profile / no
  examples).

### `GET /api/v1/models` — service discovery

Returns service identity, loaded models, and offline metrics. SAYIT calls this
on boot to gate routing away from Grok until the generator is ready.

```bash
curl http://localhost:8000/api/v1/models | jq .
```

### `POST /api/v1/generate/template` — rule-based fallback (no GPU)

```bash
curl -X POST http://localhost:8000/api/v1/generate/template \
  -H "Content-Type: application/json" \
  -d '{
    "source_sentence": "It has a high - density population because for its small territory.",
    "corrected_sentence": "It has a high - density population because of its small territory.",
    "error_type": "Preposition",
    "method": "fill_in_blanks"
  }'
```

`method` ∈ {`fill_in_blanks`, `reconstruction`}.

### `GET /health`

```json
{
  "status": "ok",
  "uptime_s": 312.4,
  "default_model": "qwen2.5-3b",
  "loaded_models": ["qwen2.5-3b"],
  "gpu": {
    "available": true,
    "device_count": 1,
    "device_name": "NVIDIA RTX 4090",
    "vram_total_mb": 24576,
    "vram_used_mb": 6500,
    "vram_free_mb": 18076
  }
}
```

## 4. Environment variables

See [.env.example](.env.example) for the full list. The most relevant ones:

| Variable           | Default            | Purpose                                                  |
|--------------------|--------------------|----------------------------------------------------------|
| `MODEL_NAME`       | `qwen2.5-3b`       | Default model when client omits the `model` field        |
| `ADAPTER_DIR`      | `/app/adapters`    | Mounted folder containing per-model LoRA subfolders      |
| `HF_HOME`          | `/app/models_cache`| HuggingFace cache (persisted via the `hf_cache` volume)  |
| `MAX_NEW_TOKENS`   | `2048`             | Generation cap                                           |
| `LOG_LEVEL`        | `INFO`             | `DEBUG` / `INFO` / `WARNING` / `ERROR`                   |
| `CORS_ORIGINS`     | `*`                | Comma-separated list, or `*` for any origin              |
| `HF_TOKEN`         | (unset)            | Only needed for gated / private base models              |

## 5. Development

Run unit tests (no GPU/model required — generator is mocked):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pytest -v
```

Run the service locally without Docker (CPU mode is slow but works):

```bash
FORCE_CPU=true uvicorn app.main:app --reload
```

Latency benchmark against a running container:

```bash
python scripts/run_benchmark.py --n 20
```

## 6. Project layout

```
.
├── Dockerfile                  # multi-stage CUDA 12.1 build
├── docker-compose.yml          # GPU-enabled service definition
├── requirements.txt
├── .env.example
├── app/
│   ├── main.py                 # FastAPI factory + lifespan
│   ├── config.py               # pydantic-settings
│   ├── api/
│   │   ├── router.py           # /api/v1/generate, /generate/template, /health
│   │   └── schemas.py
│   ├── core/
│   │   ├── model_manager.py    # base + LoRA loading, singleton
│   │   ├── generator.py        # LLM inference
│   │   ├── template_engine.py  # rule-based fallback (spaCy)
│   │   └── postprocessor.py    # JSON extraction + validation
│   └── prompts/templates.py    # SYSTEM + USER prompts (must match training)
├── adapters/qwen2.5-3b/        # LoRA weights (mounted read-only)
├── scripts/
│   ├── download_models.py
│   └── run_benchmark.py
└── tests/
    ├── test_api.py
    ├── test_generator.py
    └── test_health.py
```

## 7. SAYIT integration

This service is a drop-in replacement for the Grok-driven branch of
`error_analysis.py:generate_questions_from_errors()`. The contract is
deliberately Grok-compatible: same error-profile input, same flat
`questions[]` output.

### Networking

`docker-compose.yml` joins the existing SAYIT docker network (default name
`sayit_default`, override with `SAYIT_NETWORK_NAME`). Once both stacks are up,
the SAYIT backend resolves the generator at `http://generator:8000`.

```bash
# In SAYIT's backend env (e.g. .env.prod):
EXERCISE_GENERATOR_URL=http://generator:8000
```

### Backend client (drop into `backend/app/clients/exercise_generator_client.py`)

```python
import httpx
from app.config import settings

EXERCISE_GENERATOR_URL = settings.EXERCISE_GENERATOR_URL  # http://generator:8000


async def generate_personalized_exercises(
    user_id: str,
    error_profile: list[dict],
    language_level: str = "B1",
    max_questions: int = 15,
) -> list[dict]:
    """Replacement for the Grok-based generate_questions_from_errors()."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{EXERCISE_GENERATOR_URL}/api/v1/generate/batch",
            json={
                "user_id": user_id,
                "language_level": language_level,
                "error_profile": error_profile,
                "max_questions": max_questions,
            },
        )
        response.raise_for_status()
        return response.json()["questions"]


async def is_generator_ready() -> bool:
    """Used at startup to decide whether to route off Grok."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{EXERCISE_GENERATOR_URL}/api/v1/models")
            r.raise_for_status()
            return any(m["status"] == "loaded" for m in r.json()["models"].values())
    except Exception:
        return False
```

### Patching `error_analysis.py`

```python
# Before:
# questions = await call_grok_generate_questions(error_profile, ...)

# After:
from app.clients.exercise_generator_client import generate_personalized_exercises
questions = await generate_personalized_exercises(
    user_id=user_id,
    error_profile=error_profile,
    language_level=language_level,
    max_questions=15,
)
```

The shape of `questions` is identical to what Grok returned (`id`,
`error_type`, `text`, `options[4]`, `correct_answer`, `explanation`), so the
`/tests/personalized/start` handler and the frontend need no changes.

### Failure modes

| Situation                          | What the backend should do            |
|------------------------------------|---------------------------------------|
| Container down / 5xx               | Fall back to the static question bank |
| `/api/v1/models` not reachable     | Fall back to the static question bank |
| Response status `"partial"`        | Use the questions; log the categories from `fallback_categories` |
| Response status `"ok"`             | Use the questions as-is               |

## 8. Operational notes

- **Single-tenant inference.** The container handles one request at a time
  (synchronous `model.generate`). For higher throughput, run multiple
  replicas behind a load balancer rather than enabling internal concurrency.
- **VRAM.** Qwen2.5-3B in fp16 takes ~6 GB; with KV cache for 2048 tokens
  budget ~7–8 GB.
- **Cold start.** First request after container boot is fast because the
  model is loaded in `lifespan` — readiness comes from `/health` returning
  `loaded_models: ["qwen2.5-3b"]`.
- **Adapters are read-only.** The `adapters/` folder is mounted `:ro` so the
  container can't accidentally corrupt training artifacts.
