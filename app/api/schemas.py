from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


# ---------- Exercise payload (mirrors the JSON the model is trained to emit) ----------

class ExerciseItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    question_en: str
    options_en: Optional[List[str]] = None
    student_answer_en: str


class ExerciseContent(BaseModel):
    model_config = ConfigDict(extra="allow")

    context_text: Optional[str] = None
    items: List[ExerciseItem]
    word_bank: Optional[List[str]] = None


class ExerciseTask(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    instruction_en: str
    content_en: ExerciseContent


class Exercise(BaseModel):
    model_config = ConfigDict(extra="allow")

    target_error_category: str
    corrected_sentence: str
    task: ExerciseTask


# ---------- /api/v1/generate ----------

TaskType = Literal[
    "grammar_choice",
    "transformation",
    "vocabulary_fill",
    "matching",
    "categories",
]


class GenerateRequest(BaseModel):
    user_id: str
    message_content: str
    grammar_error: str
    explanation: str
    llm_confidence: float = Field(..., ge=0.0, le=1.0)
    task_type: TaskType = "grammar_choice"
    model: Optional[str] = Field(
        default=None,
        description="Model key (e.g. 'qwen2.5-3b'). Falls back to server default.",
    )


class GenerateResponse(BaseModel):
    status: Literal["ok", "fallback"] = "ok"
    model_used: str
    generation_time_ms: int
    exercise: Exercise
    fallback_reason: Optional[str] = None


# ---------- /api/v1/generate/template ----------

TemplateMethod = Literal["fill_in_blanks", "reconstruction"]


class TemplateRequest(BaseModel):
    source_sentence: str
    corrected_sentence: str
    error_type: str
    method: TemplateMethod = "fill_in_blanks"


class TemplateResponse(BaseModel):
    status: Literal["ok"] = "ok"
    method: TemplateMethod
    generation_time_ms: int
    exercise: Exercise


# ---------- /health ----------

class GPUInfo(BaseModel):
    available: bool
    device_count: int = 0
    device_name: Optional[str] = None
    vram_total_mb: Optional[int] = None
    vram_used_mb: Optional[int] = None
    vram_free_mb: Optional[int] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"] = "ok"
    uptime_s: float
    default_model: str
    loaded_models: List[str]
    gpu: GPUInfo


# ---------- /api/v1/generate/batch (SAYIT integration) ----------
# Replaces the Grok-driven path in SAYIT's error_analysis.py. Input shape mirrors
# the UserErrorProfile rows produced by the SAYIT backend; output shape mirrors
# what the /tests/personalized/start endpoint already passes to the frontend.


class ErrorExample(BaseModel):
    original: str
    corrected: str
    explanation: str = ""


class ErrorProfileEntry(BaseModel):
    error_type: str
    occurrences: int = 1
    examples: List[ErrorExample] = Field(default_factory=list)


class BatchGenerateRequest(BaseModel):
    user_id: str = "anonymous"
    language_level: str = "B1"  # CEFR A1..C2
    error_profile: List[ErrorProfileEntry]
    max_questions: int = Field(default=15, ge=1, le=50)
    model: Optional[str] = None


class TestQuestion(BaseModel):
    """Flat MCQ shape consumed directly by the SAYIT tests UI."""
    id: str
    error_type: str
    text: str
    options: List[str]
    correct_answer: str
    explanation: str


class BatchGenerateResponse(BaseModel):
    status: Literal["ok", "partial"] = "ok"
    model_used: str
    generation_time_ms: int
    questions: List[TestQuestion]
    # Full exercises (8-14 items, richer types) — kept optional for a future,
    # extended SAYIT UI; current frontend ignores this field.
    exercises_full: List[Dict[str, Any]] = Field(default_factory=list)
    fallback_categories: List[str] = Field(default_factory=list)


# ---------- /api/v1/models ----------

class ModelMetrics(BaseModel):
    tr_composite: Optional[float] = None
    bertscore_f1: Optional[float] = None
    perplexity: Optional[float] = None


class ModelInfo(BaseModel):
    status: Literal["loaded", "available"]
    base_model: str
    adapter: str
    vram_mb: Optional[int] = None
    metrics: Optional[ModelMetrics] = None


class ModelsResponse(BaseModel):
    service: str
    version: str
    models: Dict[str, ModelInfo]
    supported_task_types: List[str]
    fallback_methods: List[str]


# ---------- Error response ----------

class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error: str
    detail: Optional[str] = None
