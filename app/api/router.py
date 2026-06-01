from __future__ import annotations

import time
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.schemas import (
    BatchGenerateRequest,
    BatchGenerateResponse,
    ErrorResponse,
    Exercise,
    GenerateRequest,
    GenerateResponse,
    GPUInfo,
    HealthResponse,
    ModelInfo,
    ModelMetrics,
    ModelsResponse,
    TemplateRequest,
    TemplateResponse,
)
from app.config import (
    FALLBACK_METHODS,
    MODEL_METRICS,
    SERVICE_VERSION,
    SUPPORTED_MODELS,
    SUPPORTED_TASK_TYPES,
    Settings,
    get_settings,
)
from app.core.batch import generate_batch
from app.core.generator import LLMGenerator
from app.core.model_manager import ModelManager, get_manager, gpu_info
from app.core.postprocessor import parse_exercise
from app.core.template_engine import (
    build_template_exercise,
    fill_in_blanks,
)

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1")


def _get_generator(request: Request) -> LLMGenerator:
    return request.app.state.llm_generator


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def generate(
    payload: GenerateRequest,
    generator: LLMGenerator = Depends(_get_generator),
    settings: Settings = Depends(get_settings),
) -> GenerateResponse:
    if payload.model is not None and payload.model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model '{payload.model}'. Allowed: {list(SUPPORTED_MODELS)}",
        )

    try:
        raw_text, model_used, gen_ms = generator.generate_raw(
            user_id=payload.user_id,
            message_content=payload.message_content,
            grammar_error=payload.grammar_error,
            explanation=payload.explanation,
            llm_confidence=payload.llm_confidence,
            task_type=payload.task_type,
            model_key=payload.model,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001
        log.exception("llm.generate.failed")
        raise HTTPException(status_code=500, detail=f"generation_failed: {e}")

    parsed = parse_exercise(raw_text)
    if parsed.ok:
        return GenerateResponse(
            status="ok",
            model_used=model_used,
            generation_time_ms=gen_ms,
            exercise=parsed.exercise,
        )

    log.warning(
        "llm.parse.fallback",
        reason=parsed.error,
        raw_preview=raw_text[:200],
    )
    fb = fill_in_blanks(
        source_sentence=payload.message_content,
        corrected_sentence=payload.message_content,
        error_type=payload.grammar_error,
    )
    return GenerateResponse(
        status="fallback",
        model_used=model_used,
        generation_time_ms=gen_ms,
        exercise=Exercise.model_validate(fb),
        fallback_reason=parsed.error or "unknown",
    )


@router.post(
    "/generate/batch",
    response_model=BatchGenerateResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="SAYIT integration: error profile → flat MCQ questions",
)
def generate_batch_endpoint(
    payload: BatchGenerateRequest,
    generator: LLMGenerator = Depends(_get_generator),
    settings: Settings = Depends(get_settings),
) -> BatchGenerateResponse:
    if payload.model is not None and payload.model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{payload.model}'. Allowed: {list(SUPPORTED_MODELS)}",
        )
    if not payload.error_profile:
        raise HTTPException(status_code=400, detail="error_profile must be non-empty")

    t0 = time.perf_counter()
    try:
        result = generate_batch(
            generator=generator,
            user_id=payload.user_id,
            error_profile=payload.error_profile,
            max_questions=payload.max_questions,
            model_key=payload.model,
            default_model=settings.model_name,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001
        log.exception("batch.generate.failed")
        raise HTTPException(status_code=500, detail=f"batch_generation_failed: {e}")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    if not result.questions:
        raise HTTPException(
            status_code=400,
            detail="No questions could be generated — error_profile contains no usable examples.",
        )

    status_str = "partial" if result.fallback_categories else "ok"
    return BatchGenerateResponse(
        status=status_str,
        model_used=result.model_used,
        generation_time_ms=elapsed_ms,
        questions=result.questions,
        exercises_full=result.exercises_full,
        fallback_categories=result.fallback_categories,
    )


@router.post(
    "/generate/template",
    response_model=TemplateResponse,
    responses={400: {"model": ErrorResponse}},
)
def generate_template(payload: TemplateRequest) -> TemplateResponse:
    t0 = time.perf_counter()
    try:
        ex = build_template_exercise(
            method=payload.method,
            source_sentence=payload.source_sentence,
            corrected_sentence=payload.corrected_sentence,
            error_type=payload.error_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return TemplateResponse(
        status="ok",
        method=payload.method,
        generation_time_ms=elapsed_ms,
        exercise=Exercise.model_validate(ex),
    )


@router.get("/models", response_model=ModelsResponse)
def list_models(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ModelsResponse:
    manager: Optional[ModelManager] = getattr(request.app.state, "model_manager", None)
    loaded_keys = set(manager.loaded_keys) if manager else set()
    info = gpu_info()
    vram_used = info.get("vram_used_mb") if info.get("available") else None

    models: dict[str, ModelInfo] = {}
    for key, base_id in SUPPORTED_MODELS.items():
        metrics_dict = MODEL_METRICS.get(key)
        models[key] = ModelInfo(
            status="loaded" if key in loaded_keys else "available",
            base_model=base_id,
            adapter=str(settings.adapter_path_for(key)),
            vram_mb=vram_used if key in loaded_keys else None,
            metrics=ModelMetrics(**metrics_dict) if metrics_dict else None,
        )

    return ModelsResponse(
        service=settings.service_name,
        version=SERVICE_VERSION,
        models=models,
        supported_task_types=SUPPORTED_TASK_TYPES,
        fallback_methods=FALLBACK_METHODS,
    )


health_router = APIRouter()


@health_router.get("/health", response_model=HealthResponse)
def health(request: Request, settings: Settings = Depends(get_settings)) -> HealthResponse:
    started_at: float = request.app.state.started_at
    manager: Optional[ModelManager] = getattr(request.app.state, "model_manager", None)
    info = gpu_info()
    loaded = manager.loaded_keys if manager else []
    return HealthResponse(
        status="ok" if loaded else "loading",
        uptime_s=time.time() - started_at,
        default_model=settings.model_name,
        loaded_models=loaded,
        gpu=GPUInfo(**info),
    )
