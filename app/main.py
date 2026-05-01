from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import health_router, router as api_router
from app.api.schemas import ErrorResponse
from app.config import get_settings
from app.core.generator import LLMGenerator
from app.core.model_manager import init_manager


def _configure_logging(level: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)
    log = structlog.get_logger("app.lifespan")

    # HF cache must be set before importing/loading models that use it.
    os.environ.setdefault("HF_HOME", str(settings.hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(settings.hf_home))

    app.state.started_at = time.time()

    manager = init_manager()
    app.state.model_manager = manager

    log.info(
        "app.startup",
        default_model=settings.model_name,
        adapter_dir=str(settings.adapter_dir),
        device=manager.device,
    )

    try:
        manager.preload_default()
    except Exception as e:  # noqa: BLE001
        log.error("app.preload_failed", error=str(e))

    app.state.llm_generator = LLMGenerator(
        manager, max_new_tokens=settings.max_new_tokens
    )

    yield

    log.info("app.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="SAYIT Exercise Generator",
        description="LLM-driven generator of personalized English grammar exercises.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    app.include_router(health_router)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        structlog.get_logger("app.error").exception(
            "request.unhandled_exception", path=request.url.path
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_server_error", detail=str(exc)
            ).model_dump(),
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload=False)
