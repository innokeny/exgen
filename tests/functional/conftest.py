from __future__ import annotations

import time
from typing import Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.router import health_router, router as api_router
from app.api.schemas import ErrorResponse


class _FakeManager:
    loaded_keys = ["qwen2.5-3b"]
    device = "cpu"


class _FakeGenerator:

    def __init__(self, raw_output=None):
        self.raw_output = raw_output if raw_output is not None else _VALID_EXERCISE_JSON
        self.calls: list[dict] = []

    def generate_raw(self, **kwargs) -> tuple[str, str, int]:
        self.calls.append(kwargs)
        out = self.raw_output(**kwargs) if callable(self.raw_output) else self.raw_output
        return out, kwargs.get("model_key") or "qwen2.5-3b", 42


_VALID_EXERCISE_JSON = """
{
  "target_error_category": "Preposition",
  "corrected_sentence": "It has a high-density population because of its small territory.",
  "task": {
    "type": "grammar_choice",
    "instruction_en": "Choose the correct preposition.",
    "content_en": {
      "context_text": null,
      "items": [
        {
          "question_en": "It has a high-density population because ____ its small territory.",
          "options_en": ["of", "for", "to", "with"],
          "student_answer_en": "of"
        }
      ],
      "word_bank": null
    }
  }
}
"""


def _build_app(generator: _FakeGenerator) -> FastAPI:
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi import Request
    from fastapi.responses import JSONResponse
    import structlog

    app = FastAPI(title="test")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.started_at = time.time()
    app.state.model_manager = _FakeManager()
    app.state.llm_generator = generator
    app.include_router(api_router)
    app.include_router(health_router)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # noqa: ARG001
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="internal_server_error", detail=str(exc)).model_dump(),
        )

    return app


@pytest.fixture
def fake_generator() -> _FakeGenerator:
    return _FakeGenerator()


@pytest.fixture
def client(fake_generator: _FakeGenerator) -> TestClient:
    app = _build_app(fake_generator)
    return TestClient(app)


@pytest.fixture
def make_client():
    def _factory(raw_output: str) -> TestClient:
        gen = _FakeGenerator(raw_output=raw_output)
        return TestClient(_build_app(gen))

    return _factory
