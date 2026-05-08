"""
OpenAI Chat Completions shim for the SAYIT Exercise Generator.

WHY THIS EXISTS
---------------
NVIDIA AIPerf and similar industrial LLM benchmarks (LLMPerf, GuideLLM,
vLLM benchmark_serving.py) require an OpenAI-compatible /v1/chat/completions
endpoint. The SAYIT generator exposes a domain-specific /api/v1/generate
endpoint with a custom JSON schema. This shim translates between them so
AIPerf can drive load against the real backend, providing an INDEPENDENT
validation of the latency/throughput numbers measured by Locust in §3.4.

METHODOLOGY NOTE FOR §3.4
-------------------------
The SAYIT backend does NOT stream tokens — it returns the full exercise
JSON in a single response. Consequently, TTFT and ITL metrics produced
by AIPerf are NOT meaningful for the underlying system. The meaningful
metrics from an AIPerf run against this shim are:

  - Request Latency        — equivalent to Locust's "Total Average RT"
  - Request Throughput     — equivalent to Locust's "Requests/s"
  - Output Token Throughput — bonus metric AIPerf computes via its own
                              tokenizer, useful for §3.5

When stream=True is requested, the shim emulates SSE streaming by sending
the full text back in one chunk after backend inference completes. This
preserves protocol compatibility with AIPerf but the resulting TTFT/ITL
should NOT be reported as system metrics.

USAGE
-----
  pip install fastapi uvicorn httpx
  BACKEND_URL=http://localhost:8000 python shim.py
  # then in another terminal:
  curl http://localhost:8001/v1/models
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SHIM_HOST = os.getenv("SHIM_HOST", "0.0.0.0")
SHIM_PORT = int(os.getenv("SHIM_PORT", "8001"))

# Realistic backend payloads. AIPerf sends synthetic prompts that have no
# meaning for our domain, so the shim ignores their content and instead
# rotates through these fixtures — every backend call is well-formed.
DATA_DIR = Path(__file__).parent.parent / "load" / "data"

with open(DATA_DIR / "single_payloads.json", encoding="utf-8") as f:
    BACKEND_PAYLOADS: list[dict] = json.load(f)

DEFAULT_MODEL = "qwen2.5-3b"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SAYIT OpenAI Shim",
    description="Translates OpenAI Chat Completions to /api/v1/generate.",
    version="1.0.0",
)

_client: httpx.AsyncClient  # populated in startup hook


@app.on_event("startup")
async def _startup() -> None:
    global _client
    _client = httpx.AsyncClient(base_url=BACKEND_URL, timeout=120.0)


@app.on_event("shutdown")
async def _shutdown() -> None:
    await _client.aclose()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Proxies the backend /health for AIPerf's readiness check."""
    try:
        r = await _client.get("/health")
        return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Backend unreachable: {e!r}")


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing built from /api/v1/models."""
    try:
        r = await _client.get("/api/v1/models")
        backend_models = r.json().get("models", {DEFAULT_MODEL: {"status": "loaded"}})
    except httpx.HTTPError:
        backend_models = {DEFAULT_MODEL: {"status": "loaded"}}

    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": now, "owned_by": "sayit"}
            for mid in backend_models
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Translate OpenAI request -> backend, return OpenAI response."""
    body = await request.json()
    model = body.get("model", DEFAULT_MODEL)
    stream = bool(body.get("stream", False))

    # AIPerf's prompt content is synthetic; we use a real fixture as the
    # backend payload so the call is well-formed and produces deterministic
    # backend latency. The mapping is intentional: we are benchmarking the
    # backend's actual inference path, not the prompt parser.
    payload = dict(random.choice(BACKEND_PAYLOADS), model=model)

    try:
        backend_response = await _client.post("/api/v1/generate", json=payload)
        backend_response.raise_for_status()
        backend_data = backend_response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Backend error: {exc!r}")

    completion_text = json.dumps(
        backend_data.get("exercise", backend_data), ensure_ascii=False
    )

    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if not stream:
        return JSONResponse(
            {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": completion_text},
                        "finish_reason": "stop",
                    }
                ],
                # Token counts are placeholders. AIPerf computes its own
                # token counts from the supplied --tokenizer, ignoring these.
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

    # ---- streaming path -----------------------------------------------------
    # Emit the full text as one chunk. Honest about the limitation: backend
    # inference has already completed before the first SSE event leaves the
    # shim. AIPerf will record a TTFT close to total request latency and an
    # ITL close to zero. Keep this path for protocol compatibility only.
    async def sse() -> "asyncio.AsyncIterator[str]":
        first = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(first)}\n\n"

        body_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": completion_text}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(body_chunk, ensure_ascii=False)}\n\n"

        final = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"SAYIT OpenAI shim → backend at {BACKEND_URL}")
    print(f"Listening on http://{SHIM_HOST}:{SHIM_PORT}")
    uvicorn.run(app, host=SHIM_HOST, port=SHIM_PORT, log_level="info")
