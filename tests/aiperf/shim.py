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


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SHIM_HOST = os.getenv("SHIM_HOST", "0.0.0.0")
SHIM_PORT = int(os.getenv("SHIM_PORT", "8001"))

DATA_DIR = Path(__file__).parent.parent / "load" / "data"

with open(DATA_DIR / "single_payloads.json", encoding="utf-8") as f:
    BACKEND_PAYLOADS: list[dict] = json.load(f)

DEFAULT_MODEL = "qwen2.5-3b"


app = FastAPI(
    title="SAYIT OpenAI Shim",
    description="Translates OpenAI Chat Completions to /api/v1/generate.",
    version="1.0.0",
)

_client: httpx.AsyncClient


@app.on_event("startup")
async def _startup() -> None:
    global _client
    _client = httpx.AsyncClient(base_url=BACKEND_URL, timeout=120.0)


@app.on_event("shutdown")
async def _shutdown() -> None:
    await _client.aclose()


@app.get("/health")
async def health():
    try:
        r = await _client.get("/health")
        return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Backend unreachable: {e!r}")


@app.get("/v1/models")
async def list_models():
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
    body = await request.json()
    model = body.get("model", DEFAULT_MODEL)
    stream = bool(body.get("stream", False))

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
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

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


if __name__ == "__main__":
    print(f"SAYIT OpenAI shim → backend at {BACKEND_URL}")
    print(f"Listening on http://{SHIM_HOST}:{SHIM_PORT}")
    uvicorn.run(app, host=SHIM_HOST, port=SHIM_PORT, log_level="info")
