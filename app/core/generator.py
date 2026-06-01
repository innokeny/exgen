"""LLM-based exercise generation.

Wraps the trained Qwen2.5-3B + LoRA model and produces raw model output.
Parsing/validation lives in `postprocessor.py`.

Concurrency: this generator serialises access to the underlying model with
a threading.Lock. The README claims "single-tenant inference" — without
this lock, FastAPI's default thread pool happily admits multiple parallel
requests that race on the same CUDA context, producing device-side asserts
under load. The lock makes the README's promise actually hold.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import structlog
import torch

from app.core.model_manager import LoadedModel, ModelManager
from app.prompts.templates import build_messages

log = structlog.get_logger(__name__)


class LLMGenerator:
    def __init__(self, manager: ModelManager, max_new_tokens: int = 2048):
        self._manager = manager
        self._max_new_tokens = max_new_tokens
        # Single-tenant inference: serialize all model.generate calls. FastAPI
        # dispatches sync endpoints onto a threadpool, so without this lock
        # concurrent requests share KV-cache/attention buffers on the same
        # CUDA model and trigger device-side asserts.
        self._lock = threading.Lock()

    def generate_raw(
        self,
        *,
        user_id: str,
        message_content: str,
        grammar_error: str,
        explanation: str,
        llm_confidence: float,
        task_type: str,
        model_key: Optional[str] = None,
    ) -> tuple[str, str, int]:
        """Run inference. Returns (raw_text, model_used, generation_time_ms)."""
        loaded = self._manager.get(model_key)
        messages = build_messages(
            user_id=user_id,
            message_content=message_content,
            grammar_error=grammar_error,
            explanation=explanation,
            llm_confidence=llm_confidence,
            task_type=task_type,
        )

        t0 = time.perf_counter()
        wait_t0 = time.perf_counter()
        with self._lock:
            wait_ms = int((time.perf_counter() - wait_t0) * 1000)
            try:
                text, n_in, n_out = self._run(loaded, messages)
            except torch.cuda.OutOfMemoryError:
                # Clear cache so the next request has a chance to succeed,
                # but the current request is lost.
                torch.cuda.empty_cache()
                log.error("llm.generate.oom", user_id=user_id)
                raise
            except RuntimeError as e:
                # CUDA device-side asserts manifest as RuntimeError. The
                # CUDA context is now poisoned for this process — the only
                # recovery is a restart by the orchestrator. Surface it
                # clearly in the log.
                log.error(
                    "llm.generate.cuda_error",
                    user_id=user_id,
                    error=repr(e),
                    note="CUDA context likely poisoned; service restart required",
                )
                raise
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        log.info(
            "llm.generate.done",
            model_key=loaded.key,
            elapsed_ms=elapsed_ms,
            wait_ms=wait_ms,
            input_tokens=n_in,
            output_tokens=n_out,
            output_chars=len(text),
        )
        return text, loaded.key, elapsed_ms

    def _run(
        self, loaded: LoadedModel, messages: list[dict]
    ) -> tuple[str, int, int]:
        tokenizer = loaded.tokenizer
        model = loaded.model

        # Qwen tokenizers occasionally load without pad_token_id; fall back
        # to eos_token_id so generate() does not blow up at the C++ layer.
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
            if pad_token_id is None:
                raise RuntimeError(
                    "tokenizer has neither pad_token_id nor eos_token_id"
                )

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        n_in = int(inputs["input_ids"].shape[1])
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with self._lock, torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][n_in:]
        n_out = int(new_tokens.shape[0])
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, n_in, n_out